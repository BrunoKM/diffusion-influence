import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import spearmanr

from diffusion_influence.config_schemas import DatasetType, LDSScoreConfig


def searchunsorted(
    arr: np.ndarray, x: int | np.ndarray, sorter: Optional[np.ndarray] = None
) -> np.ndarray:
    arr_sorter = np.argsort(arr) if sorter is None else sorter
    sorted_arr = arr[arr_sorter]
    idxs = np.searchsorted(sorted_arr, x)
    # Permute the idxs to the original order
    return arr_sorter[idxs]


def load_measurements_and_influences_from_config(config: LDSScoreConfig):
    retrained_model_idxs_paths = [
        config.retrained_model_idxs_dir / f"sub_idx_{model_idx}.csv"
        for model_idx in range(config.num_retrained_models)
    ]
    # Set the paths to the measurements of the retrained models
    retrained_model_measurements_paths: list[list[Path]] = [
        [
            config.measurement_dir / f"subidx{model_idx}-{seed}/measurements.npy"
            for seed in range(config.num_seeds)
        ]
        for model_idx in range(config.num_retrained_models)
    ]

    match config.dataset:
        case DatasetType.cifar2 | DatasetType.cifar2deq:
            if config.train_model_idxs_path is None:
                raise ValueError("train_model_idxs_path must be specified for CIFAR2.")
            # For CIFAR2, also load the indices of the data used to train the model
            train_model_idxs = np.loadtxt(config.train_model_idxs_path, dtype=int)
        case DatasetType.cifar10 | DatasetType.cifar10deq:
            if config.train_model_idxs_path is None:
                # Model was trained on full CIFAR10 dataset
                train_model_idxs = np.arange(50_000, dtype=int)
            else:
                logging.warning("Using a subset of the CIFAR10 dataset.")
                train_model_idxs = np.loadtxt(config.train_model_idxs_path, dtype=int)
        case DatasetType.artbench:
            if config.train_model_idxs_path is None:
                train_model_idxs = np.arange(50_000, dtype=int)
            else:
                logging.warning("Using a subset of artbench dataset.")
                train_model_idxs = np.loadtxt(config.train_model_idxs_path, dtype=int)
        case _:
            raise ValueError("Only CIFAR2 and CIFAR10 are supported.")

    # Check if all the files exist
    for filepath in [
        item for sublist in retrained_model_measurements_paths for item in sublist
    ] + [config.influence_path]:
        if not filepath.exists():
            logging.info(f"Missing {filepath}")

    # Load all the data
    retrained_models_idxs = [
        np.loadtxt(filepath, dtype=int) for filepath in retrained_model_idxs_paths
    ]
    retrained_models_measurements = [
        [np.load(filepath) for filepath in seed_measurements]
        for seed_measurements in retrained_model_measurements_paths
    ]
    influence_scores: np.ndarray = np.load(
        config.influence_path
    )  # [num_queries, num_train_data]
    influence_scores = influence_scores.astype(
        np.float64
    )  # Always use float64 for numerical stability
    if np.all(influence_scores[0] == influence_scores[1]):
        raise ValueError(
            f"Influence scores are the same for all queries: Path {config.influence_path}"
        )

    # Check if all subsets of the data used for model retraining are of equal size
    # and if they are all subsets of the data used to train the original model
    for retr_model_idx in retrained_models_idxs:
        assert len(retr_model_idx) == len(retrained_models_idxs[0]), (
            f"Length mismatch: {len(retr_model_idx)} != {len(train_model_idxs)}"
        )
        assert set(retr_model_idx).issubset(set(train_model_idxs))

    return (
        train_model_idxs,
        retrained_models_idxs,
        retrained_models_measurements,
        influence_scores,
    )


def compute_influence_estimated_scores(
    influence_scores: np.ndarray,  # [num_queries, num_train_data]
    train_model_idxs: np.ndarray,  # [num_train_data]
    retrained_models_idxs: list[np.ndarray],  # [num_retrained_models, <num_train_data]
    sum_over_not_in_retrained_model_idxs: bool = False,
) -> np.ndarray:  # [num_queries]
    influence_estimated_scores = np.stack(
        [
            np.sum(
                influence_scores[
                    :,
                    ~np.isin(
                        train_model_idxs,
                        retr_model_idx,
                    )
                    if sum_over_not_in_retrained_model_idxs
                    else np.isin(
                        train_model_idxs,
                        retr_model_idx,
                    ),
                ],
                axis=1,
            )
            for retr_model_idx in retrained_models_idxs
        ]
    )  # [num_retrained_models, num_queries]
    return influence_estimated_scores  # [num_retrained_models, num_queries]


def compute_rank_correlation(
    influence_scores: np.ndarray,  # [num_queries, num_train_data]
    train_model_idxs: np.ndarray,  # [num_train_data]
    retrained_models_idxs: list[np.ndarray],  # [num_retrained_models, <num_train_data]
    retrained_models_measurements: list[
        list[np.ndarray]
    ],  # [num_retrained_models, num_seeds, num_queries]
) -> np.ndarray:  # [num_queries]
    # Compute the influence-estimated score for each retrain indices:
    influence_estimated_scores = compute_influence_estimated_scores(
        influence_scores=influence_scores,
        train_model_idxs=train_model_idxs,
        retrained_models_idxs=retrained_models_idxs,
        sum_over_not_in_retrained_model_idxs=True,
    )

    return compute_rank_correlation_from_estimated_scores(
        estimated_scores=influence_estimated_scores,
        retrained_models_measurements=retrained_models_measurements,
    )  # [num_queries]


def compute_rank_correlation_from_estimated_scores(
    estimated_scores: np.ndarray,  # [num_queries, num_train_data]
    retrained_models_measurements: list[
        list[np.ndarray]
    ],  # [num_retrained_models, num_seeds, num_queries]
) -> np.ndarray:
    # Average measurements from retrained models
    retrained_model_averaged_measurements = np.stack(
        [
            np.mean(np.stack(seeds_model_measurements, axis=0), axis=0)
            for seeds_model_measurements in retrained_models_measurements
        ]
    )  # [num_retrained_models, num_queries]
    logging.info(
        "Retrained model averaged measurements shape: "
        f"{retrained_model_averaged_measurements.shape}"
    )

    # Check if the shapes match
    if estimated_scores.shape != retrained_model_averaged_measurements.shape:
        raise ValueError(
            f"Shape mismatch: {estimated_scores.shape} "
            f"!= {retrained_model_averaged_measurements.shape}"
        )

    # Compute the Spearman rank correlations
    rank_correlations = np.zeros(estimated_scores.shape[1])
    for validation_sample_idx in range(estimated_scores.shape[1]):
        rank_correlations[validation_sample_idx] = spearmanr(
            retrained_model_averaged_measurements[:, validation_sample_idx],
            estimated_scores[:, validation_sample_idx],
        ).statistic
    return rank_correlations  # [num_queries]


def compute_lds_score_from_config(config: LDSScoreConfig):
    # Set the paths to the indices of the data used to train the retrained models
    (
        train_model_idxs,
        retrained_models_idxs,
        retrained_models_measurements,
        influence_scores,
    ) = load_measurements_and_influences_from_config(config)
    logging.info(f"Influence scores shape: {influence_scores.shape}")

    if np.any(np.isnan(influence_scores)):
        raise ValueError("Influence scores contain NaNs")

    influence_rank_correlations = compute_rank_correlation(
        influence_scores=influence_scores,
        train_model_idxs=train_model_idxs,
        retrained_models_idxs=retrained_models_idxs,
        retrained_models_measurements=retrained_models_measurements,
    )
    # Also compute the correlation across seeds:
    retrained_models_measurements_combined = np.stack(
        [
            np.stack(seeds_model_measurements, axis=0)
            for seeds_model_measurements in retrained_models_measurements
        ],
        axis=0,
    )  # [num_retrained_models, num_seeds, num_queries]

    rank_correlations_across_seeds: list[float] = []
    rank_correlations_across_averaged_seeds: list[float] = []
    for seed in range(num_seeds := retrained_models_measurements_combined.shape[1]):
        for other_seed in range(seed + 1, num_seeds):
            rank_correlations = []
            for validation_sample_idx in range(
                (
                    num_samples_to_score
                    := retrained_models_measurements_combined.shape[2]
                )
            ):
                # Spearman Rank correlation:
                rank_correlations += [
                    spearmanr(
                        retrained_models_measurements_combined[
                            :, seed, validation_sample_idx
                        ],
                        retrained_models_measurements_combined[
                            :, other_seed, validation_sample_idx
                        ],
                    ).statistic
                ]
            rank_correlations = np.array(rank_correlations)
            rank_correlations_across_seeds += [rank_correlations.mean()]
        # Also compute rank correlation to average over all other seeds:
        rank_correlations = []
        for validation_sample_idx in range(
            (num_samples_to_score := retrained_models_measurements_combined.shape[2])
        ):
            # Spearman Rank correlation:
            rank_correlations += [
                spearmanr(
                    retrained_models_measurements_combined[
                        :, seed, validation_sample_idx
                    ],
                    retrained_models_measurements_combined[
                        :,
                        [s for s in range(num_seeds) if s != seed],
                        validation_sample_idx,
                    ].mean(axis=1),
                ).statistic
            ]
        rank_correlations = np.array(rank_correlations)
        rank_correlations_across_averaged_seeds += [rank_correlations.mean()]

    rank_correlations_across_seeds_array = np.array(rank_correlations_across_seeds)
    rank_correlations_across_averaged_seeds_array = np.array(
        rank_correlations_across_averaged_seeds
    )
    return (
        influence_rank_correlations,
        rank_correlations_across_seeds_array,
        rank_correlations_across_averaged_seeds_array,
    )
