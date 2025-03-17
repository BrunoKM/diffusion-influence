import dataclasses
import logging
import math
import os
from pathlib import Path
from typing import Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import seaborn as sns
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from scipy.stats import pearsonr

from diffusion_influence.config_schemas import LDSScoreConfig
from diffusion_influence.lds import (
    compute_influence_estimated_scores,
    load_measurements_and_influences_from_config,
)


@dataclasses.dataclass
class RetrainPlotConfig(LDSScoreConfig):
    base_measurements: Optional[Path] = None
    """If given, influence estimated as influence + base_measurements"""
    num_query_to_plot: int = 16


# Register the structured dataclass config schema with Hydra
# (for type-checking, autocomplete, and validation)
cs = ConfigStore.instance()
cs.store(name="base_schema", node=RetrainPlotConfig)


@hydra.main(version_base=None, config_path="configs")
def hydra_main(omegaconf_config: omegaconf.DictConfig):
    # Parse the config into pydantic dataclasses (for validation)
    config: LDSScoreConfig = omegaconf.OmegaConf.to_object(omegaconf_config)  # type: ignore
    logging.info(f"Current working directory: {os.getcwd()}")
    main(config)


COLORS = [
    (204 / 255, 57 / 255, 42 / 255),  # palette1
    (79 / 255, 155 / 255, 143 / 255),  # palette2
    (44 / 255, 97 / 255, 194 / 255),  # palette3
    (217 / 255, 116 / 255, 89 / 255),  # palette4
    (228 / 255, 197 / 255, 119 / 255),  # palette5
    (155 / 255, 106 / 255, 145 / 255),  # palette6 converted from "#9B6A91"
    (51 / 255, 110 / 255, 49 / 255),  # palette7 converted from "#336E31"
    (198 / 255, 5 / 255, 79 / 255),  # palette8 converted from "#C6054F"
]


def main(config: RetrainPlotConfig):
    # ------------------------ Plotting setup ------------------------
    colors = COLORS

    # ICLR 2024 width: 397.48499pt
    text_width_pt = 397.48499  # in pt
    text_width = text_width_pt / 72.27  # in inches

    fs_m1 = 7  # for figure ticks
    fs = 9  # for regular figure text
    fs_p1 = 10  # figure titles

    axes_lw = 0.7

    import matplotlib
    # matplotlib.use('pgf')

    matplotlib.rc("font", size=fs)  # controls default text sizes
    matplotlib.rc("axes", titlesize=fs)  # fontsize of the axes title
    matplotlib.rc("axes", labelsize=fs)  # fontsize of the x and y labels
    matplotlib.rc("axes", linewidth=axes_lw)  # fontsize of the x and y labels
    matplotlib.rc("xtick", labelsize=fs_m1)  # fontsize of the tick labels
    matplotlib.rc("ytick", labelsize=fs_m1)  # fontsize of the tick labels
    matplotlib.rc("legend", fontsize=fs_m1 - 1)  # legend fontsize
    matplotlib.rc("figure", titlesize=fs_p1)  # fontsize of the figure title

    plt.rcParams["savefig.facecolor"] = "white"

    # matplotlib.rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
    matplotlib.rc("font", **{"family": "serif"})

    # Turn on if you've got TeX installed
    matplotlib.rc("text", usetex=True)

    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsfonts}"})
    # ^^^^^^^^^^^^^^^^^^^^^ Plotting setup ^^^^^^^^^^^^^^^^^^^^^

    # Set the paths to the indices of the data used to train the retrained models
    (
        train_model_idxs,
        retrained_models_idxs,
        retrained_models_measurements,
        influence_scores,
    ) = load_measurements_and_influences_from_config(config)
    logging.info(f"Influence scores shape: {influence_scores.shape}")
    # Make retrain_models_measurements into array
    retrained_models_measurements = np.stack(
        [
            np.stack(seed_measurements, axis=0)
            for seed_measurements in retrained_models_measurements
        ],
        axis=0,
    )  # [num_retrained_models, num_seeds, num_queries]
    # Average over seeds:
    retrained_models_measurements_seed_mean = np.mean(
        retrained_models_measurements, axis=1
    )  # [num_retrained_models, num_queries]

    num_train_data = influence_scores.shape[1]
    eps = 1 / num_train_data
    influence_scores *= eps

    if np.any(np.isnan(influence_scores)):
        raise ValueError("Influence scores contain NaNs")

    influence_estimated_scores = compute_influence_estimated_scores(
        influence_scores=influence_scores,
        train_model_idxs=train_model_idxs,
        retrained_models_idxs=retrained_models_idxs,
        sum_over_not_in_retrained_model_idxs=True,
    )  # [num_retrained_models, num_queries]
    if config.base_measurements is not None:
        base_measurements = np.load(config.base_measurements)
        assert base_measurements.ndim == 1
        if base_measurements.shape[0] != influence_estimated_scores.shape[1]:
            raise ValueError(
                f"Base measurements shape mismatch: {base_measurements.shape} != {influence_estimated_scores.shape}"
            )
        influence_estimated_scores += base_measurements[None, :]
    logging.info(f"Influence estimated score shape: {influence_estimated_scores.shape}")

    # Report the correlation between the estimated scores and the measurements:
    pearson_correlations_to_mean = [
        pearsonr(
            retrained_models_measurements_seed_mean[:, query_idx],
            influence_estimated_scores[:, query_idx],
        ).correlation
        for query_idx in range(retrained_models_measurements.shape[2])
    ]
    pearson_correlations = [
        [
            pearsonr(
                retrained_models_measurements[:, seed_idx, query_idx],
                influence_estimated_scores[:, query_idx],
            ).correlation
            for seed_idx in range(retrained_models_measurements.shape[1])
        ]
        for query_idx in range(retrained_models_measurements.shape[2])
    ]
    avg_pearson_correlation_to_mean = np.mean(pearson_correlations_to_mean)
    std_paarson_correlation_to_mean = np.std(pearson_correlations_to_mean)
    avg_pearson_correlation = np.mean(pearson_correlations)
    std_pearson_correlation = np.std(pearson_correlations)
    logging.info(
        f"Pearson correlation to ensemble mean: {avg_pearson_correlation_to_mean}"
    )
    logging.info(
        f"Pearson correlation to ensemble mean st.d.: {std_paarson_correlation_to_mean}"
    )
    logging.info(f"Per example avg. Pearson correlation: {avg_pearson_correlation}")
    logging.info(
        f"Per example avg. Pearson correlation st.d.: {std_pearson_correlation}"
    )
    if config.base_measurements:
        joint_pearson_correlations_to_mean = pearsonr(
            retrained_models_measurements_seed_mean.flatten(),
            influence_estimated_scores.flatten(),
        ).correlation
        joint_pearson_correlations = [
            pearsonr(
                retrained_models_measurements[:, seed_idx, :].flatten(),
                influence_estimated_scores.flatten(),
            ).correlation
            for seed_idx in range(retrained_models_measurements.shape[1])
        ]
        avg_joint_pearson_correlation = np.mean(joint_pearson_correlations)
        std_joint_pearson_correlation = np.std(joint_pearson_correlations)
        logging.info(
            f"Joint Pearson correlation to ensemble mean: {joint_pearson_correlations_to_mean}"
        )
        logging.info(f"Joint Pearson correlation: {avg_joint_pearson_correlation}")
        logging.info(
            f"Joint Pearson correlation st.d.: {std_joint_pearson_correlation}"
        )

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logging.info(f"Output directory: {output_dir}")
    # --- Make a joint plot with all queries:
    if config.base_measurements is not None:
        palette = sns.color_palette("deep", influence_estimated_scores.shape[1])
        plot_colors = [
            palette[query_idx]
            for retrain_idx in range(influence_estimated_scores.shape[0])
            for query_idx in range(influence_estimated_scores.shape[1])
        ]

        fig, ax = plt.subplots(figsize=(text_width, text_width))
        ax.errorbar(
            retrained_models_measurements_seed_mean.flatten(),
            influence_estimated_scores.flatten(),
            xerr=np.std(retrained_models_measurements[:, :, :], axis=1).flatten(),
            # xerr=np.stack((np.min(retrained_models_measurements[:, :, query_idx], axis=1), np.max(retrained_models_measurements[:, :, query_idx], axis=1)), axis=0),
            # yerr=0,
            fmt="o",
            color="none",
            # Set edge color
            ecolor=plot_colors,
            # Set marker color:
            markersize=1,
            alpha=0.5,
            zorder=0,
        )
        ax.set_xlabel("\\textbf{True} retrained model measurements")
        ax.set_ylabel("\\textbf{Predicted} retrained model measurement")
        fig.savefig(
            output_dir / "retrain_correlation_plot_joint.pdf",
            bbox_inches="tight",
        )

    # --- Make a plot of the measuremenets ---
    num_query_samples = config.num_query_to_plot
    num_seeds = retrained_models_measurements.shape[1]
    num_rows = math.ceil(math.sqrt(num_query_samples))
    num_cols = math.ceil(num_query_samples / num_rows)

    fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=(text_width, text_width)
    )
    for query_idx, ax in enumerate(axes.flat):
        if query_idx >= num_query_samples:
            ax.axis("off")
            continue

        # for retrain_idx in range(retrained_models_measurements.shape[0]):
        #     ax.plot(
        #         [
        #             retrained_models_measurements[retrain_idx, :, query_idx].min(axis=1),
        #             retrained_models_measurements[retrain_idx, :, query_idx].max(axis=1),
        #         ],
        #         [influence_estimated_scores[retrain_idx, query_idx]]*2,
        #         color=colors[0],
        #         linewidth=0.5,
        #     )
        ax.errorbar(
            retrained_models_measurements_seed_mean[:, query_idx],
            influence_estimated_scores[:, query_idx],
            xerr=np.std(retrained_models_measurements[:, :, query_idx], axis=1),
            # xerr=np.stack((np.min(retrained_models_measurements[:, :, query_idx], axis=1), np.max(retrained_models_measurements[:, :, query_idx], axis=1)), axis=0),
            # yerr=0,
            fmt="o",
            color=colors[0],
            markersize=2,
            zorder=0,
        )
        ax.scatter(
            retrained_models_measurements_seed_mean[:, query_idx],
            influence_estimated_scores[:, query_idx],
            color="black",
            s=1,
            zorder=1,
        )
        for seed_idx in range(num_seeds):
            ax.scatter(
                retrained_models_measurements[:, seed_idx, query_idx],
                influence_estimated_scores[:, query_idx],
                # marker=markerlist[seed_idx],
                # color=colors[seed_idx],
                color=colors[0],
                s=1.0,
                alpha=0.2,
            )
        # Get the average of the retrained models measurements and influences:
        avg_influence_for_query = np.mean(influence_estimated_scores[:, query_idx])
        min_infuence_for_query = np.min(influence_estimated_scores[:, query_idx])
        max_influence_for_query = np.max(influence_estimated_scores[:, query_idx])
        avg_measurement_for_query = np.mean(
            retrained_models_measurements[:, :, query_idx]
        )
        min_measurement_for_query = np.min(
            retrained_models_measurements[:, :, query_idx]
        )
        max_measurement_for_query = np.max(
            retrained_models_measurements[:, :, query_idx]
        )
        # Plot the line with gradient 1 going through the average point:
        x = np.linspace(min_measurement_for_query, max_measurement_for_query, 500)
        ax.plot(
            x,
            avg_influence_for_query - avg_measurement_for_query + x,
            color="gray",
            zorder=-1,
        )
        ax.set_xlim(min_measurement_for_query, max_measurement_for_query)
        ax.set_ylim(min_infuence_for_query, max_influence_for_query)
        # Remove all tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # Add text indicating correlation:
        pearson_correlation_to_mean = pearsonr(
            retrained_models_measurements_seed_mean[:, query_idx],
            influence_estimated_scores[:, query_idx],
        ).correlation
        avg_pearson_correlation = np.mean(
            [
                pearsonr(
                    retrained_models_measurements[:, seed_idx, query_idx],
                    influence_estimated_scores[:, query_idx],
                ).correlation
                for seed_idx in range(num_seeds)
            ]
        )
        ax.text(
            0.02,
            0.98,
            f"Avg. Corr.: {avg_pearson_correlation:.2f}\n"
            f"Corr. to avg.: {pearson_correlation_to_mean:.2f}",
            horizontalalignment="left",
            verticalalignment="top",
            # Add white background to text:
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="white"),
            # Set fontsize:
            fontsize=4.8,
            transform=ax.transAxes,
        )

    # Save the figure
    fig.savefig(
        output_dir / "retrain_correlation_plot.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    hydra_main()
