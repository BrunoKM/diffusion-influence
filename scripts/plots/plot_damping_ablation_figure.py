import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import hydra
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import scipy.stats
import seaborn as sns
import tqdm
from hydra.core.config_store import ConfigStore

from diffusion_influence.config_schemas import DatasetType, LDSScoreConfig
from diffusion_influence.lds import compute_lds_score_from_config

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


@dataclass
class PlotConfig:
    save_filename: str
    savedir: Path = Path(__file__).parent.parent.parent / "outputs" / "figures"
    dataset: str = "cifar2"
    damping_values_strings: list[str] = field(
        default_factory=lambda: [
            "1e-10",
            "1e-9",
            "1e-8",
            "1e-7",
            "1e-6",
            "1e-5",
            "1e-4",
            "1e-3",
            "1e-2",
            "1e-1",
            "1",
            "1e1",
        ]
    )

    measurement_dirs: list[str] = field(
        default_factory=lambda: [
            "/srv/shared/outputs/measurements/DatasetType.cifar2/ddpm_samples_loss_measurement",
        ]
        * 4
        + [
            "/srv/shared/outputs/measurements/DatasetType.cifar2/ddpm_samples_elbo_measurement",
        ]
        * 3
    )
    influence_path_format_strings: list[str] = field(
        default_factory=lambda: [
            "/srv/shared/outputs/DatasetType.cifar2/idx_train-0/influence_for_ddpm_samples_loss_1000kfac_100measurement_100loss__quantize_8bits__damping{}/influence_scores.npy"
        ]
        + [
            f"/srv/shared/outputs/DatasetType.cifar2/idx_train-0/influence_for_ddpm_samples_LOSS_1000kfac_{samples}measurement_{samples}loss__quantize_8bits__damping{{}}/influence_scores.npy"
            for samples in [250, 1000, 2500]
        ]
        + [
            f"/srv/shared/outputs/DatasetType.cifar2/idx_train-0/influence_for_ddpm_samples_ELBO_1000kfac_{samples}measurement_{samples}loss__quantize_8bits__damping{{}}/influence_scores.npy"
            for samples in [250, 1000, 2500]
        ]
    )
    legend_labels: list[str] = field(
        default_factory=lambda: list(
            map(
                lambda s: "\\texttt{" + s + "}",
                [
                    "Loss   100  samples",
                    "Loss   250  samples",
                    "Loss   1000 samples",
                    "Loss   2500 samples",
                    "ELBO   250  samples",
                    "ELBO   1000 samples",
                    "ELBO   2500 samples",
                ],
            )
        ),
    )
    legend_pre: Optional[str] = None
    color_groups: list[int] = field(default_factory=lambda: [0, 0, 0, 0, 1, 1, 1])
    linestyle_groups: Optional[list[int]] = None

    train_model_idxs_path: Optional[str] = (
        "/srv/shared/outputs/idxs/DatasetType.cifar2/idx_train.csv"
    )
    retrained_model_idxs_dir: str = (
        "/srv/shared/outputs/idxs/DatasetType.cifar2/retrain"
    )

    suptitle: str = "CIFAR-2 - K-FAC Influence "
    rescale_damping_by: float = 1.0
    num_retrained_models: int = 100
    num_seeds: int = 5
    """Num. of seeds for retraining models for LDS score computation."""


# Register the structured dataclass config schema with Hydra
# (for type-checking, autocomplete, and validation)
cs = ConfigStore.instance()
cs.store(name="base_schema", node=PlotConfig)


@hydra.main(version_base=None, config_path=None, config_name="base_schema")
def hydra_main(omegaconf_config: omegaconf.DictConfig):
    # Parse the config into pydantic dataclasses (for validation)
    config: PlotConfig = omegaconf.OmegaConf.to_object(omegaconf_config)  # type: ignore
    logging.info(f"Current working directory: {os.getcwd()}")
    main(config)


def main(config: PlotConfig):
    # ------------------------ Plotting setup ------------------------
    colors = COLORS
    sns.palplot(colors)

    golden_ratio = (1 + 5**0.5) / 2
    linestylelist = ["-", "--", "-.", ":"]

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
    # plt.rcParams.update({'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{lmodern}'})
    # matplotlib.rc('font', family='monospace', monospace=['Latin Modern Mono'])
    # ^^^^^^^^^^^^^^^^^^^^^ Plotting setup ^^^^^^^^^^^^^^^^^^^^^
    # Get colors for each result:
    num_shades_per_color_group = {
        group_idx: len([color for color in config.color_groups if color == group_idx])
        for group_idx in set(config.color_groups)
    }
    idx_within_color_group = [
        sum([group_idx_ == group_idx for group_idx_ in config.color_groups[:i]])
        for i, group_idx in enumerate(config.color_groups)
    ]
    colors = [
        sns.light_palette(
            COLORS[color_group_idx],
            n_colors=num_shades_per_color_group[color_group_idx] + 1,
        ).as_hex()[idx_within_color_group[i] + 1]
        for i, color_group_idx in enumerate(config.color_groups)
    ]
    assert (
        len(colors)
        == len(config.measurement_dirs)
        == len(config.influence_path_format_strings)
        == len(config.legend_labels)
    ), (
        f"Length mismatch: {len(colors)}, {len(config.measurement_dirs)}, {len(config.influence_path_format_strings)}, {len(config.legend_labels)}"
    )
    if config.linestyle_groups:
        assert len(config.linestyle_groups) == len(config.measurement_dirs), (
            f"Length mismatch: {len(config.linestyle_groups)}, {len(config.measurement_dirs)}"
        )

    # Damping values to iterate over
    damping_values = [float(damping) for damping in config.damping_values_strings]

    fig, ax = plt.subplots(figsize=(text_width, 0.5 * text_width / golden_ratio))
    for config_idx in range(len(config.measurement_dirs)):
        influence_path_format_string = config.influence_path_format_strings[config_idx]
        measurement_dir = config.measurement_dirs[config_idx]
        label = config.legend_labels[config_idx]
        label = label.replace(" ", "\ ")
        color = colors[config_idx]
        linestyle = (
            linestylelist[config.linestyle_groups[config_idx] % len(linestylelist)]
            if config.linestyle_groups
            else "-"
        )
        # List to hold the extracted numbers
        rank_correlations = []
        stde_values = []
        accross_seeds_correlations = []

        for damping in tqdm.tqdm(config.damping_values_strings):
            influence_path = Path(influence_path_format_string.format(damping))
            if not influence_path.parent.exists():
                # Directory not found
                raise FileNotFoundError(f"Directory {influence_path.parent} not found.")
            if not influence_path.exists():
                logging.info(
                    f"Missing influence scores for damping={damping} at {influence_path}. Treating as NaN."
                )
                rank_correlations.append(np.nan)
                stde_values.append(np.nan)
            elif np.isnan(np.load(influence_path).mean()):
                logging.info(
                    f"Some values are NaN in score for damping={damping} at {influence_path}. Treating as NaN."
                )
                rank_correlations.append(np.nan)
                stde_values.append(np.nan)
            else:
                lds_score_config = LDSScoreConfig(
                    dataset=DatasetType(config.dataset),
                    train_model_idxs_path=(
                        Path(config.train_model_idxs_path)
                        if config.train_model_idxs_path
                        else None
                    ),
                    retrained_model_idxs_dir=Path(config.retrained_model_idxs_dir),
                    measurement_dir=Path(measurement_dir),
                    influence_path=influence_path,
                    num_seeds=config.num_seeds,
                    num_retrained_models=config.num_retrained_models,
                )
                try:
                    (
                        rank_correlations_array,
                        rank_correlations_across_seeds,
                        rank_correlations_averaged_across_seeds,
                    ) = compute_lds_score_from_config(lds_score_config)

                    rank_correlation = rank_correlations_array.mean()
                    stde = scipy.stats.sem(rank_correlations_array)
                    accross_seeds_correlation = (
                        rank_correlations_averaged_across_seeds.mean()
                    )
                    rank_correlations.append(rank_correlation)
                    stde_values.append(stde)
                    accross_seeds_correlations.append(accross_seeds_correlation)
                except Exception as e:
                    logging.warn(f"Failed to extract values for damping={damping}")
                    logging.warn(e)
                    logging.warn("Treating as NaN.")
                    rank_correlations.append(np.nan)
                    stde_values.append(np.nan)
        if all([np.isnan(rank_correlation) for rank_correlation in rank_correlations]):
            raise ValueError("All rank correlations are NaN.")

        # Assert across seed correlations almost the same:
        for across_seed in accross_seeds_correlations[1:]:
            assert np.isclose(accross_seeds_correlations[0], across_seed, rtol=1e-8), (
                f"Across seed correlation values are not the same: {accross_seeds_correlations}"
            )

        if np.abs(
            min(
                rank_correlation
                for rank_correlation in rank_correlations
                if not np.isnan(rank_correlation)
            )
        ) > max(
            rank_correlation
            for rank_correlation in rank_correlations
            if not np.isnan(rank_correlation)
        ):
            # Sometimes we have inverted signs, so we invert them back :|
            # TODO: make signs consistent in influence and measurement calculations
            logging.warn("Max rank correlation negative. Inverting them.")
            rank_correlations = [
                -rank_correlation for rank_correlation in rank_correlations
            ]

        # Filter results for NaNs:
        rank_correlations_notnan = [
            rank_correlation
            for i, rank_correlation in enumerate(rank_correlations)
            if not np.isnan(rank_correlation) and not np.isnan(stde_values[i])
        ]
        stde_values_notnan = [
            stde
            for i, stde in enumerate(stde_values)
            if not np.isnan(rank_correlations[i]) and not np.isnan(stde)
        ]
        damping_notnan = [
            damping
            for i, damping in enumerate(damping_values)
            if not np.isnan(rank_correlations[i]) and not np.isnan(stde_values[i])
        ]
        damping_nan = [
            damping
            for i, damping in enumerate(damping_values)
            if np.isnan(rank_correlations[i]) or np.isnan(stde_values[i])
        ]

        # Plot the results
        ax.errorbar(
            np.array(damping_notnan) * config.rescale_damping_by,
            rank_correlations_notnan,
            yerr=stde_values_notnan,
            marker="o",
            linestyle=linestyle,
            color=color,
            label=label,
            markersize=4,
            clip_on=False,
        )
        # Mark the highest LDS score with a larger marker
        max_idx = np.argmax(rank_correlations_notnan)
        ax.scatter(
            config.rescale_damping_by * np.array(damping_notnan)[max_idx],
            rank_correlations_notnan[max_idx],
            color=color,
            marker="o",
            facecolors="none",
            edgecolors=color,
            s=50,
            alpha=0.2,
            clip_on=False,
        )
        if len(damping_nan):
            # Get the closest non-NaN value to plot the NaNs
            nan_loc = np.argmin(
                np.abs(
                    np.array(damping_notnan)[:, None] - np.array(damping_nan)[None, :]
                ),
                axis=0,
            )
            # closest_damping = [damping_notnan[loc] for loc in nan_loc]
            closest_notnan = [rank_correlations_notnan[loc] for loc in nan_loc]
            ax.scatter(
                config.rescale_damping_by * np.array(damping_nan),
                closest_notnan,
                color=color,
                marker="x",
                s=10,
                clip_on=False,
            )
        # ax.axhline(accross_seeds_correlations[0], color='black', linestyle='-', alpha=0.2, linewidth=0.5, zorder=-1, label="Exact retraining")
    # Legend with no frame
    plt.xscale("log")
    ax.set_ylim(0.0, None)
    ax.set_xlabel("Damping factor")
    ax.set_ylabel("Rank correlation (LDS)")
    # Possibly add a secondary axis for non-rescaled damping:
    if config.rescale_damping_by != 1.0:
        ax2 = ax.secondary_xaxis(
            "top",
            functions=(
                lambda x: x / config.rescale_damping_by,
                lambda x: x * config.rescale_damping_by,
            ),
        )
        ax2.set_xlabel("Damping factor (unnormalised)")

    sns.despine(ax=ax, offset=6)

    if config.legend_pre:
        legend_pre = config.legend_pre.replace(" ", "\ ")
        # Create a list of the original plot handles and labels
        handles, labels = ax.get_legend_handles_labels()

        # Define a custom patch (or any other artist) for the extra label
        extra_patch = mpatches.Patch(color="none", label=legend_pre)

        # Insert the custom patch at the beginning of the list of handles/labels
        handles = [extra_patch] + handles
        labels = [legend_pre] + labels

        # Create the legend with the modified handles and labels
        plt.legend(handles, labels, frameon=False)
    else:
        plt.legend(frameon=False)

    fig.suptitle(config.suptitle, y=1.2)
    fig.savefig(config.savedir / config.save_filename, bbox_inches="tight")
    logging.info(f"Saved figure to: {config.savedir / config.save_filename}")


if __name__ == "__main__":
    hydra_main()
