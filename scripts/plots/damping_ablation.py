"""
Deprecated: for main figure papers use `plot_damping_ablation_figure.py` instead.
"""

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import seaborn as sns
import tqdm
from hydra.core.config_store import ConfigStore


@dataclass
class PlotConfig:
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

    retrained_model_idxs_dir: str = (
        "/srv/shared/outputs/idxs/DatasetType.cifar2/retrain"
    )
    measurement_dir: str = "/srv/shared/outputs/measurements/DatasetType.cifar2/ddpm_samples_loss_measurement"
    train_model_idxs_path: Optional[str] = (
        "/srv/shared/outputs/idxs/DatasetType.cifar2/idx_train.csv "
    )
    influence_path_format_string: str = "/srv/shared/outputs/DatasetType.cifar2/idx_train-0/influence_for_ddpm_samples_loss_1000kfac_100measurement_100loss__quantize_8bits__damping{}/influence_scores.npy"

    savedir: Path = Path(__file__).parent.parent.parent / "outputs" / "figures"
    save_filename: str = "damping_ablation.pdf"


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
    colors = [
        (204 / 255, 57 / 255, 42 / 255),  # palette1
        (79 / 255, 155 / 255, 143 / 255),  # palette2
        (44 / 255, 97 / 255, 194 / 255),  # palette3
        (217 / 255, 116 / 255, 89 / 255),  # palette4
        (228 / 255, 197 / 255, 119 / 255),  # palette5
        (155 / 255, 106 / 255, 145 / 255),  # palette6 converted from "#9B6A91"
        (51 / 255, 110 / 255, 49 / 255),  # palette7 converted from "#336E31"
        (198 / 255, 5 / 255, 79 / 255),  # palette8 converted from "#C6054F"
    ]
    sns.palplot(colors)

    golden_ratio = (1 + 5**0.5) / 2

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
    matplotlib.rc("legend", fontsize=fs_m1)  # legend fontsize
    matplotlib.rc("figure", titlesize=fs_p1)  # fontsize of the figure title

    plt.rcParams["savefig.facecolor"] = "white"

    # matplotlib.rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
    matplotlib.rc("font", **{"family": "serif"})

    # Turn on if you've got TeX installed
    matplotlib.rc("text", usetex=True)

    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsfonts}"})
    # ^^^^^^^^^^^^^^^^^^^^^ Plotting setup ^^^^^^^^^^^^^^^^^^^^^

    def extract_correlation_values(
        output: str,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        # Define a regular expression to capture correlation values from the text
        rank_correlation_pattern = r"Rank correlation mean: ([0-9.-]+)"
        stde_pattern = r"Rank correlation stde: ([0-9.-]+)"
        accross_seeds_pattern = r"Accross seeds.*rank correlation mean: ([0-9.-]+)"

        # Search for the patterns and capture the group with the number
        rank_match = re.search(rank_correlation_pattern, output)
        stde_match = re.search(stde_pattern, output)
        accross_seeds_match = re.search(accross_seeds_pattern, output)

        # Use default values or the captured number if matches were found
        rank_correlation = float(rank_match.group(1)) if rank_match else None
        stde = float(stde_match.group(1)) if stde_match else None
        accross_seeds_correlation = (
            float(accross_seeds_match.group(1)) if accross_seeds_match else None
        )

        return rank_correlation, stde, accross_seeds_correlation

    # List to hold the extracted numbers
    rank_correlations = []
    stde_values = []
    accross_seeds_correlations = []

    # Damping values to iterate over
    damping_values = [float(damping) for damping in config.damping_values_strings]

    base_command = (
        f"python scripts/lds_score.py --config-name compute_lds_score_default "
        f"dataset={config.dataset} retrained_model_idxs_dir={config.retrained_model_idxs_dir} "
        f"measurement_dir={config.measurement_dir} "
        f"train_model_idxs_path={config.train_model_idxs_path if config.train_model_idxs_path else 'null'} "
        f"influence_path=" + config.influence_path_format_string
    )
    for damping in tqdm.tqdm(config.damping_values_strings):
        influence_path = Path(config.influence_path_format_string.format(damping))
        if not influence_path.parent.exists():
            # Directory not found
            raise FileNotFoundError(f"Directory not found: {influence_path.parent}")
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
            command = base_command.format(damping)
            # Execute the command and capture stdout
            output = subprocess.run(
                command, shell=True, capture_output=True, check=False
            )
            stdoutput = output.stdout.decode()
            stderror = output.stderr.decode()
            # Extract and collect the correlation values
            (
                rank_correlation,
                stde,
                accross_seeds_correlation,
            ) = extract_correlation_values(stdoutput)
            if (
                rank_correlation is None
                or stde is None
                or accross_seeds_correlation is None
            ):
                raise ValueError(
                    f"Failed to extract values for damping={damping}\n"
                    f"rank_correlation={rank_correlation}, stde={stde}, accross_seeds_correlation={accross_seeds_correlation}"
                    f"\nCommand:\n{command}"
                    f"\nOutput:\n{stdoutput}"
                    f"\nError:\n{stderror}"
                )
            rank_correlations.append(rank_correlation)
            stde_values.append(stde)
            accross_seeds_correlations.append(accross_seeds_correlation)
    if all([np.isnan(rank_correlation) for rank_correlation in rank_correlations]):
        raise ValueError("All rank correlations are NaN.")

    # Assert across seed correlations almost the same:
    for across_seed in accross_seeds_correlations[1:]:
        assert np.isclose(accross_seeds_correlations[0], across_seed, rtol=1e-8), (
            f"Across seed correlation values are not the same: {accross_seeds_correlations}"
        )

    if all(
        rank_correlation < 0.0 or np.isnan(rank_correlation)
        for rank_correlation in rank_correlations
    ):
        # Sometimes we have inverted signs, so we invert them back :|
        # TODO: make signs consistent in influence and measurement calculations
        logging.warn("All rank correlations are negative. Inverting them.")
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
    fig, ax = plt.subplots(figsize=(text_width, text_width / golden_ratio))
    ax.errorbar(
        damping_notnan,
        rank_correlations_notnan,
        yerr=stde_values_notnan,
        fmt="o-",
        color=colors[0],
        label="Influence",
    )
    if len(damping_nan):
        ax.scatter(
            damping_nan,
            [min(rank_correlations_notnan)] * len(damping_nan),
            color=colors[1],
            marker="x",
            s=10,
            label="NaN",
        )
    ax.axhline(
        accross_seeds_correlations[0],
        color="black",
        linestyle="-",
        alpha=0.2,
        linewidth=0.5,
        zorder=-1,
        label="Exact retraining",
    )
    # Legend with no frame
    plt.xscale("log")
    ax.set_ylim(0.0, None)
    ax.set_xlabel("Damping factor")
    ax.set_ylabel("Rank correlation")
    sns.despine(ax=ax, offset=6)
    plt.legend(frameon=False)

    fig.savefig(config.savedir / config.save_filename, bbox_inches="tight")
    fig.suptitle("Damping ablation on loss measurement on CIFAR-2")


if __name__ == "__main__":
    hydra_main()
