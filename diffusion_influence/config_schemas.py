import enum
import inspect
from dataclasses import field
from pathlib import Path
from typing import Any, Optional

from pydantic.dataclasses import dataclass

from diffusion_influence.compressor import (
    IdentityCompressor,
    QSVDCompressor,
    QuantizationCompressor,
    SVDCompressor,
)


class DatasetType(str, enum.Enum):
    # CIFAR2 = "CIFAR2"
    """CIFAR10, but only with 2 classes (horses (index 7) and automobiles (index 1))"""

    artbench = "artbench"
    cifar10 = "cifar10"
    cifar2 = "cifar2"
    cifar10deq = "cifar10deq"  # CIFAR-10, but with uniform dequantization
    cifar2deq = "cifar2deq"  # CIFAR-2, but with uniform dequantization
    coco = "coco"
    celeba = "celeba"
    dummy1 = "dummy1"
    dummy2 = "dummy2"


class DataRangeType(str, enum.Enum):
    UNIT_BOUNDED = "UNIT_BOUNDED"
    """Scale the data space so that it's bounded between -1.0 and 1.0"""


@dataclass
class DataConfig:
    dataset_name: DatasetType
    """Will be passed to 'load_dataset' from HuggingFace datasets"""
    data_range: DataRangeType
    cache_dir: str = "data"
    examples_idxs_path: Optional[Path] = None


@dataclass
class DataLoaderConfig:
    persistent_workers: bool
    train_batch_size: int
    eval_batch_size: int
    num_workers: int
    pin_memory: bool


class DiffusionScheduleType(str, enum.Enum):
    linear = "linear"
    scaled_linear = "scaled_linear"


class DiffusionVarianceType(str, enum.Enum):
    fixed_small = "fixed_small"


@dataclass
class DiffusionConfig:
    max_timestep: int
    schedule_type: DiffusionScheduleType
    resolution: int
    variance_type: DiffusionVarianceType
    beta_start: float
    beta_end: float
    num_inference_steps: int
    """Number of steps to use for inference (sampling)."""


class MeasurementType(str, enum.Enum):
    LOSS = "LOSS"
    ELBO = "ELBO"
    SIMPLIFIED_ELBO = "SIMPLIFIED_ELBO"
    LOG_LIKELIHOOD = "LOG_LIKELIHOOD"
    LOSS_AT_TIME = "LOSS_AT_TIME"
    SQUARE_NORM = "SQUARE_NORM"  # D-TRAK measurement
    WEIGHTED_LOSS = "WEIGHTED_LOSS"
    SAMPLING_TRAJECTORY_LOGPROB = "SAMPLING_TRAJECTORY_LOGPROB"


class OptimiserType(str, enum.Enum):
    SGD = "SGD"
    ADAM = "ADAM"
    ADAMW = "ADAMW"


class KFACType(str, enum.Enum):
    expand = "expand"
    reduce = "reduce"


class FisherType(str, enum.Enum):
    empirical = "empirical"
    mc = "mc"
    type_2 = "type-2"


class PreconditionerType(str, enum.Enum):
    identity = "identity"
    kfac = "kfac"


@dataclass
class KFACConfig:
    num_samples_for_loss: int
    use_progressbar: bool = True
    check_deterministic: bool = False
    kfac_approx: KFACType = KFACType.expand
    fisher_type: FisherType = FisherType.mc
    mc_samples: int = 1
    damping: float = 1e-8
    use_heuristic_damping: bool = False
    min_damping: float = 1e-8
    use_exact_damping: bool = True
    correct_eigenvalues: bool = False
    """Whether to use EKFAC (eigenvalue-corrected KFAC)"""


@dataclass
class OptimiserConfig:
    optimiser_type: OptimiserType
    lr: float
    """Default learning rate if a param specific lr is not specified"""
    # If specified, overrides the "global" lr with a per-parameter learning rate.
    weight_decay: float
    optimiser_kwargs: dict[str, Any] = field(default_factory=dict)
    clip_grad: float = float("inf")
    cosine_lr_schedule: bool = False
    warmup_steps: int = 0
    ema_decay: Optional[float] = None
    ema_power: float = 2 / 3
    use_ema_warmup: bool = True


@dataclass
class WandbConfig:
    entity: str
    project_name: str
    group_name: Optional[str]
    mode: str
    save_model_artifact: bool
    name: Optional[str]


class MixedPrecisionOption(str, enum.Enum):
    no = "no"
    fp16 = "fp16"
    bf16 = "bf16"


@dataclass
class AccelerateConfig:
    mixed_precision: MixedPrecisionOption
    """Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16)."""
    gradient_accumulation_steps: int
    checkpoints_total_limit: int
    """Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`.
    If `None`, no limit is set."""


@dataclass
class DiffusionTrainConfig:
    seed: int
    num_training_iter: int
    eval_frequency: int
    """Number of training iterations between evaluations."""
    data: DataConfig
    dataloader: DataLoaderConfig
    model_config_path: Path
    optimisation: OptimiserConfig
    diffusion: DiffusionConfig
    accelerate: AccelerateConfig
    wandb: WandbConfig
    log_loss_at_timesteps: list[int]
    checkpoint_frequency: Optional[int] = None
    """Number of training iterations between checkpoints. If None, there will be only one checkpoint after training."""
    resume_from_checkpoint: Optional[Path] = None
    vae_name_or_path: Optional[str] = None
    """If specified, the model will be trained as a latent diffusion model, using the VAE in
    the `/vae` subfolder of the model specified by this name or path. This will be passed to
    AutoencoderKL.from_pretrained(..., subfolder="vae")."""
    scheduler_name_or_path: Optional[str] = None
    """If specified, the scheduler to use will be loaded from HuggingFace. This will be passed to
    DDPMScheduler.from_pretrained(..., subfolder="scheduler")."""

    weighted_examples_idxs_path: Optional[Path] = None
    """List of indices of examples to weight more lightly/heavily in the loss. 
    f None, all examples are weighted equally."""
    invert_weighted_examples_idxs: bool = False
    """If True, the examples NOT in `weighted_examples_idxs` will be weighted more heavily with `weighted_examples_weight`."""
    weighted_examples_weight: float = 1.0
    """The weight to apply to the examples in `weighted_examples_idxs`. 1.0 is the 'default' weight,
    meaning all examples are weighted equally. A value < 1.0 will weight the examples more lightly,
    and a value > 1.0 will weight the examples more heavily."""


class CompressorType(str, enum.Enum):
    identity = "identity"
    quantization = "quantization"
    svd = "svd"
    qsvd = "qsvd"


# Dictionary mapping CompressorTypes to their respective classes.
COMPRESSOR_CLASS_MAPPING = {
    CompressorType.identity: IdentityCompressor,
    CompressorType.quantization: QuantizationCompressor,
    CompressorType.svd: SVDCompressor,
    CompressorType.qsvd: QSVDCompressor,
}


@dataclass
class InfluenceConfig:
    seed: int

    pretrained_model_dir_path: Path
    """Path to the directory containing the 1) pretrained model 2) corresponding config.json"""

    pretrained_model_config_path: Path
    """Used for loading in information about the training data. This is the path
    to the config.yaml file with training settings for the model, usually saved
    to the .hydra directory in the output directory of the training script."""

    preconditioner: PreconditionerType
    kfac: KFACConfig
    kfac_dataloader: DataLoaderConfig
    cache_inverse_kfac: bool
    """If true, the inverse KFAC will be cached to disk after it is computed, even if
    `cached_inverse_kfac_path` already exists (will be overridden). If false,
    and `cached_inverse_kfac_path` is not None, the cached inverse KFAC will be loaded
    from disk. If false and `cached_inverse_kfac_path` is None, the inverse KFAC will
    be computed from scratch and not cached."""
    cached_inverse_kfac_path: Optional[Path]
    """Path to store/load the cached inverse KFAC; depends on `cache_inverse_kfac`."""

    num_samples_for_loss_scoring: int
    """How many samples to use for estimating the per-example loss gradient."""
    num_samples_for_measurement_scoring: int
    """How many samples to use for estimating the per-example measurement."""

    num_loss_batch_aggregations: int
    num_measurement_batch_aggregations: int
    """How many batches to aggregate the loss/measurement gradients over. The total 
    effective number of samples will be 
    `num_samples_for_loss_scoring * num_loss_batch_aggregations` (similar for measurement),
    but doing this in multiple batches helps save memory."""

    train_gradient_compressor: CompressorType
    train_compressor_kwargs: dict[str, Any]
    query_gradient_compressor: CompressorType
    query_compressor_kwargs: dict[str, Any]
    """`ScoreCalculator` settings."""

    cache_train_gradients: bool
    """Whether to cache the train gradients TO DISK. If true, these will be
    computed and cached to disk even if `cached_train_gradients_path`
    exists (it will be overridden). If false, and 
    `cached_train_gradients_path` is None, the training gradients
    will be computed from scratch every time."""
    cached_train_gradients_path: Optional[Path]
    """Path to directory to store/load the cached predicted train gradients; depends on
    `cache_train_gradients`."""
    cached_train_gradient_queue_size: int
    """Max no. of train gradients to keep in queue on GPU when loading cached from disk.
    Will be upper-limited by the GPU memory."""
    cached_train_gradient_queue_num_workers: int
    """No. of workers to load the cached train gradients from disk to GPU queue."""

    precondition_query_gradients: bool
    """Whether to precondition the query or the train gradients. Depending on which ones
    end up being cached (if any), preconditioning the one to be cached might save
    compute."""
    cache_query_gradients: bool
    """Whether to cache the query gradients. If true, these will be cached (after compressing)
    on the GPU."""
    query_batch_size: int
    train_batch_size: int
    outer_is_query_in_score_loop: bool
    """Whether the outer loop in the score calculation should be over the query samples
    or the training samples. Generally, the outer gradient will be instantiated:
        `len(outer_dataset)` times
    , and the inner gradient will be instantiated:
        approx. `len(inner_dataset)*len(outer_dataset)` times.
    Hence, depending on whether the query or train gradients are being cached, 
    one might be desirable to be put in the inner loop to save compute.
    
    Generally, it's more desirable to make the batch_size for the outer loop larger
    compared to the inner loop (out of the `query_batch_size` and `train_batch_size`).
    """

    samples_dir_path: Path
    """Path to the directory containing the samples to score the influence for."""
    measurement: MeasurementType
    """What measurement change to try and predict with influence functions."""
    measurement_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.cache_inverse_kfac and self.cached_inverse_kfac_path is None:
            raise ValueError(
                "If `cache_inverse_kfac` is True, `cached_inverse_kfac_path` should be set."
            )
        if self.cache_train_gradients and self.cached_train_gradients_path is None:
            raise ValueError(
                "If `cache_train_gradients` is True, `cached_train_gradients_path` should be set."
            )

        if self.measurement == MeasurementType.LOSS_AT_TIME:
            assert len(self.measurement_kwargs) == 1
            assert "time" in self.measurement_kwargs
        elif self.measurement == MeasurementType.WEIGHTED_LOSS:
            assert len(self.measurement_kwargs) == 1
            assert "weighting" in self.measurement_kwargs

        for gradient_compressor, compressor_kwargs in [
            (self.train_gradient_compressor, self.train_compressor_kwargs),
            (self.query_gradient_compressor, self.query_compressor_kwargs),
        ]:
            # Make sure the keywords passed for each compressor type are appropriate for the
            # compressor type class:
            compressor_cls = COMPRESSOR_CLASS_MAPPING[gradient_compressor]
            # Check if the class has an explicit __init__ or inherits from object.
            compressor_signature = inspect.signature(compressor_cls.__init__)
            available_kwargs_and_defaults = {
                k: v
                for k, v in compressor_signature.parameters.items()
                if k != "self"  # noqa: PLR1714
                and v.kind != inspect.Parameter.VAR_KEYWORD
                and v.kind != inspect.Parameter.VAR_POSITIONAL
            }
            available_kwargs = set(available_kwargs_and_defaults.keys())
            required_kwargs = {
                k
                for k, v in available_kwargs_and_defaults.items()
                if v.default == v.empty
            }
            # Check whether **kwargs are allowed.
            accepts_wildcard_kwargs = any(
                v.kind == inspect.Parameter.VAR_KEYWORD
                for v in compressor_signature.parameters.values()
            )
            passed_compressor_kwargs = set(compressor_kwargs.keys())
            if not required_kwargs <= set(compressor_kwargs.keys()):
                raise ValueError(
                    f"Not all required kwargs for compressor {gradient_compressor} specified. "
                    f"Expected at least {required_kwargs}, but got {passed_compressor_kwargs}"
                )
            if (
                not available_kwargs >= set(compressor_kwargs.keys())
                and not accepts_wildcard_kwargs
            ):
                raise ValueError(
                    f"Unexpected kwargs for compressor {gradient_compressor} specified. "
                    f"Expected at most {available_kwargs}, but got {passed_compressor_kwargs}"
                )


@dataclass
class LDSScoreConfig:
    dataset: DatasetType
    train_model_idxs_path: Optional[Path]  # only for CIFAR2
    retrained_model_idxs_dir: Path
    measurement_dir: Path
    influence_path: Path
    num_seeds: int
    num_retrained_models: int
