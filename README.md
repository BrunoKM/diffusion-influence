<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
                 Influence Functions for Diffusion Models</h1>

---

# Summary: 
The research code for experiments in the paper [“Influence Functions for Scalable Data Attribution in Diffusion Models”](https://arxiv.org/abs/2410.13850) (by Bruno Mlodozeniec, Runa Eschenhagen, Juhan Bae, Alexander Immer, David Krueger and Richard E. Turner).


# How to run
## Setup
To get started, follow these steps:
1. **Clone the GitHub Repository:** Begin by cloning the repository using the command:
   ```shell
   git clone git@github.com:BrunoKM/diffusion-influence.git
   ```
2. **Set Up Python Environment:** Ensure you have a version `>=3.11`.
3. **Install Dependencies:** Install the dependencies by running:
   ```shell
   pip install -e .
   ```
## Replicating the results



### Commands for Linear Datamodelling Score (LDS) evaluation
We provide the commands to run experiments on `CIFAR-2`. Substitute an output directory of your choice:
```shell
PROJECT_OUTPUT_DIR=...
MODEL_TRAIN_OUTPUT_DIR="$PROJECT_OUTPUT_DIR/DatasetType.cifar2/idx_train-0";
```
#### Generate samples and retrain models for LDS benchmark:
1. **Generate indices for retraining datasets:** 
    ```shell
    python scripts/generate_retrain_idxs.py dataset_name=cifar2 seed=0 retrain_subsample_size=2500 core_subsample_size=5000 num_validation_subsamples=1000 num_subsampled_datasets=1024 idxs_save_path=$PROJECT_OUTPUT_DIR'/idxs/${dataset_name}' hydra.run.dir='${idxs_save_path}'
    ```
    This generates the following in `$PROJECT_OUTPUT_DIR/idxs/${dataset_name}`:
    - `idx_train.csv` — the (possibly subsampled) indices of the training dataset to use for training the "core" model to perform data attribution on.
    - `idx_val.csv` — the (possibly subsampled) indices of the test/validation dataset to perform data attribution on.
    - `retrain/sub_idx_{IDX}.csv` — the subsampled indices of the indices in `idx_train.csv` to use for re-training models on submsampled data for LDS evaluation
2. **Train a diffusion model on the entire dataset**
    ```shell
    accelerate launch --gpu_ids=0 --main_process_port=18888 scripts/train_unconditional.py --config-name cifar2_ddpm data.examples_idxs_path=$PROJECT_OUTPUT_DIR'/idxs/${data.dataset_name}/idx_train.csv' hydra.run.dir=$MODEL_TRAIN_OUTPUT_DIR'
    ```
    For `cifar10`: replace `--config-name cifar10_ddpm` and leave `data.examples_idxs_path` unspecified (full dataset)
3. **Generate samples**
    ```shell
    python3 scripts/sample_unconditional.py seed=0 pretrained_pipeline_path=$PROJECT_OUTPUT_DIR/DatasetType.cifar2/idx_train-0/pipeline num_samples_to_generate=1000 batch_size=64 hydra.run.dir=$PROJECT_OUTPUT_DIR/DatasetType.cifar2/idx_train-0/ddpm_samples sampling_method=DDPM num_inference_steps=1000 
    ```
4. **Retrain diffusion models on every retraining dataset**
    Launch one training run per retraining dataset:
    ```shell
    for (( SUBIDX=0; SUBIDX<100; SUBIDX++ )); do
        for (( SEED=0; SEED<5; SEED++ )); do
            accelerate launch --gpu_ids=0 --main_process_port=18888 scripts/train_unconditional.py --config-name cifar2_ddpm \
            seed=$SEED \
            num_training_iter=16000 \
            data.examples_idxs_path=$PROJECT_OUTPUT_DIR/idxs/\${data.dataset_name}/retrain/sub_idx_$SUBIDX.csv \
            hydra.run.dir=$PROJECT_OUTPUT_DIR/\${data.dataset_name}/subidx$SUBIDX-$SEED \
            eval_frequency=\${num_training_iter} \
            eval_frequency=\${divide:\${num_training_iter},4} \
            'log_loss_at_timesteps=[10,100]'
        done
    done
    ```
    - `eval_frequency=\${divide:\${num_training_iter},4} 'log_loss_at_timesteps=[10,100]'` overrides can be used for reduced logging (more infrequently and log less) for retraining models to save a bit of compute
    - `num_training_iter=16000` to halve the number of training steps as there should only be half the data
    - If you don't want `wandb` (Weights & Biases) logging, you can add the `wandb.mode=offline` flag

#### Compute true counterfactual retraining measurements
5. **Compute outputs (measurements) on the generated samples with different (retrained) models**
    First for the core trained model:
    ```shell
    python scripts/compute_measure_function.py \
        seed=0 \
        pretrained_pipeline_path=$PROJECT_OUTPUT_DIR/DatasetType.cifar2/idx_train-0/pipeline \
        samples_dir_path=$PROJECT_OUTPUT_DIR/DatasetType.cifar2/idx_train-0/ddpm_samples \
        batch_size=128 \
        dataset_name="cifar2" \
        num_samples_for_measurement=5000 \
        measurement=LOSS \
        hydra.run.dir=$PROJECT_OUTPUT_DIR/measurements/DatasetType.cifar2/'ddpm_samples_${measurement}_measurement'/idx_train-0'
    ```
    Then for all the retrained models: ... TODO
    ```shell
    for (( SUBIDX=0; SUBIDX<100; SUBIDX++ )); do
        for (( SEED=0; SEED<5; SEED++ )); do
            python scripts/compute_measure_function.py \
                seed=0 \
                pretrained_model_dir_path=$PROJECT_OUTPUT_DIR/DatasetType.cifar2/subidx$SUBIDX-$SEED/pipeline/unet \
                pretrained_model_config_path=$PROJECT_OUTPUT_DIR/DatasetType.cifar2/subidx$SUBIDX-$SEED/.hydra/config.yaml \
                samples_dir_path=$PROJECT_OUTPUT_DIR/DatasetType.cifar2/idx_train-0/ddpm_samples \
                batch_size=128 \
                dataset_name="cifar2" \
                num_samples_for_measurement=5000 \
                measurement=LOSS \
                hydra.run.dir=$PROJECT_OUTPUT_DIR/measurements/DatasetType.cifar2/'ddpm_samples_${measurement}_measurement'/subidx$SUBIDX-$SEED
        done
    done
    ```
    Change `measurement` to other values (e.g. `SIMPLIFIED_ELBO`) to evaluate other measurement functions.

#### Compute influence scores (e.g. with influence functions, TRAK, ...)
6. **Compute the scores with KFAC influence functions**
    ```shell
    python scripts/compute_influence.py --config-name compute_influence_ekfac_cifar2 pretrained_model_dir_path=$MODEL_TRAIN_OUTPUT_DIR/pipeline/unet pretrained_model_config_path=$MODEL_TRAIN_OUTPUT_DIR/.hydra/config.yaml samples_dir_path=$MODEL_TRAIN_OUTPUT_DIR/ddpm_samples hydra.run.dir=$MODEL_TRAIN_OUTPUT_DIR/influence_for_ddpm_samples;
    ```
    **Precompute K-FAC**: to avoid recomputing K-FAC every time when calling the above script, add the `cache_inverse_kfac=True` flag and select a save-path with `cached_inverse_kfac_path=` to cache the K-FAC to the chosen location. If you already have a file with cached inverse K-FAC, re-use it by setting `cache_inverse_kfac=False` and pointing `cached_inverse_kfac_path=...` to that path.
    
    Change `measurement` to other values (e.g. `SIMPLIFIED_ELBO`) to approximate changes in other measurements.
    Change `gradient_compressor` to e.g. `svd`, `quantization` or `identity` (no compression) for other compression methods.
    You can also store the training example gradients to disk for future reuse by using the `cache_preconditioned_train_gradients` and `cached_preconditioned_train_gradients_path` arguments (see `diffusion_influence/config_schemas.py` doc-strings for details).

7. (optional) **Compute the scores with TRAK**
    ```shell
    MODEL_TRAIN_OUTPUT_DIR="$PROJECT_OUTPUT_DIR/DatasetType.cifar2/idx_train-0";
    python scripts/featurise_trak_unconditional.py --config-name compute_trak_default pretrained_model_dir_path=$MODEL_TRAIN_OUTPUT_DIR/pipeline/unet pretrained_model_config_path=$MODEL_TRAIN_OUTPUT_DIR/.hydra/config.yaml hydra.run.dir=$MODEL_TRAIN_OUTPUT_DIR/trak_for_ddpm_samples model_id=0 trak.save_dir=$MODEL_TRAIN_OUTPUT_DIR/trak_for_ddpm_samples/trak_features samples_dir_path=$MODEL_TRAIN_OUTPUT_DIR/ddpm_samples
    ```
    Increment `model_id` and `pretrained_model_dir_path`, `pretrained_model_config_path` while keeping `trak.save_dir` the same if using an ensemble of models to compute the features (although, in all our experiments, we never consider ensembles).

#### Compute the training attribution metrics (LDS)
8. **Compute the LDS scores**
    ```shell
    python scripts/lds_score.py --config-name compute_lds_score_default dataset=cifar2 retrained_model_idxs_dir=$PROJECT_OUTPUT_DIR/idxs/DatasetType.cifar2/retrain measurement_dir=$PROJECT_OUTPUT_DIR/measurements/'{dataset}'/ddpm_samples_MeasurementType.LOSS_measurement train_model_idxs_path=/idxs/DatasetType.cifar2/idx_train.csv influence_path=$MODEL_TRAIN_OUTPUT_DIR/influence_for_ddpm_samples/influence_scores.npy
    ```
    - Replace `influence_path` with e.g. `$MODEL_TRAIN_OUTPUT_DIR/trak_for_ddpm_samples/scores.npy` to compute LDS scores for the influence estimated with TRAK (from step 7.), or any other method.
    - Replace `measurement_dir` appropriately with the directories computed using other measurement functions (see step 5.) if you want to measure LDS for other measurements.

### Retraining without the top influences experiments
If you would also like to try evaluating the change in loss after retraining without the top influences, you can continue-on from after step 7 above with the following commands:

9. **Generate the indices to train on**
    ```shell
    EXPERIMENT=influence_for_ddpm_samples
    SAMPLE_IDX=0;  # Generated sample to try to change the measurement of
    python scripts/generate_retrain_without_top_influences_idxs.py \
        num_influences_to_remove=100 \
        scores_path=$MODEL_TRAIN_OUTPUT_DIR/$EXPERIMENT/influence_scores.npy \
        sample_idx=$SAMPLE_IDX \
        samples_dir_path=$MODEL_TRAIN_OUTPUT_DIR/ddpm_samples \
        maximise_measurement=True \
        dataset_name=cifar2 \
        examples_idxs_path=/srv/shared/outputs/idxs/DatasetType.cifar2/idx_train.csv \
        hydra.run.dir=$PROJECT_OUTPUT_DIR/idxs/DatasetType.cifar2/counterfactual/$EXPERIMENT/'sample${sample_idx}_remove${num_influences_to_remove}'
    ```
    `maximise_measurement=True` specifies selecting examples the removal of which will maximise (increase) the measurement (loss).
10. **Retrain the model on these inidices** 
    ```shell
    RETRAINED_MODEL_DIR=$OUTPUT_DIR/DatasetType.cifar2/counterfactual/${EXPERIMENT}/sample${SAMPLE_IDX}_remove100
    accelerate launch --gpu_ids=0 --main_process_port=18888 scripts/train_unconditional.py \
        --config-name cifar2_ddpm \
        seed=$IDX \
        data.examples_idxs_path=$PROJECT_OUTPUT_DIR/idxs/DatasetType.cifar2/counterfactual/$EXPERIMENT/sample${IDX}_remove$NUMREMOVE/idx_train.csv \
        hydra.run.dir=$RETRAINED_MODEL_DIR \
        num_training_iter=32000 \
        eval_frequency=16000 \
        checkpoint_frequency=16000 \
        'log_loss_at_timesteps=[10,100]'
    ```
11. **Evaluate the measurement function after retraining**

    ```shell
    python scripts/compute_measure_function.py \
        seed=711 \
        pretrained_model_dir_path=$RETRAINED_MODEL_DIR/pipeline/unet \
        pretrained_model_config_path=${SUBDIR}/.hydra/config.yaml \
        samples_dir_path=${MODEL_TRAIN_OUTPUT_DIR}/DatasetType.cifar2/idx_train-0/ddpm_samples \
        batch_size=128 \
        dataset_name=cifar2 \
        num_samples_for_measurement=5000 \
        hydra.run.dir=$RETRAINED_MODEL_DIR/'ddpm_samples_${measurement}_measurement' \
        measurement=LOSS;
    ```
    After that, one can compare the `LOSS` measurements of the model trained on the full dataset at 
    `$PROJECT_OUTPUT_DIR/measurements/DatasetType.cifar2/ddpm_samples_MeasurementType.LOSS_measurement/idx_train-0` and for the retrained model at `$RETRAINED_MODEL_DIR/'ddpm_samples_MeasurementType.LOSS_measurement`.
    

# Citation
If you find this project useful in your research, please consider citing our [paper](https://openreview.net/forum?id=esYrEndGsr):

```bibtex
@inproceedings{
    mlodozeniec2025influence,
    title={Influence Functions for Scalable Data Attribution in Diffusion Models},
    author={Bruno Kacper Mlodozeniec and Runa Eschenhagen and Juhan Bae and Alexander Immer and David Krueger and Richard E. Turner},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://arxiv.org/abs/2410.13850},
}
```