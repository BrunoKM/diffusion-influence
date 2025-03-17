Huggingface (diffusers) models can be created from a config, but that config must be supplied as a path to `.json` file. Hence, the model configs to be loaded through the HuggingFace interface are separated out into this folder. These are:

Configs:
- `unet_cifar2.json` - the UNet2DModel class used for all `cifar2` experiments
- `unet_cifar10.json` - the UNet2DModel class used for all `cifar10` experiments
- `unet_artbench.json` - the UNet2DModel class used for all `artbench` experiments