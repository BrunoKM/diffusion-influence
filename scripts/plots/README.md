# Plot commands:

### CIFAR-2/CIFAR-10 - Influence Damping ablation LDS:
Adjust the paths in the configs at `scripts/plots/configs` to point to the right directories, and run:

```bash
python scripts/plots/plot_damping_ablation_figure.py --config-dir=scripts/plots/configs --config-name=damping_ablation_cifar2_trak
python scripts/plots/plot_damping_ablation_figure.py --config-dir=scripts/plots/configs --config-name=damping_ablation_cifar2_influence
python scripts/plots/plot_damping_ablation_figure.py --config-dir=scripts/plots/configs --config-name=damping_ablation_cifar10_trak
python scripts/plots/plot_damping_ablation_figure.py --config-dir=scripts/plots/configs --config-name=damping_ablation_cifar10_influence
```
