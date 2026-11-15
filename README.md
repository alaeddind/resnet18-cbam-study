# ResNet18–CBAM placement ablation study

Datasets: CIFAR-10 and EuroSAT RGB. Four model variants: baseline, CBAM before residual addition, CBAM after addition (before final ReLU), and CBAM at each stage exit. The before/after configurations contain eight CBAM modules each; stage-end contains four and is not a matched-count comparison.

## Install

```bash
pip install -r requirements.txt
```

## Quick pilot (run in project root)

```bash
python train.py --dataset cifar10 --model baseline --epochs 2 --batch-size 64 --seed 42
```

## Main experiment

```bash
python train.py --dataset cifar10 --model cbam_before_add --epochs 50 --seed 42 --test
python train.py --dataset eurosat --model cbam_after_add --epochs 50 --seed 42 --test
```

In Colab, upload/extract the project and run `%cd /content/resnet18-cbam-study` before commands. Use `!python train.py ...` in notebook cells. Data is downloaded automatically. Store the project on persistent Drive storage if you need checkpoints and results to survive Colab restarts.

## Experimental cautions

- The model stem is 3x3 stride 1 with no maxpool; images resized to 64x64; all variants train from scratch with the same settings.
- CIFAR-10 keeps its official test set and splits official train stratified into train/validation. EuroSAT uses a stratified 80/10/10 random split; random EuroSAT tile splits can overestimate spatial generalization because nearby satellite patches may be correlated. A spatially disjoint evaluation is a useful future extension.
- Split seed also controls initialization; for comparable experiments use the same seed across variants. EuroSAT split changes with seed, so cross-seed differences combine training and split variation.
- The `--test` flag evaluates the held-out test set once per run after checkpoint selection by validation loss. Do not select hyperparameters by test scores.
- THOP FLOPs are approximate (2x MACs) and may not count every operation, especially attention pooling/activations. Latency is hardware- and environment-dependent: report GPU/CPU, software versions, batch size and timing method.
- Training resumes are not implemented; checkpoint saves best model, not optimizer state. GPU quotas/session timeouts can interrupt runs.
