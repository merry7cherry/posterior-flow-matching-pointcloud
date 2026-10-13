# Posterior Flow Matching — Point Clouds

Joint Posterior Flow Matching (PFM) for **ShapeNet Chair and Airplane**. The model
learns conditional velocity means and diagonal variances with a PointNet encoder
and a flow-based latent prior. The same checkpoint supports ODE and posterior-driven
SDE sampling, endpoint readouts, and held-out residual/variance diagnostics.

[Project page](https://merry7cherry.github.io/posterior-flow-matching/) ·
[Paper](https://merry7cherry.github.io/posterior-flow-matching/assets/paper.pdf) ·
[Images and project hub](https://github.com/merry7cherry/posterior-flow-matching) ·
[Molecules](https://github.com/merry7cherry/posterior-flow-matching-molecules) ·
[PushT](https://github.com/merry7cherry/posterior-flow-matching-pusht)

## Install

Python 3.10 or later; Python 3.11 and PyTorch 2.8 are the release validation target.
On Linux, install the CPU PyTorch wheel first; on macOS, use ordinary `pip install torch==2.8.0`.

```bash
git clone https://github.com/merry7cherry/posterior-flow-matching-pointcloud.git
cd posterior-flow-matching-pointcloud
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[test]'
python -m pytest -q
```

No CUDA extension, cluster account, remote logger, or Git metadata is required at
runtime. CLI commands default to CPU with two threads. Full training is expensive
on CPU; the smoke configuration checks the complete workflow without a GPU.

## Data and initialization

Obtain the original `shapenet.hdf5` and the matching `GEN_chair.pt` or
`GEN_airplane.pt` from the [upstream download folder](https://drive.google.com/drive/folders/1Su0hCuGFo1AGrNb_VMNnlF7qeQwKjfhZ).
See [data and asset details](docs/data.md) for direct links, hashes, dataset terms,
and normalization. These assets are not redistributed here.

Joint training initializes the encoder and latent prior from the category's
upstream generator, and initializes the PFM coordinate network from the seed.
The upstream diffusion decoder is not used. A saved PFM checkpoint contains its
own encoder, prior, and coordinate network, so sampling does not need the original
initialization asset or dataset. No trained PFM weights are bundled in this release.
Only load trusted PyTorch checkpoint files.

## Train and resume

```bash
pfm-pointcloud-train --config configs/chair.yaml \
  --dataset /path/to/shapenet.hdf5 --condition /path/to/GEN_chair.pt \
  --output runs/chair --device cpu

# Resume the identical configuration, assets, source, and output directory.
pfm-pointcloud-train --config configs/chair.yaml \
  --dataset /path/to/shapenet.hdf5 --condition /path/to/GEN_chair.pt \
  --output runs/chair --device cpu --resume
```

For Airplane use `configs/airplane.yaml` and `GEN_airplane.pt`. Data/output paths
are caller-owned. Existing runs require explicit resume; model, optimizer,
scheduler, iteration, RNG state and input identities are saved atomically.
The PointNet encoder's BatchNorm requires at least two shapes in every training
batch, including the last batch. Choose a compatible batch size for custom subsets.

A bounded CPU run uses the **native network and 2,048 points**, four updates and
four training shapes. The subset limit is part of the run identity:

```bash
pfm-pointcloud-train --config configs/cpu_smoke.yaml \
  --dataset /path/to/shapenet.hdf5 --condition /path/to/GEN_chair.pt \
  --output runs/smoke --limit-train 4 --stop-after 2
pfm-pointcloud-train --config configs/cpu_smoke.yaml \
  --dataset /path/to/shapenet.hdf5 --condition /path/to/GEN_chair.pt \
  --output runs/smoke --limit-train 4 --resume
```

## Sample, evaluate, and inspect posterior readouts

```bash
pfm-pointcloud-sample --checkpoint runs/chair/latest.pt \
  --output runs/chair/samples-ode --solver ode --count 8
pfm-pointcloud-sample --checkpoint runs/chair/latest.pt \
  --output runs/chair/samples-sde --solver sde --diffusion-scale 1 --count 8

pfm-pointcloud-evaluate --checkpoint runs/chair/latest.pt \
  --dataset /path/to/shapenet.hdf5 --output runs/chair/evaluation

# Small end-to-end CPU check, not a quality benchmark.
pfm-pointcloud-evaluate --checkpoint runs/smoke/latest.pt \
  --dataset /path/to/shapenet.hdf5 --output runs/smoke/evaluation --limit 2 --steps 2
```

Sampling writes raw point clouds, a simple PNG gallery, and trajectory snapshots.
Evaluation samples both ODE and SDE from matched initial draws, then reports native
squared Chamfer MMD, coverage and 1-NNA after shape bounding-box normalization.
MMD averages each **reference shape's** nearest generated distance; 1-NNA is best
near 50%. It also reports grouped raw residual moments, Gaussian NLL and endpoint
moments on reference interpolations. These diagnostics do not certify free-generation
quality. EMD and JSD are not reported. Full pairwise metrics have quadratic set cost.

## Method and validation

[Architecture and configuration](docs/method.md) describes the joint KL objective,
variance bounds, SDE derivative correction, normalization and terminal endpoint
readout. [Validation](docs/validation.md) records actual CPU checks and their limits.
The Chair/Airplane configs retain the mainline architecture and training schedule,
with FP32 as the portable default. The CPU release does not reproduce paper metrics
or certify CUDA, BF16, DDP, timing, or full training convergence.

This repository contains only joint PFM for these two categories. Historical
variants, frozen training, cluster scripts, baseline training and paper figure
production are excluded.

## Citation and license

Use the [paper citation on the project page](https://merry7cherry.github.io/posterior-flow-matching/#citation).
If using the point-cloud backbone or data pipeline, also cite Luo and Hu,
*Diffusion Probabilistic Models for 3D Point Cloud Generation*, CVPR 2021.

Code is MIT licensed; retained upstream copyright notices and third-party scope
are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Data and externally
downloaded assets remain subject to their own terms.
