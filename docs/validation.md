# CPU release validation

Validated locally on **macOS arm64, Python 3.12.14, PyTorch 2.8.0**, using two CPU
threads. This is a code portability and behavior check, not a benchmark or paper
result reproduction.

## Automated suite

Seven tests pass in a clean virtual environment after installing the built wheel:
Gaussian NLL optimum/gradients; exact pointwise divergence; native encoder/prior/KL
and coordinate-network gradients; flow-prior round trip; zero-noise ODE/SDE parity;
endpoint variance scaling; CD metric reference direction; config validation; and a
complete synthetic train/save/resume/sample/evaluate workflow. The workflow test
compares uninterrupted and resumed model tensors, Adam states, scheduler, and
Torch RNG **bit for bit**. It runs both ODE and SDE and writes real PNG plots and
numeric outputs, with a smaller latent prior and synthetic shapes for CI speed.

```bash
python -m pip install '.[test]' build
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
python -m build
```

The wheel was installed into a new virtual environment, then tested from outside
the source checkout. All three CLI `--help` commands and runtime source identity
work without `.git`. The sdist includes configs, tests, documentation and notices;
the wheel contains all required third-party license texts.

## Real-data and source-equivalence checks

Separate from the synthetic suite, local CPU validation used the actual official
Chair initialization and actual ShapeNet HDF5 file. Both asset hashes match
`docs/data.md`. The native model used latent dimension 256, all 14 prior layers,
and 2,048 points, with four real training shapes and two held-out test shapes.

- Four optimizer updates completed, including a stop after update 2 and resume.
  Model, optimizer, scheduler and Torch RNG matched an uninterrupted four-update
  run bit for bit.
- ODE and SDE each completed two integration steps and final endpoint readout.
  Native CD MMD, coverage and 1-NNA, fixed/random-time posterior diagnostics, and
  both PNG galleries were produced from the real test subset.
- Under identical weights, inputs and RNG, one native-size joint training update
  (including all gradients and updated model state), ODE/SDE outputs, CD metrics,
  and real dataset ordering/normalization matched the original research code
  exactly. Both retained Chair and Airplane 800,000-update research checkpoints
  also loaded into the release model: native prior, ODE and SDE outputs matched
  the research implementation bit for bit at 2,048 points. A separate two-shape
  real Airplane test evaluation produced ODE/SDE samples, all native metrics and
  posterior diagnostics. No trained checkpoints are distributed.
  This comparison isolates pruning/packaging from fresh initialization
  RNG allocation; identical seeds across different source versions are not a
  promise of identical newly initialized weights.

The tests above do not measure sample quality. Four updates and two test shapes
are far too small for a scientific comparison, and no test metrics are presented
as paper results. The original research tree and its pre-existing uncommitted
changes were retained unchanged.

## Scope and unverified paths

The GitHub workflow uses Linux CPU wheels and Python 3.11, with a 15-minute limit
and manual dispatch support. Local macOS results do not establish Linux CI success;
the workflow result is the live authority for that environment.

Airplane shares the same architecture and pipeline, but its external official
initialization download hit the provider's temporary download quota during this
local release check. Its known asset hash is documented; a fresh official asset
load and new Airplane training are not claimed here. Existing Airplane checkpoint
sampling and real-data evaluation were checked as described above. Chair provides
the fresh-initialization training/resume evidence. Neither GPU execution, BF16, CUDA derivatives,
DDP, full training, timing comparisons nor paper-level metrics were validated.
