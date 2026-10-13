# Data and category initialization

Use the assets linked by the [official upstream repository](https://github.com/luost26/diffusion-point-cloud#datasets-and-pretrained-models).
Download them to a directory of your choice; this repository never downloads data
or checkpoints automatically and does not redistribute them. Review the original
ShapeNet dataset terms and the asset provider's terms before obtaining them.

| File | Provider link | SHA-256 |
|---|---|---|
| ShapeNet HDF5 | [shapenet.hdf5](https://drive.google.com/file/d/1WdooIyOMVt_HVd_A0QOM9DTTX41O65jQ/view) | `0dab3949d17d5b7209980f042574d7e2b5bb97b24ff7b67ecbb9cfe84a034b78` |
| Chair encoder/prior initialization | [GEN_chair.pt](https://drive.google.com/file/d/1qXtJCOevgdyhxwllJ5G4e7ZGH0sFsDXF/view) | `50de96ce2f91b363a70242960b5a0cf86695f7dd78f283705620807294f30dfc` |
| Airplane encoder/prior initialization | [GEN_airplane.pt](https://drive.google.com/file/d/1NTVl1RKcIJDildrTx-jUpBJkDNzg55ld/view) | `b1552904c2c156e777dec140d9eac7a1279d42f76242e4ebe11d832e22a65eae` |

Compute a file's hash with `shasum -a 256 FILE` (macOS) or `sha256sum FILE`
(Linux). The HDF5 and Chair hashes were rechecked during this CPU release;
the Airplane hash is the recorded identifier of the upstream asset used in the
research implementation. A fresh Airplane download encountered the provider's
quota limit; existing Airplane PFM checkpoint sampling was checked instead.
See the validation document for the exact scope.

The HDF5 groups are `03001627` (Chair) and `02691156` (Airplane), each with
`train`, `val`, and `test` arrays of `[shapes,2048,3]`. Chair has 5,602 train and
989 test shapes; Airplane has 3,438 train and 607 test shapes. The public pipeline
uses **train and test only**, preserving upstream splits. The supplied validation
split contains exact-array overlap with train/test; it is not used for checkpoint
selection. The training probe is explicitly drawn from training data.

Training subtracts each shape's mean and divides by its flattened sample standard
deviation (`shape_unit`). Evaluation separately centers and scales every generated
and reference cloud by the maximum bounding-box half-extent (`shape_bbox`).
Category data are ordered by original index then shuffled with Python seed 2020,
matching the native loader. A subset uses the prefix of that order.

Initialization imports only the category's PointNet encoder and flow-prior
weights. The coordinate PFM is initialized separately. Sampling checkpoints carry
all three models and do not depend on `GEN_*.pt`. Training/resume record full-file
hashes of data and initialization; changing either requires a new run.
