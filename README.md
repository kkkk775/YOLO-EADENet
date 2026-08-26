# YOLO-EADENet
Official PyTorch implementation of YOLO-EADENet: Edge Aware Detail Enhanced Pedestrian Detection Network, including GEIT, CA-HSFPN, and LSDECD.

This folder collects the custom modules used by the YOLO12-based model so they can be released separately from the full experimental codebase.

## Modules

- `ca_hsfpn/ca_hsfpn.py`
  - `CA_HSFPN`: coordinate attention for the HSFPN/HAFPN fusion path.
  - `Multiply` and `Add`: auxiliary fusion operators used by the YAML neck/head.

- `global_edge_information_transfer1/global_edge_information_transfer1.py`
  - `SobelConv`: edge response extraction.
  - `MutilScaleEdgeInfoGenetator`: multi-scale edge feature generation.
  - `ConvEdgeFusion`: fusion of edge features and backbone features.

- `he_lsdecd/he_lsdecd.py`
  - `DEConv`: detail-enhanced convolution.
  - `DEConv_GN`: detail-enhanced convolution with GroupNorm.
  - `Detect_LSDECD`: lightweight shared detail-enhanced convolutional detection head.

## Integration notes

These files are intended as clear release snippets. To use them inside Ultralytics YAML parsing, import the classes in the project's module registry, for example in `ultralytics/nn/tasks.py` and the relevant `__init__.py`/`extra_modules` imports, then reference the class names in the model YAML.
