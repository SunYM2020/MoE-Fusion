# Getting Started

This page provides basic tutorials about the usage of MoE-Fusion.
For installation instructions, please see [INSTALL.md](INSTALL.md).


## Prepare fusion dataset.
It is recommended to symlink the dataset root to `MoE-Fusion/data`.

Here, we give an example for data preparation of M3FD.

First, make sure your initial data are in the following structure.
```
data/M3FD
├── Vis
├── Ir
├──train.txt
├──test.txt
└── Annotation
```

## Train a model

All outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.

### Train
```python
python tools/train.py configs/MoE_Fusion.py
```

## Inference with trained models

### Test

You can use the following commands to test a dataset.

```python
python demo/demo_image_fusion.py
```

For more information on how it works, you can refer to [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md) .
