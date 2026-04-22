#  **uois_toolkit**  

A toolkit for **Unseen Object Instance Segmentation (UOIS)**  
![banner](banner.svg)

[![Sanity Check](https://github.com/jishnujayakumar/uois_toolkit/actions/workflows/sanity_check.yml/badge.svg)](https://github.com/jishnujayakumar/uois_toolkit/actions/workflows/sanity_check.yml)
[![PyPI version](https://img.shields.io/pypi/v/uois-toolkit?color=blue)](https://pypi.org/project/uois-toolkit/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/uois-toolkit?period=total&units=ABBREVIATION&left_color=BLACK&right_color=RED&left_text=downloads)](https://pepy.tech/projects/uois-toolkit)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/jishnujayakumar/uois_toolkit?style=social)](https://github.com/jishnujayakumar/uois_toolkit/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/jishnujayakumar/uois_toolkit)](https://github.com/jishnujayakumar/uois_toolkit/issues)

A PyTorch-based toolkit for loading and processing datasets for **Unseen Object Instance Segmentation (UOIS)**. This repository provides a standardized, easy-to-use interface for several popular UOIS datasets, simplifying the process of training and evaluating segmentation models.

---

## Table of Contents

- [Installation](#installation)
- [Supported Datasets](#supported-datasets)
- [Usage Example](#usage-example)
- [Testing](#testing)
- [For Maintainers](#for-maintainers)
- [License](#license)

---

## Installation

### Prerequisites
- Python 3.9+
- An environment manager like `conda` is recommended.

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jishnujayakumar/uois_toolkit.git
    cd uois_toolkit
    ```

2.  **Install the package:**
    Installing in editable mode (`-e`) allows you to modify the source code without reinstalling. The command will automatically handle all necessary dependencies listed in `pyproject.toml`.
    ```bash
    pip install -e .
    ```

**Note about detectron2**

This project depends on `detectron2` for some dataset utilities and mask handling. `detectron2` includes C++ extensions and must be built for your platform — it cannot always be installed as a pure Python wheel. Please follow the official installation instructions in the Detectron2 meta-repository and install a version compatible with your PyTorch and CUDA (or CPU-only) environment before running the tests or using the datasets:

- Detectron2 installation guide and wheels: https://github.com/facebookresearch/detectron2

On many systems you can install a compatible CPU-only wheel using the prebuilt index, or build from source if needed. If you are running on CI, ensure the runner has the necessary build tools and compatible PyTorch version.

---

## Supported Datasets

This toolkit provides dataloaders for the following datasets:

- Tabletop Object Discovery (TOD)
- OCID
- OSD
- Robot Pushing
- iTeach-HumanPlay

### Download Links

- **Main Datasets (TOD, OCID, OSD, Robot Pushing)**:
  - [**Download from Box**](https://utdallas.box.com/v/uois-datasets)
  - [**Robot Pushing**](https://utdallas.app.box.com/s/yipcemru6qsbw0wj1nsdxq1dw5mjbtiq)
- **iTeach-HumanPlay Dataset**:
  - **D5**: [Download](https://utdallas.box.com/v/iTeach-HumanPlay-D5)
  - **D40**: [Download](https://utdallas.box.com/v/iTeach-HumanPlay-D40)
  - **Test**: [Download](https://utdallas.box.com/v/iTeach-HumanPlay-Test)

### Directory Setup

It is recommended to organize the downloaded datasets into a single `DATA/` directory for convenience, though you can specify the path to each dataset individually.

---

## Usage

### Quick Start — Load Any Dataset

```python
from uois_toolkit import get_datamodule, cfg

# Available datasets: "tabletop", "ocid", "osd", "robot_pushing", "iteach_humanplay"
dm = get_datamodule(
    dataset_name="ocid",
    data_path="/path/to/OCID-dataset",
    batch_size=4,
    num_workers=2,
    config=cfg
)

dm.setup()
batch = next(iter(dm.train_dataloader()))

print("Image:", batch["image_color"].shape)   # [B, 3, H, W]
print("Depth:", batch["depth"].shape)          # [B, C, H, W]
print("Annotations:", len(batch["annotations"][0]))  # list of dicts per image
```

### Access Individual Datasets Directly

```python
from uois_toolkit.datasets import OCIDDataset, OSDDataset, TabletopDataset
from uois_toolkit.datasets import RobotPushingDataset, iTeachHumanPlayDataset
from uois_toolkit import cfg

# Load a single dataset
dataset = OCIDDataset(image_set="test", data_path="/path/to/OCID-dataset", config=cfg)
print(f"Dataset size: {len(dataset)}")

# Get a single sample
sample = dataset[0]
print("Keys:", sample.keys())
# → file_name, image_id, height, width, image_color, depth, raw_depth, annotations
```

### Batch Format

Each batch contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `image_color` | `[B, 3, H, W]` | RGB image (float, 0-255) |
| `depth` | `[B, C, H, W]` | Depth map (normalized or XYZ) |
| `raw_depth` | varies | Original depth before normalization |
| `annotations` | `List[List[Dict]]` | Per-image list of object annotations |
| `file_name` | `List[str]` | Source file paths |
| `height`, `width` | `List[int]` | Image dimensions |

Each annotation dict contains:
- `bbox`: `[x1, y1, x2, y2]` in XYXY_ABS format
- `segmentation`: RLE-encoded binary mask (pycocotools)
- `category_id`: always `1` (unseen object)

### Compute Evaluation Metrics

```python
import numpy as np
from uois_toolkit.metrics import compute_metrics, get_available_metrics

# See all available metrics
print(get_available_metrics())
# → ['precision', 'recall', 'f1_score', 'iou', 'iou_at_0.7']

# Create example masks
gt_mask = np.zeros((480, 640), dtype=np.uint8)
gt_mask[100:300, 150:400] = 1  # ground truth object region

pred_mask = np.zeros((480, 640), dtype=np.uint8)
pred_mask[110:290, 160:390] = 1  # predicted object region

# Compute metrics
results = compute_metrics(gt_mask, pred_mask, ["f1_score", "iou", "precision", "recall"])
print(results)
# → {'f1_score': 0.89, 'iou': 0.80, 'precision': 0.92, 'recall': 0.86}
```

### Use with PyTorch Lightning

```python
from uois_toolkit import get_datamodule, cfg
import pytorch_lightning as pl

dm = get_datamodule("tabletop", "/path/to/tabletop", batch_size=8, config=cfg)

model = YourLightningModel()
trainer = pl.Trainer(accelerator="auto", max_epochs=10)
trainer.fit(model, datamodule=dm)
```

---

## Testing

### Local Validation

The repository includes a `pytest` suite to verify that the dataloaders and processing pipelines are working correctly.

To run the tests, you must provide the root paths to your downloaded datasets using the `--dataset_path` argument.

```bash
python -m pytest test/test_datamodule.py -v \
  --dataset_path tabletop=/path/to/your/data/tabletop \
  --dataset_path ocid=/path/to/your/data/ocid \
  --dataset_path osd=/path/to/your/data/osd
  # Add other dataset paths as needed
```
**Note**: You only need to provide paths for the datasets you wish to test.

### Continuous Integration

This repository uses **GitHub Actions** to perform automated sanity checks on every push and pull request to the `main` branch. This workflow ensures that:
1. The package installs correctly.
2. The code adheres to basic linting standards.
3. All core modules remain importable.

This automated process helps maintain code quality and prevents the introduction of breaking changes.

---

## For Maintainers

<details>
<summary>Click to expand for PyPI publishing instructions</summary>

```bash
# 1. Install build tools
python -m pip install build twine

# 2. Clean previous builds
rm -rf build/ dist/ *.egg-info

# 3. Build the distribution files
python -m build

# 4. Upload to PyPI (requires a configured PyPI token)
twine upload dist/*
```

</details>

---

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{uois_toolkit,
  author = {Jishnu Jaykumar P and Aggarwal, Avaya and Maheshwari, Animesh},
  title = {uois_toolkit: A PyTorch Toolkit for Unseen Object Instance Segmentation},
  year = {2025},
  url = {https://github.com/jishnujayakumar/uois_toolkit}
}
```

### Dataset Citations

If you use any of the supported datasets, please also cite the original works:

<details>
<summary><b>Tabletop Object Dataset (TOD)</b></summary>

```bibtex
@inproceedings{xiang2020learning,
  title={Learning RGB-D Feature Embeddings for Unseen Object Instance Segmentation},
  author={Xiang, Yu and Xie, Christopher and Mousavian, Arsalan and Fox, Dieter},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2020}
}
```
</details>

<details>
<summary><b>OCID (Object Clutter Indoor Dataset)</b></summary>

```bibtex
@inproceedings{suchi2019easylabel,
  title={EasyLabel: A Semi-Automatic Pixel-wise Object Annotation Tool for Creating Robotic RGB-D Datasets},
  author={Suchi, Markus and Patten, Timothy and Fischinger, David and Vincze, Markus},
  booktitle={International Conference on Robotics and Automation (ICRA)},
  year={2019}
}
```
</details>

<details>
<summary><b>OSD (Object Segmentation Dataset)</b></summary>

```bibtex
@inproceedings{richtsfeld2012segmentation,
  title={Segmentation of Unknown Objects in Indoor Environments},
  author={Richtsfeld, Andreas and Morwald, Thomas and Prankl, Johann and Zillich, Michael and Vincze, Markus},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2012}
}
```
</details>

<details>
<summary><b>Robot Pushing Dataset</b></summary>

```bibtex
@inproceedings{lu2023sss,
  title={Self-Supervised Unseen Object Instance Segmentation via Long-Term Robot Interaction},
  author={Lu, Yangxiao and Khargonkar, Ninad and Xu, Zesheng and Averill, Charles and Palanisamy, Kamalesh and Hang, Kaiyu and Guo, Yunhui and Ruozzi, Nicholas and Xiang, Yu},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2023}
}
```
</details>

<details>
<summary><b>iTeach-HumanPlay Dataset</b></summary>

```bibtex
@misc{p2026iteach,
  title={iTeach: In the Wild Interactive Teaching for Failure-Driven Adaptation of Robot Perception},
  author={Jishnu Jaykumar P and Cole Salvato and Vinaya Bomnale and Jikai Wang and Yu Xiang},
  year={2026},
  eprint={2410.09072},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2410.09072}
}
```
</details>

## Contributing

Contributions are welcome! Please open an issue or submit a pull request. See [issues](https://github.com/jishnujayakumar/uois_toolkit/issues) for open tasks.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
