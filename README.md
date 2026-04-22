<p align="center">
  <img src="media/banner.svg" alt="uois_toolkit" width="100%"/>
</p>

<p align="center">
  <a href="https://github.com/jishnujayakumar/uois_toolkit/actions/workflows/sanity_check.yml"><img src="https://github.com/jishnujayakumar/uois_toolkit/actions/workflows/sanity_check.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/uois-toolkit/"><img src="https://img.shields.io/pypi/v/uois-toolkit?color=blue" alt="PyPI"></a>
  <a href="https://pepy.tech/projects/uois-toolkit"><img src="https://static.pepy.tech/personalized-badge/uois-toolkit?period=total&units=ABBREVIATION&left_color=BLACK&right_color=RED&left_text=downloads" alt="Downloads"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="https://github.com/jishnujayakumar/uois_toolkit/stargazers"><img src="https://img.shields.io/github/stars/jishnujayakumar/uois_toolkit?style=social" alt="Stars"></a>
</p>

<p align="center">
  <b>One unified interface for 5 UOIS datasets.</b><br/>
  Load, augment, train, and evaluate &mdash; in 3 lines of code.
</p>

---

## Why uois_toolkit?

| Problem | Solution |
|---------|----------|
| Every UOIS dataset ships its own format and loader | **Unified API** &mdash; one interface across all 5 datasets |
| Evaluation setup eats up research time | **Built-in metrics** &mdash; F1, IoU, Precision, Recall in a single call |
| Mixing synthetic and real data takes custom wiring | **Multi-dataset DataModule** with balanced sampling out of the box |
| Reproducing baselines means writing glue code | **Lightning-native** &mdash; drop into any training loop |

---

## Quickstart

```bash
pip install uois-toolkit
```

```python
from uois_toolkit import get_datamodule, cfg

dm = get_datamodule("ocid", "/path/to/OCID", batch_size=4, config=cfg)
dm.setup()
batch = next(iter(dm.train_dataloader()))
# batch["image_color"]  → [B, 3, H, W]
# batch["depth"]        → [B, C, H, W]
# batch["annotations"]  → per-image bboxes + RLE masks
```

Same API for all datasets &mdash; just swap the name: `"tabletop"`, `"osd"`, `"robot_pushing"`, `"iteach_humanplay"`.

---

## Supported Datasets

| Dataset | Type | Images | Setting | Source |
|---------|------|--------|---------|--------|
| **[Tabletop (TOD)](https://utdallas.box.com/v/uois-datasets)** | Synthetic | ~280K | Rendered household scenes | Xiang et al., CoRL 2020 |
| **[OCID](https://utdallas.box.com/v/uois-datasets)** | Real | 2,390 | Cluttered tabletop | Suchi et al., ICRA 2019 |
| **[OSD](https://utdallas.box.com/v/uois-datasets)** | Real | 111 | Sparse tabletop | Richtsfeld et al., IROS 2012 |
| **[Robot Pushing](https://utdallas.app.box.com/s/yipcemru6qsbw0wj1nsdxq1dw5mjbtiq)** | Real | 428 | Robot pushing objects | Lu et al., RSS 2023 |
| **[iTeach-HumanPlay](https://utdallas.box.com/v/iTeach-HumanPlay-D5)** | Real | 14K+ | Human-object interaction | P et al., arXiv 2024 |

---

## Usage

### Load a single dataset

```python
from uois_toolkit.datasets import OCIDDataset
from uois_toolkit import cfg

dataset = OCIDDataset(image_set="test", data_path="/path/to/OCID", config=cfg)
sample = dataset[0]
# Keys: file_name, image_id, height, width, image_color, depth, raw_depth, annotations
```

### Evaluate predictions

```python
import numpy as np
from uois_toolkit.metrics import compute_metrics

gt_mask = ...   # [H, W] binary
pred_mask = ... # [H, W] binary

results = compute_metrics(gt_mask, pred_mask, ["f1_score", "iou", "precision", "recall"])
# {'f1_score': 0.89, 'iou': 0.80, 'precision': 0.92, 'recall': 0.86}
```

### Train with PyTorch Lightning

```python
from uois_toolkit import get_datamodule, cfg

dm = get_datamodule("tabletop", "/data/tabletop", batch_size=8, config=cfg)
trainer = pl.Trainer(accelerator="auto", max_epochs=10)
trainer.fit(model, datamodule=dm)
```

### Batch format

| Key | Shape | Description |
|-----|-------|-------------|
| `image_color` | `[B, 3, H, W]` | RGB image (float) |
| `depth` | `[B, C, H, W]` | Depth map |
| `annotations` | `List[List[Dict]]` | Per-image object annotations |

Each annotation: `{"bbox": [x1,y1,x2,y2], "segmentation": <RLE>, "category_id": 1}`

---

## Installation from source

```bash
git clone https://github.com/jishnujayakumar/uois_toolkit.git
cd uois_toolkit
pip install -e .
```

> **Note:** [detectron2](https://github.com/facebookresearch/detectron2) is needed for mask utilities. Install a build that matches your PyTorch + CUDA version.

---

## Testing

```bash
python -m pytest test/test_datamodule.py -v \
  --dataset_path tabletop=/data/tabletop \
  --dataset_path ocid=/data/OCID \
  --dataset_path osd=/data/OSD
```

CI runs on every push and PR via GitHub Actions.

---

## Citation

```bibtex
@software{uois_toolkit,
  author = {Jishnu Jaykumar P and Aggarwal, Avaya and Maheshwari, Animesh},
  title = {uois_toolkit: A PyTorch Toolkit for Unseen Object Instance Segmentation},
  year = {2025},
  url = {https://github.com/jishnujayakumar/uois_toolkit}
}
```

<details>
<summary><b>Dataset citations</b> (click to expand)</summary>

**Tabletop (TOD)** &mdash; Xiang et al., "Learning RGB-D Feature Embeddings for Unseen Object Instance Segmentation", CoRL 2020

**OCID** &mdash; Suchi et al., "EasyLabel: A Semi-Automatic Pixel-wise Object Annotation Tool for Creating Robotic RGB-D Datasets", ICRA 2019

**OSD** &mdash; Richtsfeld et al., "Segmentation of Unknown Objects in Indoor Environments", IROS 2012

**Robot Pushing** &mdash; Lu et al., "Self-Supervised Unseen Object Instance Segmentation via Long-Term Robot Interaction", RSS 2023

**iTeach-HumanPlay** &mdash; P et al., "iTeach: In the Wild Interactive Teaching for Failure-Driven Adaptation of Robot Perception", [arXiv:2410.09072](https://arxiv.org/abs/2410.09072)

</details>

---

## Contributors

Built and polished by:

<a href="https://github.com/OnePunchMonk"><img src="https://github.com/OnePunchMonk.png" width="60" style="border-radius:50%" alt="OnePunchMonk"/></a>&nbsp;
<a href="https://github.com/AnimeshMaheshwari22"><img src="https://github.com/AnimeshMaheshwari22.png" width="60" style="border-radius:50%" alt="Animesh Maheshwari"/></a>

[**@OnePunchMonk**](https://github.com/OnePunchMonk) &bull; [**@AnimeshMaheshwari22**](https://github.com/AnimeshMaheshwari22)

PRs welcome! See [open issues](https://github.com/jishnujayakumar/uois_toolkit/issues).

## License

[MIT](LICENSE)
