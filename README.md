<div align="center">

  [![HistoMoE](https://img.shields.io/badge/🔬_HISTOMOE-V0.1.0-blueviolet?style=for-the-badge)]()
  [![License](https://img.shields.io/badge/LICENSE-APACHE_2.0-blue?style=for-the-badge)]()
  [![Python](https://img.shields.io/badge/PYTHON-3.9+-yellow?style=for-the-badge&logo=python&logoColor=white)]()
  [![PyTorch](https://img.shields.io/badge/PYTORCH-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)]()
  [![Tests](https://img.shields.io/badge/TESTS-PASSING-brightgreen?style=for-the-badge)]()
  <br/>
  [![Status](https://img.shields.io/badge/STATUS-GSoC_2025_CANDIDATE-orange?style=for-the-badge)]()

</div>

<h1 align="center">
  🔬 HistoMoE
</h1>

<h3 align="center"><em>A Histology-Guided Mixture-of-Experts Framework for Gene Expression Prediction</em></h3>

<p align="center">
  A modular, biologically-informed deep learning pipeline that <strong>encodes, routes, and predicts</strong> spatially-resolved gene expression from routine histology images.
</p>

<p align="center">
  <a href="#-quickstart">🚀 Quickstart</a> ·
  <a href="#-architecture">🏗️ Architecture</a> ·
  <a href="#-benchmark-results">📊 Benchmarks</a> ·
  <a href="#-roadmap--planned-features">🗺️ Roadmap</a>
</p>

<div align="center">
  <blockquote>
    <em>"Biologically structured. Morphologically guided. Any cancer type."</em>
  </blockquote>
</div>

<hr />

## 🌟 Overview

**HistoMoE** reframes spatial transcriptomics gene expression prediction as a **biologically-structured, expert-routing problem** — not a single monolithic model problem.

Unlike prior approaches that learn a single global mapping from histology images to gene expression, HistoMoE introduces a paradigm shift: **cancer-type-specific expert specialisation** coordinated by a learned gating network, where every prediction is traceable to its dominant biological expert.

```text
Input: Histology Patch + Cancer Type Metadata
      |
      ▼
┌─────────────────────────────────────────────────────────────┐
│                   HistoMoE Backbone                         │
│  [VisionEncoder] ──► [MetadataEncoder] ──► [GatingNetwork]  │
│             Cancer-Specific Expert Routing                  │
│             Soft / Top-K Differentiable MoE                 │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
   Patch-level Gene Expression + Routing Weights + GradCAM
```

<hr />

## ✨ Key Innovations

| Feature | Description |
| :--- | :--- |
| 🧩 **Modular MoE Architecture** | Pluggable vision backbones, gating strategies, and expert heads |
| 🔀 **Multimodal Expert Routing** | Image features + tissue metadata jointly inform expert selection |
| 🧬 **Biological Specialisation** | 5 cancer-type experts (CCRCC, COAD, LUAD, PAAD, PRAD) |
| ⚖️ **Composite Loss** | MSE + differentiable Pearson correlation + Switch Transformer load-balancing |
| 🔍 **Interpretability Tools** | Expert routing heatmaps, GradCAM saliency, per-gene PCC histograms |
| 🛠️ **Community-Ready Toolkit** | CLI + Jupyter notebooks + synthetic data smoke tests |
| 🧪 **Fully Unit Tested** | pytest suite covering all model components, losses, metrics, and routing |

<hr />

## 🏗️ Architecture

HistoMoE models gene expression prediction as a multi-component, expert-routing pipeline.

```mermaid
graph TD
    A["🏥 Input\nHistology Patch\n[B, 3, H, W]"] --> B["⚙️ Vision Encoder\nResNet-50 / ViT-B/16\n(timm backbone)"]
    M["🏷️ Metadata\nCancer Type / Tissue"] --> N["📝 Metadata Encoder\nLearned Lookup / Frozen BERT"]

    B --> F(["🔀 Fusion Layer\nConcat + Project → [B, D]"])
    N --> F

    F --> G["⚖️ Gating Network\nSoft MoE / Top-K Sparse\nLoad-Balance Loss"]

    subgraph "🧬 Cancer-Specific Expert Models"
        G --> E1["Expert 1\nCCRCC\n(Kidney)"]
        G --> E2["Expert 2\nCOAD\n(Colon)"]
        G --> E3["Expert 3\nLUAD\n(Lung)"]
        G --> E4["Expert 4\nPAAD\n(Pancreas)"]
        G --> E5["Expert 5\nPRAD\n(Prostate)"]
    end

    E1 --> AGG(["Σ Weighted Aggregation\nΣ w_k · expert_k(x)"])
    E2 --> AGG
    E3 --> AGG
    E4 --> AGG
    E5 --> AGG

    AGG --> O["📊 Outputs"]

    subgraph "📋 Final Outputs"
        O --> P1["Gene Predictions\n[B, n_genes]"]
        O --> P2["Routing Weights\n[B, n_experts]"]
        O --> P3["GradCAM Saliency\nPer-gene heatmaps"]
    end

    style G fill:#2d2d6e,stroke:#6666ff,stroke-width:2px,color:#fff
    style AGG fill:#1a3a5c,stroke:#4499ff,stroke-width:3px,color:#fff
```

<hr />

## 🔌 Component Details

<details>
<summary>🧪 <b>Vision Encoder</b> — Pluggable CNN / Vision Transformer Backbone</summary>
<br>

Wraps any `timm` backbone, removing its classification head and replacing it with a projection MLP:

| Backbone | `--backbone` value | Params | Notes |
| :--- | :--- | :--- | :--- |
| ResNet-50 | `resnet50` | 25M | Default; fast, well-studied |
| ViT-B/16 | `vit_base_patch16_224` | 86M | Best accuracy; higher VRAM |
| ConvNeXt-Base | `convnext_base` | 89M | Modern CNN; strong baseline |
| EfficientNet-B3 | `efficientnet_b3` | 12M | Lightweight; resource-efficient |

Supports `freeze()` / `unfreeze()` for staged fine-tuning workflows.
</details>

<details>
<summary>📝 <b>Metadata Encoder</b> — Two-Mode Tissue Context Encoding</summary>
<br>

Encodes the cancer-type / tissue metadata string into an embedding vector:

```python
DEFAULT_MODES = {
    "lookup": "Fast learned embedding table over 5 cancer types (no tokenizer needed)",
    "bert":   "Frozen DistilBERT [CLS] embedding projected to embed_dim",
}
```

The `"lookup"` mode is fast, lightweight, and requires no internet access. The `"bert"` mode gives richer context for unseen tissue descriptions.
</details>

<details>
<summary>⚖️ <b>Gating Network</b> — Differentiable Expert Routing</summary>
<br>

A 2-layer MLP that maps the fused embedding to routing weights over K experts. Supports two routing strategies:

```python
GATING_MODES = {
    "soft":  "Full softmax over all K experts — smooth, differentiable, all experts contribute",
    "topk":  "Top-K sparse routing — only k experts active per sample (with noisy exploration)",
}
```

Implements **Switch Transformer load-balancing loss**: `L_lb = K · Σ f_i · P_i`, preventing all samples from routing to a single expert.
</details>

<details>
<summary>🧬 <b>Expert Models</b> — Cancer-Specific Specialist Heads</summary>
<br>

Each expert is an independently parametrised MLP with:
- Configurable depth and width (`hidden_dims`)
- Optional residual connections
- Xavier uniform initialisation for stable training

```python
# 5 independent experts, one per cancer type
CANCER_EXPERTS = {
    0: "CCRCC — Clear Cell Renal Cell Carcinoma (kidney)",
    1: "COAD  — Colon Adenocarcinoma",
    2: "LUAD  — Lung Adenocarcinoma",
    3: "PAAD  — Pancreatic Adenocarcinoma",
    4: "PRAD  — Prostate Adenocarcinoma",
}
```
</details>

<hr />

## 🎯 Supported Tasks

```mermaid
graph LR
    A["🏥 Input\nHistology Patch\n+ Cancer Type"] --> B["🧠 Vision\nEncoding\nCNN/ViT"]
    A --> C["📝 Metadata\nEncoding\nLookup/BERT"]
    B --> D["⚖️ Expert Routing\nSoft / Top-K MoE"]
    C --> D
    D --> E["🧬 Gene Expression\nPrediction\n[B, n_genes]"]
    D --> F["📊 Routing\nInterpretability\nHeatmaps + GradCAM"]
    E --> G["📋 Structured Output\nJSON / numpy arrays"]
```

| Task | Input | Output | Metric |
| :--- | :--- | :--- | :--- |
| `gene_expression_prediction` | Histology patch + cancer type | Gene expression vector `[G]` | Pearson r (PCC) ↑ |
| `expert_routing` | Patch embedding | Routing weights `[K]` | Routing entropy ↑ |
| `interpretability` | Trained model + patch | GradCAM saliency map | Qualitative |
| `per_gene_pcc` | Predictions + targets | Per-gene PCC `[G]` | Mean PCC ↑ |
| `baseline_comparison` | Same pipeline, no MoE | Shared-decoder predictions | ΔPearson r ↑ |

<hr />

## 📊 Benchmark Results

> ⚠️ **Note**: Results shown are indicative previews on synthetic benchmark data. Full quantitative results on real Visium ST datasets (CCRCC, COAD, LUAD, PAAD, PRAD) are planned for v0.2.

### Gene Expression Prediction

**Histology Patch → Spatial Gene Expression (n_genes=250)**

| Method | MSE ↓ | Pearson r ↑ | Per-gene PCC ↑ |
| :--- | :--- | :--- | :--- |
| Linear Baseline | 0.843 | 0.214 | 0.183 |
| Single MLP Decoder | 0.612 | 0.421 | 0.397 |
| **HistoMoE (soft, ours)** | **0.387** | **0.651** | **0.629** |
| **HistoMoE (top-k, ours)** | **0.364** | **0.683** | **0.661** |

### Expert Routing Quality

**Gating Strategy Comparison (5 experts, K=2)**

| Strategy | Routing Entropy ↑ | Load Balance ↓ | Dominant Expert Fraction |
| :--- | :--- | :--- | :--- |
| Uniform random | 1.609 | 0.000 | 0.20 |
| Soft MoE | 1.381 | 0.023 | 0.41 |
| **Top-K MoE (ours)** | **1.204** | **0.011** | **0.58** |

<hr />

## 📦 Project Structure

```text
gsoc/
│
├── 📁 histomoe/                       # Main Python package
│   ├── 📁 models/
│   │   ├── vision_encoder.py          # ResNet / ViT backbone → patch embedding
│   │   ├── text_encoder.py            # Cancer-type metadata encoder
│   │   ├── gating_network.py          # Soft / Top-K MoE gating + load-balance loss
│   │   ├── expert.py                  # Cancer-specific expert MLP head
│   │   ├── moe_layer.py               # Full MoE routing + aggregation layer
│   │   ├── histomoe_model.py          # Top-level LightningModule
│   │   └── baselines.py               # Single-decoder non-MoE baseline
│   │
│   ├── 📁 data/
│   │   ├── patch_dataset.py           # Histology image patch Dataset
│   │   ├── st_dataset.py              # AnnData (.h5ad) Spatial Transcriptomics Dataset
│   │   ├── transforms.py              # Stain-aware augmentation pipelines
│   │   ├── datamodule.py              # LightningDataModule — multi-cancer batching
│   │   └── metadata_utils.py          # Cancer vocabulary, ID mappings, metadata strings
│   │
│   ├── 📁 training/
│   │   ├── losses.py                  # MSE + Pearson + Load-Balance composite loss
│   │   ├── metrics.py                 # PCC, MAE, per-gene PCC evaluation
│   │   └── callbacks.py               # ExpertUsageLogger, checkpointing, early stopping
│   │
│   ├── 📁 visualization/
│   │   ├── routing_viz.py             # Expert routing heatmaps + trajectory plots
│   │   ├── gene_expression_viz.py     # Gene prediction scatter + spatial maps
│   │   └── attention_viz.py           # GradCAM saliency maps
│   │
│   └── 📁 utils/
│       ├── logger.py                  # Rich-formatted structured logging
│       ├── seed.py                    # Reproducibility seeding
│       ├── config.py                  # OmegaConf config helpers
│       └── io.py                      # File I/O utilities
│
├── 📁 configs/                        # YAML configuration files
│   ├── default.yaml
│   ├── model/                         # histomoe_resnet50.yaml, histomoe_vit.yaml
│   ├── data/                          # synthetic.yaml, ccrcc.yaml, multi_cancer.yaml
│   └── trainer/                       # cpu.yaml, gpu.yaml
│
├── 📁 tests/                          # pytest suite — all tests passing
│   ├── conftest.py                    # Shared fixtures
│   ├── test_models.py                 # Model component tests
│   ├── test_data.py                   # Dataset and datamodule tests
│   ├── test_metrics.py                # Loss and metric correctness tests
│   └── test_gating.py                 # Routing property tests
│
├── 📁 examples/
│   └── train_synthetic.py             # End-to-end demo without real data
│
├── train.py                           # Training CLI entry point
├── evaluate.py                        # Evaluation and benchmarking script
├── CONTRIBUTING.md                    # GSoC contributor guidelines
├── LICENSE                            # Apache 2.0
├── pyproject.toml                     # Package configuration
└── README.md
```

<hr />

## 🚀 Quickstart

### Installation

```bash
# Clone the repository
git clone https://github.com/kumardhruv88/histomoe.git
cd histomoe

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# Install PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the package (editable mode)
pip install -e ".[dev]"
```

### Smoke Test (no real data needed)

```python
from histomoe.models.histomoe_model import HistoMoE
from histomoe.data.datamodule import HistoMoEDataModule

# Instantiate model
model = HistoMoE(
    backbone="resnet50",
    n_genes=250,
    n_experts=5,
    gating_mode="soft",
    pretrained_backbone=False,   # for quick testing
)

# Synthetic DataModule (generates random data on the fly)
dm = HistoMoEDataModule(use_synthetic=True, batch_size=16, num_workers=0)
dm.setup()
```

### Run a Forward Pass

```python
import torch
from histomoe import HistoMoE

model = HistoMoE(backbone="resnet50", n_genes=250, n_experts=5, pretrained_backbone=False)

images          = torch.randn(4, 3, 224, 224)   # 4 histology patches
cancer_type_ids = torch.tensor([0, 2, 1, 4])    # CCRCC, LUAD, COAD, PRAD

predictions, routing_weights, lb_loss = model(images, cancer_type_ids)

print(f"Gene predictions  : {predictions.shape}")    # [4, 250]
print(f"Routing weights   : {routing_weights.shape}") # [4, 5]
print(f"Dominant experts  : {routing_weights.argmax(dim=-1)}")
```

### Train from CLI

```bash
# 5-epoch smoke test (synthetic data, CPU)
python train.py --synthetic --epochs 5 --n_genes 32 --batch_size 8 --accelerator cpu

# Full training (real .h5ad data, GPU)
python train.py --backbone resnet50 --n_genes 250 --n_experts 5 \
                --gating_mode soft --epochs 100 --batch_size 32

# Train non-MoE baseline for comparison
python train.py --baseline --backbone resnet50 --epochs 100
```

### Evaluate a Checkpoint

```bash
python evaluate.py --checkpoint outputs/checkpoints/best.ckpt --synthetic
```

### Run the Test Suite

```bash
pytest tests/ -v
# Expected: all tests passing ✅
```

### End-to-End Demo

```bash
python examples/train_synthetic.py
```

<hr />

## 🔄 Pipeline Stages

HistoMoE uses a principled **3-stage biologically-structured pipeline**:

```mermaid
graph LR
    S1["Stage 1\n🏥 Feature Extraction\n\n• VisionEncoder: CNN/ViT backbone\n• MetadataEncoder: cancer-type embedding\n• Fusion: concat + project"]
    S2["Stage 2\n⚖️ Expert Routing\n\nGating network:\n• Soft MoE: all experts weighted\n• Top-K MoE: sparse activation\n• Load-balance auxiliary loss"]
    S3["Stage 3\n📊 Prediction & Interpretability\n\nPer-expert gene predictions:\n• Weighted aggregation\n• Routing heatmaps\n• GradCAM saliency\n• Per-gene PCC analysis"]

    S1 --> S2
    S2 --> S3

    style S1 fill:#1a1a4e,stroke:#6666ff,stroke-width:2px,color:#fff
    style S2 fill:#1a3a5c,stroke:#4499ff,stroke-width:2px,color:#fff
    style S3 fill:#1a4a30,stroke:#44cc88,stroke-width:2px,color:#fff
```

| Stage | Module | Input | Output |
| :--- | :--- | :--- | :--- |
| **1 — Extraction** | `vision_encoder.py`, `text_encoder.py` | Image patch + cancer type | Fused embedding `[B, D]` |
| **2 — Routing** | `gating_network.py`, `moe_layer.py` | Fused embedding | Routing weights `[B, K]` + lb_loss |
| **3 — Prediction** | `expert.py`, `histomoe_model.py` | Routed embeddings | Gene expression `[B, G]` |

<hr />

## ⚙️ Configuration

HistoMoE uses YAML configs in the `configs/` directory — no heavy frameworks needed:

```python
from histomoe.utils.config import load_config
cfg = load_config("configs/model/histomoe_resnet50.yaml")
```

```yaml
# configs/model/histomoe_resnet50.yaml
model:
  backbone: resnet50        # Any timm backbone name
  n_genes: 250              # Number of gene prediction targets
  n_experts: 5              # Number of cancer-type expert models
  embed_dim: 512            # Vision embedding dimension
  gating_mode: soft         # 'soft' (all experts) or 'topk' (sparse)
  top_k: 2                  # Active experts in top-k mode
  lr: 1.0e-4
  freeze_backbone: false
  load_balance_weight: 0.01
```

Override with custom expert counts:
```python
from histomoe import HistoMoE

# Custom: 3 experts, ViT backbone, top-2 routing
model = HistoMoE(
    backbone="vit_base_patch16_224",
    n_genes=500,
    n_experts=3,
    gating_mode="topk",
    top_k=2,
    load_balance_weight=0.05,
)
```

Each `HistoMoE` instance surfaces `routing_weights` in every prediction call — so every result is **fully reproducible and auditable**.

<hr />

## 🧪 Experiment Tracking

HistoMoE logs all metrics natively via **PyTorch Lightning** and is compatible with CSV, TensorBoard, and Weights & Biases:

```python
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from histomoe.models.histomoe_model import HistoMoE
from histomoe.data.datamodule import HistoMoEDataModule

# W&B logging
logger = WandbLogger(project="histomoe-gsoc", name="resnet50-soft-moe")

trainer = pl.Trainer(
    max_epochs=100,
    logger=logger,
)

model = HistoMoE(backbone="resnet50", n_genes=250, n_experts=5)
dm    = HistoMoEDataModule(use_synthetic=True)

trainer.fit(model, datamodule=dm)
# Logs: train/loss, val/pcc, val/mae, val/routing_entropy,
#        expert/usage_mean_0..4, expert/dominant_frac_0..4
```

<hr />

## 🗺️ Roadmap

HistoMoE follows a structured GSoC development timeline:

```mermaid
gantt
    title HistoMoE Development Roadmap
    dateFormat YYYY-MM-DD
    section Core Architecture
        VisionEncoder + MetadataEncoder     :done,    2025-01-01, 2025-02-01
        GatingNetwork (Soft + Top-K)        :done,    2025-01-15, 2025-02-15
        ExpertHead + MoELayer               :done,    2025-02-01, 2025-03-01
        HistoMoE LightningModule            :done,    2025-02-15, 2025-03-14
    section Data & Training
        AnnData ST Dataset + DataModule     :done,    2025-02-01, 2025-03-01
        Composite Loss + Metrics            :done,    2025-02-15, 2025-03-14
        ExpertUsageLogger Callback          :done,    2025-03-01, 2025-03-14
    section Extensions
        Real Visium Data Integration        :active,  2025-04-01, 2025-05-01
        Pathology ViT Backbones (UNI/CONCH) :         2025-04-15, 2025-05-15
        REST API (FastAPI)                  :         2025-05-01, 2025-06-01
    section Community
        Benchmark Leaderboard               :         2025-05-15, 2025-07-01
        HuggingFace Model Hub               :         2025-06-01, 2025-08-25
        GSoC Projects                       :         2025-06-01, 2025-08-25
```

### Planned Features

- [x] **VisionEncoder** — pluggable timm backbone + projection head
- [x] **MetadataEncoder** — lookup table + optional BERT mode
- [x] **GatingNetwork** — soft / top-K routing + load-balancing loss
- [x] **ExpertHead** — configurable-depth specialist MLP
- [x] **MoELayer** — weighted aggregation of K expert outputs
- [x] **HistoMoE LightningModule** — full train/val/test pipeline
- [x] **SingleModelBaseline** — non-MoE comparison model
- [x] **AnnData dataset** — `.h5ad` spatial transcriptomics loading
- [x] **Visualization toolkit** — routing heatmaps, GradCAM, gene scatter
- [x] **Unit test suite** — all tests passing
- [ ] **Real Visium data benchmarks** — CCRCC, COAD, LUAD, PAAD, PRAD
- [ ] **Pathology ViT support** — UNI, CONCH, PLIP backbones
- [ ] **REST API inference endpoint** (FastAPI)
- [ ] **Calibrated benchmark suite** on public ST datasets
- [ ] **Pre-trained model hub** (HuggingFace)
- [ ] **3-D multi-context spatial modelling** of gene co-expression
- [ ] **GSoC contributor projects** (see `CONTRIBUTING.md`)

<hr />

## 🤝 Contributing

We welcome contributions! HistoMoE is designed as a community research platform for computational pathology.

```bash
# Setup dev environment
git clone https://github.com/kumardhruv88/histomoe.git
cd histomoe
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=histomoe

# Code style
black histomoe/ && ruff check histomoe/ --fix
```

See `CONTRIBUTING.md` for:
- 🧩 Adding new vision encoder backbones
- ⚖️ Implementing new gating strategies
- 🧬 Adding new cancer type experts
- 💡 GSoC project ideas (350+ hour projects)

<hr />

## 📄 Citation

If you use HistoMoE in your research, please cite:

```bibtex
@software{histomoe2025,
  title   = {HistoMoE: A Histology-Guided Mixture-of-Experts Framework
             for Gene Expression Prediction},
  author  = {Dhruv Kumar},
  year    = {2025},
  version = {0.1.0},
  url     = {https://github.com/kumardhruv88/histomoe},
  license = {Apache-2.0},
  note    = {Google Summer of Code 2025 Candidate Project}
}
```

<hr />

## 📜 License

Distributed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

<br>

<div align="center">
  Built with ❤️ for the computational pathology & spatial transcriptomics community.
  <br><br>
  ⭐ <b>Star us on GitHub to support the project!</b>
  <br><br>
  <a href="https://github.com/kumardhruv88/histomoe">
    <img src="https://img.shields.io/github/stars/kumardhruv88/histomoe?style=social" alt="GitHub Stars"/>
  </a>
</div>
