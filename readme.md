# UniPruning: Unifying Local Metric and Global Feedback for Scalable Sparse LLMs

<div align="center">

[![License](https://img.shields.io/badge/License-openPangu-blue.svg)](./LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

**A unified post-training pruning framework combining local metrics with global coordination via mirror descent**

</div>

---

## Overview

UniPruning achieves state-of-the-art LLM pruning through:
- **Mirror descent optimization** for learning saliency without weight updates
- **Unified framework** supporting both unstructured (50%-70%) and semi-structured (2:4) pruning
- **One-shot mask generation** for arbitrary sparsity levels after single calibration
- **Strong performance** across LLaMA, Qwen, DeepSeek, and openPangu models

---

## 🏛️ openPangu Declaration

**This project uses [openPangu](https://openi.pcl.ac.cn/OpenPangu) models from Pengcheng Laboratory.**

This repository complies with the **openPangu Model License Agreement Version 1.0**:
- All usage follows openPangu's open-source license terms
- Proper attribution to openPangu and original model creators
- Redistribution must maintain original license
- See [LICENSE](./LICENSE) for full terms

**Tested Models:**
- openPangu-Embedded-7B-V1.1 (Ascend NPU validated)
- LLaMA2-13B, LLaMA-3.2-1B/3B
- Qwen2.5-7B/14B
- DeepSeek-R1-Distill-7B/8B

---

## Installation

```bash
# Create environment
conda create -n Uniprune python=3.10
conda activate Uniprune

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 2:4 Semi-Structured Pruning
```bash
bash train_24_final.sh
```

### Unstructured Pruning
```bash
bash final_unstructured.sh
```

### Python API
```python
from searching.main import prune_model
from searching.extract_mask_mixed import extract_mask

# Search for optimal saliency scores
model = prune_model(
    model_name="meta-llama/Llama-2-13b-hf",
    sparsity=0.6,
    calibration_samples=128
)

# Extract mask
mask = extract_mask(model, sparsity=0.6, pattern="unstructured")

# Evaluate
from searching.ppl import evaluate_perplexity
ppl = evaluate_perplexity(model, mask)
```

---

## Results

### Unstructured Pruning @ 60%

| Model | Method | WikiText PPL ↓ | Zero-shot Avg ↑ |
|-------|--------|----------------|-----------------|
| **LLaMA2-13B** | Dense | 4.57 | 0.5820 |
| | Wanda | 11.90 | 0.4709 |
| | RIA | 7.57 | 0.5075 |
| | **UniPruning** | **7.82** | **0.5106** |
| **Qwen2.5-14B** | Dense | 4.93 | 0.6218 |
| | Wanda | 11.68 | 0.5368 |
| | RIA | 9.37 | 0.5378 |
| | **UniPruning** | **8.85** | **0.5424** |

### Semi-Structured 2:4

| Model | Wanda | RIA | ProxSparse | **UniPruning** |
|-------|-------|-----|------------|----------------|
| LLaMA2-13B | 8.37 | 7.85 | 6.88 | **6.87** |
| Qwen2.5-7B | 14.77 | 13.81 | 14.06 | **10.86** |
| Qwen2.5-14B | 11.69 | 10.87 | 10.54 | **9.10** |
| DeepSeek-R1-8B | 29.77 | 30.10 | 23.74 | **20.91** |

### openPangu Results (Ascend NPU)

| Model | Method | PPL ↓ | Avg Acc ↑ |
|-------|--------|-------|-----------|
| **openPangu-7B** | Dense | 31.36 | 0.4374 |
| | Wanda (2:4) | 237.32 | 0.3834 |
| | **UniPruning (2:4)** | **106.21** | **0.4166** |
| | **UniPruning (50%)** | **49.73** | **0.5079** |

**Performance:** 1.27× end-to-end inference speedup with 2:4 sparsity on H200 GPU

---

## Repository Structure

```
UniPruning/
├── searching/
│   ├── main.py                  # Main pruning script
│   ├── extract_mask_mixed.py    # Mask extraction
│   ├── ppl.py                   # Perplexity evaluation
│   ├── TrainerSFT_SPP.py       # Custom trainer
│   ├── adamw_spp_prune.py      # Optimizer
│   └── prox_op.py              # Proximal operators
├── train_24_final.sh           # 2:4 pruning script
├── final_unstructured.sh       # Unstructured pruning script
└── requirements.txt            # Dependencies
```

---

## Citation

```bibtex
@article{unipruning2025,
  title={UniPruning: Unifying Local Metric and Global Feedback for Scalable Sparse LLMs},
  author={Anonymous},
  year={2025},
  note={Technical Report}
}
```

If you use openPangu models, please also cite:
```bibtex
@misc{openpangu2023,
  title={openPangu: Open-source Large Language Models},
  author={Pengcheng Laboratory},
  year={2023},
  url={https://openi.pcl.ac.cn/OpenPangu}
}
```

See [CITATION.bib](./CITATION.bib) for complete references.

---

## License

This project is licensed under the **openPangu Model License Agreement Version 1.0**.

- ✅ Academic research: Freely available
- ✅ Commercial use: Subject to openPangu license terms
- ✅ Redistribution: Must maintain original license and attribution

See [LICENSE](./LICENSE) for full terms.

---

## Acknowledgments

- **Pengcheng Laboratory** for openPangu models
- **Meta AI** for LLaMA, **Alibaba** for Qwen, **DeepSeek** for DeepSeek models

---

<div align="center">

**⭐ Star us if you find this project helpful! ⭐**

</div>
