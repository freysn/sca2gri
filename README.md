# Sca2Gri: Scalable Gridified Scatterplots

[![EuroVis 2025](https://img.shields.io/badge/EuroVis-2025-blue)](https://freysn.github.io/papers/sca2gri.pdf)
[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](https://freysn.github.io/papers/sca2gri.pdf)
[![Video](https://img.shields.io/badge/Video-Demo-orange)](https://freysn.github.io/videos/sca2gri_video.mp4)

**Sca2Gri** is a scalable post-processing method for large-scale scatterplots that reduces visual clutter by gridifying glyph representations. It is designed for data analysis scenarios involving millions of data points, far beyond what traditional scatterplot rendering techniques can handle effectively.

This project accompanies the paper:

> **Sca2Gri: Scalable Gridified Scatterplots**  
> _Presented at EuroVis 2025_  
> [📄 Read the paper (PDF)](https://freysn.github.io/papers/sca2gri.pdf)  
> [🎥 Watch the video demo](https://freysn.github.io/videos/sca2gri_video.mp4)

---

## 🔍 Abstract

Scatterplots are widely used in exploratory data analysis. Representing data points as glyphs is often crucial for in-depth investigation, but this can lead to significant overlap and visual clutter. Recent post-processing techniques address this issue, but their computational and/or visual scalability is generally limited to thousands of points and unable to effectively deal with large datasets in the order of millions.

**Sca2Gri** introduces a grid-based post-processing method designed for analysis scenarios where the number of data points substantially exceeds the number of glyphs that can be reasonably displayed. It supports:

- **Interactive grid generation** for large datasets
- Flexible user control of glyph size, point-to-cell mapping, and scatterplot focus
- **Scalability to millions of data points** (linear complexity w.r.t. point count)

---

## ⚙️ Installation

This project requires Python 3.8+ and the following packages:

- `matplotlib`
- `numba`
- `numpy`
- `pyobjc`
- `pyobjc-framework-Metal`
- `scikit-learn`
- `scipy`

Install dependencies using `pip`:

```bash
pip install matplotlib numba numpy pyobjc pyobjc-framework-Metal scikit-learn scipy
```

> ⚠️ **Note:** The `pyobjc` and `pyobjc-framework-Metal` packages are **macOS-only** and are used for GPU acceleration via Apple Metal. If you're running on a non-macOS system, use the `--no_metal` flag to disable Metal and run on CPU.

---

## 🚀 Usage

Run the interactive visualization GUI with the following command:

```bash
python gui_sca2gri.py --rep case_data/dropglyph_rep.pkl --emb case_data/dropglyph_emb.pkl
```

### Parameters

- `--rep` – Path to the 2D embedding (representation) of your dataset (e.g. `.pkl`)
- `--emb` – Path to the tile (glyph) representation
- `--no_metal` *(optional)* – Run on CPU without Metal (for non-macOS systems)

### Example

```bash
python gui_sca2gri.py --rep case_data/dropglyph_rep.pkl --emb case_data/dropglyph_emb.pkl --no_metal
```

---

## 📂 Example Data

The two use cases featured in the EuroVis 2025 paper can be downloaded here:

📁 [Download Example Data from Google Drive](https://drive.google.com/drive/folders/1Q-vA5yx-lrNUqPfrxhboCsHYoFUIxRNq?usp=sharing)

---

## 📖 Citation

If you use Sca2Gri in your research, please cite the following paper:

```bibtex
@article{sfrey_sca2gri,
  doi = {10.1111/cgf.70141},
  volume = {44},
  number = {3},
  journal = {Computer Graphics Forum},
  title = {{Sca2Gri}: Scalable Gridified Scatterplots},
  author = {S. Frey},
  year = {2025}
}
```

---

## 💻 Platform Notes

- ✅ **macOS (Recommended):** GPU-accelerated via Apple Metal (default)
- ⚠️ **Linux / Windows:** Must use `--no_metal` flag to run on CPU (significantly slower)

---

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

<!---
## 🙌 Acknowledgments
## 🖼️ Preview
-->
