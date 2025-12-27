<div align="center">

# ⚛️ SciViT: Scientific Vision Transformer

**End-to-End Image-to-LaTeX Generation with Swin Transformer & mBART**

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-blue?style=for-the-badge)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](./LICENSE)

<p align="center">
  <img src="assets/demo_result.gif" alt="SciViT Demo" width="800">
  <br>
  <em>Turn complex scientific papers into editable LaTeX code instantly.</em>
</p>

</div>

---

## 🚀 Introduction

**SciViT** is a robust, end-to-end framework designed to bridge the gap between visual document representation and semantic understanding. Unlike traditional OCR pipelines, SciViT directly translates rasterized scientific document images into **LaTeX markup**, handling complex structures like nested fractions, matrices, and tables with high precision.

### Why SciViT?
* **🧠 Hierarchical Vision Encoder:** Powered by **Swin Transformer**, capable of capturing both fine-grained details (symbols) and global layout structure using Shifted Window Attention.
* **🌐 Linguistically Powerful Decoder:** Leverages a **Domain-Adapted mBART**, pre-trained on multilingual corpora and fine-tuned for LaTeX syntax.
* **⚡ Efficient & Robust:** Optimized for high-resolution inputs ($896 \times 672$) and resilient to image noise.

---

## 🏗️ Architecture

SciViT adopts an encoder-decoder paradigm. We replace the traditional CNN backbone with a **Swin Transformer** to extract hierarchical visual features ($Z$), which are then cross-attended by the **mBART Decoder** to auto-regressively generate the sequence.

<div align="center">
  <img src="images/architecture.png" alt="SciViT Architecture" width="100%">
</div>

---
