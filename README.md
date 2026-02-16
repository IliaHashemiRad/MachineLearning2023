# 🧠 Machine Learning Projects Portfolio

> A curated collection of academic computer vision and probabilistic modeling projects, featuring deep learning pipelines for restoration tasks and Gaussian Mixture Model (GMM) methods for statistical inference.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red?logo=pytorch)
![Scikit--learn](https://img.shields.io/badge/ML-scikit--learn-f7931e?logo=scikitlearn)

---

## 📌 Overview

This repository contains five hands-on projects developed for computer vision and machine learning coursework. Together, they cover:

- 🖼️ **Image restoration tasks** using neural networks (denoising, super-resolution, colorization)
- 📊 **Probabilistic modeling** using Gaussian Mixture Models (EM and MAP-based estimation)
- 🧪 **Notebook-based experiments** including training, evaluation, and visual analysis

The codebase is organized around end-to-end experimental workflows, with each project in its own folder.

---

## 📂 Repository Structure

```text
MachineLearning2023/
├── Image Denoising/
│   ├── Image Denoising.ipynb
│   └── README.md
├── Image Super Resolution/
│   ├── Image Super Resolution.ipynb
│   └── README.md
├── Image Colorization/
│   ├── Image Colorization.ipynb
│   └── README.md
├── EM algorithm for GMM/
│   ├── Phase1.ipynb
│   ├── Phase 1.pdf
│   └── README.md
├── MAP of GMM/
│   ├── Phase2.ipynb
│   ├── Phase 2.pdf
│   └── README.md
└── README.md
```

---

## 🚀 Projects

### 1) 🔇 Image Denoising (AutoEncoder + PCA Baseline)
- Builds a noisy MNIST pipeline and trains an AutoEncoder for denoising.
- Includes reconstruction visualization and comparison with a PCA-based approach.
- Focus: latent representation learning for noise removal.

### 2) 🔍 Image Super Resolution (AutoEncoder)
- Trains a super-resolution model on an Unsplash-based dataset.
- Uses train/validation/test setup and visual inspection of restored outputs.
- Focus: recovering high-quality images from lower-resolution inputs.

### 3) 🎨 Image Colorization (AutoEncoder)
- Trains colorization models from grayscale landscape inputs.
- Includes at least two modeling strategies and qualitative result analysis.
- Focus: learning color mappings from structural grayscale cues.

### 4) 📈 EM Algorithm for GMM (Phase 1)
- Implements EM steps manually (initialization, E-step, M-step).
- Explores clustering behavior and parameter estimation in mixture models.
- Focus: unsupervised density estimation and iterative optimization.

### 5) 🧮 MAP of GMM for Image Denoising (Phase 2)
- Uses patch-based modeling on MNIST with GMM priors and MAP estimation.
- Includes corruption setup, posterior-based patch reconstruction, and MSE analysis.
- Focus: Bayesian restoration with probabilistic image priors.

---

## 🛠️ Tech Stack

- **Languages & Environment:** Python, Jupyter Notebook
- **Deep Learning:** PyTorch, torchvision
- **Classical ML & Statistics:** scikit-learn, SciPy, Keras/TensorFlow utilities
- **Data & Visualization:** NumPy, pandas, matplotlib, OpenCV

---

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd MachineLearning2023
   ```
2. Open any project notebook in Jupyter/Colab.
3. Install required dependencies as needed (varies slightly per notebook).
4. Run cells sequentially and follow inline instructions.

> 💡 Many notebooks include comments about optional long-running cells and model checkpoint usage.

---

## 📊 Outputs & Evaluation

Across projects, the repository demonstrates:

- Visual reconstruction quality checks (before/after comparisons)
- Learning curves for training/validation monitoring
- Quantitative metrics such as **MSE** in denoising workflows
- Comparative methods (e.g., AutoEncoder vs PCA)

---

## 📚 Notes

- This repository is experiment-oriented and intended for educational/research demonstration.
- Some notebooks are computationally intensive depending on dataset size and hardware.

---

## 👨‍💻 Author

Developed as part of university coursework in machine learning course of Sharif University of Technology in 2023.
