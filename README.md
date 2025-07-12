# ViT-ProtoNet: Few-Shot Learning with Vision Transformers and Prototypical Networks 

A lightweight PyTorch implementation of Prototypical Networks using a ViT-Small/16 backbone for few-shot classification on standard benchmarks (Mini-ImageNet, CUB-200, CIFAR-FS, FC100). Includes both production scripts for training & evaluation and Jupyter notebooks for interactive analysis.

<!-- 🚀 PERFORMANCE HIGHLIGHTS 🚀 -->

# 🏆 State-of-the-Art Few-Shot Results 🏆

![CIFAR-FS #1](https://img.shields.io/badge/CIFAR--FS-%231-brightgreen?style=for-the-badge)  
![FC100 #1](https://img.shields.io/badge/FC100-%231-brightgreen?style=for-the-badge)  
![CUB-200 #4](https://img.shields.io/badge/CUB--200-%234-yellow?style=for-the-badge)  
![Mini-ImageNet #4](https://img.shields.io/badge/Mini--ImageNet-%234-yellow?style=for-the-badge)

---

## 🎯 Benchmarks at a Glance

| Dataset           | 📈 Rank |
| ----------------- | ------: |
| **CIFAR-FS**      | **#1**  |
| **FC100**         | **#1**  |
| **CUB-200**       | **#4**  |
| **Mini-ImageNet** | **#4**  |

---

## 🚀 Features

- **Lightweight Backbone**  
  Uses ViT-Small/16 (384-dim embeddings, 6 heads, 12 layers) for fast feature extraction.
- **Prototypical Framework**  
  Episodic training with support & query sets; prototypes computed as mean CLS-token embeddings.
- **Multiple Benchmarks**  
  Ready to run on Mini-ImageNet, CUB-200, CIFAR-FS, and FC100.
- **Modular Codebase**  
  Clean separation between dataset loaders, model definitions, training loops, and utilities.
- **Interactive Notebooks**  
  Explore experiments, visualize prototype distributions, and plot learning curves.

---

## 📝 Prerequisites

- Python 3.8 or higher  
- NVIDIA GPU with CUDA 10.2+ (recommended for reasonable training speed)

---

## ⚙️ Installation

1. **Clone the repository**  
   ```
   git clone https://github.com/abdulvahapmutlu/vit-protonet.git
   cd vit-protonet
   ```

2. **Create & activate a virtual environment**

   ```
   python -m venv .venv
   source .venv/bin/activate    # Linux / macOS
   .venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**

   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 📥 Data Preparation

1. **Download**
   You must obtain the original benchmarks yourself (e.g. from official sources or Kaggle).
2. **Organize**
   Create a root folder, e.g. `/path/to/datasets`, with subdirectories:

   ```
   /path/to/datasets/
     ├── mini_imagenet/
     ├── CUB_200_2011/
     ├── cifar-fs/
     └── FC100/
   ```
3. **Point your scripts**
   Pass the dataset root path to `train.py` and `evaluate.py` via the `--data_root` argument.

---

## ▶️ Quick Start

### 1. Training

```
python src/train.py \
  --dataset cub \
  --data_root /path/to/datasets/CUB_200_2011 \
  --ways 5 \
  --shots 5 \
  --queries 15 \
  --episodes 1000 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --output_dir ./checkpoints/cub
```

* `--dataset`: one of `mini`, `cub`, `cifarfs`, `fc100`
* `--ways/--shots/--queries`: few-shot episode configuration
* `--episodes`: number of training episodes
* `--lr` & `--weight_decay`: AdamW hyperparameters
* `--batch_size`: number of episodes per gradient step
* `--output_dir`: where to save model checkpoints & logs

### 2. Evaluation

```
python src/evaluate.py \
  --dataset cub \
  --data_root /path/to/datasets/CUB_200_2011 \
  --checkpoint ./checkpoints/cub/best_model.pth \
  --ways 5 \
  --shots 5 \
  --queries 15 \
  --eval_episodes 100
```

* `--checkpoint`: path to the trained `.pth` file
* `--eval_episodes`: number of test episodes (e.g. 100 or 1000)

Results will be printed as mean accuracy ± 95% CI.

---

## 📒 Notebooks

For exploratory analysis and visualization, open one of the notebooks in the `notebooks/` folder:

* **CUB-200.ipynb**
* **FC100.ipynb**
* **CIFAR-FS.ipynb**
* **MiniImagenet.ipynb**

Each notebook walks through:

1. Loading a trained checkpoint
2. Sampling episodes
3. Computing & plotting class prototypes
4. Visualizing support/query embeddings with t-SNE / PCA
5. Plotting learning curves

---

## 🔧 Utilities

* **`src/utils.py`**

  * Logging to console & CSV
  * Plotting loss & accuracy curves
  * CLI argument parsing

Feel free to extract or extend any helper functions for your own experiments.

---


## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 📧 Contact

For questions or suggestions, open an issue or contact me at **[abdulvahapmutlu1@gmail.com](mailto:abdulvahapmutlu1@gmail.com)**.

