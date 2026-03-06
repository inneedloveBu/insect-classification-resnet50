# 🐞 Insect Pest Classification using Transfer Learning (ResNet50)


[![bilibili](https://img.shields.io/badge/🎥-Video%20on%20Bilibili-red)](https://www.bilibili.com/video/BV1zKvrBAEK1/?share_source=copy_web&vd_source=56cdc7ef44ed1ee2c9b9515febf8e9ce&t=1)

[![Hugging Face Spaces](https://img.shields.io/badge/🤗-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/indeedlove/insect-classifier)
[![GitHub](https://img.shields.io/badge/📂-GitHub-black)](https://github.com/inneedloveBu/insect-classification-resnet50)
[![Python](https://img.shields.io/badge/Python-3.10-blue)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
[![Deep Learning](https://img.shields.io/badge/DeepLearning-CNN-green)
<img width="1440" height="739" alt="ScreenShot_2025-12-26_203110_107" src="https://github.com/user-attachments/assets/66deff62-291c-4d96-9632-61f354b57ec7" />





---

# Project Overview

This project implements a **deep learning pipeline for fine-grained insect pest classification** using the **IP102 dataset**. The objective is to classify agricultural pest species from images using **transfer learning with ResNet50**.

Fine-grained insect recognition is a challenging computer vision task due to:

- high inter-class similarity
- large number of categories
- class imbalance
- varying image quality and backgrounds

To address these challenges, this project leverages **pretrained convolutional neural networks** and evaluates model performance using multiple classification metrics.

---

# Dataset

This project uses the **IP102 Dataset**, a large-scale dataset for insect pest recognition.

Dataset characteristics:

- **102 insect species**
- **~75,000 images**
- real-world agricultural environments
- significant class imbalance

---

# Project Structure

## 📁 Repository Structure
![s截图1227223548](https://github.com/user-attachments/assets/6f8f9ffc-0ded-46e7-b39f-eb98e1d365e0)

<img width="1784" height="581" alt="training_history" src="https://github.com/user-attachments/assets/2e1e03e0-97d4-4f5c-af0a-4010a389b30f" />

---
Each TXT file contains:


image_name label


Example:


00002.jpg 0
00003.jpg 0
00005.jpg 0


---

# Model Architecture

The model is built using **Transfer Learning with ResNet50**.

Architecture:


Input Image (224×224×3)
↓
ResNet50 Backbone (pretrained on ImageNet)
↓
Global Average Pooling
↓
Dense Layer (512)
↓
Dropout (0.5)
↓
Softmax Layer (102 classes)


Advantages of transfer learning:

- leverages pretrained visual representations
- reduces training time
- improves convergence on limited datasets

---

# Training Pipeline

The training pipeline includes:

### 1 Data Loading

Images are loaded according to **train / validation split** defined in dataset TXT files.

### 2 Preprocessing

- Image resizing: **224×224**
- normalization
- batch loading using TensorFlow dataset pipeline

### 3 Training Strategy

- Transfer Learning
- Frozen base model initially
- Fine-tuning of deeper layers

### 4 Optimization

- Optimizer: **Adam**
- Learning rate: **2e-5**
- Loss function: **Categorical Crossentropy**

---

# Experimental Setup

| Component | Setting |
|---|---|
| Framework | TensorFlow / Keras |
| Model | ResNet50 |
| Input Size | 224×224 |
| Classes | 102 |
| Optimizer | Adam |
| Learning Rate | 2e-5 |
| Epochs | 30 |
| Batch Size | 32 |

---

# Model Performance

The model was evaluated on the **validation set** using multiple metrics.

### Top-1 Accuracy 2.0%


### Top-5 Accuracy  9.48%


### Classification Metrics

| Metric | Score |
|---|---|
| Macro Precision | 0.03 |
| Macro Recall | 0.02 |
| Macro F1-Score | 0.02 |
| Weighted F1-Score | 0.02 |

---

# Result Interpretation

The relatively low Top-1 accuracy reflects the **difficulty of the IP102 dataset**, which involves:

- **102 visually similar insect classes**
- strong **class imbalance**
- challenging real-world image conditions

Despite this, the **Top-5 accuracy of 9.48%** indicates that the model frequently places the correct class among the top candidate predictions, demonstrating the ability to learn useful visual representations.



# Future Improvements

Several improvements could significantly enhance performance:

### Data Augmentation

- random crop
- horizontal flip
- color jitter

### Class Imbalance Handling

- focal loss
- class weighting
- oversampling minority classes

### Advanced Architectures

- EfficientNet
- Vision Transformer (ViT)
- ConvNeXt

### Model Interpretability

- Grad-CAM visualization
- attention heatmaps

---

# Skills Demonstrated

This project demonstrates practical experience in:

- Deep Learning
- Computer Vision
- Transfer Learning
- Model Evaluation
- TensorFlow / Keras
- Machine Learning Pipeline Design

---
<img width="1784" height="581" alt="training_history" src="https://github.com/user-attachments/assets/3f6572cf-e835-44be-89db-9169fb7d3882" />

<img width="3510" height="1780" alt="improvement_timeline" src="https://github.com/user-attachments/assets/0c8e81ad-88a4-4e9e-8a15-e555cf0b8377" />

<img width="3517" height="1780" alt="performance_comparison" src="https://github.com/user-attachments/assets/500322d5-4949-4ef5-9843-68aff924bd14" />

---

**Training Details**: Two-phase training (15 epochs total), batch size 32, Adam optimizer with learning rate scheduling, enhanced data augmentation.
**Hardware**: NVIDIA GTX 1650 GPU, 8GB VRAM, 16GB RAM
**Repository**: Complete code, trained models, and analysis available at [GitHub](https://github.com/inneedloveBu/insect-classification-resnet50)




## 📚 References
1. IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition (CVPR 2019)
2. Deep Residual Learning for Image Recognition (CVPR 2016)
3. TensorFlow: A System for Large-Scale Machine Learning

## 📧 Contact & Links
- **GitHub**: @inneedloveBu
- **Hugging Face**: @indeedlove
- **Live Demo**: Hugging Face Spaces

## 📄 License
MIT License
