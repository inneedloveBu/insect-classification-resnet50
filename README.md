
# IP102 Insect Classification with ResNet50V2

## 🎯 Project Overview
A deep learning-based insect classification system implemented using TensorFlow and ResNet50V2, trained on the IP102 dataset containing 102 insect classes. This project demonstrates practical computer vision skills with applications in biodiversity monitoring and agricultural technology.

## 📊 Technical Stack
- **TensorFlow 2.x** - Deep learning framework
- **ResNet50V2** - Pretrained CNN architecture
- **Gradio** - Web interface deployment
- **scikit-learn** - Model evaluation
- **Matplotlib & Seaborn** - Data visualization
- **Pandas & NumPy** - Data processing

## 🚀 Quick Start

### Installation
1. Clone the repository
2. Create and activate virtual environment
3. Install dependencies: `pip install -r requirements.txt`

### Dataset Setup
1. Download the IP102 dataset
2. Extract with proper directory structure

### Training
```bash
# Simplified training (1,000 images)
python train_simple.py

# Full training with advanced options
python train.py
```

### Deployment
```bash
# Local web application
python app_simple.py
# Visit: http://127.0.0.1:7860

# Hugging Face deployment
python app_hf.py
```

## 📈 Performance Metrics
*(Based on simplified training with 1,000 images over 5 epochs)*

| Metric | Value | Description |
|--------|-------|-------------|
| Training Accuracy | XX% | Accuracy on training subset |
| Validation Accuracy | XX% | Accuracy on validation subset |
| Test Accuracy | XX% | Accuracy on test subset |
| Top-5 Accuracy | XX% | Accuracy within top-5 predictions |
| Training Time | ~15 min | On NVIDIA GTX 1650 |

## 🎓 Relevance to QMUL MSc Data Science Curriculum

This project demonstrates skills relevant to the QMUL Data Science MSc:

### **Applied Machine Learning**
- Data Mining and preprocessing
- Deep learning implementation
- Transfer learning techniques
- Model evaluation and validation

### **Big Data & Cloud Computing**
- Large-scale data processing
- Cloud deployment (Hugging Face Spaces)
- Web application development

### **Domain-Specific Applications**
- Computer Vision: Image classification
- Data Visualization and interpretation
- Practical real-world application

## 🔬 Technical Implementation

### Model Architecture
- Base Model: ResNet50V2 pretrained on ImageNet
- Feature Extractor: Global Average Pooling
- Regularization: Batch Normalization and Dropout
- Output: 102-class probability distribution

### Training Configuration
- Optimizer: Adam with learning rate 0.001
- Loss Function: Categorical Cross-Entropy
- Batch Size: 16
- Epochs: 5 (simplified), 10+ (full training)

## 🚀 Future Enhancements

### Immediate Improvements
1. Train on full dataset (75,222 images)
2. Experiment with advanced architectures
3. Implement comprehensive data augmentation
4. Systematic hyperparameter tuning

### Advanced Features
1. Model interpretability with Grad-CAM
2. Mobile deployment with TensorFlow Lite
3. RESTful API development
4. Real-time processing capabilities

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

---

*Developed as part of preparation for QMUL MSc Data Science program.*