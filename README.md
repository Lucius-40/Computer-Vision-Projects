# Computer Vision Projects 🖼️

A comprehensive collection of computer vision projects showcasing deep learning techniques for image classification and object recognition. This repository demonstrates the progressive implementation of Convolutional Neural Networks (CNNs), from basic architectures to advanced models like ResNet-18.

## 📋 Table of Contents
- [Overview](#overview)
- [Projects](#projects)
  - [MNIST Digit Recognition](#1-mnist-digit-recognition)
  - [CIFAR-10 Image Classification](#2-cifar-10-image-classification)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Results Summary](#results-summary)
- [Learning Outcomes](#learning-outcomes)
- [Future Work](#future-work)

## 🎯 Overview

This repository documents my journey in learning computer vision and deep learning. Each project builds upon the previous one, introducing new concepts, architectures, and optimization techniques. The projects range from simple digit classification to complex multi-class image recognition using state-of-the-art architectures.

## 🚀 Projects

### 1. MNIST Digit Recognition
**Directory:** `Mnist digits classifier/`

A foundational project implementing a CNN to classify handwritten digits from the MNIST dataset.

#### Features:
- Custom CNN architecture with 2 convolutional layers
- Real-time training visualization
- Model evaluation on test data
- Prediction visualization with confidence scores

#### Model Architecture:
- **Input:** 28x28 grayscale images
- **Conv Layer 1:** 32 filters (3x3)
- **MaxPooling:** 2x2
- **Conv Layer 2:** 64 filters (3x3)
- **Fully Connected Layers:** 128 units
- **Output:** 10 classes (digits 0-9)

#### Key Techniques:
- Cross-entropy loss
- Adam optimizer
- Data normalization
- Batch processing with DataLoader

#### Notebooks:
- `MNIST_digit_recognition.ipynb` - Complete implementation with training and evaluation

[📖 Detailed Documentation](Mnist%20digits%20classifier/Readme.md)

---

### 2. CIFAR-10 Image Classification
**Directory:** `CIFAR-10/`

An advanced project series exploring various CNN architectures and optimization techniques for classifying color images across 10 categories.

#### Dataset:
- 60,000 32x32 color images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training images, 10,000 test images

#### Project Evolution:

**Phase 1: Baseline CNN** (`Object_Recognition.ipynb`)
- Custom CNN architecture
- Baseline accuracy: ~85%
- Established performance benchmarks

**Phase 2: Enhanced CNN** (`object_detection_3.ipynb`)
- Added batch normalization
- Learning rate scheduling
- Data augmentation (random flips, crops, rotations)
- Improved generalization and reduced overfitting

**Phase 3: ResNet-18 Implementation** (`Cifar_10_ResNet_18.ipynb`)
- Deep residual network with skip connections
- Combined with batch normalization, LR scheduling, and data augmentation
- **Achieved 95% accuracy** 🎉
- Detailed error analysis and visualization
- Confusion matrix analysis
- Per-class performance evaluation

#### Key Techniques:
- **Batch Normalization:** Stabilizes training and accelerates convergence
- **Learning Rate Scheduling:** Adaptive learning rate for better optimization
- **Data Augmentation:** Increases dataset diversity and model robustness
- **Residual Connections:** Enables training of deeper networks
- **Transfer Learning Concepts:** Leveraging proven architectures

#### Advanced Features:
- Comprehensive visualization of misclassifications
- Class-wise accuracy breakdown
- Training/validation loss curves
- Model checkpointing
- Performance analysis and insights

#### Notebooks:
1. `Object_Recognition.ipynb` - Baseline CNN implementation
2. `object_detection_3.ipynb` - Enhanced CNN with optimization techniques
3. `Cifar_10_ResNet_18.ipynb` - ResNet-18 architecture with full analysis

[📖 Detailed Documentation](CIFAR-10/Readme.md)

---

## 🛠️ Technologies Used

### Frameworks & Libraries:
- **PyTorch** - Deep learning framework
- **torchvision** - Computer vision datasets and transformations
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical visualization (for confusion matrices)

### Tools:
- **Jupyter Notebooks** - Interactive development environment
- **VS Code** - Code editor
- **CUDA** (optional) - GPU acceleration

## 📦 Installation

### Prerequisites:
- Python 3.7 or higher
- pip package manager

### Setup Instructions:

1. **Clone the repository:**
```bash
git clone https://github.com/Lucius-40/Computer-Vision-Projects.git
cd Computer-Vision-Projects
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv cv_env
# Windows
cv_env\Scripts\activate
# Linux/Mac
source cv_env/bin/activate
```

3. **Install dependencies:**
```bash
pip install torch torchvision matplotlib numpy seaborn jupyter
```

4. **For GPU support (optional):**
```bash
# Visit https://pytorch.org/ for CUDA-specific installation
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

5. **Launch Jupyter Notebook:**
```bash
jupyter notebook
```

## 📁 Project Structure

```
Computer-Vision-Projects/
├── README.md                          # This file
├── CIFAR-10/                          # CIFAR-10 classification project
│   ├── Cifar_10_ResNet_18.ipynb      # ResNet-18 implementation (95% accuracy)
│   ├── object_detection_3.ipynb       # Enhanced CNN with optimizations
│   ├── Object_Recognition.ipynb       # Baseline CNN (~85% accuracy)
│   └── Readme.md                      # Project-specific documentation
│
└── Mnist digits classifier/           # MNIST digit recognition project
    ├── MNIST_digit_recognition.ipynb  # Complete MNIST implementation
    ├── Readme.md                      # Project-specific documentation
    └── MNIST/                         # MNIST dataset (auto-downloaded)
        └── raw/
            ├── train-images-idx3-ubyte
            ├── train-labels-idx1-ubyte
            ├── t10k-images-idx3-ubyte
            └── t10k-labels-idx1-ubyte
```

## 📊 Results Summary

| Project | Model | Accuracy | Key Features |
|---------|-------|----------|-------------|
| MNIST | Custom CNN | ~98% | Basic CNN, 2 conv layers |
| CIFAR-10 (Baseline) | Custom CNN | ~85% | Initial implementation |
| CIFAR-10 (Enhanced) | CNN + Optimizations | ~88-90% | Batch norm, LR scheduling, augmentation |
| CIFAR-10 (Advanced) | ResNet-18 | **95%** | Residual connections, full optimization suite |

## 💡 Learning Outcomes

Throughout these projects, I gained hands-on experience with:

### Core Concepts:
- ✅ CNN architecture design and implementation
- ✅ Forward and backward propagation in deep networks
- ✅ Loss functions and optimization algorithms
- ✅ Training/validation/test splits
- ✅ Overfitting prevention techniques

### Advanced Techniques:
- ✅ Batch normalization for training stability
- ✅ Learning rate scheduling strategies
- ✅ Data augmentation for improved generalization
- ✅ Residual connections and skip connections
- ✅ Model evaluation and error analysis

### Best Practices:
- ✅ Iterative experimentation and hyperparameter tuning
- ✅ Visualization of training progress and results
- ✅ Per-class performance analysis
- ✅ Code organization and documentation
- ✅ Reproducible research practices

## 🔮 Future Work

### Planned Enhancements:
- [ ] Object detection with YOLO/Faster R-CNN
- [ ] Image segmentation (semantic and instance)
- [ ] Transfer learning with pre-trained models (VGG, EfficientNet)
- [ ] Generative models (GANs, VAEs)
- [ ] Real-time video processing
- [ ] Custom dataset creation and annotation
- [ ] Model deployment (Flask API, ONNX)
- [ ] Attention mechanisms and Vision Transformers

### Optimization Ideas:
- [ ] Mixed precision training
- [ ] Model quantization and pruning
- [ ] Distributed training experiments
- [ ] Advanced augmentation techniques (CutMix, MixUp)

## 🤝 Contributing

Suggestions and feedback are welcome! Feel free to open an issue or submit a pull request if you have ideas for improvements or new projects.

## 📧 Contact

**GitHub:** [@Lucius-40](https://github.com/Lucius-40)

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- PyTorch team for excellent documentation and tutorials
- MNIST and CIFAR-10 dataset creators
- The deep learning community for research papers and insights

---

*Last Updated: November 2025*
