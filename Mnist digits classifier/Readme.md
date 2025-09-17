# MNIST Digit Recognition Classifier

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset.

## Project Overview
- **Goal:** Classify grayscale images of handwritten digits (0-9) from the MNIST dataset.
- **Framework:** PyTorch
- **Dataset:** [MNIST](http://yann.lecun.com/exdb/mnist/)
- **Approach:** Build and train a CNN, visualize predictions, and evaluate accuracy.

## Project Structure
- `MNIST_digit_recognition.ipynb`: Main Jupyter notebook with code, explanations, and results.
- `README.md`: Project documentation (this file).

## Steps
1. **Import Libraries:** Load PyTorch, torchvision, matplotlib, and other dependencies.
2. **Data Preparation:**
   - Download and normalize the MNIST dataset.
   - Use DataLoader to batch and shuffle data for efficient training.
3. **Data Visualization:**
   - Display sample images and their labels to understand the dataset.
4. **Model Definition:**
   - Define a CNN with two convolutional layers, pooling, and fully connected layers.
5. **Training:**
   - Use cross-entropy loss and Adam optimizer.
   - Train for multiple epochs, printing loss per epoch.
6. **Evaluation:**
   - Test the model on unseen data and report accuracy.

## How to Run
1. Open `MNIST_digit_recognition.ipynb` in Jupyter Notebook or VS Code.
2. Run all cells sequentially.
3. The notebook will download the MNIST dataset automatically (if not present).
4. Training and evaluation results will be displayed in the output cells.

## Requirements
- Python 3.7+
- PyTorch
- torchvision
- matplotlib
- numpy

Install dependencies with:
```bash
pip install torch torchvision matplotlib numpy
```

## Model Architecture
- **Conv2d(1, 32, 3):** Extracts 32 features from 3x3 patches.
- **MaxPool2d(2, 2):** Reduces spatial size by half.
- **Conv2d(32, 64, 3):** Extracts 64 features from 3x3 patches.
- **MaxPool2d(2, 2):** Further reduces size.
- **Linear(1600, 64):** Fully connected layer.
- **Linear(64, 10):** Output layer for 10 digit classes.

## Notes
- The notebook includes comments and markdown cells explaining each step.
- You can modify batch size, learning rate, or number of epochs for experimentation.

## References
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
