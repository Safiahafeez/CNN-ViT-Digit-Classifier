# Handwritten Digit Recognition Using CNN and Vision Transformer

## 1. Project Overview

This project implements a deep learning-based handwritten digit recognition system using two neural network architectures: Convolutional Neural Network (CNN) and Vision Transformer (ViT).

The system is developed using PyTorch and the Scikit-learn handwritten digits dataset. Both models are trained and evaluated using a shared stratified dataset split to ensure a consistent and fair performance comparison.

A Streamlit-based web application enables users to upload handwritten digit images and obtain predictions from both trained models.

## 2. Key Features

- **Dataset Preparation:** Loading and preprocessing the Scikit-learn handwritten digits dataset.
- **Data Splitting:** Stratified training, validation, and testing datasets.
- **CNN Architecture:** Implementation and training of a Convolutional Neural Network.
- **Vision Transformer:** Implementation of patch embeddings, positional embeddings, and Transformer encoder layers.
- **Model Evaluation:** Performance assessment using accuracy, macro F1-score, confusion matrices, and a majority-class baseline.
- **Image Preprocessing:** Grayscale conversion, digit cropping, centering, resizing to 8×8 pixels, and pixel normalization.
- **Interactive Application:** Streamlit interface for uploading handwritten digit images and displaying predictions from both models.
- **Model Persistence:** Saving and loading trained model checkpoints for inference without retraining.
- **Input Validation:** Handling blank and unreadable image uploads.

## 3. Technologies and Libraries

- Python
- PyTorch
- NumPy
- Scikit-learn
- Streamlit
- Pillow (PIL)
- Matplotlib

## 4. Model Performance

Both models were evaluated using the same held-out test dataset.

| Model | Test Accuracy | Macro F1-Score |
|---|---:|---:|
| Majority-Class Baseline | 10.00% | 0.0182 |
| Convolutional Neural Network (CNN) | 96.94% | 0.9691 |
| Vision Transformer (ViT) | 95.00% | 0.9483 |

### Evaluation Summary

The CNN achieved a test accuracy of 96.94% and a macro F1-score of 0.9691, outperforming the Vision Transformer, which achieved 95.00% accuracy and a macro F1-score of 0.9483.

Both models substantially outperformed the majority-class baseline.

## 5. Project Structure

```text
Module_04_Digit_Prediction/
│
├── train.py
├── models.py
├── preprocessing.py
├── app.py
├── README.md
│
└── module4_outputs/
    ├── split_indices.npz
    ├── cnn_model.pth
    ├── vit_model.pth
    ├── evaluation_results.json
    ├── cnn_confusion_matrix.npy
    └── vit_confusion_matrix.npy
```

## 6. Installation and Execution

### Install Dependencies

Install the required Python libraries using:

```bash
pip install torch numpy scikit-learn streamlit pillow matplotlib
```

### Launch the Application

Open a terminal in the project directory and execute:

```bash
streamlit run app.py
```

The application will open in a web browser.

### Application Workflow

1. Upload a handwritten digit image in PNG, JPG, or JPEG format.
2. View the original uploaded image.
3. View the preprocessed 8×8 image.
4. Obtain predictions from the trained CNN and ViT models.
5. Compare the predictions displayed by both models.

**Note:** The application loads the previously saved model checkpoints. Model training is not required when launching the application for inference.

## 7. Conclusion

This project demonstrates the implementation and comparative evaluation of CNN and Vision Transformer architectures for handwritten digit recognition.

The experimental results indicate that the CNN achieved better performance than the Vision Transformer on the selected test dataset. The Streamlit application provides an interactive interface for image preprocessing and digit prediction using both trained models.

The project brings together dataset preparation, deep learning model development, performance evaluation, image preprocessing, and application deployment in a single workflow.