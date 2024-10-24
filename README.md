# Breast Cancer Classification Using SVM

This project implements a Support Vector Machine (SVM) model to classify radiological breast images as either **benign** or **malignant**. The project includes preprocessing of images, feature extraction using multiple techniques, training of the SVM classifier, and predictions on new images. The model achieves high accuracy and provides insights into the classification performance through confusion matrix and other metrics.

## Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Model Description](#model-description)
- [Feature Extraction Techniques](#feature-extraction-techniques)
- [Training the Model](#training-the-model)
- [Prediction on New Images](#prediction-on-new-images)
- [Evaluation](#evaluation)
- [Results](#results)
- [References](#references)

## Dataset

The dataset used consists of radiological breast images categorized into two classes:
- **Benign**: Non-cancerous images
- **Malignant**: Cancerous images

### Dataset Structure
The dataset is structured in two main folders:
```
/train
    /benign
    /malignant
/test
    /benign
    /malignant
```

Make sure the images are in grayscale or convert them before training.

## Installation

### Prerequisites
- Python 3.7 or later
- Jupyter Notebook / Google Colab
- Packages: `numpy`, `scikit-learn`, `opencv-python`, `matplotlib`, `skimage`

### Install Required Packages
```bash
pip install numpy scikit-learn opencv-python matplotlib scikit-image joblib
```

### Clone the Repository
```bash
git clone https://github.com/yourusername/breast-cancer-svm.git
cd breast-cancer-svm
```

## Model Description

The model is based on a Support Vector Machine (SVM) classifier that uses a set of extracted features from the breast images. The process consists of the following steps:

1. **Preprocessing**:
   - Resizing all images to 128x128 pixels.
   - Converting images to grayscale.
   - Normalizing pixel values to the range [0, 1].

2. **Feature Extraction**:
   The following features are extracted from the images:
   - **HOG (Histogram of Oriented Gradients)**: Captures shape and texture.
   - **SIFT (Scale-Invariant Feature Transform)**: Extracts keypoints and descriptors.
   - **LBP (Local Binary Patterns)**: Captures texture.
   - **Color Histograms**: Captures intensity distribution (though images are grayscale).

3. **SVM Classifier**:
   - A linear SVM classifier is trained on the extracted features to classify images into benign or malignant.

## Feature Extraction Techniques

1. **HOG (Histogram of Oriented Gradients)**:
   Captures the gradient orientation of the image to identify shapes and textures.

2. **SIFT (Scale-Invariant Feature Transform)**:
   Detects keypoints in the image and computes descriptors around these keypoints.

3. **LBP (Local Binary Patterns)**:
   Extracts texture features by comparing pixel intensities with their neighbors.

4. **Color Histogram**:
   Generates histograms of pixel intensity distributions.

## Training the Model

The `train_model()` function is used to train the SVM classifier. The extracted features from all training images are fed into the SVM for training.

```python
svm_model = train_svm(X_train, y_train)
joblib.dump(svm_model, 'svm_breast_cancer_model.pkl')
```

### Training Script
You can train the model using the script provided in the Jupyter notebook `breast_cancer_svm.ipynb`. Make sure the dataset is properly loaded and paths are configured.

## Prediction on New Images

To make predictions on a new image, you can use the following code:

```python
from sklearn.externals import joblib
svm_model = joblib.load('svm_breast_cancer_model.pkl')

new_image_path = '/path/to/new/image.jpg'
result = predict_image(new_image_path, svm_model)
print(f"Prediction: {result}")
```

The image will be displayed along with the predicted label (either "Benign" or "Malignant").

## Evaluation

The performance of the SVM model is evaluated using:
1. **Confusion Matrix**: Provides a visualization of the model's performance on the test set.
2. **Classification Report**: Displays precision, recall, F1-score, and support for each class.

### Example of Confusion Matrix and Classification Report:
```python
ConfusionMatrixDisplay.from_estimator(svm_model, X_test, y_test)
plt.show()

print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
```

## Results

The model achieves a test accuracy of over 85%. The precision, recall, and F1-score metrics are satisfactory, showing the model's reliability in distinguishing between benign and malignant breast images.

### Example Results:
- **Test Accuracy**: 87%
- **Confusion Matrix**: Correctly classifies both benign and malignant images with minimal false positives.
- **F1-Score**: Average F1-score of around 0.86.

## Usage Example

- Train the model:
  ```python
  svm_model = train_svm(X_train, y_train)
  joblib.dump(svm_model, 'svm_breast_cancer_model.pkl')
  ```

- Predict on a new image:
  ```python
  result = predict_image('path_to_image.jpg', svm_model)
  print(f"Prediction: {result}")
  ```

- Evaluate the model:
  ```python
  print(classification_report(y_test, y_pred))
  ```

## References

- [OpenCV Documentation](https://docs.opencv.org/)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/)
- [Breast Cancer Research Papers](https://link.springer.com/)
