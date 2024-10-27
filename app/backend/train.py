import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from skimage.feature import hog, local_binary_pattern
from matplotlib import pyplot as plt

# Setup paths and dataset
data_dir = '/content/dataset/'  # Change path if necessary
IMG_SIZE = 128  # Resize images to 128x128

# Preprocess images: resize, grayscale, normalize
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    return img_normalized

def load_images(data_dir):
    images = []
    labels = []

    for label in ['benign', 'malignant']:  # Assuming subfolders for each class
        label_dir = os.path.join(data_dir, label)
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            img = preprocess_image(img_path)
            images.append(img)
            labels.append(label)

    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Encode labels (benign = 0, malignant = 1)
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    return images, labels_encoded

images, labels = load_images(data_dir)

# HOG (Histogram of Oriented Gradients) feature extraction
def extract_hog_features(images):
    hog_features = []
    for img in images:
        hog_feat = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        hog_features.append(hog_feat)
    return np.array(hog_features)

hog_features = extract_hog_features(images)

# SIFT (Scale-Invariant Feature Transform) feature extraction
sift = cv2.SIFT_create()

def extract_sift_features(images):
    sift_features = []
    for img in images:
        # Ensure the image is in 8-bit grayscale
        if img.dtype != 'uint8':
            img = (img * 255).astype('uint8')
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            des = np.ravel(des)[:128]  # Limiting features for simplicity
            sift_features.append(des)
        else:
            sift_features.append(np.zeros(128))  # For cases with no keypoints
    return np.array(sift_features)

sift_features = extract_sift_features(images)

# LBP (Local Binary Patterns) feature extraction
def extract_lbp_features(images):
    lbp_features = []
    for img in images:
        lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
        hist = hist.astype("float")
        hist /= hist.sum()  # Normalize the histogram
        lbp_features.append(hist)
    return np.array(lbp_features)

lbp_features = extract_lbp_features(images)

# Color Histogram (for grayscale intensity)
def extract_color_histograms(images, bins=64):
    color_hist_features = []
    for img in images:
        # Convert the image to 8-bit for histogram calculation
        if img.dtype != 'uint8':
            img_8bit = (img * 255).astype('uint8')
        else:
            img_8bit = img

        # Compute the histogram
        hist = cv2.calcHist([img_8bit], [0], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()  # Normalize and flatten
        color_hist_features.append(hist)

    return np.array(color_hist_features)


color_hist_features = extract_color_histograms(images)

# Concatenate all extracted features
X = np.hstack((hog_features, sift_features, lbp_features, color_hist_features))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train SVM Classifier
svm_clf = SVC(kernel='linear')
svm_clf.fit(X_train, y_train)

# Make predictions
y_pred = svm_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {accuracy * 100:.2f}%")

# Save the model
joblib.dump(svm_clf, 'svm_breast_cancer_model.pkl')

# Prediction on a new image
def predict_image(image_path, model):
    img = preprocess_image(image_path)

    # Extract features for the new image
    hog_feat = extract_hog_features([img])
    sift_feat = extract_sift_features([img])
    lbp_feat = extract_lbp_features([img])
    color_hist_feat = extract_color_histograms([img])

    # Concatenate all features
    features = np.hstack((hog_feat, sift_feat, lbp_feat, color_hist_feat))

    # Predict
    pred = model.predict(features)
    return 'Malignant' if pred[0] == 1 else 'Benign'

# Load the model and make a prediction on a new image
svm_model = joblib.load('svm_breast_cancer_model.pkl')
new_image_path = '/Users/subbu/Desktop/ML_Project/backend/data_dir/malignant_test_image.png'  # Replace with actual image path
result = predict_image(new_image_path, svm_model)
print(f"Prediction: {result}")

# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
# import joblib
# import cv2
# import numpy as np
# from sklearn.metrics import accuracy_score

# # Part 1: Prediction on a new image and display with label
# def predict_image(image_path, model):
#     img = preprocess_image(image_path)

#     # Extract features for the new image
#     hog_feat = extract_hog_features([img])
#     sift_feat = extract_sift_features([img])
#     lbp_feat = extract_lbp_features([img])
#     color_hist_feat = extract_color_histograms([img])

#     # Concatenate all features
#     features = np.hstack((hog_feat, sift_feat, lbp_feat, color_hist_feat))

#     # Predict
#     pred = model.predict(features)
#     return 'Malignant' if pred[0] == 1 else 'Benign'

# # Function to display the image with the predicted class
# def display_image_with_label(image_path, label):
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#     # Convert image from BGR (OpenCV) to RGB (Matplotlib uses RGB format)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

#     plt.figure(figsize=(6, 6))
#     plt.imshow(img_rgb, cmap='gray')
#     plt.title(f'Prediction: {label}', fontsize=14, color='red')
#     plt.axis('off')  # Hide the axis
#     plt.show()

# # Load the trained model
# svm_model = joblib.load('svm_breast_cancer_model.pkl')

# # Replace with the path to your new image
# new_image_path = '/content/drive/MyDrive/dtaet2/valid/0/115_463070125_png.rf.d5ea5a9967a7c1a8b2dea142c17ae843.jpg'
# result = predict_image(new_image_path, svm_model)
# print(f"Prediction: {result}")

# # Display the image with the predicted label
# display_image_with_label(new_image_path, result)

# # Part 2: Evaluate the model on the test set and display confusion matrix
# y_pred = svm_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Test Accuracy: {accuracy * 100:.2f}%")

# # Generate confusion matrix
# conf_matrix = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Benign', 'Malignant'])
# disp.plot(cmap=plt.cm.Blues)
# plt.title('Confusion Matrix')
# plt.show()

# # Part 3: Display classification report (precision, recall, f1-score, support)
# classification_report_str = classification_report(y_test, y_pred, target_names=['Benign', 'Malignant'])
# print("Classification Report:\n", classification_report_str)
