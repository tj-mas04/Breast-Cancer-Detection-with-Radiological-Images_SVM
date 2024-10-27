from pathlib import Path
import joblib
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# Load the trained model
model = joblib.load('svm_breast_cancer_model.pkl')

def preprocess_image(image_bytes):
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (128, 128))
    
    # Basic preprocessing while maintaining original dimensions
    img_normalized = cv2.equalizeHist(img_resized)
    return img_normalized

def extract_hog_features(images):
    hog_features = []
    for img in images:
        # Original HOG parameters to match trained model
        hog_feat = hog(img, orientations=9,
                      pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2),
                      block_norm='L2-Hys')
        hog_features.append(hog_feat)
    return np.array(hog_features)

def extract_lbp_features(images):
    lbp_features = []
    for img in images:
        # Single-scale LBP to match original feature count
        lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        lbp_features.append(hist)
    return np.array(lbp_features)

def extract_color_histograms(images, bins=64):
    color_hist_features = []
    for img in images:
        if img.dtype != 'uint8':
            img_8bit = (img * 255).astype('uint8')
        else:
            img_8bit = img
        
        hist = cv2.calcHist([img_8bit], [0], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        color_hist_features.append(hist)
    
    return np.array(color_hist_features)

def extract_sift_features(images):
    sift = cv2.SIFT_create()
    sift_features = []
    
    for img in images:
        if img.dtype != 'uint8':
            img = (img * 255).astype('uint8')
            
        kp, des = sift.detectAndCompute(img, None)
        
        if des is not None:
            # Take mean of SIFT descriptors to get fixed-length feature
            des_mean = np.mean(des, axis=0) if des.shape[0] > 0 else np.zeros(128)
            sift_features.append(des_mean)
        else:
            sift_features.append(np.zeros(128))
            
    return np.array(sift_features)

def normalize_confidence(confidence_score):
    """
    Normalize the SVM decision function output to [0,1] range using sigmoid scaling
    """
    return 1 / (1 + np.exp(-confidence_score))

def analyze_image(image_path):
    try:
        # Preprocess image
        img = preprocess_image(image_path)
        
        # Extract features with fixed dimensions
        hog_feat = extract_hog_features([img])  # 8100 features
        sift_feat = extract_sift_features([img])  # 128 features
        lbp_feat = extract_lbp_features([img])  # 9 features
        color_hist_feat = extract_color_histograms([img])  # 64 features
        
        # Concatenate features
        features = np.hstack((hog_feat, sift_feat, lbp_feat, color_hist_feat))
        
        # Verify feature dimension
        if features.shape[1] != 8301:
            print(f"Feature dimensions: HOG={hog_feat.shape[1]}, SIFT={sift_feat.shape[1]}, "
                  f"LBP={lbp_feat.shape[1]}, Color={color_hist_feat.shape[1]}")
            raise ValueError(f"Feature vector has {features.shape[1]} features, but 8301 features are required.")
        
        # Make prediction
        pred = model.predict(features)
        
        # Get confidence score and normalize it
        confidence_raw = model.decision_function(features)[0]
        confidence = normalize_confidence(confidence_raw)
        
        # Ensure confidence is in [0,1] range
        confidence = np.clip(confidence, 0, 1)
        
        result = 'Malignant' if pred[0] == 1 else 'Benign'
        
        return result, confidence
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return "Error", 0.0