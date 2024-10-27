import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from skimage.feature import hog, local_binary_pattern
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.pipeline import Pipeline
import logging
from datetime import datetime
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class BreastCancerClassifier:
    def __init__(self, img_size=128, data_dir=None):
        self.IMG_SIZE = img_size
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.model = None
        self.feature_extractors = {}
        self.setup_feature_extractors()
        
    def setup_feature_extractors(self):
        """Initialize feature extraction methods with optimal parameters"""
        self.feature_extractors = {
            'hog': {
                'function': self.extract_hog_features,
                'params': {
                    'orientations': 9,
                    'pixels_per_cell': (8, 8),
                    'cells_per_block': (2, 2),
                    'block_norm': 'L2-Hys'
                }
            },
            'lbp': {
                'function': self.extract_lbp_features,
                'params': {
                    'P': 8,
                    'R': 1,
                    'method': 'uniform'
                }
            },
            'sift': {
                'function': self.extract_sift_features,
                'params': {
                    'n_keypoints': 100
                }
            },
            'color_hist': {
                'function': self.extract_color_histograms,
                'params': {
                    'bins': 64
                }
            }
        }

    def preprocess_image(self, image_path):
        """Enhanced image preprocessing pipeline"""
        if isinstance(image_path, str):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Handle bytes input
            img_array = np.frombuffer(image_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError("Failed to load image")
            
        # Resize
        img_resized = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_clahe = clahe.apply(img_resized)
        
        # Denoise
        img_denoised = cv2.fastNlMeansDenoising(img_clahe)
        
        # Normalize
        img_normalized = img_denoised / 255.0
        
        return img_normalized

    def extract_hog_features(self, images):
        """Extract HOG features with optimized parameters"""
        hog_features = []
        for img in images:
            hog_feat = hog(img, **self.feature_extractors['hog']['params'])
            hog_features.append(hog_feat)
        return np.array(hog_features)

    def extract_lbp_features(self, images):
        """Extract multi-scale LBP features"""
        lbp_features = []
        for img in images:
            lbp_feats = []
            # Multi-scale LBP
            for radius in [1, 2, 3]:
                params = self.feature_extractors['lbp']['params'].copy()
                params['R'] = radius
                lbp = local_binary_pattern(img, **params)
                hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
                hist = hist.astype("float")
                hist /= (hist.sum() + 1e-7)
                lbp_feats.extend(hist)
            lbp_features.append(np.array(lbp_feats))
        return np.array(lbp_features)

    def extract_sift_features(self, images):
        """Extract SIFT features with statistical aggregation"""
        sift = cv2.SIFT_create(nfeatures=self.feature_extractors['sift']['params']['n_keypoints'])
        sift_features = []
        
        for img in images:
            if img.dtype != 'uint8':
                img = (img * 255).astype('uint8')
                
            kp, des = sift.detectAndCompute(img, None)
            
            if des is not None and len(des) > 0:
                # Calculate statistical features
                des_mean = np.mean(des, axis=0)
                des_std = np.std(des, axis=0)
                des_max = np.max(des, axis=0)
                sift_feat = np.concatenate([des_mean, des_std, des_max])
            else:
                sift_feat = np.zeros(384)  # 128 * 3 for mean, std, max
                
            sift_features.append(sift_feat)
            
        return np.array(sift_features)

    def extract_color_histograms(self, images):
        """Extract enhanced color histogram features"""
        color_hist_features = []
        for img in images:
            if img.dtype != 'uint8':
                img_8bit = (img * 255).astype('uint8')
            else:
                img_8bit = img
                
            # Calculate histogram at multiple scales
            hist_features = []
            for scale in [1.0, 0.75, 0.5]:
                scaled_img = cv2.resize(img_8bit, None, fx=scale, fy=scale)
                hist = cv2.calcHist([scaled_img], [0], None, 
                                  [self.feature_extractors['color_hist']['params']['bins']], 
                                  [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                hist_features.extend(hist)
                
            color_hist_features.append(np.array(hist_features))
            
        return np.array(color_hist_features)

    def load_and_preprocess_data(self):
        """Load and preprocess the entire dataset"""
        images = []
        labels = []
        
        logging.info("Starting data loading and preprocessing...")
        
        for label in ['benign', 'malignant']:
            label_dir = os.path.join(self.data_dir, label)
            for img_file in os.listdir(label_dir):
                try:
                    img_path = os.path.join(label_dir, img_file)
                    img = self.preprocess_image(img_path)
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    logging.warning(f"Failed to process {img_file}: {str(e)}")
                    
        logging.info(f"Loaded {len(images)} images successfully")
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        # Encode labels
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        
        return images, labels_encoded

    def extract_all_features(self, images):
        """Extract all features and combine them"""
        logging.info("Starting feature extraction...")
        
        features = {}
        for name, extractor in self.feature_extractors.items():
            logging.info(f"Extracting {name} features...")
            features[name] = extractor['function'](images)
            
        # Concatenate all features
        X = np.hstack([features[name] for name in self.feature_extractors.keys()])
        logging.info(f"Final feature vector shape: {X.shape}")
        
        return X

    def build_model(self):
        """Build an ensemble model with optimized hyperparameters"""
        # Create base models
        svm_linear = SVC(kernel='linear', probability=True)
        svm_rbf = SVC(kernel='rbf', probability=True)
        svm_poly = SVC(kernel='poly', probability=True)
        
        # Create voting classifier
        self.model = VotingClassifier(
            estimators=[
                ('svm_linear', svm_linear),
                ('svm_rbf', svm_rbf),
                ('svm_poly', svm_poly)
            ],
            voting='soft'
        )

    def train(self, X, y):
        """Train the model with cross-validation and hyperparameter tuning"""
        logging.info("Starting model training...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define parameter grid for GridSearchCV
        param_grid = {
            'svm_linear__C': [0.1, 1, 10],
            'svm_rbf__C': [0.1, 1, 10],
            'svm_rbf__gamma': ['scale', 'auto'],
            'svm_poly__C': [0.1, 1, 10],
            'svm_poly__degree': [2, 3]
        }
        
        # Perform grid search with cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        
        # Save best model and parameters
        self.model = grid_search.best_estimator_
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logging.info(f"Best parameters: {grid_search.best_params_}")
        logging.info(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
        logging.info(f"Test set accuracy: {accuracy:.4f}")
        
        # Generate and save detailed report
        report = {
            'best_parameters': grid_search.best_params_,
            'cv_accuracy': grid_search.best_score_,
            'test_accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        with open('training_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        return X_test, y_test, y_pred

    def save_model(self, filename='train2_breast_cancer_model.pkl'):
        """Save the trained model and scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_extractors': self.feature_extractors,
            'img_size': self.IMG_SIZE
        }
        joblib.dump(model_data, filename)
        logging.info(f"Model saved to {filename}")

    def load_model(self, filename='train2_breast_cancer_model.pkl'):
        """Load a trained model and its components"""
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_extractors = model_data['feature_extractors']
        self.IMG_SIZE = model_data['img_size']
        logging.info(f"Model loaded from {filename}")

    def predict(self, image_path):
        """Make prediction on a single image"""
        try:
            # Preprocess image
            img = self.preprocess_image(image_path)
            
            # Extract features
            features = self.extract_all_features([img])
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            pred = self.model.predict(features_scaled)
            pred_proba = self.model.predict_proba(features_scaled)
            
            # Get confidence score
            confidence = np.max(pred_proba)
            result = 'Malignant' if pred[0] == 1 else 'Benign'
            
            return result, confidence
            
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            return "Error", 0.0

def visualize_results(X_test, y_test, y_pred):
    """Visualize model performance"""
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Feature importance (for linear SVM)
    if hasattr(clf.model.named_estimators_['svm_linear'], 'coef_'):
        feature_importance = np.abs(clf.model.named_estimators_['svm_linear'].coef_[0])
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(feature_importance)), feature_importance)
        plt.title('Feature Importance')
        plt.xlabel('Feature Index')
        plt.ylabel('Absolute Coefficient Value')
        plt.savefig('feature_importance.png')
        plt.close()

if __name__ == "__main__":
    # Initialize classifier
    clf = BreastCancerClassifier(data_dir='/Users/subbu/Desktop/ML_Project/backend/data_dir')
    
    # Load and preprocess data
    images, labels = clf.load_and_preprocess_data()
    
    # Extract features
    X = clf.extract_all_features(images)
    
    # Build and train model
    clf.build_model()
    X_test, y_test, y_pred = clf.train(X, labels)
    
    # Visualize results
    visualize_results(X_test, y_test, y_pred)
    
    # Save model
    clf.save_model()