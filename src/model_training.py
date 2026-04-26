import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix
from src.data_preprocessing import DataPreprocessor

def train_theft_detection_model():
    """Train the electricity theft detection model"""
    print("Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    df = preprocessor.load_and_combine_data()
    df = preprocessor.create_features(df)
    
    X, y = preprocessor.prepare_features(df)
    X_scaled = preprocessor.fit_transform(X)
    
    # Split data with stratified sampling to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42, shuffle=True
    )
    
    print("Training models...")
    
    # Model 1: Isolation Forest (Unsupervised anomaly detection)
    iso_forest = IsolationForest(contamination=0.15, random_state=42)
    iso_forest.fit(X_train)
    
    # Model 2: Random Forest (Supervised classification)
    rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf_classifier.fit(X_train, y_train)
    
    # Model 3: OneClassSVM for normal patterns
    oc_svm = OneClassSVM(nu=0.15, kernel='rbf', gamma='scale')
    oc_svm.fit(X_train[y_train == 0])  # Train only on normal data
    
    # Evaluate models
    rf_pred = rf_classifier.predict(X_test)
    print("\nRandom Forest Performance:")
    print(classification_report(y_test, rf_pred))
    
    # Save models and preprocessor
    joblib.dump({
        'preprocessor': preprocessor,
        'iso_forest': iso_forest,
        'rf_classifier': rf_classifier,
        'oc_svm': oc_svm
    }, 'models/theft_detector.pkl')
    
    print("Models saved successfully!")
    return preprocessor, iso_forest, rf_classifier, oc_svm

if __name__ == "__main__":
    train_theft_detection_model()