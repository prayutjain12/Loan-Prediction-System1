import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_model():
    # Load dataset
    if not os.path.exists('ml/loan_data.csv'):
        print("Dataset not found. Please run generate_dataset.py first.")
        return

    df = pd.read_csv('ml/loan_data.csv')
    
    # Preprocessing
    # Handle missing values if any
    df = df.dropna()
    
    # Label Encoding for categorical variables
    le = LabelEncoder()
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
    
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        # Save encoders for later use in prediction
        joblib.dump(le, f'ml/encoder_{col}.pkl')

    # Features and Target
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training - Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(model, 'ml/loan_prediction_model.pkl')
    print("Model saved to ml/loan_prediction_model.pkl")

if __name__ == "__main__":
    train_model()
