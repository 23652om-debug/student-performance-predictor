"""
Model Training Module - Simplified version
Trains a machine learning model to predict student performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.accuracy = 0
        
    def load_data(self, filepath):
        """Load student data from CSV file"""
        print("\n📂 Loading data...")
        try:
            data = pd.read_csv(filepath)
            print(f"✅ Data loaded successfully! Found {len(data)} students")
            print(f"📊 Features: {', '.join(data.columns[:-1])}")
            return data
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def prepare_data(self, data):
        """Prepare data for training"""
        print("\n🔄 Preparing data...")
        
        # Separate features and target
        X = data[['Study_Hours', 'Attendance', 'Previous_Score', 'Assignment_Marks']]
        y = data['Result']
        
        # Encode target labels (Pass/Fail to 1/0)
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"✅ Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"🎯 Classes: {list(self.label_encoder.classes_)}")
        
        return X, y_encoded
    
    def split_and_scale(self, X, y):
        """Split data and scale features"""
        print("\n✂️ Splitting data...")
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📚 Training set: {len(X_train)} students")
        print(f"🧪 Testing set: {len(X_test)} students")
        
        # Scale features
        print("\n📏 Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Train the Random Forest model"""
        print("\n🤖 Training model...")
        
        # Create and train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        print("✅ Model training complete!")
        
        # Show feature importance
        feature_importance = pd.DataFrame({
            'Feature': ['Study Hours', 'Attendance', 'Previous Score', 'Assignment Marks'],
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n📊 Feature Importance:")
        for _, row in feature_importance.iterrows():
            print(f"   • {row['Feature']}: {row['Importance']:.2%}")
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("\n📝 Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate accuracy
        self.accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model Accuracy: {self.accuracy:.2%}")
        print("\n📋 Detailed Report:")
        print(classification_report(
            y_test, 
            y_pred, 
            target_names=self.label_encoder.classes_
        ))
        
        return self.accuracy
    
    def save_model(self, model_dir='models'):
        """Save trained model and preprocessors"""
        print("\n💾 Saving model...")
        
        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, 'student_model.joblib')
        joblib.dump(self.model, model_path)
        
        # Save preprocessors
        preprocessor_path = os.path.join(model_dir, 'preprocessor.joblib')
        preprocessor = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }
        joblib.dump(preprocessor, preprocessor_path)
        
        print(f"✅ Model saved to: {model_path}")
        print(f"✅ Preprocessor saved to: {preprocessor_path}")
        
        return model_path, preprocessor_path
    
    def plot_feature_importance(self, save_path='models/feature_importance.png'):
        """Plot feature importance"""
        if self.model is None:
            print("❌ No model trained yet!")
            return
        
        features = ['Study Hours', 'Attendance', 'Previous Score', 'Assignment Marks']
        importance = self.model.feature_importances_
        
        plt.figure(figsize=(8, 5))
        plt.bar(features, importance, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        plt.title('Feature Importance in Student Performance Prediction', fontsize=14, fontweight='bold')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        os.makedirs('models', exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"📊 Feature importance plot saved to: {save_path}")
    
    def run_training_pipeline(self, data_path='data/students.csv'):
        """Run complete training pipeline"""
        print("\n" + "="*60)
        print("🎓 STUDENT PERFORMANCE PREDICTION - TRAINING")
        print("="*60)
        
        # Load data
        data = self.load_data(data_path)
        if data is None:
            return False
        
        # Prepare data
        X, y = self.prepare_data(data)
        
        # Split and scale
        X_train, X_test, y_train, y_test = self.split_and_scale(X, y)
        
        # Train model
        self.train_model(X_train, y_train)
        
        # Evaluate model
        self.evaluate_model(X_test, y_test)
        
        # Plot feature importance
        self.plot_feature_importance()
        
        # Save model
        self.save_model()
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return True

def main():
    """Main function to run training"""
    trainer = ModelTrainer()
    trainer.run_training_pipeline()

if __name__ == "__main__":
    main()