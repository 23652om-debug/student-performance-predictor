"""
Prediction Module - Simplified version
Makes predictions using trained model
"""

import numpy as np
import joblib
import os

class StudentPredictor:
    def __init__(self, model_dir='models'):
        self.model = None
        self.preprocessor = None
        self.model_dir = model_dir
        self.load_model()
    
    def load_model(self):
        """Load trained model and preprocessor"""
        print("🔄 Loading model...")
        
        model_path = os.path.join(self.model_dir, 'student_model.joblib')
        preprocessor_path = os.path.join(self.model_dir, 'preprocessor.joblib')
        
        try:
            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)
            print("✅ Model loaded successfully!")
            return True
        except FileNotFoundError:
            print("❌ No trained model found! Please train the model first.")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict(self, study_hours, attendance, previous_score, assignment_marks):
        """
        Predict if a student will pass or fail
        
        Parameters:
        study_hours: float - Hours studied per day
        attendance: float - Attendance percentage
        previous_score: float - Previous exam score (0-100)
        assignment_marks: float - Assignment marks (0-100)
        
        Returns:
        dict: Prediction result with details
        """
        if self.model is None or self.preprocessor is None:
            return {
                'success': False,
                'error': 'Model not loaded'
            }
        
        try:
            # Create feature array
            features = np.array([[study_hours, attendance, previous_score, assignment_marks]])
            
            # Scale features
            features_scaled = self.preprocessor['scaler'].transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            
            # Get prediction probability
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = max(probabilities)
            
            # Decode prediction
            result = self.preprocessor['label_encoder'].inverse_transform([prediction])[0]
            
            # Generate recommendations
            recommendations = self.generate_recommendations(
                result, 
                study_hours, 
                attendance, 
                previous_score, 
                assignment_marks
            )
            
            return {
                'success': True,
                'result': result,
                'confidence': confidence,
                'recommendations': recommendations,
                'features': {
                    'Study Hours': study_hours,
                    'Attendance': attendance,
                    'Previous Score': previous_score,
                    'Assignment Marks': assignment_marks
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_recommendations(self, result, study_hours, attendance, prev_score, assignments):
        """Generate recommendations based on prediction"""
        recommendations = []
        
        if result == 'Fail':
            recommendations.append("⚠️ The student is predicted to FAIL.")
            recommendations.append("\n📌 Improvement Suggestions:")
            
            if study_hours < 4:
                recommendations.append("   • Increase study hours to at least 4-5 hours per day")
            
            if attendance < 75:
                recommendations.append("   • Improve attendance to at least 80%")
            
            if prev_score < 50:
                recommendations.append("   • Focus on understanding previous exam topics")
            
            if assignments < 60:
                recommendations.append("   • Complete all assignments and seek help when needed")
        else:
            recommendations.append("✅ The student is predicted to PASS!")
            recommendations.append("\n📌 Recommendations to maintain performance:")
            
            if study_hours < 3:
                recommendations.append("   • Consider increasing study hours to maintain good performance")
            
            recommendations.append("   • Keep up the good work and stay consistent!")
        
        return "\n".join(recommendations)
    
    def predict_batch(self, students_list):
        """Predict for multiple students"""
        results = []
        for student in students_list:
            result = self.predict(
                student['study_hours'],
                student['attendance'],
                student['previous_score'],
                student['assignment_marks']
            )
            result['name'] = student.get('name', 'Unknown')
            results.append(result)
        
        return results

def main():
    """Test the predictor"""
    predictor = StudentPredictor()
    
    if predictor.model is None:
        print("\n❌ Please train the model first using: python src/train_model.py")
        return
    
    # Test predictions
    print("\n" + "="*60)
    print("🎯 TESTING PREDICTIONS")
    print("="*60)
    
    test_cases = [
        {"name": "Good Student", "hours": 8, "attendance": 95, "score": 90, "assignments": 95},
        {"name": "Average Student", "hours": 4, "attendance": 75, "score": 65, "assignments": 70},
        {"name": "At-risk Student", "hours": 1.5, "attendance": 50, "score": 35, "assignments": 40},
    ]
    
    for test in test_cases:
        print(f"\n👤 {test['name']}:")
        print(f"   Study Hours: {test['hours']}")
        print(f"   Attendance: {test['attendance']}%")
        print(f"   Previous Score: {test['score']}")
        print(f"   Assignment Marks: {test['assignments']}")
        
        result = predictor.predict(
            test['hours'],
            test['attendance'],
            test['score'],
            test['assignments']
        )
        
        if result['success']:
            print(f"   🔮 Prediction: {result['result']}")
            print(f"   📊 Confidence: {result['confidence']:.1%}")
        else:
            print(f"   ❌ Error: {result['error']}")

if __name__ == "__main__":
    main()