"""
Simple GUI Application for Student Performance Prediction
Completely offline, uses tkinter which comes with Python
"""

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os

# Add parent directory to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import StudentPredictor

class PerformancePredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎓 Student Performance Predictor")
        self.root.geometry("600x550")
        self.root.resizable(False, False)
        
        # Set colors
        self.bg_color = "#f0f0f0"
        self.primary_color = "#3498db"
        self.success_color = "#27ae60"
        self.danger_color = "#e74c3c"
        self.root.configure(bg=self.bg_color)
        
        # Load predictor
        self.predictor = StudentPredictor()
        
        # Create GUI
        self.create_widgets()
        
        # Center the window
        self.center_window()
    
    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Title
        title_frame = tk.Frame(self.root, bg=self.primary_color, height=80)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="🎓 Student Performance Prediction System",
            font=("Arial", 18, "bold"),
            bg=self.primary_color,
            fg="white"
        )
        title_label.pack(expand=True)
        
        # Subtitle
        subtitle_label = tk.Label(
            self.root,
            text="Enter student details to predict pass/fail result",
            font=("Arial", 11),
            bg=self.bg_color,
            fg="#555"
        )
        subtitle_label.pack(pady=10)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg=self.bg_color, padx=40, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Input fields
        fields = [
            ("📚 Study Hours per day:", "hours"),
            ("📊 Attendance Percentage:", "attendance"),
            ("📝 Previous Exam Score (0-100):", "prev_score"),
            ("✏️ Assignment Marks (0-100):", "assignments")
        ]
        
        self.entries = {}
        
        for i, (label_text, field_name) in enumerate(fields):
            # Label
            label = tk.Label(
                main_frame,
                text=label_text,
                font=("Arial", 11),
                bg=self.bg_color,
                anchor="w"
            )
            label.grid(row=i, column=0, sticky="w", pady=(10, 5))
            
            # Entry
            entry = tk.Entry(
                main_frame,
                font=("Arial", 11),
                width=30,
                bd=2,
                relief=tk.GROOVE
            )
            entry.grid(row=i, column=0, sticky="ew", pady=(0, 10))
            self.entries[field_name] = entry
        
        # Button frame
        button_frame = tk.Frame(main_frame, bg=self.bg_color)
        button_frame.grid(row=len(fields), column=0, pady=20)
        
        # Predict button
        self.predict_btn = tk.Button(
            button_frame,
            text="🔮 Predict Performance",
            command=self.predict,
            bg=self.primary_color,
            fg="white",
            font=("Arial", 12, "bold"),
            padx=30,
            pady=10,
            bd=0,
            cursor="hand2"
        )
        self.predict_btn.pack(side=tk.LEFT, padx=10)
        
        # Clear button
        self.clear_btn = tk.Button(
            button_frame,
            text="🗑️ Clear",
            command=self.clear_fields,
            bg="#95a5a6",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=30,
            pady=10,
            bd=0,
            cursor="hand2"
        )
        self.clear_btn.pack(side=tk.LEFT, padx=10)
        
        # Retrain button (small)
        self.retrain_btn = tk.Button(
            main_frame,
            text="🔄 Retrain Model",
            command=self.retrain_model,
            bg="#f39c12",
            fg="white",
            font=("Arial", 9),
            padx=10,
            pady=5,
            bd=0,
            cursor="hand2"
        )
        self.retrain_btn.grid(row=len(fields)+1, column=0, pady=10)
        
        # Result frame
        result_frame = tk.LabelFrame(
            main_frame,
            text="Prediction Result",
            font=("Arial", 11, "bold"),
            bg=self.bg_color,
            padx=20,
            pady=15
        )
        result_frame.grid(row=len(fields)+2, column=0, pady=20, sticky="ew")
        
        self.result_label = tk.Label(
            result_frame,
            text="Enter student details and click Predict",
            font=("Arial", 12),
            bg=self.bg_color,
            wraplength=450,
            justify="left"
        )
        self.result_label.pack()
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=("Arial", 9)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Check if model exists
        if self.predictor.model is None:
            self.status_bar.config(text="⚠️ No trained model found. Click 'Retrain Model' to train.")
    
    def validate_inputs(self):
        """Validate user inputs"""
        try:
            hours = float(self.entries['hours'].get())
            attendance = float(self.entries['attendance'].get())
            prev_score = float(self.entries['prev_score'].get())
            assignments = float(self.entries['assignments'].get())
            
            # Range checks
            if hours < 0 or hours > 24:
                messagebox.showerror("Error", "Study hours must be between 0 and 24")
                return None
            
            if attendance < 0 or attendance > 100:
                messagebox.showerror("Error", "Attendance must be between 0 and 100")
                return None
            
            if prev_score < 0 or prev_score > 100:
                messagebox.showerror("Error", "Previous score must be between 0 and 100")
                return None
            
            if assignments < 0 or assignments > 100:
                messagebox.showerror("Error", "Assignment marks must be between 0 and 100")
                return None
            
            return (hours, attendance, prev_score, assignments)
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers in all fields")
            return None
    
    def predict(self):
        """Make prediction"""
        # Validate inputs
        values = self.validate_inputs()
        if not values:
            return
        
        # Check if model is loaded
        if self.predictor.model is None:
            messagebox.showerror(
                "Error", 
                "No trained model found!\n\nPlease click 'Retrain Model' to train the model first."
            )
            return
        
        hours, attendance, prev_score, assignments = values
        
        # Update status
        self.status_bar.config(text="🤔 Making prediction...")
        self.root.update()
        
        # Make prediction
        result = self.predictor.predict(hours, attendance, prev_score, assignments)
        
        if result['success']:
            # Display result
            if result['result'] == 'Pass':
                result_text = f"✅ PREDICTION: PASS\n\n"
                result_text += f"Confidence: {result['confidence']:.1%}\n\n"
                result_text += f"{result['recommendations']}"
                self.result_label.config(fg=self.success_color)
            else:
                result_text = f"⚠️ PREDICTION: FAIL\n\n"
                result_text += f"Confidence: {result['confidence']:.1%}\n\n"
                result_text += f"{result['recommendations']}"
                self.result_label.config(fg=self.danger_color)
            
            self.result_label.config(text=result_text)
            self.status_bar.config(text="✅ Prediction completed")
        else:
            messagebox.showerror("Error", f"Prediction failed: {result['error']}")
            self.status_bar.config(text="❌ Prediction failed")
    
    def clear_fields(self):
        """Clear all input fields"""
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.result_label.config(
            text="Enter student details and click Predict",
            fg="black"
        )
        self.status_bar.config(text="Fields cleared")
    
    def retrain_model(self):
        """Retrain the model"""
        # Confirm with user
        if not messagebox.askyesno(
            "Confirm Retraining",
            "This will retrain the model using the latest data.\n\n"
            "Do you want to continue?"
        ):
            return
        
        self.status_bar.config(text="🔄 Training model...")
        self.root.update()
        
        try:
            # Import trainer
            from src.train_model import ModelTrainer
            
            # Train model
            trainer = ModelTrainer()
            success = trainer.run_training_pipeline()
            
            if success:
                # Reload predictor
                self.predictor = StudentPredictor()
                messagebox.showinfo(
                    "Success",
                    f"✅ Model retrained successfully!\n\n"
                    f"Accuracy: {trainer.accuracy:.1%}"
                )
                self.status_bar.config(text="✅ Model retrained successfully")
            else:
                messagebox.showerror("Error", "Failed to retrain model")
                self.status_bar.config(text="❌ Training failed")
                
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.status_bar.config(text="❌ Training failed")

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = PerformancePredictorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()