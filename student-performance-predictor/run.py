#!/usr/bin/env python3
"""
Main entry point for Student Performance Prediction System
Run this file to start the application
"""

import os
import sys
import subprocess
import argparse

def print_header():
    """Print application header"""
    print("\n" + "="*60)
    print("🎓 STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("="*60)
    print("An AI-based system to predict student academic performance")
    print("="*60 + "\n")

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import pandas
        import numpy
        import sklearn
        import joblib
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\n📦 Installing required packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Dependencies installed successfully!")
            return True
        except:
            print("❌ Failed to install dependencies. Please run:")
            print("   pip install -r requirements.txt")
            return False

def train_model():
    """Train the model"""
    print("\n🔄 Training model...")
    try:
        from src.train_model import main as train_main
        train_main()
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def run_gui():
    """Run the GUI application"""
    print("\n🖥️  Starting GUI application...")
    try:
        from gui.app import main as gui_main
        gui_main()
    except Exception as e:
        print(f"❌ Failed to start GUI: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Student Performance Prediction System')
    parser.add_argument('--mode', choices=['gui', 'train', 'predict'],
                       default='gui', help='Run mode (default: gui)')
    
    args = parser.parse_args()
    
    print_header()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    if args.mode == 'train':
        # Just train the model
        train_model()
    
    elif args.mode == 'predict':
        # Run prediction in console mode
        print("🔮 Prediction mode coming soon...")
        print("   For now, use --mode gui for the graphical interface")
    
    else:  # gui mode
        # Check if model exists
        if not os.path.exists('models/student_model.joblib'):
            print("⚠️  No trained model found!")
            print("🔄 Training model first...")
            if not train_model():
                print("❌ Cannot continue without trained model.")
                return
        
        # Run GUI
        run_gui()

if __name__ == "__main__":
    main()