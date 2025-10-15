# Eye Tracking Application

This project is an eye-tracking application that uses machine learning to predict gaze direction. It leverages OpenCV, dlib, and scikit-learn for face detection, landmark prediction, and gaze classification.

## Features
- Real-time face and eye region detection.
- Gaze direction prediction (left, center, right) using a Gaussian Mixture Model (GMM).
- Training mode to create a new model or load an existing one.
- Saves gaze direction data for analysis.

## Prerequisites
- Python 3.8 or higher
- OpenCV
- dlib
- scikit-learn
- numpy
- joblib

## Installation
1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd eye-tracking
   ```

2. Create a virtual environment and install the required dependencies:
   ```sh
   python -m venv venv
   # On Windows
   venv/Scripts/activate
   # On Mac/Linux
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage
1. Configure the application in `configs/app_config.py`:
   - Set `LOAD_MODEL` to `True` to load a pre-trained model or `False` to train a new one.
   - Adjust other parameters as needed.

2. Run the application:
   ```sh
   python main.py
   ```

3. Press `ESC` to exit the application.

## File Structure
- `configs/`: Configuration files.
- `core/`: Core logic for trackers and factories.
- `file_handlers/`: File handling utilities.
- `handlers/`: Modules for processing frames, faces, and eye behavior.
- `ml/`: Machine learning utilities and models.
- `renderer/`: Rendering utilities for visual feedback.
- `resources/`: External resources like the dlib model.
- `utils/`: Helper functions for mathematical operations.
- `files/`: Output files for gaze direction data.