# Spam Email Detection System

This is a Machine Learning based Spam Email Detection Web Application. It uses Natural Language Processing (NLP) to analyze email text and predict whether an email is spam or not. The application provides a simple web interface built with Flask to interact with the model.

## Features
- **Real-time Prediction:** Enter any message to check if it's spam or ham (not spam).
- **Web Interface:** A sleek and modern UI for interacting with the model.
- **Model Metrics:** Check the model's accuracy, precision, recall, and F1 score on the stats endpoint or natively through the UI.
- **Auto-Training:** The system automatically trains a model and saves it if no pre-trained model is found upon starting the application.

## Repository Structure
- `app.py`: Flask web application server and core endpoints.
- `spam_detector.py`: Core machine learning logic (preprocessing, predicting).
- `train_model.py`: Script dedicated to fetching data and training the model.
- `dataset.csv`: Dataset used for training the model.
- `requirements.txt`: Python package dependencies.
- `static/`: Contains the CSS style and JavaScript files.
- `templates/`: Contains the HTML interface templates.

## How to Run locally

1. Ensure you have Python installed on your Windows machine.
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask application:
   ```bash
   python app.py
   ```
4. Open your browser and go to `http://localhost:5000` to access the application.

*Note: Once you run `app.py`, the application will check if a model exists. If it does not exist, it will automatically train one using `dataset.csv` and save it to the directory, outputting `.pkl` files and `metrics.json`.*

## Technologies Used
- Python
- scikit-learn
- Flask
- HTML, CSS, JavaScript
