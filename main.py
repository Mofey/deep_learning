# Image Classification using CNNs (Deep Learning)

"""
This project builds a Convolutional Neural Network (CNN) to classify handwritten digits using the MNIST dataset.
It follows a modular structure to ensure separation of concerns.

Modules:
- install_requirements.py: Install necessary libraries
- data_loader.py: Load dataset
- model.py: Define CNN model
- train.py: Train the model
- test.py: Evaluate the model
- main.py: Run the full pipeline
"""

from data_loader import load_data
from model import create_cnn
from train import train_model
from test import evaluate_model

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_data()
    model = create_cnn()
    model = train_model(model, x_train, y_train)
    evaluate_model(model, x_test, y_test)