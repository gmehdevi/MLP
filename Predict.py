#! /usr/bin/env python3
from MultiLayerPerceptron import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import argparse
import tensorflow as tf

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']

    y_encoded = tf.keras.utils.to_categorical(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled.T, y_encoded.T

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def evaluate_model(model_path, X_test, y_test):
    
    mlp = create_model(model_path)

    loss, accuracy = mlp.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    y_pred = mlp.predict(X_test)
    
    return y_pred

def save_predictions(predictions, filename="predictions.csv"):
    pd.DataFrame(predictions).to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a Multi-Layer Perceptron model on a dataset.")
    parser.add_argument("--test", type=str, default="test.csv", help="Path to the test data CSV file.")
    parser.add_argument("--model_path", type=str, default="mlp_model.pkl", help="Path to the trained model file.")
    args = parser.parse_args()

    X_test, y_test = load_and_preprocess_data(args.test)

    y_pred = evaluate_model(args.model_path, X_test, y_test)

    y_pred = np.argmax(y_pred, axis=1)

    save_predictions(y_pred, "predictions.csv")

if __name__ == "__main__":
    main()
