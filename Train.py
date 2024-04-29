#! /usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from MultiLayerPerceptron import MultiLayerPerceptron
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']

    y = tf.keras.utils.to_categorical(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train.T, X_val.T, y_train.T, y_val.T

import matplotlib.pyplot as plt

def plot_learning_curves(history):
    epochs = range(1, len(history['loss']) + 1)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['accuracy'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    fig = plt.gcf()
    fig.canvas.manager.set_window_title('Learning Curves')

    plt.tight_layout()
    plt.show()



def main():
    parser = argparse.ArgumentParser(description="Train a Multi-Layer Perceptron model on a dataset.")
    parser.add_argument("--train", type=str, default="train.csv", help="Path to the training data CSV file.")
    parser.add_argument("--val", type=str, default="val.csv", help="Path to the validation data CSV file.")
    parser.add_argument("--model_path", type=str, default="mlp_model.pkl", help="Path to save the trained model.")
    parser.add_argument("--hidden_layers", "-l", type=int, nargs='+', default=[10, 10], help="Number of neurons in each hidden layer.")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--learning_rate", "-a", type=float, default=0.01, help="Learning rate for the optimizer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--plot", action='store_true', help="Plot the training validation loss and accuracy curves.")
    parser.add_argument("--param_init", type=str, default="random", help="Parameter initialization method.")
    args = parser.parse_args()

    X_train, X_val, y_train, y_val = load_and_preprocess_data('train.csv')
    
    n_features = X_train.shape[0]
    n_classes = y_train.shape[0]

    print(f"Number of features: {n_features}, Number of classes: {n_classes}")

    layer_sizes = [n_features, 10, 10, n_classes]

    activation_functions = ['relu'] * len(args.hidden_layers) + ['softmax']
    init_methods=args.param_init if len(args.hidden_layers) == len(layer_sizes) - 1 else ['random'] * len(layer_sizes)


    print(f"Layer sizes: {layer_sizes}")
    print(f"Activation functions: {activation_functions}")
    print(f"Initialization methods: {init_methods}")
    mlp = MultiLayerPerceptron(
        layer_sizes=layer_sizes,
        activations= activation_functions,
        init_methods=args.param_init if len(args.hidden_layers) == len(layer_sizes) - 1 else ['random'] * len(layer_sizes)
    )

    history = mlp.fit(X_train, y_train, learning_rate= 0.01, epochs=args.epochs, X_val=X_val, Y_val=y_val)

    if args.plot:
        plot_learning_curves(history)

    mlp.save_model(args.model_path)
    print(f"Model trained and saved to {args.model_path}")

if __name__ == "__main__":
    main()

