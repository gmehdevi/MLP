#! /usr/bin/env python3
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from MultiLayerPerceptron import MultiLayerPerceptron
DEBUG = False


def load_and_preprocess_data(csv_path):
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Split the dataset into features and target variable
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']

    # Convert labels to categorical (one-hot encoding)
    y = tf.keras.utils.to_categorical(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if DEBUG:
        return X_train, X_test, y_train, y_test
        

    return X_train.T, X_test.T, y_train.T, y_test.T

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


    plt.FigureManagerBase.set_window_title('Learning Curves')

    plt.tight_layout()
    plt.show()



def build_model(input_shape):
    if DEBUG:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])

        model.compile(loss='binary_crossentropy',  metrics=['accuracy'])
    else:
        layer_sizes=[input_shape, 10, 10, 2]
        print(f"Layer sizes: {layer_sizes}")
        activations= ['relu'] * 2 + ['softmax']
        print(f"Activation functions: {activations}")
        init_methods=['random'] * 4
        print(f"Initialization methods: {init_methods}")
        model = MultiLayerPerceptron(
            layer_sizes=layer_sizes,
            activations=activations,
            init_methods=init_methods,
            optimizer='Momentum'
        )
    return model



def main():
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data('train.csv')

    # Build the model
    if DEBUG:
        model = build_model(X_train.shape[1])
    else:
        model = build_model(X_train.shape[0])
        
    if DEBUG:
        history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
    else:
        model.fit(X_train, y_train, learning_rate=0.01, epochs=1000, X_val=X_test, Y_val=y_test)


    predictions = (model.predict(X_test))

    predictions = predictions.sum(axis=1)
    print(predictions)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()
