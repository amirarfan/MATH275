# Original Author: Andrea Mari, Updated and adapted: 28 Nov 2021 by Amir Arfan for MATH275
from re import S
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns


# Loading of the MNIST dataset
def load_mnist_data(n_train, n_test):
    mnist_dataset = keras.datasets.mnist
    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = mnist_dataset.load_data()

    # Reduce dataset size
    train_images = train_images[:n_train]
    train_labels = train_labels[:n_train]
    test_images = test_images[:n_test]
    test_labels = test_labels[:n_test]

    # Normalize pixel values within 0 and 1
    train_images = train_images / 255
    test_images = test_images / 255

    # Add extra dimension for convolution channels
    train_images = np.array(train_images[..., tf.newaxis], requires_grad=False)
    test_images = np.array(test_images[..., tf.newaxis], requires_grad=False)
    return train_images, train_labels, test_images, test_labels


# Quantum circuit as a convolution kernel
@qml.qnode(qml.device("default.qubit", wires=4))
def circuit(phi, rand_params):
    # Encoding of 4 classical input values
    for j in range(4):
        qml.RY(np.pi * phi[j], wires=j)

    # Random quantum circuit
    RandomLayers(rand_params, wires=list(range(4)))

    # Measurement producing 4 classical output values
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]


# Quanvolution scheme
def quanv(image, rand_params):
    """Convolves the input image with many applications of the same quantum circuit."""
    out = np.zeros((14, 14, 4))

    # Loop over the coordinates of the top-left pixel of 2X2 squares
    for j in range(0, 28, 2):
        for k in range(0, 28, 2):
            # Process a squared 2x2 region of the image with a quantum circuit
            q_results = circuit(
                [
                    image[j, k, 0],
                    image[j, k + 1, 0],
                    image[j + 1, k, 0],
                    image[j + 1, k + 1, 0],
                ],
                rand_params,
            )
            # Assign expectation values to different channels of the output pixel (j/2, k/2)
            for c in range(4):
                out[j // 2, k // 2, c] = q_results[c]
    return out


# Quantum pre-processing of the dataset


# Hybrid quantum-classical model
def MyModel():
    """Initializes and returns a custom Keras model
    which is ready to be trained."""
    model = keras.models.Sequential(
        [keras.layers.Flatten(), keras.layers.Dense(10, activation="softmax")]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    n_epochs = 100  # Number of optimization epochs
    n_layers = 20  # Number of random layers
    n_train = 50  # Size of the train dataset
    n_test = 30  # Size of the test dataset

    PREPROCESS = (
        False  # If False, skip quantum processing and load data from SAVE_PATH
    )
    np.random.seed(123)  # Seed for NumPy random number generator
    tf.random.set_seed(123)  # Seed for TensorFlow random number generator

    rand_params = np.random.uniform(
        high=2 * np.pi, size=(n_layers, 4)
    )  # For the random layers

    train_images, train_labels, test_images, test_labels = load_mnist_data(
        n_train, n_test
    )

    if PREPROCESS:
        q_train_images = []
        print("Quantum pre-processing of train images:")
        for idx, img in enumerate(train_images):
            print("{}/{}        ".format(idx + 1, n_train), end="\r")
            q_train_images.append(quanv(img, rand_params))
        q_train_images = np.asarray(q_train_images)

        q_test_images = []
        print("\nQuantum pre-processing of test images:")
        for idx, img in enumerate(test_images):
            print("{}/{}        ".format(idx + 1, n_test), end="\r")
            q_test_images.append(quanv(img, rand_params))
        q_test_images = np.asarray(q_test_images)

        # Save pre-processed images
        np.save("q_train_images.npy", q_train_images)
        np.save("q_test_images.npy", q_test_images)
    else:
        q_train_images = np.load("q_train_images.npy")
        q_test_images = np.load("q_test_images.npy")

    q_model = MyModel()

    q_history = q_model.fit(
        q_train_images,
        train_labels,
        validation_data=(q_test_images, test_labels),
        batch_size=4,
        epochs=n_epochs,
        verbose=2,
    )

    c_model = MyModel()

    c_history = c_model.fit(
        train_images,
        train_labels,
        validation_data=(test_images, test_labels),
        batch_size=4,
        epochs=n_epochs,
        verbose=2,
    )
    sns.set_style("whitegrid")
    sns.color_palette("bright")
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 9))
    sns.lineplot(
        data=q_history.history["val_accuracy"],
        ax=ax1,
        label="With Quantum Layer",
    )
    sns.lineplot(
        data=c_history.history["val_accuracy"],
        ax=ax1,
        label="Without Quantum Layer",
    )
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([0, 1])
    ax1.set_xlabel("Epoch")
    ax1.legend()
    plt.savefig("accuracy.png")
    plt.tight_layout()
    plt.show()
