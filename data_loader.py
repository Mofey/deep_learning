from tensorflow.keras.datasets import mnist

def load_data():
    """Loads the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train / 255.0, y_train), (x_test / 255.0, y_test)
