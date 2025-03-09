from model import create_cnn

def train_model(model, x_train, y_train):
    """Trains the CNN model."""
    model.fit(x_train, y_train, epochs=5, validation_split=0.1)
    return model