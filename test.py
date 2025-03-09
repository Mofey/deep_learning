def evaluate_model(model, x_test, y_test):
    """Evaluates the trained CNN model."""
    loss, acc = model.evaluate(x_test, y_test)
    print(f'Test Accuracy: {acc:.4f}')