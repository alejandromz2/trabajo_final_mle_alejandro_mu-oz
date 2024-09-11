from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def evaluate_model(model, X_test_scaled, y_test):
    # Evaluar el modelo en el conjunto de prueba
    loss, accuracy = model.evaluate(X_test_scaled, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

def plot_accuracy(history):
    # Gráfica de la precisión
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
