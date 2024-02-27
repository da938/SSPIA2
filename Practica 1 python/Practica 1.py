import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.1, max_epochs=100):
        # Inicialización de pesos aleatorios y otros parámetros
        self.weights = np.random.rand(num_inputs + 1)  # +1 para el sesgo (bias)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def train(self, X, y):
        # Entrenamiento del perceptrón
        X = np.insert(X, 0, 1, axis=1)  # Agregando sesgo
        epochs = 0
        while epochs < self.max_epochs:
            error_count = 0
            for i in range(X.shape[0]):
                # Predicción del perceptrón
                prediction = np.dot(X[i], self.weights) > 0
                error = y[i] - prediction
                if error != 0:
                    # Actualización de pesos si hay error
                    self.weights += self.learning_rate * error * X[i]
                    error_count += 1
            if error_count == 0:
                break
            epochs += 1

    def predict(self, X):
        # Predicción de nuevas instancias
        X = np.insert(X, 0, 1, axis=1)  # Agregando sesgo
        predictions = []
        for i in range(X.shape[0]):
            prediction = np.dot(X[i], self.weights) > 0
            predictions.append(prediction)
        return np.array(predictions)


def read_data(file_name):
    try:
        # Lectura de datos desde un archivo CSV
        data = np.genfromtxt(file_name, delimiter=',')
        X = data[:, :-1]
        y = data[:, -1]
        return X, y
    except IOError:
        # Manejo de errores si no se puede abrir el archivo
        print(f"Error: No se pudo abrir el archivo {file_name}")
        return None, None


def train_perceptron(X_train, y_train, num_inputs, learning_rate=0.1, max_epochs=100):
    # Función para entrenar el perceptrón y devolver el modelo entrenado
    perceptron = Perceptron(num_inputs=num_inputs, learning_rate=learning_rate, max_epochs=max_epochs)
    perceptron.train(X_train, y_train)
    return perceptron


def evaluate_accuracy(perceptron, X_test, y_test):
    # Evaluación de la precisión del perceptrón en el conjunto de prueba
    predictions = perceptron.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print("Accuracy on test set:", accuracy)


def plot_data(X, y, weights=None):
    # Función para visualizar los datos y, opcionalmente, la línea de decisión
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
    if weights is not None:
        # Visualización de la línea de decisión si se proporcionan los pesos
        plt.plot([-2, 2], [(weights[0] + weights[1] * (-2)) / -weights[2], (weights[0] + weights[1] * 2) / -weights[2]],
                 color='green', label='Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Data and Decision Boundary')
    plt.legend()
    plt.show()


def main():
    # Lectura de datos de entrenamiento y prueba
    X_train, y_train = read_data('XORtrn.csv')
    X_test, y_test = read_data('XORtst.csv')

    if X_train is not None and X_test is not None:
        # Entrenamiento del perceptrón
        perceptron = train_perceptron(X_train, y_train, num_inputs=X_train.shape[1])

        # Evaluación de la precisión del perceptrón en el conjunto de prueba
        evaluate_accuracy(perceptron, X_test, y_test)

        # Visualización de los datos y la línea de decisión
        plot_data(X_train, y_train, perceptron.weights)


if __name__ == '__main__':
    main()
