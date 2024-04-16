import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    """
    Clase que representa una red neuronal multicapa.
    """

    def _init_(self, layer_sizes):
        """
        Inicializa la red neuronal con capas de pesos y sesgos aleatorios.

        Args:
        layer_sizes (list): Una lista que especifica el número de neuronas en cada capa de la red.
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        # Inicialización aleatoria de pesos y sesgos
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) for i in range(self.num_layers - 1)]
        self.biases = [np.random.randn(1, layer_sizes[i+1]) for i in range(self.num_layers - 1)]

    def forward(self, X):
        """
        Propaga las entradas hacia adelante a través de la red.

        Args:
        X (array): Entradas de la red neuronal.

        Returns:
        array: Salida de la red neuronal después de la propagación hacia adelante.
        """
        # Propagación hacia adelante
        self.activations = [X]
        for i in range(self.num_layers - 1):
            weighted_input = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.activations.append(self.sigmoid(weighted_input))
        return self.activations[-1]

    def backward(self, X, y, output, learning_rate):
        """
        Realiza la retropropagación del error para ajustar los pesos y sesgos.

        Args:
        X (array): Entradas de la red neuronal.
        y (array): Etiquetas reales correspondientes a las entradas.
        output (array): Salida de la red neuronal.
        learning_rate (float): Tasa de aprendizaje para ajustar los pesos y sesgos.
        """
        # Retropropagación
        error = y - output
        deltas = [error * self.sigmoid_derivative(output)]

        for i in range(self.num_layers - 2, 0, -1):
            error = deltas[-1].dot(self.weights[i].T)
            deltas.append(error * self.sigmoid_derivative(self.activations[i]))
        deltas.reverse()

        # Actualización de pesos y sesgos
        for i in range(self.num_layers - 1):
            self.weights[i] += np.dot(self.activations[i].T, deltas[i]) * learning_rate
            self.biases[i] += np.sum(deltas[i], axis=0) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        """
        Entrena la red neuronal utilizando el algoritmo de retropropagación.

        Args:
        X (array): Entradas de entrenamiento de la red neuronal.
        y (array): Etiquetas de entrenamiento correspondientes a las entradas.
        epochs (int): Número de épocas de entrenamiento.
        learning_rate (float): Tasa de aprendizaje para ajustar los pesos y sesgos.
        """
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)

    def predict(self, X):
        """
        Realiza predicciones para nuevas entradas.

        Args:
        X (array): Entradas para las cuales realizar predicciones.

        Returns:
        array: Salida predicha para las entradas dadas.
        """
        return np.round(self.forward(X))

    @staticmethod
    def sigmoid(x):
        """
        Función de activación sigmoide.

        Args:
        x (array): Entradas a la función sigmoide.

        Returns:
        array: Salida de la función sigmoide.
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        """
        Derivada de la función de activación sigmoide.

        Args:
        x (array): Entradas a la derivada de la función sigmoide.

        Returns:
        array: Salida de la derivada de la función sigmoide.
        """
        return x * (1 - x)


# Cargar datos
data = np.genfromtxt('mydataset.csv', delimiter=',')
X = data[:, :-1]
y = data[:, -1]

# Definir la arquitectura de la red neuronal
layer_sizes = [X.shape[1], 5, 1]

# Crear y entrenar la red neuronal
model = NeuralNetwork(layer_sizes)
model.train(X, y.reshape(-1, 1), epochs=8000, learning_rate=0.05)

# Predecir y graficar resultados
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Red Neuronal para Clasificación')
plt.show()