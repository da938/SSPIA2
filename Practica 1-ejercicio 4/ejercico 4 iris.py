import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class MLP:
    def __init__(self, layers):
        self.layers = layers
        self.weights = [np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2.0 / layers[i]) for i in
                        range(len(layers) - 1)]
        self.biases = [np.zeros((1, layers[i + 1])) for i in range(len(layers) - 1)]

    def activation(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def activation_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.layers) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            a = self.softmax(z) if i == len(self.layers) - 2 else self.activation(z)
            self.z_values.append(z)
            self.activations.append(a)

    def backward(self, y, learning_rate):
        errors = [y - self.activations[-1]]
        deltas = [errors[-1]]

        for i in range(len(self.layers) - 2, 0, -1):
            error = deltas[-1].dot(self.weights[i].T)
            delta = error * self.activation_derivative(self.activations[i])
            errors.append(error)
            deltas.append(delta)

        deltas.reverse()

        for i in range(len(self.layers) - 1):
            self.weights[i] += self.activations[i].T.dot(deltas[i]) * learning_rate
            self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(y, learning_rate)

    def predict(self, X):
        self.forward(X)
        return self.activations[-1]

    def evaluate_loo(self, X, y):
        loo = LeaveOneOut()
        accuracies = []

        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.train(X_train, y_train, epochs=1000, learning_rate=0.2)
            y_pred = self.predict(X_test)
            accuracies.append(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))

        return np.mean(accuracies), np.std(accuracies)

    def evaluate_kfold(self, X, y, k):
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        accuracies = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.train(X_train, y_train, epochs=1000, learning_rate=0.2)
            y_pred = self.predict(X_test)
            accuracies.append(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))

        return np.mean(accuracies), np.std(accuracies)


# Cargar los datos
data = pd.read_csv('irisbin.csv', header=None)

# Separar características y etiquetas
X = data.iloc[:, :-3].values
y = data.iloc[:, -3:].values

# Normalizar los datos
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Definir la estructura de la red
layers = [X.shape[1], 8, 3]
mlp = MLP(layers)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
mlp.train(X_train, y_train, epochs=1000, learning_rate=0.2)

# Predicciones en el conjunto de prueba
predictions = mlp.predict(X_test)
accuracy_test = accuracy_score(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))
print(f"Exactitud en el conjunto de prueba: {accuracy_test:.2f}")

# Evaluación Leave-One-Out
loo_mean_accuracy, loo_std_accuracy = mlp.evaluate_loo(X, y)
loo_error = 1 - loo_mean_accuracy
print("\nLeave-One-Out Cross-Validation:")
print(f"Error esperado: {loo_error:.2f}")
print(f"Promedio: {loo_mean_accuracy:.2f}")
print(f"Desviación estándar: {loo_std_accuracy:.2f}")

# Evaluación K-Fold (K=5)
kfold_mean_accuracy, kfold_std_accuracy = mlp.evaluate_kfold(X, y, k=5)
kfold_error = 1 - kfold_mean_accuracy
print("\nK-Fold Cross-Validation (k=5):")
print(f"Error esperado: {kfold_error:.2f}")
print(f"Promedio: {kfold_mean_accuracy:.2f}")
print(f"Desviación estándar: {kfold_std_accuracy:.2f}")

# Mostrar predicciones y especies reales
print("\nPredicciones y Especies Reales:")
for i, (prediction, true_species) in enumerate(zip(predictions, y_test)):
    species_pred = ['Virginica', 'Versicolor', 'Setosa'][np.argmax(prediction)]
    species_real = ['Virginica', 'Versicolor', 'Setosa'][np.argmax(true_species)]
    print(f"{i + 1}: Predicción={species_pred}, Especie real={species_real}")

# Visualización de los datos de prueba
plt.scatter(X_test[y_test[:, 0] == 1, 0], X_test[y_test[:, 0] == 1, 1], color='red', label='Setosa', alpha=0.7)
plt.scatter(X_test[y_test[:, 1] == 1, 0], X_test[y_test[:, 1] == 1, 1], color='green', label='Versicolor', alpha=0.7)
plt.scatter(X_test[y_test[:, 2] == 1, 0], X_test[y_test[:, 2] == 1, 1], color='blue', label='Virginica', alpha=0.7)
plt.xlabel('Característica 1 - Pétalo')
plt.ylabel('Característica 2 - Sépalo')
plt.title('Clasificación con MLP para Especies de Iris')
plt.legend(loc='lower right', bbox_transform=plt.gcf().transFigure)
plt.show()
