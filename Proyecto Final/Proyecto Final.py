import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(file_path):
    """
    Carga y preprocesa el conjunto de datos Zoo.

    Args:
        file_path (str): Ruta del archivo CSV del conjunto de datos Zoo.

    Returns:
        tuple: Conjuntos de entrenamiento y prueba para características (X) y etiquetas (y).
    """
    dataset = pd.read_csv(file_path)
    X = dataset.drop(['animal_name', 'type'], axis=1)
    y = dataset['type']

    # Normalizar las características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def evaluate_model(model_name, model, X_test, y_test):
    """
    Evalúa el modelo y muestra las métricas de rendimiento.

    Args:
        model_name (str): Nombre del modelo.
        model (sklearn estimator): Modelo entrenado.
        X_test (array): Datos de prueba.
        y_test (array): Etiquetas de prueba.
    """
    y_pred = model.predict(X_test)

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Calcular specificity
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn = conf_matrix[0][0]  # Verdaderos negativos
    fp = conf_matrix[0][1]  # Falsos positivos
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    # Imprimir métricas
    print(f"\n-------- {model_name} --------\n")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"F1 Score: {f1:.3f}\n")

    # Graficar las métricas
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score']
    metrics_values = [accuracy, precision, recall, specificity, f1]
    plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.title(model_name)
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    for i, value in enumerate(metrics_values):
        plt.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
    plt.show()


def logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    evaluate_model('Logistic Regression', model, X_test, y_test)


def k_nearest_neighbors(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    evaluate_model('K-Nearest Neighbors', model, X_test, y_test)


def support_vector_machine(X_train, X_test, y_train, y_test):
    model = SVC(C=1.0)
    model.fit(X_train, y_train)
    evaluate_model('Support Vector Machine', model, X_test, y_test)


def naive_bayes(X_train, X_test, y_train, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    evaluate_model('Naive Bayes', model, X_test, y_test)


def neural_network(X_train, X_test, y_train, y_test):
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    model.fit(X_train, y_train)
    evaluate_model('Neural Network', model, X_test, y_test)


# Ruta del archivo CSV del conjunto de datos Zoo
file_path = 'zoo.csv'

# Cargar y preprocesar los datos
X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

# Evaluación de los clasificadores
logistic_regression(X_train, X_test, y_train, y_test)
k_nearest_neighbors(X_train, X_test, y_train, y_test)
support_vector_machine(X_train, X_test, y_train, y_test)
naive_bayes(X_train, X_test, y_train, y_test)
neural_network(X_train, X_test, y_train, y_test)
