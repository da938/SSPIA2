import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

# Desactivar las advertencias de Sklearn
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Cargar los datasets desde los archivos CSV
swedish_auto_data = pd.read_csv('AutoInsurSweden.csv')
wine_quality_data = pd.read_csv('wine-Quality.csv')
pima_diabetes_data = pd.read_csv('pima-indians-diabetes.csv')

# Categorizar los valores de Y en Swedish Auto Insurance Dataset
quartiles = swedish_auto_data['Y'].quantile([0.25, 0.5, 0.75])
low_limit = quartiles.iloc[0]
medium_limit = quartiles.iloc[1]
high_limit = quartiles.iloc[2]


def categorize_y(y):
    """Categoriza el valor de Y en bajo, medio o alto según los cuartiles."""
    if y <= low_limit:
        return 'bajo'
    elif y <= medium_limit:
        return 'medio'
    else:
        return 'alto'


swedish_auto_data['Y_category'] = swedish_auto_data['Y'].apply(categorize_y)


def evaluate_classifier(X, y, classifier):
    """
    Entrena y evalúa un clasificador en los datos proporcionados, imprimiendo
    las métricas de evaluación.

    Parámetros:
    X (DataFrame): Características del conjunto de datos.
    y (Series): Etiquetas del conjunto de datos.
    classifier (Class): Clase del clasificador a usar.
    """
    # Dividir los datos en conjunto de entrenamiento y conjunto de prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instanciar y entrenar el clasificador
    model = classifier(max_iter=1000) if classifier == LogisticRegression else classifier()
    model.fit(X_train, y_train)

    # Hacer predicciones
    y_pred = model.predict(X_test)

    # Calcular métricas de evaluación
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division='warn')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Imprimir métricas
    print(f"\nMétricas para {classifier.__name__}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


# Swedish Auto Insurance Dataset
print("\nSwedish Auto Insurance Dataset:")
X_swedish = swedish_auto_data[['X']]
y_swedish = swedish_auto_data['Y_category']
evaluate_classifier(X_swedish, y_swedish, LogisticRegression)
evaluate_classifier(X_swedish, y_swedish, KNeighborsClassifier)
evaluate_classifier(X_swedish, y_swedish, SVC)
evaluate_classifier(X_swedish, y_swedish, GaussianNB)
evaluate_classifier(X_swedish, y_swedish, MLPClassifier)

# Wine Quality Dataset
print("\nWine Quality Dataset:")
X_wine = wine_quality_data.drop('quality', axis=1)
y_wine = wine_quality_data['quality']
evaluate_classifier(X_wine, y_wine, LogisticRegression)
evaluate_classifier(X_wine, y_wine, KNeighborsClassifier)
evaluate_classifier(X_wine, y_wine, SVC)
evaluate_classifier(X_wine, y_wine, GaussianNB)
evaluate_classifier(X_wine, y_wine, MLPClassifier)

# Pima Indians Diabetes Dataset
print("\nPima Indians Diabetes Dataset:")
X_pima = pima_diabetes_data.drop('Class variable (0 or 1)', axis=1)
y_pima = pima_diabetes_data['Class variable (0 or 1)']
evaluate_classifier(X_pima, y_pima, LogisticRegression)
evaluate_classifier(X_pima, y_pima, KNeighborsClassifier)
evaluate_classifier(X_pima, y_pima, SVC)
evaluate_classifier(X_pima, y_pima, GaussianNB)
evaluate_classifier(X_pima, y_pima, MLPClassifier)
