 import numpy as np  # Importa la librería NumPy para trabajar con matrices y operaciones matemáticas
from sklearn.model_selection import train_test_split  # Importa la función train_test_split de scikit-learn para dividir el conjunto de datos en entrenamiento y prueba
from sklearn.metrics import accuracy_score  # Importa la función accuracy_score de scikit-learn para evaluar la precisión del modelo

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.1, max_epochs=100):
        """
        Inicializa un objeto de la clase Perceptron.
        
        Parámetros:
            - num_inputs: Número de características de entrada.
            - learning_rate: Tasa de aprendizaje del perceptrón.
            - max_epochs: Número máximo de épocas para el entrenamiento.
        """
        self.weights = np.random.rand(num_inputs + 1)  # Inicializa los pesos con valores aleatorios
        self.learning_rate = learning_rate  # Asigna la tasa de aprendizaje
        self.max_epochs = max_epochs  # Asigna el número máximo de épocas de entrenamiento

    def train(self, X, y):
        """
        Entrena el perceptrón utilizando el algoritmo de aprendizaje del perceptrón.
        
        Parámetros:
            - X: Matriz de características de entrada.
            - y: Vector de etiquetas de salida.
        """
        X = np.insert(X, 0, 1, axis=1)  # Agrega un término de sesgo a las características de entrada
        epochs = 0
        while epochs < self.max_epochs:  # Itera hasta que se alcance el número máximo de épocas
            error_count = 0
            for i in range(X.shape[0]):  # Itera sobre todas las muestras de entrada
                prediction = np.dot(X[i], self.weights) > 0  # Realiza la predicción del perceptrón
                error = y[i] - prediction  # Calcula el error de predicción
                if error != 0:  # Si hay error en la predicción
                    self.weights += self.learning_rate * error * X[i]  # Actualiza los pesos del perceptrón
                    error_count += 1  # Incrementa el contador de errores
            if error_count == 0:  # Si no hay errores en ninguna muestra
                break  # Termina el bucle de entrenamiento
            epochs += 1  # Incrementa el contador de épocas

    def predict(self, X):
        """
        Realiza predicciones utilizando el perceptrón entrenado.
        
        Parámetros:
            - X: Matriz de características de entrada.
        
        Retorna:
            - predictions: Vector de predicciones binarias.
        """
        X = np.insert(X, 0, 1, axis=1)  # Agrega un término de sesgo a las características de entrada
        predictions = []
        for i in range(X.shape[0]):  # Itera sobre todas las muestras de entrada
            prediction = np.dot(X[i], self.weights) > 0  # Realiza la predicción del perceptrón
            predictions.append(prediction)  # Agrega la predicción al vector de predicciones
        return np.array(predictions)  # Convierte las predicciones en un array de NumPy

def read_data(file_name):
    """
    Lee los datos del archivo CSV especificado.
    
    Parámetros:
        - file_name: Nombre del archivo CSV.
    
    Retorna:
        - X: Matriz de características de entrada.
        - y: Vector de etiquetas de salida.
    """
    data = np.genfromtxt(file_name, delimiter=',')  # Lee los datos desde el archivo CSV
    X = data[:, :-1]  # Extrae las características de entrada
    y = data[:, -1]  # Extrae las etiquetas de salida
    return X, y

def generate_partitions(X, y, num_partitions, train_size):
    """
    Genera particiones de entrenamiento y prueba utilizando train_test_split.
    
    Parámetros:
        - X: Matriz de características de entrada.
        - y: Vector de etiquetas de salida.
        - num_partitions: Número de particiones a generar.
        - train_size: Proporción de datos a utilizar para entrenamiento.
    
    Retorna:
        - partitions: Lista de tuplas que contienen las particiones de entrenamiento y prueba.
    """
    partitions = []
    for _ in range(num_partitions):  # Itera sobre el número de particiones especificado
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=None)  # Divide los datos en entrenamiento y prueba
        partitions.append((X_train, X_test, y_train, y_test))  # Agrega las particiones a la lista
    return partitions

def main():
    datasets = ['spheres1d10.csv', 'spheres2d10.csv', 'spheres2d50.csv', 'spheres2d70.csv']  # Lista de nombres de archivos de dataset
    perturbations = [10, 10, 50, 70]  # Lista de niveles de perturbación

    for dataset, perturbation in zip(datasets, perturbations):  # Itera sobre los nombres de archivos y los niveles de perturbación
        print(f"\nDataset: {dataset}")
        X, y = read_data(dataset)  # Lee los datos del archivo CSV
        num_partitions = 10  # Número de particiones a generar
        train_size = 0.8  # Proporción de datos a utilizar para entrenamiento

        partitions = generate_partitions(X, y, num_partitions, train_size)  # Genera las particiones de entrenamiento y prueba

        for i, partition in enumerate(partitions):  # Itera sobre las particiones generadas
            X_train, X_test, y_train, y_test = partition  # Obtiene las particiones de entrenamiento y prueba

            perceptron = Perceptron(num_inputs=X_train.shape[1])  # Inicializa un perceptrón
            perceptron.train(X_train, y_train)  # Entrena el perceptrón con los datos de entrenamiento

            predictions = perceptron.predict(X_test)  # Realiza predicciones con los datos de prueba
            accuracy = accuracy_score(y_test, predictions)  # Calcula la precisión del modelo
            print(f"Partition {i+1}: Accuracy on test set: {accuracy}")  # Imprime la precisión del modelo para la partición actual

if __name__ == '__main__':
    main()  # Ejecuta la función principal si el script se ejecuta como programa principal
