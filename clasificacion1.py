import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Cargar el archivo CSV
file_path = 'categorias.csv'
df = pd.read_csv(file_path)

# Graficar la distribución de clases
class_counts = df['is_dead'].value_counts()

# Crear un gráfico de barras
plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Classes')
plt.xlabel('Class (0: Alive, 1: Dead)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Separar las características (X) y la variable objetivo (y)
X = df.drop(columns=['is_dead', 'categoria_edad'])
y = df['is_dead']

# Realizar la partición estratificada
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Definir el modelo de árbol de decisión
clf = DecisionTreeClassifier(random_state=42)

# Definir los parámetros para la búsqueda de cuadrícula
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'auto', 'sqrt', 'log2']
}

# Realizar la búsqueda de cuadrícula
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Obtener los mejores hiperparámetros
best_params = grid_search.best_params_
print(f'Mejores hiperparámetros: {best_params}')

# Ajustar el modelo con los mejores hiperparámetros
best_clf = grid_search.best_estimator_
best_clf.fit(X_train, y_train)

# Realizar predicciones sobre el conjunto de prueba
y_pred = best_clf.predict(X_test)

# Calcular la precisión
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy del árbol de decisión con los mejores hiperparámetros sobre el conjunto de prueba: {accuracy:.2f}')
