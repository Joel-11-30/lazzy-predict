# =============================
# LazyPredict con datos de diabetes
# =============================

import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar el dataset
df = pd.read_csv("D:/2025/estadistica computacional/lazypredict/diabetes.csv")

# Separar características y variable objetivo
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Inicializar LazyClassifier
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)

# Entrenar y evaluar modelos
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# Mostrar los resultados ordenados por precisión
print("Modelos evaluados en datos de diabetes:\n")
print(models.sort_values("Accuracy", ascending=False))
