import pandas as pd
import numpy as np
import pathlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from joblib import dump, load

# Cargar el dataset de detección de fraude
print("Cargando el dataset..")
df = pd.read_csv(pathlib.Path('data/creditcard_2023.csv'))

# Eliminar la columna 'id' si existe en el dataset
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Separar las características (X) y la variable objetivo (y)
X = df.drop(columns=['Class'])  # Eliminar la columna 'Class' (es la variable objetivo)
y = df['Class']                 # La columna 'Class' es la variable objetivo

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Escalar los datos para mejorar el rendimiento del modelo
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar el modelo
print("Entrenando el modelo..")
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train_scaled, y_train)

# Evaluar el modelo
print("Evaluación del modelo..")
y_pred = clf.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Guardar el modelo entrenado y el escalador
print("Guardando el modelo..")
dump(clf, pathlib.Path('model/creditcard-fraud-v1.joblib'))
dump(scaler, pathlib.Path('model/scaler.joblib'))  # Guardar también el escalador para usarlo en predicciones futuras

print("Entrenamiento y guardado del modelo completo.")

# Cargar el modelo entrenado y el escalador
loaded_model = load(pathlib.Path('model/creditcard-fraud-v1.joblib'))
scaler = load(pathlib.Path('model/scaler.joblib'))  # Cargar el escalador guardado

# Definir los nuevos datos para predecir (sin la columna 'id')
new_data = pd.DataFrame({
    'V1': [1.2060142158925],
    'V2': [-0.883519266886649],
    'V3': [1.02129930498789],
    'V4': [-1.41004682543458],
    'V5': [-0.341590726704414],
    'V6': [0.101490927939643],
    'V7': [0.036318183259545],
    'V8': [-0.141480335152521],
    'V9': [-0.348832472179471],
    'V10': [1.49943749721418],
    'V11': [-0.0206632743973843],
    'V12': [0.270369734598419],
    'V13': [-0.449711040783195],
    'V14': [0.652047585488573],
    'V15': [0.217553489682765],
    'V16': [0.485169455744755],
    'V17': [0.528776754387637],
    'V18': [0.996338123810281],
    'V19': [-0.407218557226148],
    'V20': [-0.566847648753014],
    'V21': [-0.106932274662551],
    'V22': [0.252150942310138],
    'V23': [0.00434958921036528],
    'V24': [1.06593325661107],
    'V25': [0.33009979963702],
    'V26': [-0.289690732999582],
    'V27': [-0.188490496247],
    'V28': [-0.0608150030191554],
    'Amount': [4741.71]
})

# Escalar los nuevos datos usando el escalador cargado
new_data_scaled = scaler.transform(new_data)

# Realizar la predicción con los datos escalados
predictions = loaded_model.predict(new_data_scaled)
print(f'First prediction: {predictions}')
