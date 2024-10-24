from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

# Permitir cualquier origen (ajusta según sea necesario)
origins = ['*']

app = FastAPI(title='Credit Card Fraud Detection API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# Cargar el modelo y el escalador
model = load(pathlib.Path('model/creditcard-fraud-v1.joblib'))
scaler = load(pathlib.Path('model/scaler.joblib'))

# Definir la entrada de datos según las columnas del dataset (exceptuando 'Class')
class InputData(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

class OutputData(BaseModel):
    score: float

@app.post('/score', response_model=OutputData)
def score(data: InputData):
    # Convertir los datos de entrada en un array y escalarlos
    model_input = np.array([v for v in data.dict().values()]).reshape(1, -1)
    model_input_scaled = scaler.transform(model_input)
    
    # Obtener la probabilidad de predicción
    result = model.predict_proba(model_input_scaled)[:, -1]

    return {'score': result[0]}
