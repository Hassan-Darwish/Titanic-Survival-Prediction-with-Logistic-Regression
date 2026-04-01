from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

weights = np.load('trained_weights.npy')
bias = np.load('trained_bias.npy')

class Passenger(BaseModel):
    class_: float
    sex: int
    age: float
    fare: float
    family_size: float
    embarked_c: int
    embarked_q: int
    embarked_s: int

@app.post('/predict')
async def predict(passenger: Passenger):
    features = np.array([
        passenger.class_,
        passenger.sex,
        passenger.age,
        passenger.fare,
        passenger.family_size,
        passenger.embarked_c,
        passenger.embarked_q,
        passenger.embarked_s
    ])

    z = np.dot(features, weights) + bias

    probability = 1 / (1 + np.exp(-z))

    prediction = 1 if probability > 0.5 else 0
    status = 'survived' if prediction == 1 else 'died'

    return {
        'survival_probability': float(probability),
        'predicition' : prediction,
        'status' : status
    }