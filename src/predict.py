import numpy as np

def predict(model, new_data, scaler):
    # Escalar los datos nuevos usando el mismo scaler
    new_data_scaled = scaler.transform(new_data)

    # Realizar predicciones con el modelo entrenado
    predictions = model.predict(new_data_scaled)
    
    # Convertir las probabilidades en predicciones binarias
    predictions_bin = (predictions > 0.5).astype(int)
    
    return predictions_bin
