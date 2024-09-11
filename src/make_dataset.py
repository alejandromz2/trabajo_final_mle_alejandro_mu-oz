import pandas as pd

def make_dataset(file_path):
    # Convertimos nuestro archivo local al formato dataframe necesario para trabajar en Python
    data = pd.read_excel(file_path)
    
    # Transformar "sat_general" en una variable binaria
    data['sat_general_bin'] = data['sat_general'].apply(lambda x: 1 if x == 5 else 0)
    
    # Selección de las variables independientes y dependiente
    X = data[['sat1', 'sat2', 'sat3', 'sat4', 'sat5', 'sat6', 'sat7', 'sat8', 'medicion', 'N_días_hosp', 'sexo', 'edad', 'hospital']]
    y = data['sat_general_bin']
    
    return X, y
