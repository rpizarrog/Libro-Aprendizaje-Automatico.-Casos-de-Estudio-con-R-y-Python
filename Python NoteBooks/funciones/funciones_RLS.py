# funciones_RLS.py
# ------------------------------------------------------------
# Funciones para implementar y evaluar modelos de
# Regresión Lineal Simple en Python
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def f_cargar_datos(ruta_archivo):
    #------------------------------------------------------------
    # Importar datos desde un archivo CSV.
    #------------------------------------------------------------
    datos = pd.read_csv(ruta_archivo)
    return datos


def f_visualizar_head_tail_reducido_word(datos, n=10):
    total_columnas = datos.shape[1]

    idx_prim = list(range(min(4, total_columnas)))
    idx_ult = list(range(max(total_columnas - 3, 0), total_columnas))

    columnas = sorted(set(idx_prim + idx_ult))

    datos_reducidos = datos.iloc[:, columnas]

    head_datos = datos_reducidos.head(n).astype(str)
    tail_datos = datos_reducidos.tail(n).astype(str)

    fila_puntos = pd.DataFrame([["..."] * datos_reducidos.shape[1]],
                              columns=datos_reducidos.columns)

    tabla_final = pd.concat([head_datos, fila_puntos, tail_datos])

    return tabla_final


def f_describir_datos(datos):
    describe = datos.describe()

    from io import StringIO
    buffer = StringIO()
    datos.info(buf=buffer)
    structure = buffer.getvalue()

    return {
        "describe": describe,
        "structure": structure
    }


def f_particionar_datos(datos, proporcion_entrenamiento=0.7):
    train, test = train_test_split(
        datos,
        train_size=proporcion_entrenamiento,
        random_state=2026
    )

    return {
        "datos_entrenamiento": train,
        "datos_validacion": test
    }


def f_construir_modelo(datos_entrenamiento, variable_independiente, variable_dependiente):
    X = datos_entrenamiento[[variable_independiente]]
    y = datos_entrenamiento[variable_dependiente]

    modelo = LinearRegression()
    modelo.fit(X, y)

    return modelo


def f_diagrama_dispersion_tendencia(modelo, datos, x, y):
    x_vals = datos[x]
    y_vals = datos[y]

    r = np.corrcoef(x_vals, y_vals)[0,1]

    y_pred = modelo.predict(datos[[x]])
    r2 = r2_score(y_vals, y_pred)

    plt.figure(figsize=(8,6))
    plt.scatter(x_vals, y_vals)

    x_sorted = np.sort(x_vals)
    y_line = modelo.predict(x_sorted.reshape(-1,1))

    plt.plot(x_sorted, y_line)

    plt.title("Dispersión y tendencia")
    plt.suptitle(f"{x} vs {y} ; r={r:.3f} ; R²={r2:.3f}")

    plt.xlabel(x)
    plt.ylabel(y)

    plt.show()


def f_evaluar_modelo(modelo, datos_validacion, variable_dependiente, variable_independiente):
    X = datos_validacion[[variable_independiente]]
    y_real = datos_validacion[variable_dependiente]

    pred = modelo.predict(X)

    mse = mean_squared_error(y_real, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_real, pred)

    n = len(y_real)
    p = 1

    r2_adj = 1 - (1 - r2)*(n - 1)/(n - p - 1)

    resultado = pd.DataFrame({
        "R_square": [round(r2,4)],
        "R_square_ajustado": [round(r2_adj,4)],
        "MSE": [round(mse,4)],
        "RMSE": [round(rmse,4)]
    })

    return resultado
