# FUNCIONES PARA MODELOS DE REGRESIÓN EXPONENCIAL LINEAL, POLINOMIAL y LOGARITMICA
# Archivo: funciones_MExp.py
#  Autor: Rubén Pizarro Gurrola (adaptado)
# ============================================================
# Librerías

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import scipy.stats as stats

from statsmodels.stats.stattools import durbin_watson

from sklearn.model_selection import train_test_split # Partir datos
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf # Para modelo logarítmico

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.pipeline import Pipeline
from statsmodels.nonparametric.smoothers_lowess import lowess

from statsmodels.stats.stattools import durbin_watson


# Funciones
# Funciones para ejecución del caso
def f_cargar_datos(ruta_archivo):
    #------------------------------------------------------------
    # Importar datos desde un archivo CSV.
    # Argumentos:
    #   ruta_archivo: ruta del archivo a cargar.
    # Retorna:
    #   DataFrame listo para análisis.
    #------------------------------------------------------------

    datos = pd.read_csv(ruta_archivo)
    return datos

def f_visualizar_head_tail_reducido_word(datos, n=10):
    #------------------------------------------------------------
    # Mostrar primeros n y últimos n registros en una sola tabla
    # con columnas reducidas:
    #   - primeras 4 columnas
    #   - últimas 3 columnas
    #------------------------------------------------------------

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
    #------------------------------------------------------------
    # Generar estadísticas descriptivas básicas
    #------------------------------------------------------------

    describe = datos.describe()

    structure = datos.info()

    return {
        "describe": describe,
        "structure": structure
    }

def f_particionar_datos(datos, proporcion_entrenamiento=0.7):
    #------------------------------------------------------------
    # Dividir dataset en entrenamiento y validación
    #------------------------------------------------------------

    train, test = train_test_split(
        datos,
        train_size=proporcion_entrenamiento,
        random_state=2026
    )

    return {
        "datos_entrenamiento": train,
        "datos_validacion": test
    }


def f_construir_modelo(datos_entrenamiento, variable_independiente, variable_dependiente, grado=1):
    #------------------------------------------------------------
    # Construir modelo de regresión lineal o polinómico
    # usando Pipeline (evita warnings automáticamente)
    #------------------------------------------------------------
    X = datos_entrenamiento[[variable_independiente]]
    y = datos_entrenamiento[variable_dependiente]

    modelo = Pipeline([
        ("poly", PolynomialFeatures(degree=grado, include_bias=False)),
        ("lr", LinearRegression())
    ])

    modelo.fit(X, y)

    # 🔥 METADATA (ESTO FALTABA)
    modelo.tipo = "polinomial"
    modelo.variable_x = variable_independiente
    modelo.variable_y = variable_dependiente
    modelo.grado = grado

    return modelo    

def f_construir_modelo_log(datos, x, y):
    """
    Construye un modelo de regresión logarítmica (lin-log)

    datos: DataFrame (pandas)
    x: nombre variable independiente (str)
    y: nombre variable dependiente (str)
    """
    formula = f"{y} ~ np.log({x})"

    modelo = smf.ols(formula=formula, data=datos).fit()

    modelo.tipo = "logaritmico"
    modelo.variable_x = x
    modelo.variable_y = y

    return modelo


def f_construir_modelo_exp(datos, x, y):
  # Constuye modelo para regesión exponencial
    import numpy as np
    import statsmodels.formula.api as smf

    if (datos[y] <= 0).any():
        raise ValueError("La variable dependiente debe ser positiva")

    formula = f"np.log({y}) ~ {x}"

    modelo = smf.ols(formula=formula, data=datos).fit()

    # 🔥 METADATA
    modelo.tipo = "exponencial"
    modelo.variable_x = x
    modelo.variable_y = y

    return modelo

def f_diagrama_dispersion_tendencia(modelo, datos, x, y):

    x_vals = datos[x]
    y_vals = datos[y]

    # correlación
    r = np.corrcoef(x_vals, y_vals)[0,1]

    # predicción directa (Pipeline ya gestiona todo)
    y_pred = modelo.predict(datos[[x]])

    r2 = r2_score(y_vals, y_pred)

    plt.figure(figsize=(8,6))

    # dispersión
    plt.scatter(x_vals, y_vals, alpha=0.6)

    # ordenar X
    x_sorted = np.sort(x_vals.values)
    x_sorted_df = pd.DataFrame(x_sorted, columns=[x])

    y_line = modelo.predict(x_sorted_df)

    # curva
    plt.plot(x_sorted, y_line, color='red', linewidth=2,
             label=f"Grado {modelo.grado}")

    plt.title("Dispersión y tendencia")
    plt.suptitle(f"{x} vs {y} ; r={r:.3f} ; R²={r2:.3f}")

    plt.xlabel(x)
    plt.ylabel(y)

    plt.legend()

    plt.show()

def f_diagrama_dispersion_matriz(modelos, datos, x, y, titulos=None):

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score

    if len(modelos) != 4:
        raise ValueError("Debes proporcionar exactamente 4 modelos")

    if titulos is None:
        titulos = [f"Modelo {i+1}" for i in range(4)]

    x_vals = datos[x]
    y_vals = datos[y]

    r = np.corrcoef(x_vals, y_vals)[0,1]

    fig, axes = plt.subplots(2, 2, figsize=(12,10))
    axes = axes.flatten()

    for i, modelo in enumerate(modelos):

        ax = axes[i]

        ax.scatter(x_vals, y_vals, alpha=0.6)

        # ordenar x para curva suave
        x_sorted = np.sort(x_vals.values)
        x_sorted_df = pd.DataFrame(x_sorted, columns=[x])

        #----------------------------------------------------
        # 🔥 DETECTAR SI ES MODELO EXPONENCIAL
        #----------------------------------------------------
        es_exponencial = False

        if hasattr(modelo, "model"):  # statsmodels
            formula_str = str(modelo.model.formula)
            if "np.log" in formula_str.split("~")[0]:
                es_exponencial = True

        #----------------------------------------------------
        # PREDICCIONES
        #----------------------------------------------------
        y_pred = modelo.predict(datos[[x]])
        y_line = modelo.predict(x_sorted_df)

        # 🔥 Ajuste si es exponencial
        if es_exponencial:
            y_pred = np.exp(y_pred)
            y_line = np.exp(y_line)

        #----------------------------------------------------
        # MÉTRICAS
        #----------------------------------------------------
        r2 = r2_score(y_vals, y_pred)

        #----------------------------------------------------
        # GRÁFICA
        #----------------------------------------------------
        ax.plot(x_sorted, y_line, color='red', linewidth=2)

        ax.set_title(f"{titulos[i]}\nr={r:.3f} ; R²={r2:.3f}")
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    plt.tight_layout()
    plt.show()

def f_ecuaciones_modelos(modelos, nombres_modelos=None):

    import numpy as np

    if nombres_modelos is None:
        nombres_modelos = [f"Modelo {i+1}" for i in range(len(modelos))]

    for modelo, nombre in zip(modelos, nombres_modelos):

        print("\n" + "="*60)
        print(f" {nombre}")
        print("="*60)

        tipo = getattr(modelo, "tipo", "desconocido")

        #--------------------------------------------------------
        # 🔥 CASO 1: POLINOMIAL (sklearn)
        #--------------------------------------------------------
        if hasattr(modelo, "named_steps"):

            poly = modelo.named_steps["poly"]
            lr = modelo.named_steps["lr"]

            nombres_vars = poly.get_feature_names_out([modelo.variable_x])
            coeficientes = lr.coef_
            intercepto = lr.intercept_

            print("\nCoeficientes del modelo:")
            print(f"Intercepto: {round(intercepto,4)}")

            for nombre_var, coef in zip(nombres_vars, coeficientes):
                print(f"{nombre_var}: {round(coef,4)}")

            # ecuación
            ecuacion = f"ŷ = {round(intercepto,4)}"

            for c, nombre_var in zip(coeficientes, nombres_vars):

                signo = "+" if c >= 0 else "-"
                nombre_var = nombre_var.replace(" ", "")

                ecuacion += f" {signo} {abs(round(c,4))}·{nombre_var}"

        #--------------------------------------------------------
        # 🔥 CASO 2: EXPONENCIAL
        #--------------------------------------------------------
        elif tipo == "exponencial":

            params = modelo.params

            b0 = params.iloc[0]   # ln(a)
            b1 = params.iloc[1]

            a = np.exp(b0)
            b = b1

            nombre_var = modelo.variable_x

            print("\nCoeficientes del modelo:")
            print(f"ln(a): {round(b0,4)}")
            print(f"b: {round(b,4)}")

            ecuacion = f"ŷ = {round(a,4)} · e^({round(b,4)}·{nombre_var})"

        #--------------------------------------------------------
        # 🔥 CASO 3: LOGARÍTMICO
        #--------------------------------------------------------
        elif tipo == "logaritmico":

            params = modelo.params

            intercepto = params.iloc[0]
            coef = params.iloc[1]

            nombre_var = modelo.variable_x

            print("\nCoeficientes del modelo:")
            print(f"Intercepto: {round(intercepto,4)}")
            print(f"log({nombre_var}): {round(coef,4)}")

            ecuacion = f"ŷ = {round(intercepto,4)} + {round(coef,4)}·ln({nombre_var})"

        #--------------------------------------------------------
        # 🔥 CASO 4: FALLBACK (por si acaso)
        #--------------------------------------------------------
        else:

            params = modelo.params
            intercepto = params.iloc[0]
            coeficientes = params.iloc[1:]
            nombres_vars = params.index[1:]

            print("\nCoeficientes del modelo:")
            print(f"Intercepto: {round(intercepto,4)}")

            for nombre_var, coef in zip(nombres_vars, coeficientes):
                print(f"{nombre_var}: {round(coef,4)}")

            ecuacion = f"ŷ = {round(intercepto,4)}"

            for c, nombre_var in zip(coeficientes, nombres_vars):
                signo = "+" if c >= 0 else "-"
                ecuacion += f" {signo} {abs(round(c,4))}·{nombre_var}"

        #--------------------------------------------------------
        # 🔥 OUTPUT FINAL
        #--------------------------------------------------------
        print("\nEcuación del modelo:")
        print(ecuacion)

def f_evaluar_modelo(modelo, datos_validacion, variable_dependiente, variable_independiente):

    #------------------------------------------------------------
    # DATOS
    #------------------------------------------------------------
    y_real = datos_validacion[variable_dependiente].values

    tipo = getattr(modelo, "tipo", "desconocido")
    x_var = getattr(modelo, "variable_x", variable_independiente)

    #------------------------------------------------------------
    #  PREDICCIÓN CORRECTA SEGÚN MODELO
    #------------------------------------------------------------
    if hasattr(modelo, "named_steps"):
        # sklearn → polinomial
        X_poly = modelo.named_steps["poly"].transform(datos_validacion[[x_var]])
        pred = modelo.named_steps["lr"].predict(X_poly)

    elif tipo == "exponencial":
        pred_log = modelo.predict(datos_validacion)
        pred = np.exp(pred_log)

    elif tipo == "logaritmico":
        pred = modelo.predict(datos_validacion)

    else:
        raise ValueError("Tipo de modelo no reconocido")

    #------------------------------------------------------------
    # MÉTRICAS
    #------------------------------------------------------------
    mse = mean_squared_error(y_real, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_real, pred)

    #------------------------------------------------------------
    #  NÚMERO DE PARÁMETROS (p)
    #------------------------------------------------------------
    if hasattr(modelo, "named_steps"):
        poly = modelo.named_steps["poly"]
        p = len(poly.get_feature_names_out())

    else:
        # statsmodels → número de coeficientes - 1 (sin intercepto)
        p = len(modelo.params) - 1

    n = len(y_real)

    #------------------------------------------------------------
    # R² AJUSTADO
    #------------------------------------------------------------
    r2_adj = 1 - (1 - r2)*(n - 1)/(n - p - 1)

    #------------------------------------------------------------
    # RESULTADO
    #------------------------------------------------------------
    resultado = pd.DataFrame({
        "R_square": [round(r2,4)],
        "R_square_ajustado": [round(r2_adj,4)],
        "MSE": [round(mse,4)],
        "RMSE": [round(rmse,4)],
        "Parametros (p)": [p]
    })

    return resultado

def f_evaluar_modelos_varios(modelos, datos_validacion, y, x, nombres_modelos=None):

    import pandas as pd

    if nombres_modelos is None:
        nombres_modelos = [f"Modelo {i+1}" for i in range(len(modelos))]

    resultados = []

    for modelo, nombre in zip(modelos, nombres_modelos):

        try:
            res = f_evaluar_modelo(modelo, datos_validacion, y, x)
            res["Modelo"] = nombre
            resultados.append(res)

        except Exception as e:
            print(f"Error en modelo {nombre}: {e}")

    df_final = pd.concat(resultados, ignore_index=True)

    #------------------------------------------------------------
    #  ORDENAMIENTO
    #------------------------------------------------------------
    df_final = df_final.sort_values(by="RMSE")

    return df_final


# Funciones para validar el modelo con los supuestos
# del modelo polinomial
def f_verificar_linealidad(datos, x, y):
    #------------------------------------------------------------
    # Evaluar relación entre variables mediante scatter plot
    #------------------------------------------------------------

    plt.figure(figsize=(8,6))
    plt.scatter(datos[x], datos[y], alpha=0.6)

    plt.title("Verificación de linealidad")
    plt.xlabel(x)
    plt.ylabel(y)

    plt.show()

def f_matriz_verificar_homocedasticidad(modelos, datos, x, y, nombres_modelos=None):

    if len(modelos) != 4:
        raise ValueError("Debes proporcionar exactamente 4 modelos")

    if nombres_modelos is None:
        nombres_modelos = [f"Modelo {i+1}" for i in range(4)]

    fig, axes = plt.subplots(2, 2, figsize=(12,10))
    axes = axes.flatten()

    for i, modelo in enumerate(modelos):

        ax = axes[i]

        y_real = datos[y].values

        #--------------------------------------------------------
        #  PREDICCIÓN SEGÚN TIPO
        #--------------------------------------------------------
        tipo = getattr(modelo, "tipo", "desconocido")
        x_var = getattr(modelo, "variable_x", x)

        #--------------------------------------------------------
        #  DETECCIÓN INTELIGENTE
        #--------------------------------------------------------
        if hasattr(modelo, "named_steps"):
            # sklearn Pipeline → polinomial
            X_poly = modelo.named_steps["poly"].transform(datos[[x_var]])
            y_pred = modelo.named_steps["lr"].predict(X_poly)

        elif tipo == "exponencial":
            y_pred_log = modelo.predict(datos)
            y_pred = np.exp(y_pred_log)

        elif tipo == "logaritmico":
            y_pred = modelo.predict(datos)

        else:
            raise ValueError("Tipo de modelo no reconocido")

        #--------------------------------------------------------
        # RESIDUOS
        #--------------------------------------------------------
        residuos = y_real - y_pred

        ax.scatter(y_pred, residuos, alpha=0.5)
        ax.axhline(y=0, linestyle='--')

        # LOWESS
        curva = lowess(residuos, y_pred, frac=0.3)
        ax.plot(curva[:,0], curva[:,1], linewidth=2, alpha=0.5)

        # Banda
        std = np.std(residuos)
        x_sorted = np.sort(y_pred)

        ax.fill_between(x_sorted, -2*std, 2*std, alpha=0.05)

        ax.set_title(nombres_modelos[i])
        ax.set_xlabel("Valores ajustados")
        ax.set_ylabel("Residuos")

    plt.tight_layout()
    plt.show()

def f_matriz_normalidad_modelos(modelos, datos, x, y, nombres_modelos=None):
    #------------------------------------------------------------
    # Evaluación de normalidad de residuos (robusta)
    #------------------------------------------------------------

    if len(modelos) != 4:
        raise ValueError("Debes proporcionar exactamente 4 modelos")

    if nombres_modelos is None:
        nombres_modelos = [f"Modelo {i+1}" for i in range(4)]

    resultados = []

    fig, axes = plt.subplots(4, 2, figsize=(12,16))

    for i, modelo in enumerate(modelos):

        #--------------------------------------------------------
        #  DETECCIÓN INTELIGENTE DEL MODELO
        #--------------------------------------------------------
        tipo = getattr(modelo, "tipo", "desconocido")
        x_var = getattr(modelo, "variable_x", x)

        try:
            if hasattr(modelo, "named_steps"):
                # sklearn (polinomial)
                X_poly = modelo.named_steps["poly"].transform(datos[[x_var]])
                y_pred = modelo.named_steps["lr"].predict(X_poly)

            elif tipo == "exponencial":
                y_pred_log = modelo.predict(datos)
                y_pred = np.exp(y_pred_log)

            elif tipo == "logaritmico":
                y_pred = modelo.predict(datos)

            else:
                raise ValueError("Tipo de modelo no reconocido")

        except Exception as e:
            print(f"Error en modelo {i}: {e}")
            continue

        #--------------------------------------------------------
        # RESIDUOS
        #--------------------------------------------------------
        y_real = datos[y].values
        residuos = y_real - y_pred

        #--------------------------------------------------------
        #  SHAPIRO (ROBUSTO)
        #--------------------------------------------------------
        if len(residuos) > 5000:
            residuos_test = np.random.choice(residuos, 5000, replace=False)
        else:
            residuos_test = residuos

        W, p_value = stats.shapiro(residuos_test)

        interpretacion = "Normal" if p_value > 0.05 else "No normal"

        resultados.append({
            "Modelo": nombres_modelos[i],
            "W": round(W,4),
            "p_value": round(p_value,4),
            "Normalidad": interpretacion
        })

        #--------------------------------------------------------
        #  HISTOGRAMA
        #--------------------------------------------------------
        sns.histplot(residuos, kde=True, ax=axes[i,0], color="gray")
        axes[i,0].set_title(
            f"{nombres_modelos[i]}\nHistograma\n"
            f"W={W:.3f} | p={p_value:.3f} | {interpretacion}"
        )

        #--------------------------------------------------------
        #  QQ-PLOT
        #--------------------------------------------------------
        stats.probplot(residuos, dist="norm", plot=axes[i,1])
        axes[i,1].set_title(
            f"{nombres_modelos[i]}\nQ-Q Plot\n"
            f"W={W:.3f} | p={p_value:.3f} | {interpretacion}"
        )

    plt.tight_layout()
    plt.show()

    #------------------------------------------------------------
    # RESULTADOS
    #------------------------------------------------------------
    df_resultados = pd.DataFrame(resultados)

    df_resultados["Ranking"] = df_resultados["p_value"].rank(ascending=False)
    df_resultados = df_resultados.sort_values(by="Ranking")

    return df_resultados


def f_verificar_independencia_residuos(modelos, datos, x, y, nombres_modelos=None, graficar=True):
    #------------------------------------------------------------
    # Evaluación de independencia de residuos (Durbin-Watson)
    #------------------------------------------------------------

    if len(modelos) != 4:
        raise ValueError("Debes proporcionar exactamente 4 modelos")

    if nombres_modelos is None:
        nombres_modelos = [f"Modelo {i+1}" for i in range(4)]

    resultados = []

    if graficar:
        fig, axes = plt.subplots(2, 2, figsize=(12,8))
        axes = axes.flatten()

    for i, modelo in enumerate(modelos):

        #--------------------------------------------------------
        #  DETECCIÓN INTELIGENTE
        #--------------------------------------------------------
        tipo = getattr(modelo, "tipo", "desconocido")
        x_var = getattr(modelo, "variable_x", x)

        try:
            if hasattr(modelo, "named_steps"):
                # sklearn → polinomial
                X_poly = modelo.named_steps["poly"].transform(datos[[x_var]])
                y_pred = modelo.named_steps["lr"].predict(X_poly)

            elif tipo == "exponencial":
                y_pred_log = modelo.predict(datos)
                y_pred = np.exp(y_pred_log)

            elif tipo == "logaritmico":
                y_pred = modelo.predict(datos)

            else:
                raise ValueError("Tipo de modelo no reconocido")

        except Exception as e:
            print(f"Error en modelo {nombres_modelos[i]}: {e}")
            continue

        #--------------------------------------------------------
        # RESIDUOS
        #--------------------------------------------------------
        y_real = datos[y].values
        residuos = y_real - y_pred

        #--------------------------------------------------------
        #  DURBIN-WATSON
        #--------------------------------------------------------
        dw = durbin_watson(residuos)

        # interpretación
        if 1.5 <= dw <= 2.5:
            interpretacion = "Independencia"
        elif dw < 1.5:
            interpretacion = "Autocorrelación positiva"
        else:
            interpretacion = "Autocorrelación negativa"

        resultados.append({
            "Modelo": nombres_modelos[i],
            "Durbin_Watson": round(dw,4),
            "Interpretacion": interpretacion
        })

        #--------------------------------------------------------
        #  GRÁFICO
        #--------------------------------------------------------
        if graficar:
            ax = axes[i]

            ax.plot(residuos, marker='o', linestyle='-', alpha=0.6)
            ax.axhline(y=0, linestyle='--')

            ax.set_title(
                f"{nombres_modelos[i]}\nDW={dw:.3f}\n{interpretacion}"
            )
            ax.set_xlabel("Orden")
            ax.set_ylabel("Residuo")

    if graficar:
        plt.tight_layout()
        plt.show()

    #------------------------------------------------------------
    # RESULTADOS
    #------------------------------------------------------------
    df_resultados = pd.DataFrame(resultados)

    # ranking (más cercano a 2 es mejor)
    df_resultados["Distancia_2"] = abs(df_resultados["Durbin_Watson"] - 2)
    df_resultados["Ranking"] = df_resultados["Distancia_2"].rank()

    df_resultados = df_resultados.sort_values("Ranking")

    return df_resultados