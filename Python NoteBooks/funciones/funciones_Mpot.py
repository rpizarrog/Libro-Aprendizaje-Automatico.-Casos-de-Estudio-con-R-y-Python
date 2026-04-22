# FUNCIONES PARA MODELOS DE REGRESIÓN POTENCIAL, EXPONENCIAL LINEAL, POLINOMIAL y LOGARITMICA
# Archivo: funciones_Mpot.py
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

def f_construir_modelo_pot(datos, x, y):
    # Construye modelo de regresión POTENCIAL
    import numpy as np
    import statsmodels.formula.api as smf

    #--------------------------------------------------------
    # 🔥 VALIDACIONES
    #--------------------------------------------------------
    if (datos[y] <= 0).any():
        raise ValueError("La variable dependiente debe ser positiva")

    if (datos[x] <= 0).any():
        raise ValueError("La variable independiente debe ser positiva")

    #--------------------------------------------------------
    # 🔥 FORMULA (log-log)
    #--------------------------------------------------------
    formula = f"np.log({y}) ~ np.log({x})"

    modelo = smf.ols(formula=formula, data=datos).fit()

    #--------------------------------------------------------
    # 🔥 METADATA (muy buena práctica)
    #--------------------------------------------------------
    modelo.tipo = "potencial"
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

    n = len(modelos)

    #----------------------------------------------------
    # VALIDACIÓN
    #----------------------------------------------------
    if n == 0:
        raise ValueError("Debes proporcionar al menos un modelo")

    if titulos is None:
        titulos = [f"Modelo {i+1}" for i in range(n)]

    if len(titulos) != n:
        raise ValueError("La cantidad de títulos debe coincidir con los modelos")

    x_vals = datos[x]
    y_vals = datos[y]

    # correlación real
    r = np.corrcoef(x_vals, y_vals)[0,1]

    #----------------------------------------------------
    # MATRIZ DINÁMICA
    #----------------------------------------------------
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
    axes = np.array(axes).reshape(-1)

    #----------------------------------------------------
    # ITERAR MODELOS
    #----------------------------------------------------
    for i, modelo in enumerate(modelos):

        ax = axes[i]

        ax.scatter(x_vals, y_vals, alpha=0.6)

        #----------------------------------------------------
        # CURVA SUAVE
        #----------------------------------------------------
        x_sorted = np.linspace(x_vals.min(), x_vals.max(), 200)
        x_sorted_df = pd.DataFrame({x: x_sorted})

        #----------------------------------------------------
        # DETECCIÓN DE MODELO (ROBUSTA)
        #----------------------------------------------------
        tipo = "lineal"

        if hasattr(modelo, "model"):
            formula_str = str(modelo.model.formula)

            lhs = formula_str.split("~")[0]
            rhs = formula_str.split("~")[1]

            es_log_y = "np.log" in lhs
            es_log_x = "np.log" in rhs

            if es_log_y and es_log_x:
                tipo = "potencial"
            elif es_log_y:
                tipo = "exponencial"
            elif es_log_x:
                tipo = "logaritmico"
            elif "poly" in rhs or "**" in rhs:
                tipo = "polinomial"

        #----------------------------------------------------
        # PREDICCIONES CORRECTAS (CLAVE)
        #----------------------------------------------------
        try:
            X_pred = datos[[x]]
            X_line = x_sorted_df

            y_pred = modelo.predict(X_pred)
            y_line = modelo.predict(X_line)

            #------------------------------------------------
            # CORRECCIÓN DE ESCALA
            #------------------------------------------------
            if tipo in ["exponencial", "potencial"]:
                y_pred = np.exp(y_pred)
                y_line = np.exp(y_line)

            #------------------------------------------------
            # CONTROL NUMÉRICO
            #------------------------------------------------
            if np.any(~np.isfinite(y_pred)):
                r2 = np.nan
            else:
                r2 = r2_score(y_vals, y_pred)

        except Exception:
            y_line = np.full_like(x_sorted, np.nan)
            r2 = np.nan

        #----------------------------------------------------
        # GRAFICAR CURVA
        #----------------------------------------------------
        ax.plot(x_sorted, y_line, color='red', linewidth=2)

        #----------------------------------------------------
        # TÍTULO LIMPIO
        #----------------------------------------------------
        ax.set_title(
            f"{titulos[i]}\nTipo: {tipo} | r={r:.3f} | R²={r2:.3f}",
            fontsize=10
        )

        ax.set_xlabel(x)
        ax.set_ylabel(y)

    #----------------------------------------------------
    # ELIMINAR SUBPLOTS VACÍOS
    #----------------------------------------------------
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def f_ecuaciones_modelos(modelos, nombres):
    """
    Genera ecuaciones matemáticas de modelos:
    - Lineal
    - Logarítmico
    - Exponencial
    - Potencial
    - Polinomial (sklearn y statsmodels)

    Retorna:
    - DataFrame con ecuaciones
    """

    import numpy as np
    import pandas as pd

    resultados = []

    for i, modelo in enumerate(modelos):

        try:
            ecuacion = ""
            tipo = "lineal"

            #--------------------------------------------------------
            # 🔥 MODELOS SKLEARN (POLINOMIALES)
            #--------------------------------------------------------
            if hasattr(modelo, "named_steps"):

                poly = modelo.named_steps["poly"]
                lr = modelo.named_steps["lr"]

                coefs = lr.coef_
                intercepto = lr.intercept_

                nombres_vars = poly.get_feature_names_out([modelo.variable_x])

                tipo = "polinomial"

                ecuacion = f"ŷ = {round(intercepto,4)}"

                for j in range(len(coefs)):
                    coef = coefs[j]
                    if abs(coef) < 1e-6:
                        continue

                    signo = "+" if coef >= 0 else "-"
                    termino = nombres_vars[j]

                    ecuacion += f" {signo} {abs(round(coef,4))}*{termino}"

            #--------------------------------------------------------
            # 🔥 MODELOS STATSMODELS
            #--------------------------------------------------------
            elif hasattr(modelo, "params"):

                coefs = modelo.params.values
                nombres_vars = modelo.params.index.tolist()

                formula = str(modelo.model.formula)

                lhs, rhs = formula.split("~")

                es_log_y = "np.log" in lhs
                es_log_x = "np.log" in rhs

                #----------------------------------------------------
                # 🔥 POTENCIAL
                # log(y) = b0 + b1 log(x)
                # y = exp(b0) * x^b1
                #----------------------------------------------------
                if es_log_y and es_log_x:

                    tipo = "potencial"

                    b0 = coefs[0]
                    b1 = coefs[1]

                    a = np.exp(b0)

                    var = rhs.replace("np.log(", "").replace(")", "").strip()

                    ecuacion = f"ŷ = {round(a,4)} * {var}^{round(b1,4)}"

                #----------------------------------------------------
                # 🔥 EXPONENCIAL
                # log(y) = b0 + b1 x
                # y = exp(b0) * e^(b1 x)
                #----------------------------------------------------
                elif es_log_y:

                    tipo = "exponencial"

                    b0 = coefs[0]
                    b1 = coefs[1]

                    a = np.exp(b0)

                    var = rhs.strip()

                    ecuacion = f"ŷ = {round(a,4)} * e^({round(b1,4)}*{var})"

                #----------------------------------------------------
                # 🔥 LOGARÍTMICO
                # y = b0 + b1 log(x)
                #----------------------------------------------------
                elif es_log_x:

                    tipo = "logaritmico"

                    ecuacion = f"ŷ = {round(coefs[0],4)}"

                    for j in range(1, len(coefs)):
                        coef = coefs[j]
                        signo = "+" if coef >= 0 else "-"
                        termino = nombres_vars[j].replace("np.log(", "ln(")

                        ecuacion += f" {signo} {abs(round(coef,4))}*{termino}"

                #----------------------------------------------------
                # 🔥 LINEAL / POLINOMIAL (statsmodels)
                #----------------------------------------------------
                else:

                    tipo = "lineal"

                    ecuacion = f"ŷ = {round(coefs[0],4)}"

                    for j in range(1, len(coefs)):
                        coef = coefs[j]
                        signo = "+" if coef >= 0 else "-"
                        termino = nombres_vars[j]

                        ecuacion += f" {signo} {abs(round(coef,4))}*{termino}"

            else:
                ecuacion = "No reconocido"
                tipo = "No identificado"

            resultados.append({
                "Modelo": nombres[i],
                "Tipo": tipo,
                "Ecuacion": ecuacion
            })

        except Exception as e:
            resultados.append({
                "Modelo": nombres[i],
                "Tipo": "Error",
                "Ecuacion": "Error al generar"
            })

    return pd.DataFrame(resultados)

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

def f_evaluar_modelos_varios(
    modelos,
    datos_validacion,
    x,
    y,
    nombres
):
    """
    Evaluación de modelos:
    - R²
    - R² ajustado
    - MSE
    - RMSE
    - Número de parámetros

    Compatible con:
    - statsmodels
    - sklearn (Pipeline)
    """

    import numpy as np
    import pandas as pd

    resultados = []

    for i, modelo in enumerate(modelos):

        try:
            #--------------------------------------------------------
            # 🔥 DETECTAR TIPO
            #--------------------------------------------------------
            tipo = "lineal"

            if hasattr(modelo, "tipo"):
                tipo = modelo.tipo

            elif hasattr(modelo, "model"):
                formula = str(modelo.model.formula)
                lhs, rhs = formula.split("~")

                es_log_y = "np.log" in lhs
                es_log_x = "np.log" in rhs

                if es_log_y and es_log_x:
                    tipo = "potencial"
                elif es_log_y:
                    tipo = "exponencial"
                elif es_log_x:
                    tipo = "logaritmico"
                elif "poly" in rhs:
                    tipo = "polinomial"

            #--------------------------------------------------------
            # 🔥 PREDICCIONES (ESCALA ORIGINAL)
            #--------------------------------------------------------
            X_val = datos_validacion[[x]]
            y_real = datos_validacion[y].values

            if hasattr(modelo, "named_steps"):  # sklearn
                y_pred = modelo.predict(X_val)

                # número de parámetros
                p = len(modelo.named_steps["lr"].coef_) + 1

            else:  # statsmodels
                y_pred = modelo.predict(datos_validacion)

                # número de parámetros
                p = len(modelo.params)

            #--------------------------------------------------------
            # 🔥 CORRECCIÓN ESCALA (SOLO PARA PREDICCIÓN)
            #--------------------------------------------------------
            if tipo in ["exponencial", "potencial"]:
                y_pred = np.exp(y_pred)

            #--------------------------------------------------------
            # 🔥 CONTROL NUMÉRICO
            #--------------------------------------------------------
            y_pred = np.asarray(y_pred).flatten()
            y_real = np.asarray(y_real).flatten()

            mask = np.isfinite(y_pred) & np.isfinite(y_real)
            y_pred = y_pred[mask]
            y_real = y_real[mask]

            #--------------------------------------------------------
            # 🔥 MÉTRICAS
            #--------------------------------------------------------
            errores = y_real - y_pred

            mse = np.mean(errores**2)
            rmse = np.sqrt(mse)

            sst = np.sum((y_real - np.mean(y_real))**2)
            sse = np.sum((y_real - y_pred)**2)

            r2 = 1 - (sse / sst)

            n = len(y_real)
            r2_adj = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

            resultados.append({
                "R_square": round(r2, 4),
                "R_square_ajustado": round(r2_adj, 4),
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4),
                "Parametros (p)": p,
                "Modelo": nombres[i]
            })

        except Exception as e:
            resultados.append({
                "R_square": None,
                "R_square_ajustado": None,
                "MSE": None,
                "RMSE": None,
                "Parametros (p)": None,
                "Modelo": nombres[i]
            })

    df = pd.DataFrame(resultados)

    #--------------------------------------------------------
    # 🔥 ORDENAR (MEJOR = MENOR RMSE)
    #--------------------------------------------------------
    df = df.sort_values("RMSE")

    df.reset_index(drop=True, inplace=True)

    return df


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

def f_tukey_linealidad_modelos(modelos, datos, x, y, nombres=None):

    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    n = len(modelos)

    if nombres is None:
        nombres = [f"Modelo {i+1}" for i in range(n)]

    resultados = []

    for i, modelo in enumerate(modelos):

        try:
            #------------------------------------------
            # 🔥 PREDICCIONES DEL MODELO
            #------------------------------------------
            y_hat = modelo.predict(datos[[x]])

            #------------------------------------------
            # 🔥 DETECTAR TIPO
            #------------------------------------------
            tipo = "lineal"

            if hasattr(modelo, "model"):
                formula_str = str(modelo.model.formula)

                lhs = formula_str.split("~")[0]
                rhs = formula_str.split("~")[1]

                es_log_y = "np.log" in lhs
                es_log_x = "np.log" in rhs

                if es_log_y and es_log_x:
                    tipo = "potencial"
                elif es_log_y:
                    tipo = "exponencial"
                elif es_log_x:
                    tipo = "logaritmico"
                elif "poly" in rhs:
                    tipo = "polinomial"

            #------------------------------------------
            # 🔥 CORRECCIÓN DE ESCALA
            #------------------------------------------
            if tipo in ["exponencial", "potencial"]:
                y_hat = np.exp(y_hat)

            #------------------------------------------
            # 🔥 MODELO AUXILIAR (Tukey)
            #------------------------------------------
            df_aux = pd.DataFrame({
                "y": datos[y],
                "y_hat": y_hat
            })

            df_aux["y_hat2"] = df_aux["y_hat"]**2

            X = sm.add_constant(df_aux[["y_hat", "y_hat2"]])
            y_real = df_aux["y"]

            modelo_aux = sm.OLS(y_real, X).fit()

            p_value = modelo_aux.pvalues["y_hat2"]

            #------------------------------------------
            # 🔥 INTERPRETACIÓN
            #------------------------------------------
            if p_value > 0.05:
                conclusion = "✔ Linealidad"
            else:
                conclusion = "✖ No linealidad"

            resultados.append({
                "Modelo": nombres[i],
                "Tipo": tipo,
                "p_value": round(p_value, 4),
                "Resultado": conclusion
            })

        except Exception:
            resultados.append({
                "Modelo": nombres[i],
                "Tipo": "Error",
                "p_value": None,
                "Resultado": "Error en cálculo"
            })

    df_resultados = pd.DataFrame(resultados)

    # ranking (mejor = mayor p-value)
    df_resultados["Ranking"] = df_resultados["p_value"].rank(ascending=False)

    return df_resultados.sort_values("Ranking")

def f_tukey_linealidad_modelos_plot(modelos, datos, x, y, nombres=None):
    """
    Evalúa linealidad con:
      - Gráfica residuos vs ajustados + LOWESS
      - Prueba de Tukey (no aditividad)
    Muestra todo en una matriz de gráficos (3 columnas).
    Devuelve un DataFrame con p-values y conclusión.
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from statsmodels.nonparametric.smoothers_lowess import lowess

    n = len(modelos)

    if nombres is None:
        nombres = [f"Modelo {i+1}" for i in range(n)]

    if len(nombres) != n:
        raise ValueError("El número de nombres debe coincidir con los modelos")

    #----------------------------------------------------
    # LAYOUT
    #----------------------------------------------------
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
    axes = np.array(axes).reshape(-1)

    resultados = []

    for i, modelo in enumerate(modelos):

        ax = axes[i]

        try:
            #------------------------------------------
            # 🔥 DETECTAR TIPO
            #------------------------------------------
            tipo = "lineal"

            if hasattr(modelo, "model"):
                formula_str = str(modelo.model.formula)
                lhs = formula_str.split("~")[0]
                rhs = formula_str.split("~")[1]

                es_log_y = "np.log" in lhs
                es_log_x = "np.log" in rhs

                if es_log_y and es_log_x:
                    tipo = "potencial"
                elif es_log_y:
                    tipo = "exponencial"
                elif es_log_x:
                    tipo = "logaritmico"
                elif "poly" in rhs:
                    tipo = "polinomial"

            #------------------------------------------
            # 🔥 PREDICCIONES (correctas)
            #------------------------------------------
            X_pred = datos[[x]]
            y_hat = modelo.predict(X_pred)

            #------------------------------------------
            # 🔥 CORRECCIÓN ESCALA
            #------------------------------------------
            if tipo in ["exponencial", "potencial"]:
                y_hat = np.exp(y_hat)

            y_real = datos[y]

            #------------------------------------------
            # 🔥 RESIDUOS
            #------------------------------------------
            residuos = y_real - y_hat

            #------------------------------------------
            # 🔥 LOWESS (curvatura)
            #------------------------------------------
            lowess_fit = lowess(residuos, y_hat, frac=0.3)

            #------------------------------------------
            # 🔥 PRUEBA DE TUKEY
            #------------------------------------------
            df_aux = pd.DataFrame({
                "y": y_real,
                "y_hat": y_hat
            })

            df_aux["y_hat2"] = df_aux["y_hat"]**2

            X_tukey = sm.add_constant(df_aux[["y_hat", "y_hat2"]])
            modelo_aux = sm.OLS(df_aux["y"], X_tukey).fit()

            p_value = modelo_aux.pvalues["y_hat2"]

            if p_value > 0.05:
                conclusion = "✔ Lineal"
            else:
                conclusion = "✖ No lineal"

            #------------------------------------------
            # 🔥 GRÁFICA
            #------------------------------------------
            ax.scatter(y_hat, residuos, alpha=0.5)
            ax.plot(lowess_fit[:,0], lowess_fit[:,1], color="red", linewidth=2)
            ax.axhline(0, linestyle="--", color="black")

            ax.set_title(
                f"{nombres[i]}\nTipo: {tipo} | p={p_value:.4f} | {conclusion}",
                fontsize=10
            )

            ax.set_xlabel("Valores ajustados")
            ax.set_ylabel("Residuos")

            resultados.append({
                "Modelo": nombres[i],
                "Tipo": tipo,
                "p_value": round(p_value, 4),
                "Resultado": conclusion
            })

        except Exception:
            ax.set_title(f"{nombres[i]}\nError en modelo")
            resultados.append({
                "Modelo": nombres[i],
                "Tipo": "Error",
                "p_value": None,
                "Resultado": "Error"
            })

    #----------------------------------------------------
    # LIMPIAR EJES VACÍOS
    #----------------------------------------------------
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    #----------------------------------------------------
    # RESULTADOS TABLA
    #----------------------------------------------------
    df_resultados = pd.DataFrame(resultados)

    # ranking: mejor = mayor p-value
    df_resultados["Ranking"] = df_resultados["p_value"].rank(ascending=False)

    return df_resultados.sort_values("Ranking")

def f_matriz_verificar_homocedasticidad(modelos, datos, x, y, nombres_modelos=None):

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.nonparametric.smoothers_lowess import lowess

    n = len(modelos)

    if n == 0:
        raise ValueError("Debes proporcionar al menos un modelo")

    if nombres_modelos is None:
        nombres_modelos = [f"Modelo {i+1}" for i in range(n)]

    if len(nombres_modelos) != n:
        raise ValueError("Los nombres deben coincidir con el número de modelos")

    #--------------------------------------------------------
    # MATRIZ DINÁMICA
    #--------------------------------------------------------
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
    axes = np.array(axes).reshape(-1)

    y_real = datos[y].values

    for i, modelo in enumerate(modelos):

        ax = axes[i]

        try:
            #--------------------------------------------------------
            # 🔥 DETECCIÓN DE TIPO (ROBUSTA)
            #--------------------------------------------------------
            tipo = "lineal"

            if hasattr(modelo, "tipo"):
                tipo = modelo.tipo

            elif hasattr(modelo, "model"):  # statsmodels
                formula_str = str(modelo.model.formula)
                lhs, rhs = formula_str.split("~")

                es_log_y = "np.log" in lhs
                es_log_x = "np.log" in rhs

                if es_log_y and es_log_x:
                    tipo = "potencial"
                elif es_log_y:
                    tipo = "exponencial"
                elif es_log_x:
                    tipo = "logaritmico"
                elif "poly" in rhs:
                    tipo = "polinomial"

            elif hasattr(modelo, "named_steps"):
                tipo = "polinomial"

            #--------------------------------------------------------
            # 🔥 PREDICCIONES
            #--------------------------------------------------------
            if hasattr(modelo, "named_steps"):
                # sklearn pipeline
                X_poly = modelo.named_steps["poly"].transform(datos[[x]])
                y_pred = modelo.named_steps["lr"].predict(X_poly)

            else:
                X_pred = datos[[x]]
                y_pred = modelo.predict(X_pred)

                # corrección escala
                if tipo in ["exponencial", "potencial"]:
                    y_pred = np.exp(y_pred)

            #--------------------------------------------------------
            # 🔥 CONTROL NUMÉRICO
            #--------------------------------------------------------
            y_pred = np.array(y_pred).astype(float)

            mask = np.isfinite(y_pred) & np.isfinite(y_real)
            y_pred = y_pred[mask]
            residuos = y_real[mask] - y_pred

            #--------------------------------------------------------
            # 🔥 LOWESS
            #--------------------------------------------------------
            curva = lowess(residuos, y_pred, frac=0.3)

            #--------------------------------------------------------
            # 🔥 DESVIACIÓN
            #--------------------------------------------------------
            std = np.std(residuos)

            #--------------------------------------------------------
            # 🔥 GRÁFICA
            #--------------------------------------------------------
            ax.scatter(y_pred, residuos, alpha=0.5)
            ax.plot(curva[:,0], curva[:,1], linewidth=2, color="blue")

            ax.axhline(0, linestyle='--', color="black")
            ax.axhline(2*std, linestyle=':', color="gray")
            ax.axhline(-2*std, linestyle=':', color="gray")

            #--------------------------------------------------------
            # 🔥 INTERPRETACIÓN AUTOMÁTICA
            #--------------------------------------------------------
            pendiente = np.polyfit(y_pred, residuos, 1)[0]

            if abs(pendiente) < 0.05:
                interpretacion = "✔ Homo"
            else:
                interpretacion = "✖ Hetero"

            ax.set_title(f"{nombres_modelos[i]}\nTipo: {tipo} | {interpretacion}", fontsize=10)
            ax.set_xlabel("Valores ajustados")
            ax.set_ylabel("Residuos")

        except Exception as e:
            ax.set_title(f"{nombres_modelos[i]}\nError", fontsize=10)

    #--------------------------------------------------------
    # LIMPIAR EJES VACÍOS
    #--------------------------------------------------------
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def f_pruebas_homocedasticidad(modelos, nombres, datos, x, y):

    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white

    resultados = []

    for i, modelo in enumerate(modelos):

        try:
            #--------------------------------------------------------
            # 🔥 DETECTAR SI ES SKLEARN O STATSMODELS
            #--------------------------------------------------------
            es_sklearn = hasattr(modelo, "named_steps")

            #--------------------------------------------------------
            # 🔥 PREDICCIONES
            #--------------------------------------------------------
            if es_sklearn:
                X = datos[[x]]
                y_real = datos[y].values

                y_pred = modelo.predict(X)

            else:
                y_real = np.asarray(modelo.model.endog)
                y_pred = np.asarray(modelo.fittedvalues)

            #--------------------------------------------------------
            # 🔥 RESIDUOS
            #--------------------------------------------------------
            residuos = y_real - y_pred

            #--------------------------------------------------------
            # 🔥 MATRIZ X ROBUSTA (SOLO fitted values)
            #--------------------------------------------------------
            X_test = sm.add_constant(y_pred)

            #--------------------------------------------------------
            # 🔥 BREUSCH–PAGAN
            #--------------------------------------------------------
            bp = het_breuschpagan(residuos, X_test)
            bp_p_value = bp[1]

            #--------------------------------------------------------
            # 🔥 WHITE
            #--------------------------------------------------------
            try:
                white = het_white(residuos, X_test)
                white_p_value = white[1]
            except:
                white_p_value = np.nan

            #--------------------------------------------------------
            # 🔥 INTERPRETACIÓN
            #--------------------------------------------------------
            bp_resultado = "✔ Homo" if bp_p_value > 0.05 else "✖ Hetero"

            if np.isnan(white_p_value):
                white_resultado = "No aplica"
            else:
                white_resultado = "✔ Homo" if white_p_value > 0.05 else "✖ Hetero"

        except Exception as e:
            bp_p_value = np.nan
            white_p_value = np.nan
            bp_resultado = "Error"
            white_resultado = "Error"

        resultados.append({
            "Modelo": nombres[i],
            "BP_p_value": round(bp_p_value, 6) if not np.isnan(bp_p_value) else np.nan,
            "White_p_value": round(white_p_value, 6) if not np.isnan(white_p_value) else np.nan,
            "BP_resultado": bp_resultado,
            "White_resultado": white_resultado
        })

    return pd.DataFrame(resultados)

def f_matriz_normalidad_modelos(modelos, datos, x, y, nombres=None):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats

    n = len(modelos)

    if nombres is None:
        nombres = [f"Modelo {i+1}" for i in range(n)]

    if len(nombres) != n:
        raise ValueError("Los nombres deben coincidir con el número de modelos")

    #----------------------------------------------------
    # MATRIZ 3x3 DINÁMICA
    #----------------------------------------------------
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
    axes = np.array(axes).reshape(-1)

    resultados = []

    for i, modelo in enumerate(modelos):

        ax = axes[i]

        try:
            #--------------------------------------------------------
            # 🔥 DETECTAR TIPO
            #--------------------------------------------------------
            tipo = "lineal"

            if hasattr(modelo, "tipo"):
                tipo = modelo.tipo

            elif hasattr(modelo, "model"):
                formula = str(modelo.model.formula)
                lhs, rhs = formula.split("~")

                es_log_y = "np.log" in lhs
                es_log_x = "np.log" in rhs

                if es_log_y and es_log_x:
                    tipo = "potencial"
                elif es_log_y:
                    tipo = "exponencial"
                elif es_log_x:
                    tipo = "logaritmico"
                elif "poly" in rhs:
                    tipo = "polinomial"

            #--------------------------------------------------------
            # 🔥 PREDICCIONES
            #--------------------------------------------------------
            if hasattr(modelo, "named_steps"):  # sklearn
                X = datos[[x]]
                y_real = datos[y].values
                y_pred = modelo.predict(X)

            else:  # statsmodels
                y_real = np.asarray(modelo.model.endog)
                y_pred = np.asarray(modelo.fittedvalues)

            #--------------------------------------------------------
            # 🔥 CORRECCIÓN ESCALA (CLAVE)
            #--------------------------------------------------------
            if tipo in ["exponencial", "potencial"]:
                y_pred = np.exp(y_pred)

            #--------------------------------------------------------
            # 🔥 RESIDUOS
            #--------------------------------------------------------
            residuos = y_real - y_pred

            # limpieza
            residuos = residuos[np.isfinite(residuos)]

            # estandarizar (mejora gráfica)
            residuos_std = (residuos - np.mean(residuos)) / np.std(residuos)

            #--------------------------------------------------------
            # 🔥 SHAPIRO
            #--------------------------------------------------------
            if len(residuos_std) > 5000:
                residuos_test = np.random.choice(residuos_std, 5000, replace=False)
            else:
                residuos_test = residuos_std

            shapiro = stats.shapiro(residuos_test)
            p_value = shapiro.pvalue

            resultado = "✔ Normal" if p_value > 0.05 else "✖ No normal"

            #--------------------------------------------------------
            # 🔥 GRÁFICOS
            #--------------------------------------------------------
            sns.histplot(residuos_std, kde=True, ax=ax, color="gray")

            # Q-Q sobre el mismo eje (truco visual compacto)
            stats.probplot(residuos_std, dist="norm", plot=ax)

            ax.set_title(
                f"{nombres[i]}\nTipo: {tipo} | p={p_value:.4f} | {resultado}",
                fontsize=10
            )

            resultados.append({
                "Modelo": nombres[i],
                "Tipo": tipo,
                "p_value": round(p_value, 4),
                "Normalidad": resultado
            })

        except Exception:
            ax.set_title(f"{nombres[i]}\nError")

            resultados.append({
                "Modelo": nombres[i],
                "Tipo": "Error",
                "p_value": None,
                "Normalidad": "Error"
            })

    #----------------------------------------------------
    # LIMPIAR SUBPLOTS VACÍOS
    #----------------------------------------------------
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    #----------------------------------------------------
    # TABLA RESULTADOS
    #----------------------------------------------------
    df = pd.DataFrame(resultados)
    df["Ranking"] = df["p_value"].rank(ascending=False)

    return df.sort_values("Ranking")

def f_prueba_shapiro(modelos, nombres, datos, x, y):
    """
    Prueba de normalidad Shapiro-Wilk para múltiples modelos.

    Parámetros:
    - modelos: lista de modelos (statsmodels o sklearn)
    - nombres: lista de nombres de modelos
    - datos: DataFrame
    - x: variable independiente (str)
    - y: variable dependiente (str)

    Retorna:
    - DataFrame con estadístico W, p-value e interpretación
    """

    import numpy as np
    import pandas as pd
    from scipy.stats import shapiro

    resultados = []

    for i, modelo in enumerate(modelos):

        try:
            #--------------------------------------------------------
            # 🔥 DETECTAR TIPO DE MODELO
            #--------------------------------------------------------
            tipo = "lineal"

            if hasattr(modelo, "tipo"):
                tipo = modelo.tipo

            elif hasattr(modelo, "model"):
                formula = str(modelo.model.formula)
                lhs, rhs = formula.split("~")

                es_log_y = "np.log" in lhs
                es_log_x = "np.log" in rhs

                if es_log_y and es_log_x:
                    tipo = "potencial"
                elif es_log_y:
                    tipo = "exponencial"
                elif es_log_x:
                    tipo = "logaritmico"
                elif "poly" in rhs:
                    tipo = "polinomial"

            #--------------------------------------------------------
            # 🔥 PREDICCIONES
            #--------------------------------------------------------
            if hasattr(modelo, "named_steps"):  # sklearn
                X = datos[[x]]
                y_real = datos[y].values
                y_pred = modelo.predict(X)

            else:  # statsmodels
                y_real = np.asarray(modelo.model.endog)
                y_pred = np.asarray(modelo.fittedvalues)

            #--------------------------------------------------------
            # 🔥 CORRECCIÓN ESCALA
            #--------------------------------------------------------
            if tipo in ["exponencial", "potencial"]:
                y_pred = np.exp(y_pred)

            #--------------------------------------------------------
            # 🔥 RESIDUOS
            #--------------------------------------------------------
            residuos = y_real - y_pred
            residuos = residuos[np.isfinite(residuos)]

            #--------------------------------------------------------
            # 🔥 SHAPIRO-WILK
            #--------------------------------------------------------
            if len(residuos) > 5000:
                np.random.seed(123)
                residuos_test = np.random.choice(residuos, 5000, replace=False)
            else:
                residuos_test = residuos

            W, p_value = shapiro(residuos_test)

            interpretacion = "✔ Normal" if p_value > 0.05 else "✖ No normal"

            resultados.append({
                "Modelo": nombres[i],
                "Tipo": tipo,
                "Shapiro_W": round(W, 4),
                "p_value": round(p_value, 4),
                "Normalidad": interpretacion
            })

        except Exception:
            resultados.append({
                "Modelo": nombres[i],
                "Tipo": "Error",
                "Shapiro_W": None,
                "p_value": None,
                "Normalidad": "Error"
            })

    df = pd.DataFrame(resultados)

    # ranking: mejor = mayor p-value
    df["Ranking"] = df["p_value"].rank(ascending=False)

    return df.sort_values("Ranking")

def f_prueba_anderson(modelos, nombres, datos, x, y):
    """
    Prueba de normalidad Anderson-Darling para múltiples modelos.

    Retorna:
    - Estadístico A²
    - p-value
    - Interpretación
    """

    import numpy as np
    import pandas as pd
    from statsmodels.stats.diagnostic import normal_ad

    resultados = []

    for i, modelo in enumerate(modelos):

        try:
            #--------------------------------------------------------
            # 🔥 DETECTAR TIPO DE MODELO
            #--------------------------------------------------------
            tipo = "lineal"

            if hasattr(modelo, "tipo"):
                tipo = modelo.tipo

            elif hasattr(modelo, "model"):
                formula = str(modelo.model.formula)
                lhs, rhs = formula.split("~")

                es_log_y = "np.log" in lhs
                es_log_x = "np.log" in rhs

                if es_log_y and es_log_x:
                    tipo = "potencial"
                elif es_log_y:
                    tipo = "exponencial"
                elif es_log_x:
                    tipo = "logaritmico"
                elif "poly" in rhs:
                    tipo = "polinomial"

            #--------------------------------------------------------
            # 🔥 PREDICCIONES
            #--------------------------------------------------------
            if hasattr(modelo, "named_steps"):  # sklearn
                X = datos[[x]]
                y_real = datos[y].values
                y_pred = modelo.predict(X)

            else:  # statsmodels
                y_real = np.asarray(modelo.model.endog)
                y_pred = np.asarray(modelo.fittedvalues)

            #--------------------------------------------------------
            # 🔥 CORRECCIÓN ESCALA
            #--------------------------------------------------------
            if tipo in ["exponencial", "potencial"]:
                y_pred = np.exp(y_pred)

            #--------------------------------------------------------
            # 🔥 RESIDUOS
            #--------------------------------------------------------
            residuos = y_real - y_pred
            residuos = residuos[np.isfinite(residuos)]

            #--------------------------------------------------------
            # 🔥 ANDERSON–DARLING (con p-value)
            #--------------------------------------------------------
            stat, p_value = normal_ad(residuos)

            interpretacion = "✔ Normal" if p_value > 0.05 else "✖ No normal"

            resultados.append({
                "Modelo": nombres[i],
                "Tipo": tipo,
                "AD_stat": round(stat, 4),
                "p_value": round(p_value, 4),
                "Normalidad": interpretacion
            })

        except Exception:
            resultados.append({
                "Modelo": nombres[i],
                "Tipo": "Error",
                "AD_stat": None,
                "p_value": None,
                "Normalidad": "Error"
            })

    df = pd.DataFrame(resultados)

    # ranking: mejor = mayor p-value
    df["Ranking"] = df["p_value"].rank(ascending=False)

    return df.sort_values("Ranking")  

def f_prueba_kolmogorov(modelos, nombres, datos, x, y):
    """
    Prueba de normalidad Kolmogorov-Smirnov para múltiples modelos.

    Retorna:
    - Estadístico D
    - p-value
    - Interpretación
    """

    import numpy as np
    import pandas as pd
    from scipy.stats import kstest

    resultados = []

    for i, modelo in enumerate(modelos):

        try:
            #--------------------------------------------------------
            # 🔥 DETECTAR TIPO DE MODELO
            #--------------------------------------------------------
            tipo = "lineal"

            if hasattr(modelo, "tipo"):
                tipo = modelo.tipo

            elif hasattr(modelo, "model"):
                formula = str(modelo.model.formula)
                lhs, rhs = formula.split("~")

                es_log_y = "np.log" in lhs
                es_log_x = "np.log" in rhs

                if es_log_y and es_log_x:
                    tipo = "potencial"
                elif es_log_y:
                    tipo = "exponencial"
                elif es_log_x:
                    tipo = "logaritmico"
                elif "poly" in rhs:
                    tipo = "polinomial"

            #--------------------------------------------------------
            # 🔥 PREDICCIONES
            #--------------------------------------------------------
            if hasattr(modelo, "named_steps"):  # sklearn
                X = datos[[x]]
                y_real = datos[y].values
                y_pred = modelo.predict(X)

            else:  # statsmodels
                y_real = np.asarray(modelo.model.endog)
                y_pred = np.asarray(modelo.fittedvalues)

            #--------------------------------------------------------
            # 🔥 CORRECCIÓN ESCALA
            #--------------------------------------------------------
            if tipo in ["exponencial", "potencial"]:
                y_pred = np.exp(y_pred)

            #--------------------------------------------------------
            # 🔥 RESIDUOS
            #--------------------------------------------------------
            residuos = y_real - y_pred
            residuos = residuos[np.isfinite(residuos)]

            #--------------------------------------------------------
            # 🔥 ESTANDARIZAR (CLAVE para K-S)
            #--------------------------------------------------------
            residuos_std = (residuos - np.mean(residuos)) / np.std(residuos)

            #--------------------------------------------------------
            # 🔥 KOLMOGOROV-SMIRNOV
            #--------------------------------------------------------
            stat, p_value = kstest(residuos_std, 'norm')

            interpretacion = "✔ Normal" if p_value > 0.05 else "✖ No normal"

            resultados.append({
                "Modelo": nombres[i],
                "Tipo": tipo,
                "KS_stat": round(stat, 4),
                "p_value": round(p_value, 4),
                "Normalidad": interpretacion
            })

        except Exception:
            resultados.append({
                "Modelo": nombres[i],
                "Tipo": "Error",
                "KS_stat": None,
                "p_value": None,
                "Normalidad": "Error"
            })

    df = pd.DataFrame(resultados)

    # ranking: mejor = mayor p-value
    df["Ranking"] = df["p_value"].rank(ascending=False)

    return df.sort_values("Ranking")      

def f_verificar_independencia_residuos(
    modelos,
    datos,
    x,
    y,
    nombres_modelos=None,
    graficar=True
):
    """
    Evaluación de independencia de residuos (Durbin-Watson)
    Compatible con statsmodels y sklearn.
    Consistente con R (usa residuos del modelo).
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.stats.diagnostic import acorr_ljungbox

    n = len(modelos)

    if nombres_modelos is None:
        nombres_modelos = [f"Modelo {i+1}" for i in range(n)]

    resultados = []

    #------------------------------------------------------------
    # MATRIZ 3x3
    #------------------------------------------------------------
    if graficar:
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
        axes = np.array(axes).reshape(-1)

    for i, modelo in enumerate(modelos):

        try:
            #--------------------------------------------------------
            # 🔥 DETECTAR TIPO
            #--------------------------------------------------------
            tipo = "lineal"

            if hasattr(modelo, "tipo"):
                tipo = modelo.tipo

            elif hasattr(modelo, "model"):
                formula = str(modelo.model.formula)
                lhs, rhs = formula.split("~")

                es_log_y = "np.log" in lhs
                es_log_x = "np.log" in rhs

                if es_log_y and es_log_x:
                    tipo = "potencial"
                elif es_log_y:
                    tipo = "exponencial"
                elif es_log_x:
                    tipo = "logaritmico"
                elif "poly" in rhs:
                    tipo = "polinomial"

            #--------------------------------------------------------
            # 🔥 RESIDUOS CORRECTOS (CLAVE 🔥)
            #--------------------------------------------------------
            if hasattr(modelo, "named_steps"):  # sklearn
                X = datos[[x]]
                y_real = datos[y].values
                y_pred = modelo.predict(X)
                residuos = y_real - y_pred

            else:  # statsmodels
                residuos = np.asarray(modelo.resid)

            residuos = residuos[np.isfinite(residuos)]

            #--------------------------------------------------------
            # 🔥 DURBIN-WATSON
            #--------------------------------------------------------
            dw = durbin_watson(residuos)

            #--------------------------------------------------------
            # 🔥 p-value (aproximación tipo R)
            #--------------------------------------------------------
            lb = acorr_ljungbox(residuos, lags=[1], return_df=True)
            p_value = lb["lb_pvalue"].values[0]

            #--------------------------------------------------------
            # 🔥 INTERPRETACIÓN
            #--------------------------------------------------------
            if 1.5 <= dw <= 2.5:
                interpretacion = "✔ Independencia"
            elif dw < 1.5:
                interpretacion = "✖ Autocorrelación positiva"
            else:
                interpretacion = "✖ Autocorrelación negativa"

            decision = "No se rechaza H0" if p_value > 0.05 else "Se rechaza H0"

            resultados.append({
                "Modelo": nombres_modelos[i],
                "Tipo": tipo,
                "DW": round(dw, 4),
                "p_value": round(p_value, 4),
                "Interpretacion": interpretacion,
                "Decision": decision
            })

            #--------------------------------------------------------
            # 🔥 GRÁFICA
            #--------------------------------------------------------
            if graficar:
                ax = axes[i]

                ax.plot(residuos, marker='o', linestyle='-', alpha=0.6)
                ax.axhline(0, linestyle='--', color='black')

                ax.set_title(
                    f"{nombres_modelos[i]}\nDW={dw:.3f} | p={p_value:.3f}\n{interpretacion}",
                    fontsize=10
                )

                ax.set_xlabel("Orden")
                ax.set_ylabel("Residuo")

        except Exception as e:

            resultados.append({
                "Modelo": nombres_modelos[i],
                "Tipo": "Error",
                "DW": None,
                "p_value": None,
                "Interpretacion": "Error",
                "Decision": "Error"
            })

            if graficar:
                axes[i].set_title(f"{nombres_modelos[i]}\nError", fontsize=10)

    #------------------------------------------------------------
    # LIMPIAR EJES VACÍOS
    #------------------------------------------------------------
    if graficar:
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    #------------------------------------------------------------
    # TABLA RESULTADOS
    #------------------------------------------------------------
    df = pd.DataFrame(resultados)

    # ranking: más cercano a 2 es mejor
    df["Distancia_2"] = abs(df["DW"] - 2)
    df["Ranking"] = df["Distancia_2"].rank()

    return df.sort_values("Ranking")