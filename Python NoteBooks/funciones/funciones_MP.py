# FUNCIONES PARA MODELOS DE REGRESIÓN LINEAL Y POLINOMIAL
# Archivo: funciones_MP.py
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

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.pipeline import Pipeline
from statsmodels.nonparametric.smoothers_lowess import lowess


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
    
    # metadatos útiles
    modelo.grado = grado
    modelo.variable_x = variable_independiente
    
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
        
        ax.scatter(x_vals, y_vals, alpha=0.5)
        
        x_sorted = np.sort(x_vals.values)
        x_sorted_df = pd.DataFrame(x_sorted, columns=[x])
        
        y_pred = modelo.predict(datos[[x]])
        y_line = modelo.predict(x_sorted_df)
        
        r2 = r2_score(y_vals, y_pred)
        
        ax.plot(x_sorted, y_line, color='red', linewidth=2)
        
        ax.set_title(f"{titulos[i]}\nr={r:.3f} ; R²={r2:.3f}")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    
    plt.tight_layout()
    plt.show() 

def f_ecuaciones_modelos(modelos, nombres_modelos=None):
    #------------------------------------------------------------
    # f_ecuaciones_modelos()
    #
    # Objetivo:
    #   Mostrar ecuaciones matemáticas y coeficientes
    #   de múltiples modelos (lineal y polinomiales)
    #
    # Argumentos:
    #   modelos          : lista de modelos
    #   nombres_modelos  : nombres de los modelos
    #
    #------------------------------------------------------------
    
    if nombres_modelos is None:
        nombres_modelos = [f"Modelo {i+1}" for i in range(len(modelos))]
    
    for modelo, nombre in zip(modelos, nombres_modelos):
        
        print("\n" + "="*60)
        print(f" {nombre}")
        print("="*60)
        
        # extraer pipeline
        poly = modelo.named_steps["poly"]
        lr = modelo.named_steps["lr"]
        
        nombres_vars = poly.get_feature_names_out([modelo.variable_x])
        coeficientes = lr.coef_
        intercepto = lr.intercept_
        
        #--------------------------------------------------------
        # Mostrar coeficientes
        #--------------------------------------------------------
        print("\nCoeficientes del modelo:")
        print(f"Intercepto: {round(intercepto,4)}")
        
        for nombre_var, coef in zip(nombres_vars, coeficientes):
            print(f"{nombre_var}: {round(coef,4)}")
        
        #--------------------------------------------------------
        # Construir ecuación
        #--------------------------------------------------------
        ecuacion = f"ŷ = {round(intercepto,4)}"
        
        for c, nombre_var in zip(coeficientes, nombres_vars):
            signo = "+" if c >= 0 else "-"
            ecuacion += f" {signo} {abs(round(c,4))}·{nombre_var}"
        
        print("\nEcuación del modelo:")
        print(ecuacion)    

def f_evaluar_modelo(modelo, datos_validacion, variable_dependiente, variable_independiente):
        
    #------------------------------------------------------------
    # 1. Datos
    #------------------------------------------------------------
    X = datos_validacion[[variable_independiente]]
    y_real = datos_validacion[variable_dependiente]
    
    #------------------------------------------------------------
    # 2. Predicción (Pipeline ya maneja polinomio)
    #------------------------------------------------------------
    pred = modelo.predict(X)
    
    #------------------------------------------------------------
    # 3. Métricas
    #------------------------------------------------------------
    mse = mean_squared_error(y_real, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_real, pred)
    
    #------------------------------------------------------------
    # 4. Número de parámetros (p)
    #------------------------------------------------------------
    if hasattr(modelo, "named_steps"):
        poly = modelo.named_steps["poly"]
        p = len(poly.get_feature_names_out())
    else:
        p = 1
    
    n = len(y_real)
    
    #------------------------------------------------------------
    # 5. R² ajustado
    #------------------------------------------------------------
    r2_adj = 1 - (1 - r2)*(n - 1)/(n - p - 1)
    
    #------------------------------------------------------------
    # 6. Resultado
    #------------------------------------------------------------
    resultado = pd.DataFrame({
        "R_square": [round(r2,4)],
        "R_square_ajustado": [round(r2_adj,4)],
        "MSE": [round(mse,4)],
        "RMSE": [round(rmse,4)],
        "Parametros (p)": [p]
    })
    
    return resultado

# función que evalúa varios modelos al mismo tiempo jerarquiza ordenando con RMSE 
# y devuelve una tabla 
# Recibe el modelo, los datos de validación, 
# las variables independiente y dependiente y el nombre del modelo
# Manda llamar f_evaluar_modelo() que evalua un modelo a la vez
def f_evaluar_modelos_varios(modelos, datos_validacion, y, x, nombres_modelos=None):
       
    if nombres_modelos is None:
        nombres_modelos = [f"Modelo {i+1}" for i in range(len(modelos))]
    
    resultados = []
    
    for modelo, nombre in zip(modelos, nombres_modelos):
        
        res = f_evaluar_modelo(modelo, datos_validacion, y, x)
        res["Modelo"] = nombre
        
        resultados.append(res)
    
    df_final = pd.concat(resultados, ignore_index=True)
    
    # ordenar por mejor RMSE
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
    #------------------------------------------------------------
    # Visualización 2x2 de residuos con curva suavizada (LOWESS)
    #------------------------------------------------------------
        
    if len(modelos) != 4:
        raise ValueError("Debes proporcionar exactamente 4 modelos")
    
    if nombres_modelos is None:
        nombres_modelos = [f"Modelo {i+1}" for i in range(4)]
    
    fig, axes = plt.subplots(2, 2, figsize=(12,10))
    axes = axes.flatten()
    
    for i, modelo in enumerate(modelos):
        
        ax = axes[i]
        
        # valores reales
        y_real = datos[y]
        
        # predicción
        if hasattr(modelo, "poly"):
            X_poly = modelo.poly.transform(datos[[x]])
            y_pred = modelo.predict(X_poly)
        else:
            y_pred = modelo.predict(datos[[x]])
        
        # residuos
        residuos = y_real - y_pred
        
        # scatter
        ax.scatter(y_pred, residuos, alpha=0.5)
        
        # línea horizontal en cero
        ax.axhline(y=0, linestyle='--')
        
        #--------------------------------------------------------
        # 🔥 CURVA SUAVIZADA (LOWESS)
        #--------------------------------------------------------
        curva = lowess(residuos, y_pred, frac=0.3)
        
        ax.plot(curva[:,0], curva[:,1],
                linewidth=2,
                alpha=0.4,
                linestyle='-')
        
        #--------------------------------------------------------
        # 🔥 BANDA VISUAL (opcional tipo nube)
        #--------------------------------------------------------
        std = np.std(residuos)
        
        ax.fill_between(
            np.sort(y_pred),
            -2*std,
            2*std,
            alpha=0.05
        )
        
        # títulos
        ax.set_title(nombres_modelos[i])
        ax.set_xlabel("Valores ajustados")
        ax.set_ylabel("Residuos")
    
    plt.tight_layout()
    plt.show()

def f_matriz_normalidad_modelos(modelos, datos, x, y, nombres_modelos=None):
    #------------------------------------------------------------
    # f_matriz_normalidad_modelos()
    #
    # Objetivo:
    #   Evaluar normalidad de residuos en 4 modelos:
    #   - Histograma + densidad
    #   - Q-Q plot
    #   - Shapiro-Wilk
    #   - Ranking automático
    #
    #------------------------------------------------------------
    
    
    if len(modelos) != 4:
        raise ValueError("Debes proporcionar exactamente 4 modelos")
    
    if nombres_modelos is None:
        nombres_modelos = [f"Modelo {i+1}" for i in range(4)]
    
    resultados = []
    
    fig, axes = plt.subplots(4, 2, figsize=(12,16))
    
    for i, modelo in enumerate(modelos):
        
        #--------------------------------------------------------
        # Predicciones
        #--------------------------------------------------------
        if hasattr(modelo, "poly"):
            X_poly = modelo.poly.transform(datos[[x]])
            y_pred = modelo.predict(X_poly)
        else:
            y_pred = modelo.predict(datos[[x]])
        
        y_real = datos[y]
        residuos = y_real - y_pred
        
        #--------------------------------------------------------
        # Shapiro-Wilk
        #--------------------------------------------------------
        W, p_value = stats.shapiro(residuos)
        
        if p_value > 0.05:
            interpretacion = "Normal"
        else:
            interpretacion = "No normal"
        
        # guardar resultados
        resultados.append({
            "Modelo": nombres_modelos[i],
            "W": W,
            "p_value": p_value,
            "Normalidad": interpretacion
        })
        
        #--------------------------------------------------------
        # Gráficos
        #--------------------------------------------------------
        
        # Histograma
        sns.histplot(residuos, kde=True, ax=axes[i,0])
        axes[i,0].set_title(
            f"{nombres_modelos[i]}\nHistograma\n"
            f"W={W:.3f} | p={p_value:.3f} | {interpretacion}"
        )
        
        # Q-Q plot
        stats.probplot(residuos, dist="norm", plot=axes[i,1])
        axes[i,1].set_title(
            f"{nombres_modelos[i]}\nQ-Q Plot\n"
            f"W={W:.3f} | p={p_value:.3f} | {interpretacion}"
        )
    
    plt.tight_layout()
    plt.show()
    
    #------------------------------------------------------------
    # DataFrame de resultados
    #------------------------------------------------------------
    df_resultados = pd.DataFrame(resultados)
    
    #------------------------------------------------------------
    # Ranking (mejor normalidad = mayor p-value)
    #------------------------------------------------------------
    df_resultados["Ranking"] = df_resultados["p_value"].rank(ascending=False)
    
    df_resultados = df_resultados.sort_values(by="Ranking")
    
    return df_resultados


def f_verificar_independencia_residuos(modelos, datos, x, y, nombres_modelos=None, graficar=True):
    #------------------------------------------------------------
    # f_verificar_independencia_residuos()
    #
    # Objetivo:
    #   Evaluar independencia de residuos mediante:
    #   - Prueba Durbin-Watson
    #   - Interpretación automática
    #   - (Opcional) gráfico de residuos vs orden
    #
    # Argumentos:
    #   modelos         : lista de modelos (4)
    #   datos           : DataFrame
    #   x               : variable independiente
    #   y               : variable dependiente
    #   nombres_modelos : nombres de los modelos
    #   graficar        : True/False
    #
    # Retorna:
    #   DataFrame con resultados y ranking
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
        # Predicción
        #--------------------------------------------------------
        if hasattr(modelo, "poly"):
            X_poly = modelo.poly.transform(datos[[x]])
            y_pred = modelo.predict(X_poly)
        else:
            y_pred = modelo.predict(datos[[x]])
        
        y_real = datos[y]
        residuos = y_real - y_pred
        
        #--------------------------------------------------------
        # Durbin-Watson
        #--------------------------------------------------------
        dw = durbin_watson(residuos)
        
        # interpretación
        if 1.5 <= dw <= 2.5:
            interpretacion = "Independencia (sin autocorrelación)"
        elif dw < 1.5:
            interpretacion = "Autocorrelación positiva"
        else:
            interpretacion = "Autocorrelación negativa"
        
        resultados.append({
            "Modelo": nombres_modelos[i],
            "Durbin_Watson": dw,
            "Interpretacion": interpretacion
        })
        
        #--------------------------------------------------------
        # Gráfico (opcional)
        #--------------------------------------------------------
        if graficar:
            ax = axes[i]
            
            ax.plot(residuos, marker='o', alpha=0.6)
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
    # DataFrame resultados
    #------------------------------------------------------------
    df_resultados = pd.DataFrame(resultados)
    
    #------------------------------------------------------------
    # Ranking (más cercano a 2 es mejor)
    #------------------------------------------------------------
    df_resultados["Distancia_2"] = abs(df_resultados["Durbin_Watson"] - 2)
    df_resultados["Ranking"] = df_resultados["Distancia_2"].rank()
    
    df_resultados = df_resultados.sort_values("Ranking")
    
    return df_resultados  
