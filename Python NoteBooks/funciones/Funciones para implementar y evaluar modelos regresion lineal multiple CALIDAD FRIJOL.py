# Funciones para implementar y evaluar modelos de regresión múltiple en Python
# Rubén Pizarro Gurrola
# Mayo 2026

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from scipy.stats import shapiro
from scipy.stats import kstest
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import linear_reset
import statsmodels.api as sm

#========================================================
# CARGAR DATOS
#========================================================

def f_cargar_datos(ruta_archivo):

    datos = pd.read_csv(ruta_archivo)

    return datos

#========================================================
# VISUALIZAR HEAD Y TAIL
#========================================================

def f_visualizar_head_tail_reducido(
        datos,
        n = 6
):

    #----------------------------------------------------
    # Total columnas
    #----------------------------------------------------

    total_columnas = datos.shape[1]

    #----------------------------------------------------
    # Primeras 4 columnas
    #----------------------------------------------------

    idx_prim = list(
        range(
            min(4, total_columnas)
        )
    )

    #----------------------------------------------------
    # Últimas 4 columnas
    #----------------------------------------------------

    idx_ult = list(
        range(
            max(total_columnas - 4, 0),
            total_columnas
        )
    )

    #----------------------------------------------------
    # Evitar duplicados
    #----------------------------------------------------

    idx_ult = [
        i for i in idx_ult
        if i not in idx_prim
    ]

    #----------------------------------------------------
    # Subconjuntos
    #----------------------------------------------------

    datos_prim = datos.iloc[:, idx_prim]

    datos_ult = datos.iloc[:, idx_ult]

    #----------------------------------------------------
    # HEAD
    #----------------------------------------------------

    head_prim = (
        datos_prim
        .head(n)
        .astype(str)
        .reset_index(drop = True)
    )

    head_ult = (
        datos_ult
        .head(n)
        .astype(str)
        .reset_index(drop = True)
    )

    #----------------------------------------------------
    # TAIL
    #----------------------------------------------------

    tail_prim = (
        datos_prim
        .tail(n)
        .astype(str)
        .reset_index(drop = True)
    )

    tail_ult = (
        datos_ult
        .tail(n)
        .astype(str)
        .reset_index(drop = True)
    )

    #----------------------------------------------------
    # Separadores
    #----------------------------------------------------

    sep_head = pd.DataFrame({
        "...": ["..."] * n
    })

    sep_tail = pd.DataFrame({
        "...": ["..."] * n
    })

    #----------------------------------------------------
    # Combinar HEAD
    #----------------------------------------------------

    head_comb = pd.concat(

        [
            head_prim,
            sep_head,
            head_ult
        ],

        axis = 1
    )

    #----------------------------------------------------
    # Combinar TAIL
    #----------------------------------------------------

    tail_comb = pd.concat(

        [
            tail_prim,
            sep_tail,
            tail_ult
        ],

        axis = 1
    )

    #----------------------------------------------------
    # Fila separadora
    #----------------------------------------------------

    fila_sep = pd.DataFrame(

        [["..."] * head_comb.shape[1]],

        columns = head_comb.columns
    )

    #----------------------------------------------------
    # Tabla final
    #----------------------------------------------------

    tabla = pd.concat(

        [
            head_comb,
            fila_sep,
            tail_comb
        ],

        ignore_index = True
    )

    return tabla

#========================================================
# DESCRIBIR DATOS
#========================================================

def f_describir_datos(datos):

    describe = datos.describe(include = 'all')

    structure = datos.dtypes

    return {
        "describe": describe,
        "structure": structure
    }

def f_convertir_dummis(datos, variable_dependiente):
    
    #----------------------------------------------------------
    # f_convertir_dummis()
    #
    # Objetivo:
    #   Convertir variables categóricas y booleanas
    #   a variables dummy, manteniendo la variable
    #   dependiente al final del DataFrame.
    #
    # Argumentos:
    #   datos                : DataFrame de pandas
    #   variable_dependiente : nombre de la variable objetivo
    #
    # Valor de retorno:
    #   DataFrame transformado
    #
    #----------------------------------------------------------
    
    import pandas as pd
    
    
    #----------------------------------------------------------
    # Copiar datos
    #----------------------------------------------------------
    
    datos_dummis = datos.copy()
    
    
    #----------------------------------------------------------
    # Guardar variable dependiente
    #----------------------------------------------------------
    
    y = datos_dummis[variable_dependiente]
    
    
    #----------------------------------------------------------
    # Eliminar temporalmente variable dependiente
    #----------------------------------------------------------
    
    datos_dummis = datos_dummis.drop(
        columns = [variable_dependiente]
    )
    
    
    #----------------------------------------------------------
    # Detectar variables categóricas y booleanas
    #----------------------------------------------------------
    
    variables_convertir = datos_dummis.select_dtypes(
        include = ["object", "bool"]
    ).columns
    
    
    #----------------------------------------------------------
    # Convertir a variables dummy
    #----------------------------------------------------------
    
    datos_dummis = pd.get_dummies(
        datos_dummis,
        columns = variables_convertir,
        drop_first = True,
        dtype = int
    )
    
    
    #----------------------------------------------------------
    # Agregar variable dependiente al final
    #----------------------------------------------------------
    
    datos_dummis[variable_dependiente] = y
    
    
    #----------------------------------------------------------
    # Devolver resultado
    #----------------------------------------------------------
    
    return datos_dummis

#========================================================
# PARTICIONAR DATOS
#========================================================

def f_particionar_datos(datos,
                         proporcion_entrenamiento = 0.7):

    datos_entrenamiento, datos_validacion = train_test_split(
        datos,
        train_size = proporcion_entrenamiento,
        random_state = 2026
    )

    return {
        "datos_entrenamiento": datos_entrenamiento,
        "datos_validacion": datos_validacion
    }

#========================================================
# CONVERTIR FACTOR
#========================================================

def f_convertir_factor(datos):

    datos_mod = datos.copy()

    for col in datos_mod.columns:

        if datos_mod[col].dtype == 'object':

            datos_mod[col] = datos_mod[col].astype('category')

        if datos_mod[col].dtype == 'bool':

            datos_mod[col] = datos_mod[col].astype(int)

    return datos_mod

#========================================================
# REDONDEAR VARIABLES NUMÉRICAS
#========================================================

def f_redondear_numericas(datos,
                          decimales = 2):

    datos_out = datos.copy()

    columnas_num = datos_out.select_dtypes(include = np.number).columns

    datos_out[columnas_num] = datos_out[columnas_num].round(decimales)

    return datos_out

#========================================================
# MODELO REGRESIÓN LINEAL MÚLTIPLE
#========================================================

def f_construir_modelo_RLM(datos,
                           variable_dependiente,
                           ver_resumen = True):

    X = datos.drop(columns = [variable_dependiente])

    y = datos[variable_dependiente]

    X = pd.get_dummies(X,
                       drop_first = True)

    modelo = LinearRegression()

    modelo.fit(X, y)

    if ver_resumen:

        X_sm = sm.add_constant(X)

        modelo_sm = sm.OLS(y, X_sm).fit()

        print(modelo_sm.summary())

    return modelo

#========================================================
# MODELO REGRESIÓN LINEAL MÚLTIPLE
# CON STATSMODELS
#========================================================

def f_construir_modelo_RLM_statsmodels(

        datos,

        variable_dependiente,

        ver_resumen = True
):

    #----------------------------------------------------
    # Variables independientes
    #----------------------------------------------------

    X = datos.drop(
        columns = [variable_dependiente]
    )

    #----------------------------------------------------
    # Variable dependiente
    #----------------------------------------------------

    y = datos[variable_dependiente]

    #----------------------------------------------------
    # Variables dummy
    #----------------------------------------------------

    X = pd.get_dummies(

        X,

        drop_first = True
    )

    #----------------------------------------------------
    # Constante
    #----------------------------------------------------

    X = sm.add_constant(X)

    #----------------------------------------------------
    # Modelo OLS
    #----------------------------------------------------

    modelo = sm.OLS(

        y,

        X

    ).fit()

    #----------------------------------------------------
    # Resumen
    #----------------------------------------------------

    if ver_resumen:

        print(modelo.summary())

    return modelo

#========================================================
# MODELO POLINOMIAL MÚLTIPLE
#========================================================

def f_multiple_polinomial(datos,
                          variable_dependiente,
                          orden = 2,
                          ver_resumen = True):

    X = datos.drop(columns = [variable_dependiente])

    y = datos[variable_dependiente]

    X = pd.get_dummies(X,
                       drop_first = True)

    poly = PolynomialFeatures(
        degree = orden,
        include_bias = False
    )

    X_poly = poly.fit_transform(X)

    modelo = LinearRegression()

    modelo.fit(X_poly, y)

    if ver_resumen:

        print("\n============================")
        print(f"Modelo Polinomial Orden {orden}")
        print("============================")

        print("Número de términos:", X_poly.shape[1])

    return {
        "modelo": modelo,
        "poly": poly
    }

#========================================================
# ESTANDARIZAR Y ESCALAR
#========================================================

def f_estandarizar_escalar(datos,
                           decimales = 4):

    datos_est = datos.copy()
    datos_esc = datos.copy()

    columnas_num = datos.select_dtypes(include = np.number).columns

    scaler_est = StandardScaler()

    scaler_minmax = MinMaxScaler()

    datos_est[columnas_num] = np.round(
        scaler_est.fit_transform(datos[columnas_num]),
        decimales
    )

    datos_esc[columnas_num] = np.round(
        scaler_minmax.fit_transform(datos[columnas_num]),
        decimales
    )

    return {
        "datos_estandarizados": datos_est,
        "datos_escalados": datos_esc
    }

#========================================================
# MODELO LASSO
#========================================================

def f_construir_modelo_lasso(datos,
                             variable_dependiente,
                             ver_resumen = True):

    X = datos.drop(columns = [variable_dependiente])

    y = datos[variable_dependiente]

    X = pd.get_dummies(X,
                       drop_first = True)

    modelo = LassoCV(
        cv = 10,
        random_state = 2026
    )

    modelo.fit(X, y)

    if ver_resumen:

        print("\n============================")
        print("Modelo LASSO")
        print("============================")

        print("Lambda óptimo:", modelo.alpha_)

        print("Coeficientes:")
        print(modelo.coef_)

    return modelo

#========================================================
# MODELO RIDGE
#========================================================

def f_construir_modelo_ridge(datos,
                             variable_dependiente,
                             ver_resumen = True):

    X = datos.drop(columns = [variable_dependiente])

    y = datos[variable_dependiente]

    X = pd.get_dummies(X,
                       drop_first = True)

    alphas = np.logspace(-4, 4, 100)

    modelo = RidgeCV(
        alphas = alphas,
        cv = 10
    )

    modelo.fit(X, y)

    if ver_resumen:

        print("\n============================")
        print("Modelo RIDGE")
        print("============================")

        print("Lambda óptimo:", modelo.alpha_)

        print("Coeficientes:")
        print(modelo.coef_)

    return modelo

#========================================================
# MULTICOLINEALIDAD VIF
#========================================================

def f_multicolinealidad(datos,
                        variable_dependiente):

    X = datos.drop(columns = [variable_dependiente])

    X = pd.get_dummies(X,
                       drop_first = True)

    X = sm.add_constant(X)

    vif = pd.DataFrame()

    vif["Variable"] = X.columns

    vif["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]

    return vif

#========================================================
# DIAGNÓSTICO DE LINEALIDAD
#========================================================

def f_linealidad(modelo):

    #----------------------------------------------------
    # Valores ajustados
    #----------------------------------------------------

    ajustados = modelo.fittedvalues

    #----------------------------------------------------
    # Residuos
    #----------------------------------------------------

    residuos = modelo.resid

    #----------------------------------------------------
    # Gráfico
    #----------------------------------------------------

    plt.figure(

        figsize = (6,4)
    )

    plt.scatter(

        ajustados,

        residuos,

        color = "blue"
    )

    plt.axhline(

        0,

        linestyle = "--",

        color = "red"
    )

    plt.title(

        "Residuos vs Valores Ajustados"
    )

    plt.xlabel(

        "Valores ajustados"
    )

    plt.ylabel(

        "Residuos"
    )

    plt.show()

    #----------------------------------------------------
    # Interpretación
    #----------------------------------------------------

    print("\n============================")

    print("Diagnóstico de Linealidad")

    print("============================")

    print(
        "- Los residuos deben distribuirse "
        "aleatoriamente alrededor de 0."
    )

    print(
        "- No deben observarse patrones "
        "curvos."
    )

    print(
        "- Curvaturas sugieren "
        "no linealidad."
    )

#========================================================
# TEST DE LINEALIDAD
# RAMSEY RESET
#========================================================

def f_linealidad_test(

        modelo

):

    #----------------------------------------------------
    # Ramsey RESET
    #----------------------------------------------------

    resultado = linear_reset(

        modelo,

        power = 2,

        use_f = True
    )

    #----------------------------------------------------
    # Mostrar resultado
    #----------------------------------------------------

    print("\n============================")

    print("Test de Linealidad (Ramsey RESET)")

    print("============================")

    print(resultado)

    #----------------------------------------------------
    # Interpretación
    #----------------------------------------------------

    p_valor = resultado.pvalue

    print("\nInterpretación:")

    if p_valor > 0.05:

        print(
            "✔ No se rechaza H0 → "
            "El modelo es lineal "
            "(no hay evidencia de curvatura)"
        )

    else:

        print(
            "❌ Se rechaza H0 → "
            "Existe evidencia de no linealidad"
        )

    return resultado

#========================================================
# HOMOCEDASTICIDAD
#========================================================

def f_homocedasticidad(

        modelo

):

    #----------------------------------------------------
    # Residuos
    #----------------------------------------------------

    residuos = modelo.resid

    #----------------------------------------------------
    # Valores ajustados
    #----------------------------------------------------

    ajustados = modelo.fittedvalues

    #----------------------------------------------------
    # Matriz X usada en el modelo
    #----------------------------------------------------

    X = modelo.model.exog

    #----------------------------------------------------
    # Breusch-Pagan
    #----------------------------------------------------

    bp = het_breuschpagan(

        residuos,

        X
    )

    #----------------------------------------------------
    # Resultados
    #----------------------------------------------------

    resultado = pd.DataFrame({

        "Prueba": [

            "Breusch-Pagan"
        ],

        "LM Statistic": [

            round(bp[0], 4)
        ],

        "LM p-value": [

            round(bp[1], 4)
        ],

        "F Statistic": [

            round(bp[2], 4)
        ],

        "F p-value": [

            round(bp[3], 4)
        ]
    })

    #----------------------------------------------------
    # Gráfico
    #----------------------------------------------------

    plt.figure(

        figsize = (6,4)
    )

    plt.scatter(

        ajustados,

        residuos,

        color = "blue"
    )

    plt.axhline(

        0,

        linestyle = "--",

        color = "red"
    )

    plt.title(

        "Residuos vs Valores Ajustados"
    )

    plt.xlabel(

        "Valores ajustados"
    )

    plt.ylabel(

        "Residuos"
    )

    plt.show()

    #----------------------------------------------------
    # Interpretación
    #----------------------------------------------------

    print("\n============================")

    print("Diagnóstico de Homocedasticidad")

    print("============================")

    print(resultado)

    print("\nInterpretación:")

    if bp[1] > 0.05:

        print(
            "✔ No se rechaza H0 → "
            "Existe homocedasticidad"
        )

    else:

        print(
            "❌ Se rechaza H0 → "
            "Existe heterocedasticidad"
        )

    return resultado

#========================================================
# NORMALIDAD DE RESIDUOS
#========================================================

from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import anderson

def f_normalidad(

        modelo

):

    #----------------------------------------------------
    # Residuos estandarizados
    #----------------------------------------------------

    residuos = modelo.resid

    residuos = (

        residuos - np.mean(residuos)

    ) / np.std(residuos)

    #----------------------------------------------------
    # SHAPIRO-WILK
    #----------------------------------------------------

    shapiro_test = shapiro(

        residuos
    )

    #----------------------------------------------------
    # KOLMOGOROV-SMIRNOV
    #----------------------------------------------------

    ks_test = kstest(

        residuos,

        'norm'
    )

    #----------------------------------------------------
    # ANDERSON-DARLING
    #----------------------------------------------------

    ad_test = anderson(

        residuos,

        dist = 'norm'
    )

    #----------------------------------------------------
    # Aproximación interpretación
    # scipy no devuelve p-value directo
    # usamos nivel 5%
    #----------------------------------------------------

    ad_critico_5 = ad_test.critical_values[2]

    ad_interpretacion = (

        "Normalidad"

        if ad_test.statistic < ad_critico_5

        else

        "No normalidad"
    )

    #----------------------------------------------------
    # RESULTADOS
    #----------------------------------------------------

    resultado = pd.DataFrame({

        "Prueba": [

            "Shapiro-Wilk",

            "Kolmogorov-Smirnov",

            "Anderson-Darling"
        ],

        "Estadistico": [

            round(shapiro_test.statistic, 4),

            round(ks_test.statistic, 4),

            round(ad_test.statistic, 4)
        ],

        "p_value": [

            round(shapiro_test.pvalue, 4),

            round(ks_test.pvalue, 4),

            np.nan
        ],

        "Interpretacion": [

            "Normalidad"

            if shapiro_test.pvalue > 0.05

            else

            "No normalidad",

            "Normalidad"

            if ks_test.pvalue > 0.05

            else

            "No normalidad",

            ad_interpretacion
        ]
    })

    #----------------------------------------------------
    # HISTOGRAMA
    #----------------------------------------------------

    plt.figure(

        figsize = (6,4)
    )

    plt.hist(

        residuos,

        bins = 15,

        density = True,

        alpha = 0.7
    )

    plt.title(

        "Histograma de residuos"
    )

    plt.xlabel(

        "Residuos estandarizados"
    )

    plt.ylabel(

        "Frecuencia"
    )

    plt.show()

    #----------------------------------------------------
    # QQ-PLOT
    #----------------------------------------------------

    sm.qqplot(

        residuos,

        line = '45'
    )

    plt.title(

        "QQ-Plot residuos"
    )

    plt.show()

    #----------------------------------------------------
    # RESULTADOS
    #----------------------------------------------------

    print("\n============================")

    print("Diagnóstico de Normalidad")

    print("============================")

    print(resultado)

    #----------------------------------------------------
    # Anderson detalle
    #----------------------------------------------------

    print("\nAnderson-Darling:")

    print(

        f"Estadístico = "
        f"{round(ad_test.statistic,4)}"
    )

    print(

        f"Valor crítico 5% = "
        f"{round(ad_critico_5,4)}"
    )

    print("\nInterpretación:")

    for i in range(len(resultado)):

        prueba = resultado.iloc[i]["Prueba"]

        interpretacion = resultado.iloc[i]["Interpretacion"]

        if interpretacion == "Normalidad":

            print(

                f"✔ {prueba}: Normalidad"
            )

        else:

            print(

                f"❌ {prueba}: No normalidad"
            )

    return resultado


#========================================================
# INDEPENDENCIA DE RESIDUOS
#========================================================

from statsmodels.stats.stattools import durbin_watson

def f_independencia(

        modelo

):

    #----------------------------------------------------
    # Residuos
    #----------------------------------------------------

    residuos = modelo.resid

    #----------------------------------------------------
    # Durbin-Watson
    #----------------------------------------------------

    dw = durbin_watson(

        residuos
    )

    #----------------------------------------------------
    # DataFrame resultado
    #----------------------------------------------------

    resultado = pd.DataFrame({

        "Prueba": [

            "Durbin-Watson"
        ],

        "Estadistico": [

            round(dw,4)
        ],

        "Interpretacion": [

            "Sin autocorrelación"

            if 1.5 <= dw <= 2.5

            else

            "Posible autocorrelación"
        ]
    })

    #----------------------------------------------------
    # Gráfico residuos
    #----------------------------------------------------

    plt.figure(

        figsize = (7,4)
    )

    plt.plot(

        residuos,

        color = "black",

        linewidth = 1
    )

    plt.axhline(

        0,

        linestyle = "--",

        color = "red"
    )

    plt.title(

        "Residuos en secuencia"
    )

    plt.xlabel(

        "Observación"
    )

    plt.ylabel(

        "Residuo"
    )

    plt.show()

    #----------------------------------------------------
    # Resultados
    #----------------------------------------------------

    print("\n============================")

    print("Diagnóstico de Independencia")

    print("============================")

    print(resultado)

    #----------------------------------------------------
    # Interpretación ampliada
    #----------------------------------------------------

    print("\nInterpretación:")

    if 1.5 <= dw <= 2.5:

        print(
            "✔ No existe evidencia "
            "de autocorrelación"
        )

    elif dw < 1.5:

        print(
            "❌ Existe posible "
            "autocorrelación positiva"
        )

    else:

        print(
            "❌ Existe posible "
            "autocorrelación negativa"
        )

    return resultado

#========================================================
# ECUACIÓN DEL MODELO
#========================================================

def f_ecuacion_modelo(

        modelo,

        redondeo = 4

):

    #----------------------------------------------------
    # Coeficientes
    #----------------------------------------------------

    coefs = modelo.params

    #----------------------------------------------------
    # Redondear
    #----------------------------------------------------

    coefs = round(

        coefs,

        redondeo
    )

    #----------------------------------------------------
    # Intercepto
    #----------------------------------------------------

    intercepto = coefs.iloc[0]

    ecuacion = f"ŷ = {intercepto}"

    #----------------------------------------------------
    # Construcción términos
    #----------------------------------------------------

    for variable in coefs.index[1:]:

        valor = coefs[variable]

        signo = (

            " + "

            if valor >= 0

            else

            " - "
        )

        termino = (

            f"{signo}"

            f"{abs(valor)}"

            f"*{variable}"
        )

        ecuacion += termino

    #----------------------------------------------------
    # Mostrar
    #----------------------------------------------------

    print("\n============================")

    print("ECUACIÓN DEL MODELO")

    print("============================\n")

    print(ecuacion)

    return ecuacion

#========================================================
# EVALUACIÓN DEL MODELO
#========================================================

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

def f_evaluacion_modelo(

        modelo,

        datos_validacion,

        variable_dependiente,

        redondeo = 4

):

    #----------------------------------------------------
    # Variable dependiente real
    #----------------------------------------------------

    y_real = datos_validacion[
        variable_dependiente
    ]

    #----------------------------------------------------
    # Variables independientes
    #----------------------------------------------------

    X_val = datos_validacion.drop(
        columns = [variable_dependiente]
    )

    #----------------------------------------------------
    # Variables dummy
    #----------------------------------------------------

    X_val = pd.get_dummies(

        X_val,

        drop_first = True
    )

    #----------------------------------------------------
    # Agregar constante
    #----------------------------------------------------

    X_val = sm.add_constant(

        X_val,

        has_constant = "add"
    )

    #----------------------------------------------------
    # Reordenar columnas
    # IMPORTANTÍSIMO
    #----------------------------------------------------

    X_val = X_val.reindex(

        columns = modelo.model.exog_names,

        fill_value = 0
    )

    #----------------------------------------------------
    # Predicciones
    #----------------------------------------------------

    pred = modelo.predict(

        X_val
    )

    #----------------------------------------------------
    # MÉTRICAS
    #----------------------------------------------------

    mse = mean_squared_error(

        y_real,

        pred
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(

        y_real,

        pred
    )

    r2 = r2_score(

        y_real,

        pred
    )

    #----------------------------------------------------
    # R² AJUSTADO
    #----------------------------------------------------

    n = len(y_real)

    p = X_val.shape[1] - 1

    r2_adj = 1 - (

        (1 - r2)

        * (n - 1)

        / (n - p - 1)
    )

    #----------------------------------------------------
    # RESULTADOS
    #----------------------------------------------------

    resultado = pd.DataFrame({

        "R_square": [

            round(r2, redondeo)
        ],

        "R_square_ajustado": [

            round(r2_adj, redondeo)
        ],

        "MSE": [

            round(mse, redondeo)
        ],

        "RMSE": [

            round(rmse, redondeo)
        ],

        "MAE": [

            round(mae, redondeo)
        ]
    })

    #----------------------------------------------------
    # Mostrar
    #----------------------------------------------------

    print("\n============================")

    print("Evaluación del Modelo")

    print("============================")

    print(resultado)

    return resultado
