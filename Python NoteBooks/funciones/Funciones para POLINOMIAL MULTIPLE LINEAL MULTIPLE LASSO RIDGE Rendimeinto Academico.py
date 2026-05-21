# Funciones para implementar y evaluar modelos de regresión múltiple en Python
# Se crean regresión regresión lineal mpultiple con datos originales 
# Se crean modelos de regresi[]on polinomial múltiple con datos originales
# Se crean regresión lineal múltiple, Lasso y Ridge con datos normalizados
# Rubén Pizarro Gurrola
# Mayo 2026

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
    # Copia directa
    # SIN VARIABLES DUMMY
    #----------------------------------------------------

    X = X.copy()

    #----------------------------------------------------
    # Constante
    #----------------------------------------------------

    X = sm.add_constant(

        X,

        has_constant = "add"
    )

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

        print(

            modelo.summary()
        )

    return modelo

#========================================================
# REGRESIÓN MÚLTIPLE POLINOMIAL
# CON STATSMODELS
#========================================================

from sklearn.preprocessing import (
    PolynomialFeatures
)

def f_multiple_polinomial(

        datos,

        variable_dependiente,

        grado = 2,

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
    # Variables originales
    #----------------------------------------------------

    nombres_originales = X.columns

    #----------------------------------------------------
    # Transformación polinomial
    #----------------------------------------------------

    poly = PolynomialFeatures(

        degree = grado,

        include_bias = False
    )

    X_poly = poly.fit_transform(

        X
    )

    #----------------------------------------------------
    # Nombres variables polinomiales
    #----------------------------------------------------

    nombres_poly = poly.get_feature_names_out(

        nombres_originales
    )

    #----------------------------------------------------
    # DataFrame polinomial
    #----------------------------------------------------

    X_poly = pd.DataFrame(

        X_poly,

        columns = nombres_poly,

        index = X.index
    )

    #----------------------------------------------------
    # Constante
    #----------------------------------------------------

    X_poly = sm.add_constant(

        X_poly,

        has_constant = "add"
    )

    #----------------------------------------------------
    # Modelo OLS
    #----------------------------------------------------

    modelo = sm.OLS(

        y,

        X_poly

    ).fit()

    #----------------------------------------------------
    # Guardar estructura
    # IMPORTANTÍSIMO
    #----------------------------------------------------

    modelo.columnas_entrenamiento = X_poly.columns

    modelo.poly_transformador = poly

    modelo.nombres_originales = nombres_originales

    #----------------------------------------------------
    # Información
    #----------------------------------------------------

    print("\n============================")

    print(

        f"MODELO POLINOMIAL GRADO {grado}"
    )

    print("============================")

    print(

        f"Número de variables originales: "
        f"{len(nombres_originales)}"
    )

    print(

        f"Número de términos polinomiales: "
        f"{X_poly.shape[1]-1}"
    )

    #----------------------------------------------------
    # Resumen
    #----------------------------------------------------

    if ver_resumen:

        print(

            modelo.summary()
        )

    return modelo

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


def f_construir_modelo_lasso(

        datos,

        variable_dependiente,

        cv = 10,

        random_state = 2026,

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
    # Modelo LASSO CV
    #----------------------------------------------------

    modelo = LassoCV(

        cv = cv,

        random_state = random_state

    )

    modelo.fit(

        X,

        y
    )

    #----------------------------------------------------
    # Coeficientes
    #----------------------------------------------------

    coeficientes = pd.DataFrame({

        "Variable":

            X.columns,

        "Coeficiente":

            np.round(
                modelo.coef_,
                4
            )
    })

    #----------------------------------------------------
    # Información
    #----------------------------------------------------

    if ver_resumen:

        print("\n============================")

        print("MODELO LASSO")

        print("============================")

        print(

            f"Lambda óptimo: "
            f"{round(modelo.alpha_,4)}"
        )

        print(

            f"Intercepto: "
            f"{round(modelo.intercept_,4)}"
        )

        print("\nCoeficientes:")

        print(coeficientes)

    #----------------------------------------------------
    # Guardar columnas
    #----------------------------------------------------

    modelo.columnas_entrenamiento = X.columns

    return modelo

  #========================================================
# MODELO RIDGE
#========================================================



def f_construir_modelo_ridge(
        datos,
        variable_dependiente,
        alphas = np.logspace(-4, 4, 100),
        cv = 10,
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
    # Modelo RIDGE CV
    #----------------------------------------------------

    modelo = RidgeCV(

        alphas = alphas,

        cv = cv
    )

    modelo.fit(

        X,

        y
    )

    #----------------------------------------------------
    # Coeficientes
    #----------------------------------------------------

    coeficientes = pd.DataFrame({

        "Variable":

            X.columns,

        "Coeficiente":

            np.round(
                modelo.coef_,
                4
            )
    })

    #----------------------------------------------------
    # Información
    #----------------------------------------------------

    if ver_resumen:

        print("\n============================")

        print("MODELO RIDGE")

        print("============================")

        print(

            f"Lambda óptimo: "
            f"{round(modelo.alpha_,4)}"
        )

        print(

            f"Intercepto: "
            f"{round(modelo.intercept_,4)}"
        )

        print("\nCoeficientes:")

        print(coeficientes)

    #----------------------------------------------------
    # Guardar columnas entrenamiento
    #----------------------------------------------------

    modelo.columnas_entrenamiento = X.columns

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
# VALIDAR POSTULADOS MULTIMODELO
#========================================================

#========================================================
# VALIDAR POSTULADOS
# OLS + LASSO + RIDGE
#========================================================

def f_validar_postulados_modelos(

        modelos,

        X_datos = None,

        y_datos = None,

        nombres = None,

        redondeo = 4

):

    #----------------------------------------------------
    # Lista modelos
    #----------------------------------------------------

    if not isinstance(modelos, list):

        modelos = [modelos]

    #----------------------------------------------------
    # Nombres
    #----------------------------------------------------

    if nombres is None:

        nombres = [

            f"Modelo_{i+1}"

            for i in range(len(modelos))
        ]

    #----------------------------------------------------
    # Resultados
    #----------------------------------------------------

    resultados = []

    #====================================================
    # RECORRER MODELOS
    #====================================================

    for i, modelo in enumerate(modelos):

        #================================================
        # MODELOS OLS STATSMODELS
        #================================================

        if hasattr(modelo, "model"):

            X = modelo.model.exog

            residuos = modelo.resid

            #--------------------------------------------
            # VIF
            #--------------------------------------------

            vif_valores = []

            for j in range(1, X.shape[1]):

                vif = variance_inflation_factor(

                    X,

                    j
                )

                vif_valores.append(vif)

            vif_max = max(vif_valores)

            #--------------------------------------------
            # Interpretación VIF
            #--------------------------------------------

            if vif_max < 5:

                interpretacion_vif = "Cumple"

            elif vif_max < 10:

                interpretacion_vif = "Moderada"

            else:

                interpretacion_vif = "No cumple"

            #--------------------------------------------
            # Linealidad
            #--------------------------------------------

            reset = linear_reset(

                modelo,

                power = 2,

                use_f = True
            )

            p_lineal = reset.pvalue

            linealidad = (

                "Cumple"

                if p_lineal > 0.05

                else

                "No cumple"
            )

            #--------------------------------------------
            # Homocedasticidad
            #--------------------------------------------

            bp = het_breuschpagan(

                residuos,

                X
            )

            p_homo = bp[1]

            homocedasticidad = (

                "Cumple"

                if p_homo > 0.05

                else

                "No cumple"
            )

        #================================================
        # MODELOS LASSO / RIDGE
        #================================================

        else:

            #--------------------------------------------
            # Validación
            #--------------------------------------------

            if X_datos is None or y_datos is None:

                raise ValueError(

                    "LASSO/RIDGE requieren "
                    "X_datos e y_datos"
                )

            #--------------------------------------------
            # Predicciones
            #--------------------------------------------

            pred = modelo.predict(

                X_datos
            )

            residuos = y_datos - pred

            #--------------------------------------------
            # VIF
            #--------------------------------------------

            vif_valores = []

            for j in range(X_datos.shape[1]):

                vif = variance_inflation_factor(

                    X_datos.values,

                    j
                )

                vif_valores.append(vif)

            vif_max = max(vif_valores)

            #--------------------------------------------
            # Interpretación VIF
            #--------------------------------------------

            if vif_max < 5:

                interpretacion_vif = "Cumple"

            elif vif_max < 10:

                interpretacion_vif = "Moderada"

            else:

                interpretacion_vif = "No cumple"

            #--------------------------------------------
            # No aplica formalmente
            #--------------------------------------------

            linealidad = "NA"

            homocedasticidad = "NA"

        #================================================
        # NORMALIDAD
        #================================================

        shapiro_test = shapiro(

            residuos
        )

        normalidad = (

            "Cumple"

            if shapiro_test.pvalue > 0.05

            else

            "No cumple"
        )

        #================================================
        # INDEPENDENCIA
        #================================================

        dw = durbin_watson(

            residuos
        )

        independencia = (

            "Cumple"

            if 1.5 <= dw <= 2.5

            else

            "No cumple"
        )

        #================================================
        # RESULTADO
        #================================================

        resultados.append({

            "Modelo":

                nombres[i],

            "Multicolinealidad":

                interpretacion_vif,

            "VIF_Max":

                round(vif_max, redondeo),

            "Linealidad":

                linealidad,

            "Homocedasticidad":

                homocedasticidad,

            "Normalidad":

                normalidad,

            "Independencia":

                independencia
        })

    #----------------------------------------------------
    # DataFrame final
    #----------------------------------------------------

    resultados = pd.DataFrame(

        resultados
    )

    #----------------------------------------------------
    # Mostrar
    #----------------------------------------------------

    print("\n============================")

    print("Validación de Postulados")

    print("============================")

    print(resultados)

    return resultados

#========================================================
# ECUACIÓN DEL MODELO
#========================================================

#========================================================
# ECUACIONES DE MODELOS
# OLS + LASSO + RIDGE
#========================================================

def f_ecuaciones_modelos(

        modelos,

        nombres = None,

        redondeo = 4

):

    #----------------------------------------------------
    # Convertir a lista
    #----------------------------------------------------

    if not isinstance(modelos, list):

        modelos = [modelos]

    #----------------------------------------------------
    # Nombres modelos
    #----------------------------------------------------

    if nombres is None:

        nombres = [

            f"Modelo_{i+1}"

            for i in range(len(modelos))
        ]

    #----------------------------------------------------
    # Lista ecuaciones
    #----------------------------------------------------

    ecuaciones = []

    #====================================================
    # RECORRER MODELOS
    #====================================================

    for i, modelo in enumerate(modelos):

        print("\n============================")

        print(

            f"ECUACIÓN: {nombres[i]}"
        )

        print("============================\n")

        #================================================
        # MODELOS OLS (STATSMODELS)
        #================================================

        if hasattr(modelo, "params"):

            coefs = modelo.params.round(

                redondeo
            )

            intercepto = coefs.iloc[0]

            variables = coefs.index[1:]

            valores = coefs.iloc[1:]

        #================================================
        # MODELOS LASSO / RIDGE
        #================================================

        else:

            intercepto = round(

                modelo.intercept_,

                redondeo
            )

            variables = modelo.columnas_entrenamiento

            valores = np.round(

                modelo.coef_,

                redondeo
            )

        #================================================
        # ECUACIÓN
        #================================================

        ecuacion = f"ŷ = {intercepto}"

        #------------------------------------------------
        # Construcción términos
        #------------------------------------------------

        for variable, valor in zip(

                variables,

                valores
        ):

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

        #------------------------------------------------
        # Mostrar
        #------------------------------------------------

        print(ecuacion)

        #------------------------------------------------
        # Guardar
        #------------------------------------------------

        ecuaciones.append({

            "Modelo":

                nombres[i],

            "Ecuacion":

                ecuacion
        })

    #----------------------------------------------------
    # DataFrame final
    #----------------------------------------------------

    ecuaciones = pd.DataFrame(

        ecuaciones
    )

    return ecuaciones

#========================================================
# EVALUACIÓN MULTIMODELO
# OLS + LASSO + RIDGE
#========================================================

from sklearn.metrics import (

    mean_squared_error,

    mean_absolute_error,

    r2_score
)

#========================================================
# EVALUACIÓN MULTIMODELO
# OLS + POLINOMIAL + LASSO + RIDGE
#========================================================

from sklearn.metrics import (

    mean_squared_error,

    mean_absolute_error,

    r2_score
)

def f_evaluacion_modelos(

        modelos,

        datos_validacion,

        variable_dependiente,

        nombres = None,

        redondeo = 4

):

    #----------------------------------------------------
    # Convertir a lista
    #----------------------------------------------------

    if not isinstance(modelos, list):

        modelos = [modelos]

    #----------------------------------------------------
    # Nombres modelos
    #----------------------------------------------------

    if nombres is None:

        nombres = [

            f"Modelo_{i+1}"

            for i in range(len(modelos))
        ]

    #----------------------------------------------------
    # Resultados
    #----------------------------------------------------

    resultados = []

    #====================================================
    # DATOS VALIDACIÓN
    #====================================================

    y_real = datos_validacion[

        variable_dependiente
    ]

    X_val = datos_validacion.drop(

        columns = [variable_dependiente]
    )

    #====================================================
    # RECORRER MODELOS
    #====================================================

    for i, modelo in enumerate(modelos):

        #================================================
        # MODELOS OLS / POLINOMIALES
        #================================================

        if hasattr(modelo, "model"):

            #--------------------------------------------
            # MODELO POLINOMIAL
            #--------------------------------------------

            if hasattr(modelo, "poly_transformador"):

                #----------------------------------------
                # Variables originales
                #----------------------------------------

                X_base = X_val[

                    modelo.nombres_originales
                ]

                #----------------------------------------
                # Transformación polinomial
                #----------------------------------------

                X_poly = modelo.poly_transformador.transform(

                    X_base
                )

                #----------------------------------------
                # Nombres polinomiales
                #----------------------------------------

                nombres_poly = (

                    modelo.poly_transformador
                    .get_feature_names_out(

                        modelo.nombres_originales
                    )
                )

                #----------------------------------------
                # DataFrame
                #----------------------------------------

                X_pred = pd.DataFrame(

                    X_poly,

                    columns = nombres_poly,

                    index = X_base.index
                )

            #--------------------------------------------
            # MODELO LINEAL NORMAL
            #--------------------------------------------

            else:

                X_pred = X_val.copy()

            #--------------------------------------------
            # Constante
            #--------------------------------------------

            X_pred = sm.add_constant(

                X_pred,

                has_constant = "add"
            )

            #--------------------------------------------
            # Reordenar columnas
            #--------------------------------------------

            X_pred = X_pred.reindex(

                columns = modelo.model.exog_names,

                fill_value = 0
            )

            #--------------------------------------------
            # Predicciones
            #--------------------------------------------

            pred = modelo.predict(

                X_pred
            )

            p = X_pred.shape[1] - 1

        #================================================
        # MODELOS LASSO / RIDGE
        #================================================

        else:

            #--------------------------------------------
            # Copia directa
            #--------------------------------------------

            X_pred = X_val.copy()

            #--------------------------------------------
            # Reordenar columnas
            #--------------------------------------------

            X_pred = X_pred.reindex(

                columns = modelo.columnas_entrenamiento,

                fill_value = 0
            )

            #--------------------------------------------
            # Predicción
            #--------------------------------------------

            pred = modelo.predict(

                X_pred
            )

            p = X_pred.shape[1]

        #================================================
        # MÉTRICAS
        #================================================

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

        #================================================
        # R² AJUSTADO
        #================================================

        n = len(y_real)

        r2_adj = 1 - (

            (1 - r2)

            * (n - 1)

            / (n - p - 1)
        )

        #================================================
        # RESULTADOS
        #================================================

        resultados.append({

            "Modelo":

                nombres[i],

            "R_square":

                round(r2, redondeo),

            "R_square_ajustado":

                round(r2_adj, redondeo),

            "MSE":

                round(mse, redondeo),

            "RMSE":

                round(rmse, redondeo),

            "MAE":

                round(mae, redondeo)
        })

    #----------------------------------------------------
    # DataFrame final
    #----------------------------------------------------

    resultados = pd.DataFrame(

        resultados
    )

    #----------------------------------------------------
    # Mostrar
    #----------------------------------------------------

    print("\n============================")

    print("Evaluación de Modelos")

    print("============================")

    print(resultados)

    return resultados