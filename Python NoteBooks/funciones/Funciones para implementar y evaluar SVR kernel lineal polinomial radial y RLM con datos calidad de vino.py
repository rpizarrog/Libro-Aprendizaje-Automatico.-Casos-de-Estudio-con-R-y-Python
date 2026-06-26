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
# FUNCIÓN
# f_crear_SVR_lineal()
#========================================================

def f_crear_SVR_lineal(
    
    datos_entrenamiento,
    
    variables_independientes,
    
    variable_dependiente,
    
    epsilon = 0.1,
    
    cost = 1,
    
    gamma = None
):
    
    #----------------------------------------------------
    # LIBRERÍAS
    #----------------------------------------------------
    
    import pandas as pd
    
    import numpy as np
    
    from sklearn.svm import SVR
    
    #----------------------------------------------------
    # VARIABLES X e y
    #----------------------------------------------------
    
    X = datos_entrenamiento[
        
        variables_independientes
    ]
    
    y = datos_entrenamiento[
        
        variable_dependiente
    ]
    
    #----------------------------------------------------
    # GAMMA
    # similar a e1071
    #----------------------------------------------------
    
    if gamma is None:
        
        gamma = 1 / len(
            
            variables_independientes
        )
    
    #----------------------------------------------------
    # CONSTRUIR MODELO
    #----------------------------------------------------
    
    modelo = SVR(
        
        kernel = "linear",
        
        epsilon = epsilon,
        
        C = cost,
        
        gamma = gamma
    )
    
    #----------------------------------------------------
    # ENTRENAR
    #----------------------------------------------------
    
    modelo.fit(
        
        X,
        
        y
    )
    
    #====================================================
    # RESUMEN
    #====================================================
    
    print("\n")
    
    print("=====================================")
    
    print("MODELO SVR KERNEL LINEAL")
    
    print("=====================================")
    
    #----------------------------------------------------
    # Variables independientes
    #----------------------------------------------------
    
    print("\nVARIABLES INDEPENDIENTES:")
    
    print(
        variables_independientes
    )
    
    #----------------------------------------------------
    # Variable dependiente
    #----------------------------------------------------
    
    print("\nVARIABLE DEPENDIENTE:")
    
    print(
        variable_dependiente
    )
    
    #----------------------------------------------------
    # Kernel
    #----------------------------------------------------
    
    print("\nKERNEL:")
    
    print(
        modelo.kernel
    )
    
    #----------------------------------------------------
    # Gamma
    #----------------------------------------------------
    
    print("\nGAMMA:")
    
    print(
        modelo.gamma
    )
    
    #----------------------------------------------------
    # Epsilon
    #----------------------------------------------------
    
    print("\nEPSILON:")
    
    print(
        modelo.epsilon
    )
    
    #----------------------------------------------------
    # Cost
    #----------------------------------------------------
    
    print("\nCOST:")
    
    print(
        modelo.C
    )
    
    #----------------------------------------------------
    # Intercepto
    #----------------------------------------------------
    
    print("\nINTERCEPTO b:")
    
    print(
        modelo.intercept_
    )
    
    #----------------------------------------------------
    # Número vectores soporte
    #----------------------------------------------------
    
    print("\nNÚMERO VECTORES SOPORTE:")
    
    print(
        len(modelo.support_)
    )
    
    #----------------------------------------------------
    # Índices vectores soporte
    #----------------------------------------------------
    
    print("\nÍNDICES VECTORES SOPORTE:")
    
    print(
        modelo.support_
    )
    
    #----------------------------------------------------
    # Vectores soporte
    #----------------------------------------------------
    
    print("\nVECTORES SOPORTE:")
    
    print(
        modelo.support_vectors_
    )
    
    #----------------------------------------------------
    # Coeficientes alpha
    #----------------------------------------------------
    
    print("\nALPHAS / COEFICIENTES:")
    
    print(
        modelo.dual_coef_
    )
    
    #----------------------------------------------------
    # Coeficientes lineales
    #----------------------------------------------------
    
    print("\nPESOS w:")
    
    print(
        modelo.coef_
    )
    
    #----------------------------------------------------
    # RETORNO
    #----------------------------------------------------
    
    return modelo

#========================================================
# FUNCIÓN
# f_crear_SVR_polinomial()
#========================================================

def f_crear_SVR_polinomial(
    
    datos_entrenamiento,
    
    variables_independientes,
    
    variable_dependiente,
    
    grado = 2,
    
    epsilon = 0.1,
    
    cost = 1,
    
    gamma = None
):
    
    #----------------------------------------------------
    # LIBRERÍAS
    #----------------------------------------------------
    
    import pandas as pd
    
    import numpy as np
    
    from sklearn.svm import SVR
    
    #----------------------------------------------------
    # VARIABLES X e y
    #----------------------------------------------------
    
    X = datos_entrenamiento[
        
        variables_independientes
    ]
    
    y = datos_entrenamiento[
        
        variable_dependiente
    ]
    
    #----------------------------------------------------
    # GAMMA
    # similar a e1071
    #----------------------------------------------------
    
    if gamma is None:
        
        gamma = 1 / len(
            
            variables_independientes
        )
    
    #----------------------------------------------------
    # CONSTRUIR MODELO
    #----------------------------------------------------
    
    modelo = SVR(
        
        kernel = "poly",
        
        degree = grado,
        
        epsilon = epsilon,
        
        C = cost,
        
        gamma = gamma
    )
    
    #----------------------------------------------------
    # ENTRENAR
    #----------------------------------------------------
    
    modelo.fit(
        
        X,
        
        y
    )
    
    #====================================================
    # RESUMEN
    #====================================================
    
    print("\n")
    
    print("=====================================")
    
    print(
        f"MODELO SVR KERNEL POLINOMIAL GRADO {grado}"
    )
    
    print("=====================================")
    
    #----------------------------------------------------
    # Variables independientes
    #----------------------------------------------------
    
    print("\nVARIABLES INDEPENDIENTES:")
    
    print(
        variables_independientes
    )
    
    #----------------------------------------------------
    # Variable dependiente
    #----------------------------------------------------
    
    print("\nVARIABLE DEPENDIENTE:")
    
    print(
        variable_dependiente
    )
    
    #----------------------------------------------------
    # Kernel
    #----------------------------------------------------
    
    print("\nKERNEL:")
    
    print(
        modelo.kernel
    )
    
    #----------------------------------------------------
    # Degree
    #----------------------------------------------------
    
    print("\nGRADO POLINOMIAL:")
    
    print(
        modelo.degree
    )
    
    #----------------------------------------------------
    # Gamma
    #----------------------------------------------------
    
    print("\nGAMMA:")
    
    print(
        modelo.gamma
    )
    
    #----------------------------------------------------
    # Epsilon
    #----------------------------------------------------
    
    print("\nEPSILON:")
    
    print(
        modelo.epsilon
    )
    
    #----------------------------------------------------
    # Cost
    #----------------------------------------------------
    
    print("\nCOST:")
    
    print(
        modelo.C
    )
    
    #----------------------------------------------------
    # Intercepto
    #----------------------------------------------------
    
    print("\nINTERCEPTO b:")
    
    print(
        modelo.intercept_
    )
    
    #----------------------------------------------------
    # Número vectores soporte
    #----------------------------------------------------
    
    print("\nNÚMERO VECTORES SOPORTE:")
    
    print(
        len(modelo.support_)
    )
    
    #----------------------------------------------------
    # Índices vectores soporte
    #----------------------------------------------------
    
    print("\nÍNDICES VECTORES SOPORTE:")
    
    print(
        modelo.support_
    )
    
    #----------------------------------------------------
    # Vectores soporte
    #----------------------------------------------------
    
    print("\nVECTORES SOPORTE:")
    
    print(
        modelo.support_vectors_
    )
    
    #----------------------------------------------------
    # Coeficientes alpha
    #----------------------------------------------------
    
    print("\nALPHAS / COEFICIENTES:")
    
    print(
        modelo.dual_coef_
    )
    
    #----------------------------------------------------
    # RETORNO
    #----------------------------------------------------
    
    return modelo

#========================================================
# FUNCIÓN
# f_crear_SVR_radial()
#========================================================

def f_crear_SVR_radial(
    
    datos_entrenamiento,
    
    variables_independientes,
    
    variable_dependiente,
    
    epsilon = 0.1,
    
    cost = 1,
    
    gamma = None
):
    
    #----------------------------------------------------
    # LIBRERÍAS
    #----------------------------------------------------
    
    import pandas as pd
    
    import numpy as np
    
    from sklearn.svm import SVR
    
    #----------------------------------------------------
    # VARIABLES X e y
    #----------------------------------------------------
    
    X = datos_entrenamiento[
        
        variables_independientes
    ]
    
    y = datos_entrenamiento[
        
        variable_dependiente
    ]
    
    #----------------------------------------------------
    # GAMMA
    # similar a e1071
    #----------------------------------------------------
    
    if gamma is None:
        
        gamma = 1 / len(
            
            variables_independientes
        )
    
    #----------------------------------------------------
    # CONSTRUIR MODELO
    #----------------------------------------------------
    
    modelo = SVR(
        
        kernel = "rbf",
        
        epsilon = epsilon,
        
        C = cost,
        
        gamma = gamma
    )
    
    #----------------------------------------------------
    # ENTRENAR
    #----------------------------------------------------
    
    modelo.fit(
        
        X,
        
        y
    )
    
    #====================================================
    # RESUMEN
    #====================================================
    
    print("\n")
    
    print("=====================================")
    
    print("MODELO SVR KERNEL RADIAL")
    
    print("=====================================")
    
    #----------------------------------------------------
    # Variables independientes
    #----------------------------------------------------
    
    print("\nVARIABLES INDEPENDIENTES:")
    
    print(
        variables_independientes
    )
    
    #----------------------------------------------------
    # Variable dependiente
    #----------------------------------------------------
    
    print("\nVARIABLE DEPENDIENTE:")
    
    print(
        variable_dependiente
    )
    
    #----------------------------------------------------
    # Kernel
    #----------------------------------------------------
    
    print("\nKERNEL:")
    
    print(
        modelo.kernel
    )
    
    #----------------------------------------------------
    # Gamma
    #----------------------------------------------------
    
    print("\nGAMMA:")
    
    print(
        modelo.gamma
    )
    
    #----------------------------------------------------
    # Epsilon
    #----------------------------------------------------
    
    print("\nEPSILON:")
    
    print(
        modelo.epsilon
    )
    
    #----------------------------------------------------
    # Cost
    #----------------------------------------------------
    
    print("\nCOST:")
    
    print(
        modelo.C
    )
    
    #----------------------------------------------------
    # Intercepto
    #----------------------------------------------------
    
    print("\nINTERCEPTO b:")
    
    print(
        modelo.intercept_
    )
    
    #----------------------------------------------------
    # Número vectores soporte
    #----------------------------------------------------
    
    print("\nNÚMERO VECTORES SOPORTE:")
    
    print(
        len(modelo.support_)
    )
    
    #----------------------------------------------------
    # Índices vectores soporte
    #----------------------------------------------------
    
    print("\nÍNDICES VECTORES SOPORTE:")
    
    print(
        modelo.support_
    )
    
    #----------------------------------------------------
    # Vectores soporte
    #----------------------------------------------------
    
    print("\nVECTORES SOPORTE:")
    
    print(
        modelo.support_vectors_
    )
    
    #----------------------------------------------------
    # Coeficientes alpha
    #----------------------------------------------------
    
    print("\nALPHAS / COEFICIENTES:")
    
    print(
        modelo.dual_coef_
    )
    
    #----------------------------------------------------
    # RETORNO
    #----------------------------------------------------
    
    return modelo

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
# FUNCIÓN
# f_validar_postulados_modelos()
#========================================================

def f_validar_postulados_modelos(
    
    modelos,
    
    datos_list,
    
    variable_dependiente,
    
    nombres = None
):
    
    #----------------------------------------------------
    # LIBRERÍAS
    #----------------------------------------------------
    
    import pandas as pd
    
    import numpy as np
    
    import statsmodels.api as sm
    
    from scipy.stats import shapiro
    
    from statsmodels.stats.stattools import durbin_watson
    
    from statsmodels.stats.diagnostic import het_breuschpagan
    
    from statsmodels.stats.outliers_influence import (
        
        variance_inflation_factor
    )
    
    #----------------------------------------------------
    # CONVERTIR A LISTA
    #----------------------------------------------------
    
    if not isinstance(modelos, list):
        
        modelos = [modelos]
    
    if not isinstance(datos_list, list):
        
        datos_list = [datos_list]
    
    #----------------------------------------------------
    # VALIDAR LONGITUD
    #----------------------------------------------------
    
    if len(modelos) != len(datos_list):
        
        raise ValueError(
            
            "Modelos y datasets deben tener la misma longitud"
        )
    
    #----------------------------------------------------
    # NOMBRES
    #----------------------------------------------------
    
    if nombres is None:
        
        nombres = [
            
            f"Modelo_{i+1}"
            
            for i in range(len(modelos))
        ]
    
    #----------------------------------------------------
    # RESULTADOS
    #----------------------------------------------------
    
    resultados = []
    
    #====================================================
    # RECORRER MODELOS
    #====================================================
    
    for i in range(len(modelos)):
        
        modelo = modelos[i]
        
        datos = datos_list[i]
        
        #------------------------------------------------
        # VARIABLE DEPENDIENTE
        #------------------------------------------------
        
        y_real = datos[
            
            variable_dependiente
        ]
        
        #------------------------------------------------
        # VARIABLES INDEPENDIENTES
        #------------------------------------------------
        
        X = datos.drop(
            
            columns = [variable_dependiente]
        )
        
        #================================================
        # IDENTIFICAR TIPO MODELO
        #================================================
        
        nombre_clase = type(modelo).__name__
        
        #------------------------------------------------
        # MODELOS SVR
        #------------------------------------------------
        
        if "SVR" in nombre_clase:
            
            tipo_modelo = "svm"
        
        #------------------------------------------------
        # MODELOS statsmodels
        #------------------------------------------------
        
        else:
            
            tipo_modelo = "lineal"
        
        #================================================
        # PREDICCIONES
        #================================================
        
        #------------------------------------------------
        # MODELOS LINEALES statsmodels
        #------------------------------------------------
        
        if tipo_modelo == "lineal":
            
            X_const = sm.add_constant(
                
                X,
                
                has_constant = "add"
            )
            
            pred = modelo.predict(
                
                X_const
            )
        
        #------------------------------------------------
        # MODELOS SVR
        #------------------------------------------------
        
        else:
            
            pred = modelo.predict(X)
        
        #------------------------------------------------
        # RESIDUOS
        #------------------------------------------------
        
        residuos = y_real - pred
        
        #================================================
        # 1. MULTICOLINEALIDAD
        #================================================
        
        if tipo_modelo == "lineal":
            
            try:
                
                X_vif = sm.add_constant(
                    
                    X,
                    
                    has_constant = "add"
                )
                
                vif_val = max([
                    
                    variance_inflation_factor(
                        
                        X_vif.values,
                        
                        j
                    )
                    
                    for j in range(
                        
                        1,
                        
                        X_vif.shape[1]
                    )
                ])
                
            except:
                
                vif_val = np.nan
        
        else:
            
            vif_val = np.nan
        
        #================================================
        # 2. LINEALIDAD
        #================================================
        
        if tipo_modelo == "lineal":
            
            try:
                
                corr = np.corrcoef(
                    
                    pred,
                    
                    residuos
                )[0,1]
                
                p_lineal = (
                    
                    0.99
                    
                    if abs(corr) < 0.3
                    
                    else 0.01
                )
                
            except:
                
                p_lineal = np.nan
        
        else:
            
            p_lineal = np.nan
        
        #================================================
        # 3. HOMOCEDASTICIDAD
        #================================================
        
        if tipo_modelo == "lineal":
            
            try:
                
                modelo_aux = sm.OLS(
                    
                    y_real,
                    
                    sm.add_constant(
                        
                        X,
                        
                        has_constant = "add"
                    )
                ).fit()
                
                bp_test = het_breuschpagan(
                    
                    modelo_aux.resid,
                    
                    modelo_aux.model.exog
                )
                
                p_homo = bp_test[1]
                
            except:
                
                p_homo = np.nan
        
        else:
            
            p_homo = np.nan
        
        #================================================
        # 4. NORMALIDAD
        #================================================
        
        try:
            
            p_norm = shapiro(
                
                residuos
            )[1]
            
        except:
            
            p_norm = np.nan
        
        #================================================
        # 5. INDEPENDENCIA
        #================================================
        
        try:
            
            dw = durbin_watson(
                
                residuos
            )
            
            p_dw = (
                
                0.99
                
                if 1.5 <= dw <= 2.5
                
                else 0.01
            )
            
        except:
            
            p_dw = np.nan
        
        #================================================
        # INTERPRETACIÓN
        #================================================
        
        def interpretar(p):
            
            if pd.isna(p):
                
                return "NA"
            
            if p > 0.05:
                
                return "Cumple"
            
            return "No cumple"
        
        #================================================
        # RESULTADOS
        #================================================
        
        resultados.append({
            
            "Modelo":
                nombres[i],
            
            "Tipo":
                tipo_modelo,
            
            "VIF_Max":
                round(vif_val, 4)
                if not pd.isna(vif_val)
                else np.nan,
            
            "Linealidad":
                interpretar(p_lineal),
            
            "Homocedasticidad":
                interpretar(p_homo),
            
            "Normalidad":
                interpretar(p_norm),
            
            "Independencia":
                interpretar(p_dw)
        })
    
    #====================================================
    # DATAFRAME FINAL
    #====================================================
    
    resultados = pd.DataFrame(
        
        resultados
    )
    
    #====================================================
    # MOSTRAR
    #====================================================
    
    print("\n============================")
    
    print("VALIDACIÓN DE POSTULADOS")
    
    print("============================")
    
    print(resultados)
    
    #----------------------------------------------------
    # RETORNO
    #----------------------------------------------------
    
    return resultados

 #========================================================
# FUNCIÓN
# f_evaluacion_modelos()
# ROBUSTA PARA:
# - statsmodels OLS
# - modelos polinomiales
# - Ridge / Lasso
# - SVR
#========================================================

def f_evaluacion_modelos(

        modelos,

        datos_validacion,

        variable_dependiente,

        nombres = None,

        redondeo = 4

):

    #----------------------------------------------------
    # LIBRERÍAS
    #----------------------------------------------------

    import pandas as pd

    import numpy as np

    import statsmodels.api as sm

    from sklearn.metrics import (

        mean_squared_error,

        mean_absolute_error,

        r2_score
    )

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
        # IDENTIFICAR TIPO MODELO
        #================================================

        nombre_clase = type(modelo).__name__

        #================================================
        # MODELOS SVR
        #================================================

        if "SVR" in nombre_clase:

            #--------------------------------------------
            # Reordenar columnas
            #--------------------------------------------

            if hasattr(modelo, "feature_names_in_"):

                X_pred = X_val.reindex(

                    columns = modelo.feature_names_in_,

                    fill_value = 0
                )

            else:

                X_pred = X_val.copy()

            #--------------------------------------------
            # Predicción
            #--------------------------------------------

            pred = modelo.predict(

                X_pred
            )

            #--------------------------------------------
            # Parámetros aproximados
            # vectores soporte
            #--------------------------------------------

            p = len(

                modelo.support_
            )

        #================================================
        # MODELOS OLS / POLINOMIALES
        #================================================

        elif hasattr(modelo, "model"):

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
        # MODELOS RIDGE / LASSO
        #================================================

        else:

            #--------------------------------------------
            # Copia directa
            #--------------------------------------------

            X_pred = X_val.copy()

            #--------------------------------------------
            # Reordenar columnas
            #--------------------------------------------

            if hasattr(modelo, "columnas_entrenamiento"):

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

        #------------------------------------------------
        # EVITAR DIVISIÓN INVÁLIDA
        #------------------------------------------------

        if (n - p - 1) <= 0:

            r2_adj = np.nan

        else:

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

            "Parametros":

                p,

            "R_square":

                round(r2, redondeo),

            "R_square_ajustado":

                (
                    round(r2_adj, redondeo)

                    if not np.isnan(r2_adj)

                    else np.nan
                ),

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

    #----------------------------------------------------
    # Retorno
    #----------------------------------------------------

    return resultados 

#========================================================
# FUNCIÓN
# f_visualizar_RMSE()
# PYTHON
#
# ACEPTA:
# - statsmodels
# - SVR
# - Ridge / Lasso
# - múltiples modelos
# - GRID AUTOMÁTICO
#========================================================

#========================================================
# FUNCIÓN
# f_visualizar_RMSE()
# CORREGIDA
#
# Compatible con:
# - statsmodels
# - SVR
# - sklearn
#

def f_visualizar_RMSE(

    modelos,

    datos,

    variable_dependiente,

    nombres_modelos = None,

    ncol = 3,

    figsize = (18, 10)
):

    #----------------------------------------------------
    # LIBRERÍAS
    #----------------------------------------------------

    import numpy as np

    import pandas as pd

    import matplotlib.pyplot as plt

    import statsmodels.api as sm

    from sklearn.metrics import (

        mean_squared_error,

        r2_score
    )

    #----------------------------------------------------
    # CONVERTIR A LISTA
    #----------------------------------------------------

    if not isinstance(modelos, list):

        modelos = [modelos]

    #----------------------------------------------------
    # NOMBRES
    #----------------------------------------------------

    if nombres_modelos is None:

        nombres_modelos = [

            f"Modelo_{i+1}"

            for i in range(len(modelos))
        ]

    #----------------------------------------------------
    # VARIABLES
    #----------------------------------------------------

    y_real = np.array(

        datos[
            variable_dependiente
        ]

    ).flatten()

    X_val = datos.drop(

        columns = [variable_dependiente]
    )

    #----------------------------------------------------
    # GRID
    #----------------------------------------------------

    n_modelos = len(modelos)

    nrow = int(

        np.ceil(n_modelos / ncol)
    )

    fig, axes = plt.subplots(

        nrow,

        ncol,

        figsize = figsize
    )

    #----------------------------------------------------
    # AJUSTE EJES
    #----------------------------------------------------

    axes = np.array(axes).reshape(-1)

    #====================================================
    # RECORRER MODELOS
    #====================================================

    for i, modelo in enumerate(modelos):

        ax = axes[i]

        nombre_modelo = nombres_modelos[i]

        #================================================
        # IDENTIFICAR TIPO
        #================================================

        nombre_clase = type(modelo).__name__

        #================================================
        # MODELOS SVR
        #================================================

        if "SVR" in nombre_clase:

            #--------------------------------------------
            # Reordenar columnas
            #--------------------------------------------

            if hasattr(modelo, "feature_names_in_"):

                X_pred = X_val.reindex(

                    columns = modelo.feature_names_in_,

                    fill_value = 0
                )

            else:

                X_pred = X_val.copy()

            #--------------------------------------------
            # Predicción
            #--------------------------------------------

            pred = np.array(

                modelo.predict(X_pred)

            ).flatten()

        #================================================
        # MODELOS statsmodels
        #================================================

        elif hasattr(modelo, "model"):

            #--------------------------------------------
            # Modelo polinomial
            #--------------------------------------------

            if hasattr(modelo, "poly_transformador"):

                X_base = X_val[
                    
                    modelo.nombres_originales
                ]

                X_poly = (

                    modelo.poly_transformador
                    .transform(X_base)
                )

                nombres_poly = (

                    modelo.poly_transformador
                    .get_feature_names_out(

                        modelo.nombres_originales
                    )
                )

                X_pred = pd.DataFrame(

                    X_poly,

                    columns = nombres_poly,

                    index = X_base.index
                )

            else:

                X_pred = X_val.copy()

            #--------------------------------------------
            # Agregar constante
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
            # Predicción
            #--------------------------------------------

            pred = np.array(

                modelo.predict(X_pred)

            ).flatten()

        #================================================
        # MODELOS sklearn
        #================================================

        else:

            pred = np.array(

                modelo.predict(X_val)

            ).flatten()

        #================================================
        # MÉTRICAS
        #================================================

        rmse = np.sqrt(

            mean_squared_error(

                y_real,

                pred
            )
        )

        r2 = r2_score(

            y_real,

            pred
        )

        #================================================
        # EJE X ORDENADO
        #================================================

        orden = np.arange(

            len(y_real)
        )

        #================================================
        # VISUALIZACIÓN
        #================================================

        #------------------------------------------------
        # REALES
        #------------------------------------------------

        ax.plot(

            orden,

            y_real,

            color = "yellow",

            linewidth = 1,

            label = "Valores reales"
        )

        ax.scatter(

            orden,

            y_real,

            color = "yellow",

            s = 10
        )

        #------------------------------------------------
        # PREDICCIONES
        #------------------------------------------------

        ax.plot(

            orden,

            pred,

            color = "blue",

            linewidth = 1,

            label = "Predicciones"
        )

        ax.scatter(

            orden,

            pred,

            color = "blue",

            s = 10
        )

        #------------------------------------------------
        # TÍTULOS
        #------------------------------------------------

        ax.set_title(

            f"{nombre_modelo}\nRMSE = {rmse:.4f} | R² = {r2:.4f}",

            fontsize = 11
        )

        ax.set_xlabel(

            "Observación"
        )

        ax.set_ylabel(

            variable_dependiente
        )

        ax.legend()

    #----------------------------------------------------
    # ELIMINAR EJES SOBRANTES
    #----------------------------------------------------

    for j in range(

        i + 1,

        len(axes)
    ):

        fig.delaxes(

            axes[j]
        )

    #----------------------------------------------------
    # AJUSTE FINAL
    #----------------------------------------------------

    plt.tight_layout()

    plt.show()