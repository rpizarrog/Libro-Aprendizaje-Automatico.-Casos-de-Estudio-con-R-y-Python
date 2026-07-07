# Funciones para implementar y evaluar modelos KNN ADA BOOST Y BAYES 

# Permite construir modelos de clasificación
# SVM varios kermels
# Bosques aleatorios
# Arboles
# de regresión logística Python

# Rubén Pizarro Gurrola
# Junio 2026

#========================================================
# CARGAR DATOS
#========================================================

def f_cargar_datos(ruta_archivo):

    datos = pd.read_csv(ruta_archivo)

    return datos

#=========================================================
# FUNCIÓN
# f_redondear()
#
# ACEPTA:
# - datos (DataFrame)
#
# DEVUELVE:
# - DataFrame con las variables numéricas
#   redondeadas a dos decimales
#=========================================================

import pandas as pd
import numpy as np

def f_redondear(datos):
    """
    Redondea a dos posiciones decimales únicamente
    las variables numéricas de un DataFrame.

    Parámetros
    ----------
    datos : pandas.DataFrame

    Devuelve
    --------
    pandas.DataFrame
    """

    #---------------------------------------------
    # VALIDACIONES
    #---------------------------------------------

    if not isinstance(datos, pd.DataFrame):
        raise TypeError(
            "datos debe ser un DataFrame de pandas."
        )

    #---------------------------------------------
    # COPIA
    #---------------------------------------------

    datos_redondeados = datos.copy()

    #---------------------------------------------
    # VARIABLES NUMÉRICAS
    #---------------------------------------------

    columnas_numericas = datos_redondeados.select_dtypes(
        include=[np.number]
    ).columns

    #---------------------------------------------
    # REDONDEAR
    #---------------------------------------------

    datos_redondeados[columnas_numericas] = (
        datos_redondeados[columnas_numericas]
        .round(2)
    )

    return datos_redondeados

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

#========================================================
# FUNCIÓN
# f_describir_datos()
#========================================================

def f_describir_datos(datos):

    import pandas as pd

    #----------------------------------------------------
    # ESTRUCTURA
    #----------------------------------------------------

    estructura = datos.dtypes

    #----------------------------------------------------
    # VARIABLES NUMÉRICAS
    #----------------------------------------------------

    variables_numericas = datos.select_dtypes(
        include=["number"]
    )

    describe_numericas = None

    if variables_numericas.shape[1] > 0:

        describe_numericas = (
            variables_numericas
            .describe()
            .T
            .round(4)
        )

    #----------------------------------------------------
    # VARIABLES CATEGÓRICAS
    #----------------------------------------------------

    variables_categoricas = datos.select_dtypes(
        include=["object", "category"]
    )

    frecuencias = {}

    for variable in variables_categoricas.columns:

        tabla = pd.DataFrame({

            "Frecuencia":
                datos[variable]
                .value_counts(),

            "Porcentaje":
                round(
                    datos[variable]
                    .value_counts(normalize=True)
                    * 100,
                    2
                )

        })

        frecuencias[variable] = tabla

    #----------------------------------------------------
    # RESULTADO
    #----------------------------------------------------

    return {
        "describe": describe_numericas,
        "frecuencias": frecuencias,
        "structure": estructura
    }



def f_convertir_categorias(datos):
    
    datos = datos.copy()

    if "felicidad" in datos.columns:

        datos["felicidad"] = (
            datos["felicidad"]
            .replace({
                0:"Baja",
                1:"Media",
                2:"Alta"
            })
        )

    return datos

#=========================================================
# FUNCIÓN
# f_frecuencias_clases()
#=========================================================

def f_frecuencias_clases(
        datos,
        ncols = 1,
        figsize = (18, 12),
        hspace = 0.60,
        wspace = 0.30):

    """
    Genera diagramas de barras para todas las variables
    categóricas (object, category y bool).

    Argumentos:
    ------------------------------------------------------
    datos    : DataFrame
    ncols    : Número de columnas del grid
    figsize  : Tamaño de la figura
    hspace   : Espacio vertical entre filas
    wspace   : Espacio horizontal entre columnas
    """

    #-----------------------------------------------------
    # LIBRERÍAS
    #-----------------------------------------------------

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    #-----------------------------------------------------
    # VARIABLES CATEGÓRICAS
    #-----------------------------------------------------

    variables = datos.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    if len(variables) == 0:

        print(
            "No existen variables categóricas en el conjunto de datos."
        )

        return

    #-----------------------------------------------------
    # GRID
    #-----------------------------------------------------

    nvars = len(variables)

    nrows = int(
        np.ceil(
            nvars / ncols
        )
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize = figsize
    )

    #-----------------------------------------------------
    # ASEGURAR VECTOR DE EJES
    #-----------------------------------------------------

    if nvars == 1:

        axes = np.array([axes])

    else:

        axes = np.array(axes).reshape(-1)

    #-----------------------------------------------------
    # GRÁFICOS
    #-----------------------------------------------------

    for i, variable in enumerate(variables):

        frecuencias = (
            datos[variable]
            .value_counts(dropna = False)
        )

        frecuencias.plot(
            kind = "bar",
            ax = axes[i]
        )

        #-------------------------------------------------
        # TÍTULO
        #-------------------------------------------------

        axes[i].set_title(
            variable,
            fontsize = 11,
            pad = 12
        )

        axes[i].set_xlabel("")

        axes[i].set_ylabel(
            "Frecuencia"
        )

        #-------------------------------------------------
        # ROTACIÓN ETIQUETAS
        #-------------------------------------------------

        axes[i].tick_params(
            axis = "x",
            rotation = 45,
            labelsize = 8
        )

        #-------------------------------------------------
        # ETIQUETAS SOBRE BARRAS
        #-------------------------------------------------

        for barra in axes[i].patches:

            altura = barra.get_height()

            axes[i].annotate(

                f"{int(altura):,}",

                (
                    barra.get_x() +
                    barra.get_width()/2,
                    altura
                ),

                ha = "center",

                va = "bottom",

                fontsize = 8

            )

    #-----------------------------------------------------
    # ELIMINAR EJES SOBRANTES
    #-----------------------------------------------------

    for j in range(
        len(variables),
        len(axes)
    ):

        fig.delaxes(
            axes[j]
        )

    #-----------------------------------------------------
    # ESPACIADO
    #-----------------------------------------------------

    plt.subplots_adjust(

        hspace = hspace,

        wspace = wspace

    )

    plt.tight_layout(
        pad = 2.5
    )

    plt.show()


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

#=========================================================
# FUNCIÓN
# f_crear_modelo_KNN()
#=========================================================

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

def f_crear_modelo_KNN(
    datos_entrenamiento,
    variable_dependiente,
    k=5,
    escalar=False
):
    """
    Construye un modelo de clasificación KNN.

    Parámetros
    ----------
    datos_entrenamiento : DataFrame
    variable_dependiente : str
    k : int
    escalar : bool

    Devuelve
    --------
    modelo : Pipeline o KNeighborsClassifier
    """

    #-----------------------------------------------------
    # VALIDACIONES
    #-----------------------------------------------------

    if not isinstance(datos_entrenamiento, pd.DataFrame):

        raise TypeError(
            "datos_entrenamiento debe ser un DataFrame."
        )

    if variable_dependiente not in datos_entrenamiento.columns:

        raise ValueError(
            f"La variable '{variable_dependiente}' no existe."
        )

    if k <= 0:

        raise ValueError(
            "k debe ser mayor que cero."
        )

    #-----------------------------------------------------
    # VARIABLES
    #-----------------------------------------------------

    X = datos_entrenamiento.drop(
        columns=[variable_dependiente]
    )

    y = datos_entrenamiento[
        variable_dependiente
    ]

    #-----------------------------------------------------
    # VALIDAR VARIABLES NUMÉRICAS
    #-----------------------------------------------------

    if not all(
        X.dtypes.apply(
            lambda x: x.kind in "if"
        )
    ):

        raise ValueError(

            "Todas las variables predictoras deben ser numéricas."

        )

    #-----------------------------------------------------
    # MODELO
    #-----------------------------------------------------

    if escalar:

        modelo = Pipeline([

            ("escalador",
             StandardScaler()),

            ("knn",
             KNeighborsClassifier(
                 n_neighbors=k
             ))

        ])

    else:

        modelo = KNeighborsClassifier(
            n_neighbors=k
        )

    #-----------------------------------------------------
    # ENTRENAMIENTO
    #-----------------------------------------------------

    modelo.fit(X, y)

    #-----------------------------------------------------
    # INFORMACIÓN
    #-----------------------------------------------------

    print()

    print("---------------------------------------")

    print("Modelo KNN construido correctamente")

    print("---------------------------------------")

    print(f"Observaciones : {X.shape[0]}")

    print(f"Variables     : {X.shape[1]}")

    print(f"k             : {k}")

    print(f"Escalamiento  : {escalar}")

    print("---------------------------------------")

    print()

    return modelo

#=========================================================
# FUNCIÓN
# f_crear_modelo_Ada_Boost()
#=========================================================

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

def f_crear_modelo_Ada_Boost(
    datos_entrenamiento,
    variable_dependiente,
    n_estimators=50,
    learning_rate=1.0,
    random_state=123
):
    """
    Construye un modelo de clasificación AdaBoost.

    Parámetros
    ----------
    datos_entrenamiento : DataFrame
    variable_dependiente : str
    n_estimators : int
    learning_rate : float
    random_state : int

    Devuelve
    --------
    modelo : AdaBoostClassifier
    """

    #-----------------------------------------------------
    # VALIDACIONES
    #-----------------------------------------------------

    if not isinstance(datos_entrenamiento, pd.DataFrame):

        raise TypeError(
            "datos_entrenamiento debe ser un DataFrame."
        )

    if variable_dependiente not in datos_entrenamiento.columns:

        raise ValueError(
            f"La variable '{variable_dependiente}' no existe."
        )

    if n_estimators <= 0:

        raise ValueError(
            "n_estimators debe ser mayor que cero."
        )

    #-----------------------------------------------------
    # VARIABLES
    #-----------------------------------------------------

    X = datos_entrenamiento.drop(
        columns=[variable_dependiente]
    )

    y = datos_entrenamiento[
        variable_dependiente
    ]

    #-----------------------------------------------------
    # VALIDAR VARIABLES NUMÉRICAS
    #-----------------------------------------------------

    if not all(
        X.dtypes.apply(
            lambda x: x.kind in "if"
        )
    ):

        raise ValueError(
            "Todas las variables predictoras deben ser numéricas."
        )

    #-----------------------------------------------------
    # CLASIFICADOR BASE
    #-----------------------------------------------------

    clasificador_base = DecisionTreeClassifier(
        max_depth=1,
        random_state=random_state
    )

    #-----------------------------------------------------
    # MODELO
    #-----------------------------------------------------

    modelo = AdaBoostClassifier(

        estimator=clasificador_base,

        n_estimators=n_estimators,

        learning_rate=learning_rate,

        random_state=random_state

    )

    #-----------------------------------------------------
    # ENTRENAMIENTO
    #-----------------------------------------------------

    modelo.fit(X, y)

    #-----------------------------------------------------
    # INFORMACIÓN
    #-----------------------------------------------------

    print()

    print("---------------------------------------")

    print("Modelo AdaBoost construido correctamente")

    print("---------------------------------------")

    print(f"Observaciones : {X.shape[0]}")

    print(f"Variables     : {X.shape[1]}")

    print(f"Clasificadores: {n_estimators}")

    print(f"Learning Rate : {learning_rate}")

    print(f"Random State  : {random_state}")

    print("---------------------------------------")

    print()

    return modelo


#=========================================================
# FUNCIÓN
# f_crear_modelo_Bayes()
#=========================================================

from sklearn.naive_bayes import GaussianNB
import pandas as pd

def f_crear_modelo_Bayes(
    datos_entrenamiento,
    variable_dependiente
):
    """
    Construye un modelo de clasificación
    Naïve Bayes Gaussiano.

    Parámetros
    ----------
    datos_entrenamiento : DataFrame
    variable_dependiente : str

    Devuelve
    --------
    modelo : GaussianNB
    """

    #-----------------------------------------------------
    # VALIDACIONES
    #-----------------------------------------------------

    if not isinstance(datos_entrenamiento, pd.DataFrame):

        raise TypeError(
            "datos_entrenamiento debe ser un DataFrame."
        )

    if variable_dependiente not in datos_entrenamiento.columns:

        raise ValueError(
            f"La variable '{variable_dependiente}' no existe."
        )

    #-----------------------------------------------------
    # VARIABLES
    #-----------------------------------------------------

    X = datos_entrenamiento.drop(
        columns=[variable_dependiente]
    )

    y = datos_entrenamiento[
        variable_dependiente
    ]

    #-----------------------------------------------------
    # VALIDAR VARIABLES NUMÉRICAS
    #-----------------------------------------------------

    if not all(
        X.dtypes.apply(
            lambda x: x.kind in "if"
        )
    ):

        raise ValueError(
            "GaussianNB requiere variables predictoras numéricas."
        )

    #-----------------------------------------------------
    # MODELO
    #-----------------------------------------------------

    modelo = GaussianNB()

    #-----------------------------------------------------
    # ENTRENAMIENTO
    #-----------------------------------------------------

    modelo.fit(X, y)

    #-----------------------------------------------------
    # INFORMACIÓN
    #-----------------------------------------------------

    print()

    print("---------------------------------------")

    print("Modelo Naïve Bayes construido correctamente")

    print("---------------------------------------")

    print(f"Observaciones : {X.shape[0]}")

    print(f"Variables     : {X.shape[1]}")

    print(f"Clases        : {len(pd.unique(y))}")

    print("---------------------------------------")

    print()

    return modelo

#=========================================================
# FUNCIÓN
# f_crear_modelo_SVM_lineal()
#
# ACEPTA:
# - datos
# - variable_dependiente
# - C = 0.1
#
# DEVUELVE:
# - modelo SVM Lineal
#=========================================================

import pandas as pd

from sklearn.svm import SVC


#=========================================================
# FUNCIÓN
# f_crear_modelo_SVM_lineal()
#
# ACEPTA:
# - datos
# - variable_dependiente
# - C = 0.1
#
# DEVUELVE:
# - modelo SVM Lineal
#=========================================================

import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC


def f_crear_modelo_SVM_lineal(
        datos,
        variable_dependiente,
        C=0.1):
    """
    Construye un modelo SVM con kernel lineal.

    Parámetros
    ----------
    datos : pandas.DataFrame

    variable_dependiente : str

    C : float, default=0.1

    Devuelve
    --------
    modelo : LinearSVC
    """

    #------------------------------------------------------
    # VALIDACIONES
    #------------------------------------------------------

    if not isinstance(datos, pd.DataFrame):

        raise TypeError(
            "datos debe ser un DataFrame de pandas."
        )

    if variable_dependiente not in datos.columns:

        raise ValueError(
            f"La variable '{variable_dependiente}' "
            "no existe en los datos."
        )

    #------------------------------------------------------
    # VARIABLES INDEPENDIENTES
    #------------------------------------------------------

    X = datos.drop(
        columns=[variable_dependiente]
    )

    y = datos[
        variable_dependiente
    ]

    #------------------------------------------------------
    # VALIDAR VARIABLES NUMÉRICAS
    #------------------------------------------------------

    if not np.all(
        [np.issubdtype(t, np.number)
         for t in X.dtypes]
    ):

        raise ValueError(
            "Todas las variables independientes "
            "deben ser numéricas."
        )

    #------------------------------------------------------
    # MODELO
    #------------------------------------------------------

    modelo_SVM = LinearSVC(

        C=C,

        dual="auto",

        random_state=123,

        max_iter=10000

    )

    modelo_SVM.fit(X, y)

    #------------------------------------------------------
    # INFORMACIÓN
    #------------------------------------------------------

    print("\n======================================")
    print("MODELO SVM LINEAL")
    print("======================================")
    print(f"Kernel              : Lineal")
    print(f"Implementación      : LinearSVC (LIBLINEAR)")
    print(f"Cost (C)            : {C}")
    print(f"Número de clases    : {len(modelo_SVM.classes_)}")
    print(f"Iteraciones         : {modelo_SVM.n_iter_}")
    print("======================================\n")

    return modelo_SVM

#=========================================================
# FUNCIÓN
# f_crear_modelo_SVM_polinomial()
#
# ACEPTA:
# - datos
# - variable_dependiente
# - grado = 2
# - C = 0.1
# - coef0 = 1
#
# DEVUELVE:
# - modelo SVM Polinomial
#=========================================================

import pandas as pd
import numpy as np

from sklearn.svm import SVC


def f_crear_modelo_SVM_polinomial(
        datos,
        variable_dependiente,
        grado=2,
        C=0.1,
        coef0=1):

    """
    Construye un modelo SVM con kernel polinomial.

    Parámetros
    ----------
    datos : pandas.DataFrame

    variable_dependiente : str

    grado : int, default=2

    C : float, default=0.1

    coef0 : float, default=1

    Devuelve
    --------
    modelo : sklearn.svm.SVC
    """

    #------------------------------------------------------
    # VALIDACIONES
    #------------------------------------------------------

    if not isinstance(datos, pd.DataFrame):

        raise TypeError(
            "datos debe ser un DataFrame de pandas."
        )

    if variable_dependiente not in datos.columns:

        raise ValueError(
            f"La variable '{variable_dependiente}' "
            "no existe en los datos."
        )

    #------------------------------------------------------
    # VARIABLES
    #------------------------------------------------------

    X = datos.drop(
        columns=[variable_dependiente]
    )

    y = datos[
        variable_dependiente
    ]

    #------------------------------------------------------
    # VALIDAR VARIABLES NUMÉRICAS
    #------------------------------------------------------

    if not np.all(
        [np.issubdtype(t, np.number)
         for t in X.dtypes]
    ):

        raise ValueError(
            "Todas las variables independientes "
            "deben ser numéricas."
        )

    #------------------------------------------------------
    # MODELO
    #------------------------------------------------------

    modelo_SVM = SVC(

        kernel="poly",

        degree=grado,

        C=C,

        coef0=coef0,

        gamma="scale",

        probability=True,

        random_state=123

    )

    modelo_SVM.fit(X, y)

    #------------------------------------------------------
    # INFORMACIÓN
    #------------------------------------------------------

    print("\n========================================")
    print("MODELO SVM POLINOMIAL")
    print("========================================")
    print(f"Kernel               : Polinomial")
    print(f"Grado                : {grado}")
    print(f"Cost (C)             : {C}")
    print(f"Coef0                : {coef0}")
    print(f"Gamma                : scale")
    print(f"Número de clases     : {len(modelo_SVM.classes_)}")
    print(f"Vectores de soporte  : {modelo_SVM.support_vectors_.shape[0]}")
    print("========================================\n")

    return modelo_SVM

#=========================================================
# FUNCIÓN
# f_crear_modelo_SVM_radial()
#
# ACEPTA:
# - datos
# - variable_dependiente
# - C = 0.1
# - gamma = "scale"
#
# DEVUELVE:
# - modelo SVM Radial
#=========================================================

import pandas as pd
import numpy as np

from sklearn.svm import SVC


def f_crear_modelo_SVM_radial(
        datos,
        variable_dependiente,
        C=0.1,
        gamma="scale"):

    """
    Construye un modelo SVM con kernel radial (RBF).

    Parámetros
    ----------
    datos : pandas.DataFrame

    variable_dependiente : str

    C : float, default=0.1

    gamma : {"scale","auto"} o float,
            default="scale"

    Devuelve
    --------
    modelo : sklearn.svm.SVC
    """

    #------------------------------------------------------
    # VALIDACIONES
    #------------------------------------------------------

    if not isinstance(datos, pd.DataFrame):

        raise TypeError(
            "datos debe ser un DataFrame de pandas."
        )

    if variable_dependiente not in datos.columns:

        raise ValueError(
            f"La variable '{variable_dependiente}' "
            "no existe en los datos."
        )

    #------------------------------------------------------
    # VARIABLES
    #------------------------------------------------------

    X = datos.drop(
        columns=[variable_dependiente]
    )

    y = datos[
        variable_dependiente
    ]

    #------------------------------------------------------
    # VALIDAR VARIABLES NUMÉRICAS
    #------------------------------------------------------

    if not np.all(
        [np.issubdtype(t, np.number)
         for t in X.dtypes]
    ):

        raise ValueError(
            "Todas las variables independientes "
            "deben ser numéricas."
        )

    #------------------------------------------------------
    # MODELO
    #------------------------------------------------------

    modelo_SVM = SVC(

        kernel="rbf",

        C=C,

        gamma=gamma,

        probability=True,

        random_state=123

    )

    modelo_SVM.fit(X, y)

    #------------------------------------------------------
    # INFORMACIÓN
    #------------------------------------------------------

    print("\n========================================")
    print("MODELO SVM RADIAL")
    print("========================================")
    print(f"Kernel               : Radial (RBF)")
    print(f"Cost (C)             : {C}")
    print(f"Gamma                : {gamma}")
    print(f"Número de clases     : {len(modelo_SVM.classes_)}")
    print(f"Vectores de soporte  : {modelo_SVM.support_vectors_.shape[0]}")
    print("========================================\n")

    return modelo_SVM

#=========================================================
# FUNCIÓN
# f_convertir_dummys()
#=========================================================

def f_convertir_dummys(datos):

    """
    Convierte automáticamente todas las variables
    categóricas (object, category y bool)
    en variables dummy.

    Argumentos
    ----------
    datos : DataFrame

    Retorna
    -------
    DataFrame con variables dummy
    """

    import pandas as pd

    #-----------------------------------------------------
    # COPIA
    #-----------------------------------------------------

    datos_dummys = datos.copy()

    #-----------------------------------------------------
    # VARIABLES CATEGÓRICAS
    #-----------------------------------------------------

    variables_categoricas = (
        datos_dummys
        .select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        )
        .columns
    )

    #-----------------------------------------------------
    # DUMMIES
    #-----------------------------------------------------

    datos_dummys = pd.get_dummies(

        datos_dummys,

        columns = variables_categoricas,

        drop_first = True,

        dtype = int

    )

    #-----------------------------------------------------
    # INFORMACIÓN
    #-----------------------------------------------------

    print()

    print("="*40)

    print(" CONVERSIÓN A VARIABLES DUMMY ")

    print("="*40)

    print(
        "Variables originales :",
        datos.shape[1]
    )

    print(
        "Variables finales    :",
        datos_dummys.shape[1]
    )

    print(
        "Observaciones        :",
        datos_dummys.shape[0]
    )

    print("="*40)

    return datos_dummys

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

#=========================================================
# FUNCIÓN
# f_construir_arbol_clasificacion()
#=========================================================

def f_construir_arbol_clasificacion(
        datos,
        variable_dependiente,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=123):

    #-----------------------------------------------------
    # LIBRERÍAS
    #-----------------------------------------------------

    import pandas as pd

    from sklearn.tree import DecisionTreeClassifier

    #-----------------------------------------------------
    # VALIDACIONES
    #-----------------------------------------------------

    if variable_dependiente not in datos.columns:

        raise ValueError(
            f"La variable '{variable_dependiente}' no existe."
        )

    #-----------------------------------------------------
    # X e y
    #-----------------------------------------------------

    X = datos.drop(
        columns=[variable_dependiente]
    )

    y = datos[
        variable_dependiente
    ]

    #-----------------------------------------------------
    # MODELO
    #-----------------------------------------------------

    modelo = DecisionTreeClassifier(

        criterion=criterion,

        max_depth=max_depth,

        min_samples_split=min_samples_split,

        min_samples_leaf=min_samples_leaf,

        random_state=random_state
    )

    modelo.fit(X, y)

    #-----------------------------------------------------
    # METADATOS
    #-----------------------------------------------------

    modelo.variable_dependiente = variable_dependiente

    modelo.n_clases = y.nunique()

    modelo.clases = list(y.unique())

    modelo.frecuencias_clases = (
        y.value_counts()
        .sort_index()
        .to_dict()
    )

    modelo.n_variables = X.shape[1]

    #-----------------------------------------------------
    # RESUMEN
    #-----------------------------------------------------

    print()
    print("====================================")
    print(" ÁRBOL DE CLASIFICACIÓN")
    print("====================================")
    print("Variable objetivo :", variable_dependiente)
    print("Número clases     :", modelo.n_clases)
    print("Variables         :", modelo.n_variables)
    print("Criterio          :", criterion)
    print("Observaciones     :", len(datos))
    print()

    print("Frecuencia de clases")

    for k, v in modelo.frecuencias_clases.items():

        print(f"{k}: {v}")

    print("====================================")

    return modelo


#=========================================================
# FUNCIÓN
# f_visualizar_arbol()
#=========================================================

def f_visualizar_arbol(
        modelo,
        figsize=(22,12),
        profunidad=3,
        fontsize=9):

    #-----------------------------------------------------
    # LIBRERÍAS
    #-----------------------------------------------------

    import matplotlib.pyplot as plt

    from sklearn.tree import plot_tree

    from sklearn.tree import DecisionTreeClassifier

    #-----------------------------------------------------
    # VALIDACIÓN
    #-----------------------------------------------------

    if not isinstance(
        modelo,
        DecisionTreeClassifier
    ):

        raise ValueError(
            "El modelo debe ser DecisionTreeClassifier."
        )

    #-----------------------------------------------------
    # FIGURA
    #-----------------------------------------------------

    plt.figure(
        figsize=figsize
    )

    plot_tree(

        modelo,

        filled=True,

        rounded=True,

        feature_names=modelo.feature_names_in_,

        class_names=[
            str(x)
            for x in modelo.classes_
        ],

        max_depth=profunidad,

        fontsize=fontsize
    )

    plt.title(
        "Árbol de Clasificación"
    )

    plt.show()


#=========================================================
# FUNCIÓN
# f_variables_importantes()
#=========================================================

def f_variables_importantes(
        modelos,
        nombres_modelos=None,
        top=10,
        ncols=2,
        figsize=(14,8)):

    #-----------------------------------------------------
    # LIBRERÍAS
    #-----------------------------------------------------

    import pandas as pd
    import matplotlib.pyplot as plt
    import math

    #-----------------------------------------------------
    # CONVERTIR A LISTA
    #-----------------------------------------------------

    if not isinstance(modelos, list):

        modelos = [modelos]

    #-----------------------------------------------------
    # NOMBRES
    #-----------------------------------------------------

    if nombres_modelos is None:

        nombres_modelos = [

            f"Modelo {i+1}"

            for i in range(
                len(modelos)
            )
        ]

    #-----------------------------------------------------
    # GRID
    #-----------------------------------------------------

    n_modelos = len(modelos)

    nrows = math.ceil(
        n_modelos / ncols
    )

    fig, axes = plt.subplots(

        nrows=nrows,

        ncols=ncols,

        figsize=figsize

    )

    #-----------------------------------------------------
    # AJUSTAR AXES
    #-----------------------------------------------------

    if n_modelos == 1:

        axes = [axes]

    else:

        axes = axes.flatten()

    #-----------------------------------------------------
    # TABLA GLOBAL
    #-----------------------------------------------------

    tabla_global = []

    #-----------------------------------------------------
    # RECORRER MODELOS
    #-----------------------------------------------------

    for i, modelo in enumerate(modelos):

        importancia = pd.DataFrame({

            "Variable":
                modelo.feature_names_in_,

            "Importancia":
                modelo.feature_importances_

        })

        #---------------------------------------------
        # PORCENTAJE
        #---------------------------------------------

        importancia["Importancia"] = (

            importancia["Importancia"]

            * 100

        )

        importancia = (

            importancia

            .sort_values(

                by="Importancia",

                ascending=False

            )

            .head(top)

            .reset_index(drop=True)

        )

        #---------------------------------------------
        # REDONDEAR
        #---------------------------------------------

        importancia["Importancia"] = (

            importancia["Importancia"]

            .round(2)

        )

        importancia["Modelo"] = (

            nombres_modelos[i]

        )

        tabla_global.append(
            importancia
        )

        #---------------------------------------------
        # GRÁFICA
        #---------------------------------------------

        ax = axes[i]

        ax.barh(

            importancia["Variable"],

            importancia["Importancia"]

        )

        #---------------------------------------------
        # ETIQUETAS
        #---------------------------------------------

        for j, valor in enumerate(

            importancia["Importancia"]

        ):

            ax.text(

                valor,

                j,

                f"{valor:.2f}",

                va="center"

            )

        ax.invert_yaxis()

        ax.set_title(

            nombres_modelos[i]

        )

        ax.set_xlabel(

            "Importancia (%)"

        )

        ax.set_ylabel("")

    #-----------------------------------------------------
    # ELIMINAR EJES VACÍOS
    #-----------------------------------------------------

    for j in range(

        n_modelos,

        len(axes)

    ):

        fig.delaxes(
            axes[j]
        )

    #-----------------------------------------------------
    # AJUSTES
    #-----------------------------------------------------

    plt.tight_layout()

    plt.show()

    #-----------------------------------------------------
    # TABLA CONSOLIDADA
    #-----------------------------------------------------

    tabla_global = pd.concat(

        tabla_global,

        ignore_index=True

    )

    return tabla_global


#=========================================================
# FUNCIÓN
# f_construir_random_forest()
#=========================================================

def f_construir_random_forest(
        datos,
        variable_dependiente,
        n_estimators=500,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=123):

    #-----------------------------------------------------
    # LIBRERÍAS
    #-----------------------------------------------------

    import pandas as pd

    from sklearn.ensemble import RandomForestClassifier

    #-----------------------------------------------------
    # VALIDACIONES
    #-----------------------------------------------------

    if variable_dependiente not in datos.columns:

        raise ValueError(
            f"La variable '{variable_dependiente}' no existe."
        )

    #-----------------------------------------------------
    # X e y
    #-----------------------------------------------------

    X = datos.drop(
        columns=[variable_dependiente]
    )

    y = datos[
        variable_dependiente
    ]

    #-----------------------------------------------------
    # MODELO
    #-----------------------------------------------------

    modelo = RandomForestClassifier(

        n_estimators=n_estimators,

        criterion=criterion,

        max_depth=max_depth,

        min_samples_split=min_samples_split,

        min_samples_leaf=min_samples_leaf,

        max_features=max_features,

        random_state=random_state,

        n_jobs=-1
    )

    modelo.fit(X, y)

    #-----------------------------------------------------
    # METADATOS
    #-----------------------------------------------------

    modelo.variable_dependiente = (
        variable_dependiente
    )

    modelo.n_clases = y.nunique()

    modelo.clases = list(
        y.unique()
    )

    modelo.frecuencias_clases = (
        y.value_counts()
         .sort_index()
         .to_dict()
    )

    modelo.n_variables = X.shape[1]

    #-----------------------------------------------------
    # RESUMEN
    #-----------------------------------------------------

    print()
    print("==================================================")
    print(" RANDOM FOREST")
    print("==================================================")

    print(
        f"Variable objetivo : {variable_dependiente}"
    )

    print(
        f"Número de clases  : {modelo.n_clases}"
    )

    print(
        f"Variables         : {modelo.n_variables}"
    )

    print(
        f"Árboles           : {n_estimators}"
    )

    print(
        f"Criterio          : {criterion}"
    )

    print(
        f"Observaciones     : {len(datos)}"
    )

    print()
    print("Frecuencias:")

    for clase, frecuencia in (
        modelo.frecuencias_clases.items()
    ):

        print(
            f"{clase}: {frecuencia}"
        )

    print("==================================================")

    return modelo




#=========================================================
# FUNCIÓN
# f_estandarizar_train_transf_valid()
#=========================================================

def f_estandarizar_train_transf_valid(
        datos_entrenamiento,
        datos_validacion,
        variable_dependiente):

    #-----------------------------------------------------
    # LIBRERÍAS
    #-----------------------------------------------------

    import pandas as pd

    from sklearn.preprocessing import StandardScaler

    #-----------------------------------------------------
    # COPIAS
    #-----------------------------------------------------

    entrenamiento = datos_entrenamiento.copy()

    validacion = datos_validacion.copy()

    #-----------------------------------------------------
    # X TRAIN
    #-----------------------------------------------------

    X_train = entrenamiento.drop(
        columns=[variable_dependiente]
    )

    y_train = entrenamiento[
        variable_dependiente
    ]

    #-----------------------------------------------------
    # X VALID
    #-----------------------------------------------------

    X_valid = validacion.drop(
        columns=[variable_dependiente]
    )

    y_valid = validacion[
        variable_dependiente
    ]

    #-----------------------------------------------------
    # ESCALADOR
    #-----------------------------------------------------

    scaler = StandardScaler()

    #-----------------------------------------------------
    # AJUSTAR SOLO TRAIN
    #-----------------------------------------------------

    X_train_std = scaler.fit_transform(
        X_train
    )

    #-----------------------------------------------------
    # TRANSFORMAR VALIDACIÓN
    #-----------------------------------------------------

    X_valid_std = scaler.transform(
        X_valid
    )

    #-----------------------------------------------------
    # DATAFRAMES
    #-----------------------------------------------------

    X_train_std = pd.DataFrame(
        X_train_std,
        columns=X_train.columns,
        index=X_train.index
    )

    X_valid_std = pd.DataFrame(
        X_valid_std,
        columns=X_valid.columns,
        index=X_valid.index
    )

    #-----------------------------------------------------
    # RECONSTRUIR DATASETS
    #-----------------------------------------------------

    datos_entrenamiento_std = pd.concat(
        [X_train_std, y_train],
        axis=1
    )

    datos_validacion_std = pd.concat(
        [X_valid_std, y_valid],
        axis=1
    )

    #-----------------------------------------------------
    # RESULTADO
    #-----------------------------------------------------

    return {

        "datos_entrenamiento":
            datos_entrenamiento_std,

        "datos_validacion":
            datos_validacion_std,

        "scaler":
            scaler

    }

#=========================================================
# FUNCIÓN
# f_crear_modelo_regresion_logistica()
#=========================================================

def f_crear_modelo_regresion_logistica(
        datos,
        variable_dependiente,
        tipo="binomial",
        balanceo="ninguno",
        semilla=123):

    #-----------------------------------------------------
    # LIBRERÍAS
    #-----------------------------------------------------

    import pandas as pd
    import numpy as np

    from sklearn.linear_model import LogisticRegression

    from sklearn.utils import resample

    #-----------------------------------------------------
    # VALIDACIONES
    #-----------------------------------------------------

    if variable_dependiente not in datos.columns:

        raise ValueError(
            f"La variable '{variable_dependiente}' no existe."
        )

    #-----------------------------------------------------
    # COPIA
    #-----------------------------------------------------

    np.random.seed(semilla)

    datos = datos.copy()

    datos_originales = datos.copy()

    #-----------------------------------------------------
    # VARIABLE DEPENDIENTE
    #-----------------------------------------------------

    y = datos[variable_dependiente]

    X = datos.drop(
        columns=[variable_dependiente]
    )

    #-----------------------------------------------------
    # FRECUENCIAS ORIGINALES
    #-----------------------------------------------------

    frecuencias_originales = (
        y
        .value_counts()
        .to_dict()
    )

    porcentajes_originales = (
        y
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .to_dict()
    )

    #-----------------------------------------------------
    # NÚMERO DE CLASES
    #-----------------------------------------------------

    n_clases = y.nunique()

    #-----------------------------------------------------
    # PONDERACIONES
    #-----------------------------------------------------

    class_weight = None

    #-----------------------------------------------------
    # BALANCEO
    #-----------------------------------------------------

    if balanceo.lower() == "undersampling":

        frecuencia_min = y.value_counts().min()

        datos_balanceados = []

        for clase in y.unique():

            temp = datos[
                datos[variable_dependiente] == clase
            ]

            temp = resample(
                temp,
                replace=False,
                n_samples=frecuencia_min,
                random_state=semilla
            )

            datos_balanceados.append(temp)

        datos = pd.concat(
            datos_balanceados,
            axis=0
        )

        datos = datos.sample(
            frac=1,
            random_state=semilla
        )

        y = datos[variable_dependiente]

        X = datos.drop(
            columns=[variable_dependiente]
        )

    #-----------------------------------------------------

    elif balanceo.lower() == "oversampling":

        frecuencia_max = y.value_counts().max()

        datos_balanceados = []

        for clase in y.unique():

            temp = datos[
                datos[variable_dependiente] == clase
            ]

            temp = resample(
                temp,
                replace=True,
                n_samples=frecuencia_max,
                random_state=semilla
            )

            datos_balanceados.append(temp)

        datos = pd.concat(
            datos_balanceados,
            axis=0
        )

        datos = datos.sample(
            frac=1,
            random_state=semilla
        )

        y = datos[variable_dependiente]

        X = datos.drop(
            columns=[variable_dependiente]
        )

    #-----------------------------------------------------

    elif balanceo.lower() == "smote":

        from imblearn.over_sampling import SMOTE

        smote = SMOTE(
            random_state=semilla
        )

        X, y = smote.fit_resample(
            X,
            y
        )

    #-----------------------------------------------------

    elif balanceo.lower() == "ponderacion":

        class_weight = "balanced"

    #-----------------------------------------------------

    elif balanceo.lower() == "ninguno":

        pass

    #-----------------------------------------------------

    else:

        raise ValueError(
            "balanceo debe ser: "
            "'ninguno', "
            "'undersampling', "
            "'oversampling', "
            "'SMOTE' o "
            "'ponderacion'"
        )

    #-----------------------------------------------------
    # FRECUENCIAS ENTRENAMIENTO
    #-----------------------------------------------------

    frecuencias_entrenamiento = (
        pd.Series(y)
        .value_counts()
        .to_dict()
    )

    porcentajes_entrenamiento = (
        pd.Series(y)
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .to_dict()
    )

    #-----------------------------------------------------
    # MODELO BINOMIAL
    #-----------------------------------------------------

    if tipo.lower() == "binomial":

        if n_clases != 2:

            raise ValueError(
                "La regresión logística binomial "
                "requiere exactamente 2 clases."
            )

        modelo = LogisticRegression(

            class_weight=class_weight,

            max_iter=5000,

            random_state=semilla

        )

    #-----------------------------------------------------
    # MODELO MULTINOMIAL
    #-----------------------------------------------------

    elif tipo.lower() == "multinomial":

        modelo = LogisticRegression(

            # multi_class="multinomial",

            solver="lbfgs",

            class_weight=class_weight,

            max_iter=3000,

            random_state=semilla

        )

    #-----------------------------------------------------

    else:

        raise ValueError(
            "tipo debe ser "
            "'binomial' o "
            "'multinomial'"
        )

    #-----------------------------------------------------
    # AJUSTE
    #-----------------------------------------------------

    modelo.fit(X, y)

    #-----------------------------------------------------
    # METADATOS
    #-----------------------------------------------------

    modelo.variable_dependiente = variable_dependiente

    modelo.tipo_modelo = tipo

    modelo.balanceo = balanceo

    modelo.n_clases = n_clases

    modelo.frecuencias_originales = frecuencias_originales

    modelo.porcentajes_originales = porcentajes_originales

    modelo.frecuencias_entrenamiento = frecuencias_entrenamiento

    modelo.porcentajes_entrenamiento = porcentajes_entrenamiento

    #-----------------------------------------------------
    # RESUMEN
    #-----------------------------------------------------

    print()

    print("="*50)

    print(" REGRESIÓN LOGÍSTICA ")

    print("="*50)

    print("Tipo               :", tipo)

    print("Balanceo           :", balanceo)

    print("Variable objetivo  :", variable_dependiente)

    print("Número de clases   :", n_clases)

    print()

    print("Frecuencias originales:")

    print(frecuencias_originales)

    print()

    print("Frecuencias entrenamiento:")

    print(frecuencias_entrenamiento)

    print()

    print("Observaciones usadas:", len(y))

    print("="*50)

    return modelo

#=========================================================
# FUNCIÓN
# f_predicciones()
#=========================================================

def f_predicciones(
        modelo,
        datos_validacion,
        variable_dependiente):

    import pandas as pd
    import numpy as np

    #-----------------------------------------------------
    # VARIABLES INDEPENDIENTES
    #-----------------------------------------------------

    X = datos_validacion.drop(
        columns=[variable_dependiente]
    )

    #-----------------------------------------------------
    # CLASES PREDICHAS
    #-----------------------------------------------------

    pred = modelo.predict(X)

    #-----------------------------------------------------
    # PROBABILIDADES
    #-----------------------------------------------------

    prob = modelo.predict_proba(X)

    #-----------------------------------------------------
    # BINOMIAL
    #-----------------------------------------------------

    if len(modelo.classes_) == 2:

        probabilidad = prob[:,1]

    #-----------------------------------------------------
    # MULTINOMIAL
    #-----------------------------------------------------

    else:

        probabilidad = np.max(
            prob,
            axis=1
        )

    #-----------------------------------------------------
    # RESULTADO
    #-----------------------------------------------------

    resultado = pd.DataFrame({

        "Real":
            datos_validacion[
                variable_dependiente
            ].values,

        "Prediccion":
            pred,

        "Probabilidad":
            np.round(
                probabilidad,
                4
            )

    })

    resultado["Porcentual"] = (

        resultado["Probabilidad"] * 100

    ).round(2).astype(str) + " %"

    return resultado

#=========================================================
# FUNCIÓN
# f_matriz_confusion()
#
# PARTE 1
#
# Compatible con:
# - LogisticRegression
# - DecisionTreeClassifier
# - RandomForestClassifier
# - LinearSVC
# - SVC (lineal, poly, rbf, sigmoid)
# - KNN
# - AdaBoost
# - GradientBoosting
# - Naive Bayes
# - MLPClassifier
#=========================================================

def f_matriz_confusion(
        modelo,
        datos_validacion,
        variable_dependiente,
        clase_interes=None):

    #-----------------------------------------------------
    # LIBRERÍAS
    #-----------------------------------------------------

    import numpy as np

    import pandas as pd

    from sklearn.base import ClassifierMixin

    from sklearn.metrics import (

        confusion_matrix,

        accuracy_score,

        cohen_kappa_score

    )

    #-----------------------------------------------------
    # VALIDACIONES
    #-----------------------------------------------------

    if not isinstance(datos_validacion,
                      pd.DataFrame):

        raise TypeError(

            "datos_validacion debe ser un DataFrame."

        )

    if variable_dependiente not in datos_validacion.columns:

        raise ValueError(

            f"La variable '{variable_dependiente}' "

            "no existe en los datos."

        )

    if not isinstance(modelo,
                      ClassifierMixin):

        raise TypeError(

            "El objeto recibido no corresponde "

            "a un clasificador de scikit-learn."

        )

    #-----------------------------------------------------
    # VARIABLES
    #-----------------------------------------------------

    X = datos_validacion.drop(

        columns=[variable_dependiente]

    )

    y_real = datos_validacion[
        variable_dependiente
    ]

    #-----------------------------------------------------
    # PREDICCIONES
    #-----------------------------------------------------

    y_pred = modelo.predict(X)

    #-----------------------------------------------------
    # CLASES
    #-----------------------------------------------------

    clases = np.unique(

        np.concatenate(

            (
                np.asarray(y_real),
                np.asarray(y_pred)
            )

        )

    )

    n_clases = len(clases)

    #-----------------------------------------------------
    # MATRIZ DE CONFUSIÓN
    #-----------------------------------------------------

    matriz = confusion_matrix(

        y_real,

        y_pred,

        labels=clases

    )

    tabla_mc = pd.DataFrame(

        matriz,

        index=[

            f"Real_{x}"

            for x in clases

        ],

        columns=[

            f"Pred_{x}"

            for x in clases

        ]

    )

    #-----------------------------------------------------
    # MÉTRICAS GENERALES
    #-----------------------------------------------------

    accuracy = accuracy_score(

        y_real,

        y_pred

    )

    kappa = cohen_kappa_score(

        y_real,

        y_pred

    )

    #-----------------------------------------------------
    # MATRICES UNO CONTRA EL RESTO
    #-----------------------------------------------------

    resultados_clases = []

    total = np.sum(matriz)

    for i in range(n_clases):

        #---------------------------------------------
        # TP
        #---------------------------------------------

        TP = matriz[i, i]

        #---------------------------------------------
        # FN
        #---------------------------------------------

        FN = np.sum(

            matriz[i, :]

        ) - TP

        #---------------------------------------------
        # FP
        #---------------------------------------------

        FP = np.sum(

            matriz[:, i]

        ) - TP

        #---------------------------------------------
        # TN
        #---------------------------------------------

        TN = (

            total

            - TP

            - FN

            - FP

        )

        #---------------------------------------------
        # GUARDAR RESULTADOS
        #---------------------------------------------

        resultados_clases.append({

            "Clase": clases[i],

            "TP": TP,

            "TN": TN,

            "FP": FP,

            "FN": FN

        })

    #-----------------------------------------------------
    # DATAFRAME
    #-----------------------------------------------------

    resultados_clases = pd.DataFrame(

        resultados_clases

    )

    #-----------------------------------------------------
    # CONTINÚA EN LA PARTE 2
    #-----------------------------------------------------
        #-----------------------------------------------------
    # MÉTRICAS POR CLASE
    #-----------------------------------------------------

    precision_clases = []

    recall_clases = []

    sensitivity_clases = []

    specificity_clases = []

    f1_clases = []

    balanced_accuracy_clases = []

    for i in range(len(resultados_clases)):

        TP = resultados_clases.loc[i, "TP"]

        TN = resultados_clases.loc[i, "TN"]

        FP = resultados_clases.loc[i, "FP"]

        FN = resultados_clases.loc[i, "FN"]

        #---------------------------------------------
        # PRECISION
        #---------------------------------------------

        if (TP + FP) > 0:

            precision = TP / (TP + FP)

        else:

            precision = 0

        #---------------------------------------------
        # RECALL
        #---------------------------------------------

        if (TP + FN) > 0:

            recall = TP / (TP + FN)

        else:

            recall = 0

        #---------------------------------------------
        # SENSITIVITY
        #---------------------------------------------

        sensitivity = recall

        #---------------------------------------------
        # SPECIFICITY
        #---------------------------------------------

        if (TN + FP) > 0:

            specificity = TN / (TN + FP)

        else:

            specificity = 0

        #---------------------------------------------
        # F1
        #---------------------------------------------

        if (precision + recall) > 0:

            f1 = (

                2 *

                precision *

                recall

            ) / (

                precision +

                recall

            )

        else:

            f1 = 0

        #---------------------------------------------
        # BALANCED ACCURACY
        #---------------------------------------------

        balanced_accuracy = (

            sensitivity +

            specificity

        ) / 2

        #---------------------------------------------
        # GUARDAR
        #---------------------------------------------

        precision_clases.append(

            precision

        )

        recall_clases.append(

            recall

        )

        sensitivity_clases.append(

            sensitivity

        )

        specificity_clases.append(

            specificity

        )

        f1_clases.append(

            f1

        )

        balanced_accuracy_clases.append(

            balanced_accuracy

        )

    #-----------------------------------------------------
    # PROMEDIOS MACRO
    #-----------------------------------------------------

    precision = np.mean(

        precision_clases

    )

    recall = np.mean(

        recall_clases

    )

    sensitivity = np.mean(

        sensitivity_clases

    )

    specificity = np.mean(

        specificity_clases

    )

    f1 = np.mean(

        f1_clases

    )

    balanced_accuracy = np.mean(

        balanced_accuracy_clases

    )

    #-----------------------------------------------------
    # DATAFRAME RESULTADOS
    #-----------------------------------------------------

    estadisticos = pd.DataFrame({

        "Accuracy":
            [round(accuracy,4)],

        "Kappa":
            [round(kappa,4)],

        "Precision":
            [round(precision,4)],

        "Recall":
            [round(recall,4)],

        "Sensitivity":
            [round(sensitivity,4)],

        "Specificity":
            [round(specificity,4)],

        "F1":
            [round(f1,4)],

        "Balanced_Accuracy":
            [round(
                balanced_accuracy,
                4
            )]

    })

    #-----------------------------------------------------
    # RETORNO
    #-----------------------------------------------------

    return {

        "matriz_confusion":
            tabla_mc,

        "estadisticos":
            estadisticos,

        "estadisticas_por_clase":
            resultados_clases.assign(

                Precision=np.round(
                    precision_clases,
                    4
                ),

                Recall=np.round(
                    recall_clases,
                    4
                ),

                Sensitivity=np.round(
                    sensitivity_clases,
                    4
                ),

                Specificity=np.round(
                    specificity_clases,
                    4
                ),

                F1=np.round(
                    f1_clases,
                    4
                ),

                Balanced_Accuracy=np.round(
                    balanced_accuracy_clases,
                    4
                )

            )

    }


#=========================================================
# FUNCIÓN
# f_evaluacion()
#=========================================================

def f_evaluacion(
        modelos,
        datos_validacion,
        variable_dependiente,
        clase_interes=None,
        nombres_modelos=None):

    import pandas as pd

    #-----------------------------------------------------
    # CONVERTIR MODELO A LISTA
    #-----------------------------------------------------

    if not isinstance(modelos, list):

        modelos = [modelos]

    #-----------------------------------------------------
    # DATOS VALIDACIÓN
    #-----------------------------------------------------

    if not isinstance(datos_validacion, list):

        datos_validacion = (
            [datos_validacion]
            * len(modelos)
        )

    #-----------------------------------------------------
    # NOMBRES
    #-----------------------------------------------------

    if nombres_modelos is None:

        nombres_modelos = [

            f"Modelo {i+1}"

            for i in range(
                len(modelos)
            )

        ]

    #-----------------------------------------------------
    # RESULTADOS
    #-----------------------------------------------------

    resultados = []

    #-----------------------------------------------------
    # RECORRER MODELOS
    #-----------------------------------------------------

    for i in range(len(modelos)):

        modelo = modelos[i]

        datos_val = datos_validacion[i]

        nombre = nombres_modelos[i]

        #---------------------------------------------
        # MATRIZ DE CONFUSIÓN
        #---------------------------------------------

        resultado_mc = (

            f_matriz_confusion(

                modelo=modelo,

                datos_validacion=datos_val,

                variable_dependiente=
                    variable_dependiente,

                clase_interes=
                    clase_interes

            )

        )

        est = (
            resultado_mc[
                "estadisticos"
            ]
            .copy()
        )

        est.insert(
            0,
            "Modelo",
            nombre
        )

        #---------------------------------------------
        # BALANCEO
        #---------------------------------------------

        if hasattr(
            modelo,
            "balanceo"
        ):

            est.insert(

                1,

                "Balanceo",

                modelo.balanceo

            )

        resultados.append(est)

    #-----------------------------------------------------
    # UNIR
    #-----------------------------------------------------

    resultados = pd.concat(

        resultados,

        ignore_index=True

    )

    return resultados