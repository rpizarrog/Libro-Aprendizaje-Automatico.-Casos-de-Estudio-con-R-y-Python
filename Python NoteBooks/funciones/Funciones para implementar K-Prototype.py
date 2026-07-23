# Funciones para implementar # K-Prototypes

# Rubén Pizarro Gurrola
# Julio 2026

# CARGAR DATOS
def f_cargar_datos(ruta_archivo):
    datos = pd.read_csv(ruta_archivo)
    return datos

# FUNCIÓN
# f_redondear()
#
# ACEPTA:
# - datos (DataFrame)
#
# DEVUELVE:
# - DataFrame con las variables numéricas
#   redondeadas a dos decimales

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
    # VALIDACIONES
    if not isinstance(datos, pd.DataFrame):
        raise TypeError(
            "datos debe ser un DataFrame de pandas."
        )

    # COPIA
    datos_redondeados = datos.copy()
    # VARIABLES NUMÉRICAS

    columnas_numericas = datos_redondeados.select_dtypes(
        include=[np.number]).columns

    # REDONDEAR
    datos_redondeados[columnas_numericas] = (
        datos_redondeados[columnas_numericas]
        .round(2))

    return datos_redondeados


# VISUALIZAR HEAD Y TAIL
def f_visualizar_head_tail_reducido(
        datos,
        n = 6):
    # Total columnas
    total_columnas = datos.shape[1]

    # Primeras 4 columnas
    idx_prim = list(
        range(
            min(4, total_columnas)
        )
    )

    # Últimas 4 columnas
    idx_ult = list(
        range(
            max(total_columnas - 4, 0),
            total_columnas
        )
    )

    # Evitar duplicados
    idx_ult = [
        i for i in idx_ult
        if i not in idx_prim
    ]

    # Subconjuntos
    datos_prim = datos.iloc[:, idx_prim]
    datos_ult = datos.iloc[:, idx_ult]

    # HEAD
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

    # TAIL
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

    # Separadores
    sep_head = pd.DataFrame({
        "...": ["..."] * n
    })
    sep_tail = pd.DataFrame({
        "...": ["..."] * n
    })

    # Combinar HEAD
    head_comb = pd.concat(
        [
            head_prim,
            sep_head,
            head_ult
        ],
        axis = 1
    )

    # Combinar TAIL

    tail_comb = pd.concat(
        [
            tail_prim,
            sep_tail,
            tail_ult
        ],
        axis = 1
    )
    # Fila separadora
    fila_sep = pd.DataFrame(
        [["..."] * head_comb.shape[1]],
        columns = head_comb.columns
    )

    # Tabla final
    tabla = pd.concat(
        [
            head_comb,
            fila_sep,
            tail_comb
        ],
        ignore_index = True
    )

    return tabla


# DESCRIBIR DATOS
# FUNCIÓN
# f_describir_datos()
def f_describir_datos(datos):

    import pandas as pd
    # ESTRUCTURA
    estructura = datos.dtypes
    # VARIABLES NUMÉRICAS
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

    # VARIABLES CATEGÓRICAS
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
    # RESULTADO
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


# FUNCIÓN
# f_frecuencias_clases()
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
    # LIBRERÍAS
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    # VARIABLES CATEGÓRICAS

    variables = datos.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    if len(variables) == 0:
        print("No existen variables categóricas en el conjunto de datos.")

        return
    # GRID

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
    # ASEGURAR VECTOR DE EJES
    if nvars == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).reshape(-1)
    # GRÁFICOS
    for i, variable in enumerate(variables):
        frecuencias = (
            datos[variable]
            .value_counts(dropna = False)
        )
        frecuencias.plot(
            kind = "bar",
            ax = axes[i]
        )
        # TÍTULO
        axes[i].set_title(
            variable,
            fontsize = 11,
            pad = 12
        )
        axes[i].set_xlabel("")
        axes[i].set_ylabel(
            "Frecuencia"
        )
        # ROTACIÓN ETIQUETAS
        axes[i].tick_params(
            axis = "x",
            rotation = 45,
            labelsize = 8
        )
        # ETIQUETAS SOBRE BARRAS
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
    # ELIMINAR EJES SOBRANTES
    for j in range(
        len(variables),
        len(axes)
    ):
        fig.delaxes(
            axes[j]
        )
    # ESPACIADO
    plt.subplots_adjust(
        hspace = hspace,
        wspace = wspace
    )
    plt.tight_layout(
        pad = 2.5
    )
    plt.show()

# Función para estandarizar y escalar
# Recibe datos y devuelve un diccionarios con 
# datos estandarizados y escalados por default 4 decimales
# Los escaladores estandarizados y escalados
def f_estandarizar_escalar(datos, variables_numericas, decimales=4):
    
    # Copias de los datos originales
    
    datos_est = datos.copy()
    datos_esc = datos.copy()
    
    # Validar que las variables existan en los datos
    
    for variable in variables_numericas:
        if variable not in datos.columns:
            raise ValueError(f"La variable '{variable}' no existe en los datos.")
    
    # Crear escaladores
    
    escalador_est = StandardScaler()
    escalador_minmax = MinMaxScaler()
    
    # Estandarización Z-score
    # Media = 0, desviación estándar = 1
    
    datos_est[variables_numericas] = np.round(
        escalador_est.fit_transform(datos[variables_numericas]),
        decimales
    )
    
    # Escalamiento Min-Max
    # Rango entre 0 y 1
    
    datos_esc[variables_numericas] = np.round(
        escalador_minmax.fit_transform(datos[variables_numericas]),
        decimales
    )
    
    # Resultado

    return {
        "datos_estandarizados": datos_est[variables_numericas],
        "escalador_est":escalador_est, 
        "datos_escalados": datos_esc[variables_numericas],
        "escalador_minmax":escalador_minmax
    }

#=========================================================
# FUNCIÓN
# f_crear_KPrototypes()
#
# OBJETIVO:
# - Crear un modelo K-Prototypes para datos mixtos.
# - Acepta variables numéricas y categóricas.
# - Usa medias para variables numéricas.
# - Usa modas para variables categóricas.
# - Devuelve clústeres, prototipos finales, frecuencia y costo.
#
# PAQUETE:
# - kmodes.kprototypes.KPrototypes
#=========================================================

def f_crear_KPrototypes(
    datos,
    variables=None,
    variables_numericas=None,
    variables_categoricas=None,
    n_clusters=3,
    init="Huang",
    n_init=25,
    max_iter=100,
    gamma=None,
    random_state=2026,
    estandarizar=True,
    nombre_cluster="cluster_KPrototypes"
):
    
    #-----------------------------------------------------
    # Validaciones básicas
    #-----------------------------------------------------
    
    if not isinstance(datos, pd.DataFrame):
        raise TypeError("El objeto 'datos' debe ser un DataFrame de pandas.")
    
    if n_clusters < 2:
        raise ValueError("El número de clústeres debe ser al menos 2.")
    
    if n_init < 1:
        raise ValueError("n_init debe ser al menos 1.")
    
    if max_iter < 1:
        raise ValueError("max_iter debe ser al menos 1.")
    
    if n_clusters > datos.shape[0]:
        raise ValueError(
            "El número de clústeres no puede ser mayor que el número de registros."
        )
    
    #-----------------------------------------------------
    # Selección de variables
    #-----------------------------------------------------
    
    if variables is None:
        
        if variables_numericas is None:
            variables_numericas = datos.select_dtypes(
                include=["number"]
            ).columns.tolist()
        
        if variables_categoricas is None:
            variables_categoricas = datos.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()
        
        variables = variables_numericas + variables_categoricas
    
    else:
        
        for variable in variables:
            if variable not in datos.columns:
                raise ValueError(f"La variable '{variable}' no existe en los datos.")
        
        if variables_numericas is None:
            variables_numericas = [
                variable for variable in variables
                if pd.api.types.is_numeric_dtype(datos[variable])
            ]
        
        if variables_categoricas is None:
            variables_categoricas = [
                variable for variable in variables
                if (
                    pd.api.types.is_object_dtype(datos[variable]) or
                    pd.api.types.is_categorical_dtype(datos[variable]) or
                    pd.api.types.is_bool_dtype(datos[variable])
                )
            ]
    
    #-----------------------------------------------------
    # Validar que existan ambos tipos de variables
    #-----------------------------------------------------
    
    if len(variables_numericas) == 0:
        raise ValueError(
            "K-Prototypes requiere al menos una variable numérica."
        )
    
    if len(variables_categoricas) == 0:
        raise ValueError(
            "K-Prototypes requiere al menos una variable categórica."
        )
    
    #-----------------------------------------------------
    # Validar existencia y tipo de variables numéricas
    #-----------------------------------------------------
    
    for variable in variables_numericas:
        
        if variable not in datos.columns:
            raise ValueError(f"La variable numérica '{variable}' no existe.")
        
        if not pd.api.types.is_numeric_dtype(datos[variable]):
            raise TypeError(f"La variable '{variable}' no es numérica.")
    
    #-----------------------------------------------------
    # Validar existencia y tipo de variables categóricas
    #-----------------------------------------------------
    
    for variable in variables_categoricas:
        
        if variable not in datos.columns:
            raise ValueError(f"La variable categórica '{variable}' no existe.")
        
        if not (
            pd.api.types.is_object_dtype(datos[variable]) or
            pd.api.types.is_categorical_dtype(datos[variable]) or
            pd.api.types.is_bool_dtype(datos[variable])
        ):
            raise TypeError(f"La variable '{variable}' no es categórica.")
    
    #-----------------------------------------------------
    # Datos para el modelo
    #-----------------------------------------------------
    
    datos_modelo = datos[
        variables_numericas + variables_categoricas
    ].copy()
    
    if datos_modelo.isna().any().any():
        raise ValueError(
            "Existen valores perdidos. Deben tratarse antes de aplicar K-Prototypes."
        )
    
    #-----------------------------------------------------
    # Guardar parámetros para estandarizar
    #-----------------------------------------------------
    
    medias = None
    desviaciones = None
    
    if estandarizar:
        
        medias = datos_modelo[variables_numericas].mean()
        desviaciones = datos_modelo[variables_numericas].std(ddof=0)
        
        if (desviaciones == 0).any():
            variables_constantes = desviaciones[
                desviaciones == 0
            ].index.tolist()
            
            raise ValueError(
                "Existen variables numéricas con desviación estándar cero: "
                + ", ".join(variables_constantes)
            )
        
        datos_modelo[variables_numericas] = (
            datos_modelo[variables_numericas] - medias
        ) / desviaciones
    
    #-----------------------------------------------------
    # Convertir variables categóricas a texto
    #-----------------------------------------------------
    
    for variable in variables_categoricas:
        datos_modelo[variable] = datos_modelo[variable].astype(str)
    
    #-----------------------------------------------------
    # Convertir a matriz para KPrototypes
    #-----------------------------------------------------
    
    matriz_modelo = datos_modelo.to_numpy()
    
    # Índices de variables categóricas dentro de la matriz
    indices_categoricas = [
        datos_modelo.columns.get_loc(variable)
        for variable in variables_categoricas
    ]
    
    #-----------------------------------------------------
    # Crear modelo K-Prototypes
    #-----------------------------------------------------
    
    modelo_KPrototypes = KPrototypes(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        gamma=gamma,
        random_state=random_state
    )
    
    modelo_KPrototypes.fit(
        matriz_modelo,
        categorical=indices_categoricas
    )
    
    #-----------------------------------------------------
    # Clúster asignado
    #-----------------------------------------------------
    
    cluster = modelo_KPrototypes.labels_ + 1
    
    #-----------------------------------------------------
    # Datos con clúster
    #-----------------------------------------------------
    
    datos_cluster = datos.copy()
    datos_cluster[nombre_cluster] = cluster
    
    #-----------------------------------------------------
    # Prototipos finales
    #-----------------------------------------------------
    
    centros = modelo_KPrototypes.cluster_centroids_
    
    centros_numericos = pd.DataFrame(
        centros[:, :len(variables_numericas)].astype(float),
        columns=variables_numericas
    )
    
    centros_categoricos = pd.DataFrame(
        centros[:, len(variables_numericas):],
        columns=variables_categoricas
    )
    
    prototipos_modelo = pd.concat(
        [
            centros_numericos,
            centros_categoricos
        ],
        axis=1
    )
    
    prototipos_modelo.insert(
        0,
        "cluster",
        range(1, n_clusters + 1)
    )
    
    #-----------------------------------------------------
    # Prototipos en escala original
    #-----------------------------------------------------
    
    prototipos_finales = prototipos_modelo.copy()
    
    if estandarizar:
        
        for variable in variables_numericas:
            
            prototipos_finales[variable] = (
                prototipos_modelo[variable].astype(float) *
                desviaciones[variable] +
                medias[variable]
            )
    
    #-----------------------------------------------------
    # Frecuencia por clúster
    #-----------------------------------------------------
    
    frecuencia_cluster = (
        datos_cluster[nombre_cluster]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    
    frecuencia_cluster.columns = [
        "cluster",
        "n"
    ]
    
    #-----------------------------------------------------
    # Costo
    #-----------------------------------------------------
    
    costo_total = modelo_KPrototypes.cost_
    costo_promedio = costo_total / datos_modelo.shape[0]
    
    #-----------------------------------------------------
    # Resultado
    #-----------------------------------------------------
    
    resultado = {
        "modelo": modelo_KPrototypes,
        "cluster": cluster,
        "datos_cluster": datos_cluster,
        "prototipos_modelo": prototipos_modelo,
        "prototipos_finales": prototipos_finales,
        "frecuencia_cluster": frecuencia_cluster,
        "costo": costo_total,
        "costo_total": costo_total,
        "costo_promedio": costo_promedio,
        "variables": variables_numericas + variables_categoricas,
        "variables_numericas": variables_numericas,
        "variables_categoricas": variables_categoricas,
        "indices_categoricas": indices_categoricas,
        "n_clusters": n_clusters,
        "init": init,
        "n_init": n_init,
        "max_iter": max_iter,
        "gamma": modelo_KPrototypes.gamma,
        "random_state": random_state,
        "estandarizar": estandarizar,
        "medias": medias,
        "desviaciones": desviaciones,
        "nombre_cluster": nombre_cluster
    }
    
    return resultado



#=========================================================
# LIBRERÍAS
#=========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math

#=========================================================
# FUNCIÓN
# f_dispersion_variables_clusters()
#=========================================================

def f_dispersion_variables_clusters(
    datos,
    variable_cluster="cluster",
    variables=None,
    centroides=None,
    titulo="Diagramas de dispersión por clúster",
    ncol=2,
    tam_puntos=12,
    alpha=0.50,
    tam_centroides=80
):
    """
    Genera diagramas de dispersión por pares de variables numéricas,
    coloreando los puntos según el clúster.

    Parámetros:
    ----------
    datos : pandas.DataFrame
        DataFrame con los datos y la variable de clúster.

    variable_cluster : str
        Nombre de la columna que contiene el clúster asignado.

    variables : list o None
        Lista de variables numéricas a graficar.
        Si es None, selecciona automáticamente las numéricas.

    centroides : pandas.DataFrame, numpy.ndarray o None
        Centroides del modelo. Deben contener las mismas variables.
        Si se proporciona, se grafican como puntos negros.

    titulo : str
        Título general de la figura.

    ncol : int
        Número de columnas en el arreglo de gráficos.

    tam_puntos : int o float
        Tamaño de los puntos de los registros.

    alpha : float
        Transparencia de los puntos.

    tam_centroides : int o float
        Tamaño de los puntos de los centroides.

    Retorna:
    -------
    fig : matplotlib.figure.Figure
        Figura generada.
    """

    #-----------------------------------------------------
    # Validaciones
    #-----------------------------------------------------

    if not isinstance(datos, pd.DataFrame):
        raise TypeError("El objeto 'datos' debe ser un DataFrame de pandas.")

    if variable_cluster not in datos.columns:
        raise ValueError(
            f"La variable de clúster '{variable_cluster}' no existe en los datos."
        )

    #-----------------------------------------------------
    # Seleccionar variables numéricas
    #-----------------------------------------------------

    if variables is None:

        variables = datos.select_dtypes(include=np.number).columns.tolist()

        if variable_cluster in variables:
            variables.remove(variable_cluster)

    else:

        for variable in variables:
            if variable not in datos.columns:
                raise ValueError(
                    f"La variable '{variable}' no existe en los datos."
                )

    if len(variables) < 2:
        raise ValueError(
            "Se requieren al menos dos variables numéricas para generar diagramas de dispersión."
        )

    #-----------------------------------------------------
    # Copiar datos y convertir clúster a categoría
    #-----------------------------------------------------

    datos_graf = datos.copy()
    datos_graf[variable_cluster] = datos_graf[variable_cluster].astype(str)

    #-----------------------------------------------------
    # Preparar centroides si existen
    #-----------------------------------------------------

    centroides_graf = None

    if centroides is not None:

        if isinstance(centroides, np.ndarray):

            centroides_graf = pd.DataFrame(
                centroides,
                columns=variables
            )

        elif isinstance(centroides, pd.DataFrame):

            centroides_graf = centroides.copy()

        else:
            raise TypeError(
                "Los centroides deben ser un DataFrame de pandas o un arreglo numpy."
            )

        for variable in variables:
            if variable not in centroides_graf.columns:
                raise ValueError(
                    "Los centroides deben contener las mismas variables numéricas seleccionadas."
                )

    #-----------------------------------------------------
    # Crear combinaciones de pares de variables
    #-----------------------------------------------------

    combinaciones = list(itertools.combinations(variables, 2))

    n_graficos = len(combinaciones)
    nfilas = math.ceil(n_graficos / ncol)

    #-----------------------------------------------------
    # Crear figura
    #-----------------------------------------------------

    fig, axes = plt.subplots(
        nfilas,
        ncol,
        figsize=(6 * ncol, 5 * nfilas)
    )

    # Cuando solo hay un gráfico o una fila, ajustar axes
    axes = np.array(axes).reshape(-1)

    #-----------------------------------------------------
    # Crear colores por clúster
    #-----------------------------------------------------

    clusters = sorted(datos_graf[variable_cluster].unique())

    colores = plt.cm.tab10(np.linspace(0, 1, len(clusters)))

    mapa_colores = {
        cluster: colores[i]
        for i, cluster in enumerate(clusters)
    }

    #-----------------------------------------------------
    # Crear cada gráfico
    #-----------------------------------------------------

    for i, (var_x, var_y) in enumerate(combinaciones):

        ax = axes[i]

        # Graficar puntos por clúster
        for cluster in clusters:

            datos_cluster = datos_graf[
                datos_graf[variable_cluster] == cluster
            ]

            ax.scatter(
                datos_cluster[var_x],
                datos_cluster[var_y],
                s=tam_puntos,
                alpha=alpha,
                color=mapa_colores[cluster],
                label=f"Clúster {cluster}"
            )

        # Agregar centroides como puntos negros
        if centroides_graf is not None:

            ax.scatter(
                centroides_graf[var_x],
                centroides_graf[var_y],
                s=tam_centroides,
                color="black",
                marker="o",
                label="Centroides"
            )

        ax.set_title(f"{var_x} vs {var_y}")
        ax.set_xlabel(var_x)
        ax.set_ylabel(var_y)
        ax.grid(alpha=0.30)
        ax.legend()

    #-----------------------------------------------------
    # Eliminar gráficos vacíos si sobran ejes
    #-----------------------------------------------------

    for j in range(n_graficos, len(axes)):
        fig.delaxes(axes[j])

    #-----------------------------------------------------
    # Título general
    #-----------------------------------------------------

    fig.suptitle(titulo, fontsize=16, y=1.02)
    plt.tight_layout()

    return fig

#=========================================================
# LIBRERÍAS
#=========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

#=========================================================
# FUNCIÓN
# f_diagramas_cajas()
#=========================================================

def f_diagramas_cajas(
    datos,
    variable_cluster="cluster",
    variables=None,
    titulo="Diagramas de caja por clúster",
    ncol=2,
    tam_puntos=8,
    alpha_puntos=0.50,
    alpha_caja=0.50
):
    """
    Genera diagramas de caja para comparar variables numéricas
    dentro de cada clúster.

    Parámetros:
    ----------
    datos : pandas.DataFrame
        DataFrame con los datos y la variable de clúster.

    variable_cluster : str
        Nombre de la columna que contiene el clúster asignado.

    variables : list o None
        Lista de variables numéricas a graficar.
        Si es None, selecciona automáticamente las variables numéricas.

    titulo : str
        Título general de la figura.

    ncol : int
        Número de columnas en el arreglo de gráficos.

    tam_puntos : int o float
        Tamaño de los puntos individuales.

    alpha_puntos : float
        Transparencia de los puntos individuales.

    alpha_caja : float
        Transparencia de las cajas.

    Retorna:
    -------
    fig : matplotlib.figure.Figure
        Figura generada.
    """

    #-----------------------------------------------------
    # Validaciones
    #-----------------------------------------------------

    if not isinstance(datos, pd.DataFrame):
        raise TypeError("El objeto 'datos' debe ser un DataFrame de pandas.")

    if variable_cluster not in datos.columns:
        raise ValueError(
            f"La variable de clúster '{variable_cluster}' no existe en los datos."
        )

    #-----------------------------------------------------
    # Seleccionar variables numéricas
    #-----------------------------------------------------

    if variables is None:

        variables = datos.select_dtypes(include=np.number).columns.tolist()

        if variable_cluster in variables:
            variables.remove(variable_cluster)

    else:

        for variable in variables:
            if variable not in datos.columns:
                raise ValueError(
                    f"La variable '{variable}' no existe en los datos."
                )

    if len(variables) < 1:
        raise ValueError(
            "Se requiere al menos una variable numérica para generar diagramas de caja."
        )

    #-----------------------------------------------------
    # Copiar datos y convertir clúster a texto
    #-----------------------------------------------------

    datos_graf = datos.copy()
    datos_graf[variable_cluster] = datos_graf[variable_cluster].astype(str)

    #-----------------------------------------------------
    # Ordenar clústeres
    #-----------------------------------------------------

    clusters = sorted(datos_graf[variable_cluster].unique())

    #-----------------------------------------------------
    # Definir filas y columnas
    #-----------------------------------------------------

    n_graficos = len(variables)
    nfilas = math.ceil(n_graficos / ncol)

    fig, axes = plt.subplots(
        nfilas,
        ncol,
        figsize=(6 * ncol, 5 * nfilas)
    )

    axes = np.array(axes).reshape(-1)

    #-----------------------------------------------------
    # Crear colores por clúster
    #-----------------------------------------------------

    colores = plt.cm.tab10(np.linspace(0, 1, len(clusters)))

    mapa_colores = {
        cluster: colores[i]
        for i, cluster in enumerate(clusters)
    }

    #-----------------------------------------------------
    # Crear diagramas de caja
    #-----------------------------------------------------

    for i, variable in enumerate(variables):

        ax = axes[i]

        datos_por_cluster = [
            datos_graf.loc[
                datos_graf[variable_cluster] == cluster,
                variable
            ].dropna()
            for cluster in clusters
        ]

        #-------------------------------------------------
        # Diagrama de caja
        #-------------------------------------------------

        box = ax.boxplot(
            datos_por_cluster,
            tick_labels=[f"Clúster {cluster}" for cluster in clusters],
            patch_artist=True,
            showfliers=True
        )

        # Colorear cajas
        for patch, cluster in zip(box["boxes"], clusters):
            patch.set_facecolor(mapa_colores[cluster])
            patch.set_alpha(alpha_caja)

        #-------------------------------------------------
        # Agregar puntos individuales con jitter
        #-------------------------------------------------

        for pos, cluster in enumerate(clusters, start=1):

            valores = datos_graf.loc[
                datos_graf[variable_cluster] == cluster,
                variable
            ].dropna()

            jitter = np.random.normal(
                loc=pos,
                scale=0.05,
                size=len(valores)
            )

            ax.scatter(
                jitter,
                valores,
                s=tam_puntos,
                alpha=alpha_puntos,
                color=mapa_colores[cluster]
            )

        #-------------------------------------------------
        # Etiquetas
        #-------------------------------------------------

        ax.set_title(f"Distribución de {variable}")
        ax.set_xlabel("Clúster")
        ax.set_ylabel(variable)
        ax.grid(alpha=0.30)

    #-----------------------------------------------------
    # Eliminar ejes vacíos si sobran
    #-----------------------------------------------------

    for j in range(n_graficos, len(axes)):
        fig.delaxes(axes[j])

    #-----------------------------------------------------
    # Título general
    #-----------------------------------------------------

    fig.suptitle(titulo, fontsize=16, y=1.02)
    plt.tight_layout()

    return fig

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


# FUNCIÓN
# f_frecuencias_clases()
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
    # LIBRERÍAS
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    # VARIABLES CATEGÓRICAS

    variables = datos.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    if len(variables) == 0:
        print("No existen variables categóricas en el conjunto de datos.")

        return
    # GRID

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
    # ASEGURAR VECTOR DE EJES
    if nvars == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).reshape(-1)
    # GRÁFICOS
    for i, variable in enumerate(variables):
        frecuencias = (
            datos[variable]
            .value_counts(dropna = False)
        )
        frecuencias.plot(
            kind = "bar",
            ax = axes[i]
        )
        # TÍTULO
        axes[i].set_title(
            variable,
            fontsize = 11,
            pad = 12
        )
        axes[i].set_xlabel("")
        axes[i].set_ylabel(
            "Frecuencia"
        )
        # ROTACIÓN ETIQUETAS
        axes[i].tick_params(
            axis = "x",
            rotation = 45,
            labelsize = 8
        )
        # ETIQUETAS SOBRE BARRAS
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
    # ELIMINAR EJES SOBRANTES
    for j in range(
        len(variables),
        len(axes)
    ):
        fig.delaxes(
            axes[j]
        )
    # ESPACIADO
    plt.subplots_adjust(
        hspace = hspace,
        wspace = wspace
    )
    plt.tight_layout(
        pad = 2.5
    )
    plt.show()

#=========================================================
# FUNCIÓN
# f_visualizar_clusters_categoricos()
#
# OBJETIVO:
# - Visualizar variables categóricas por clúster.
# - Útil para modelos K-Prototypes y variables categóricas.
# - Cada panel corresponde a una variable.
# - El eje X muestra las categorías.
# - El color representa el clúster.
# - Muestra porcentajes dentro de cada clúster.
#=========================================================

#=========================================================
# FUNCIÓN
# f_visualizar_clusters_categoricos()
#
# OBJETIVO:
# - Visualizar variables categóricas por clúster.
# - Útil para modelos K-Modes.
# - Cada panel corresponde a una variable.
# - El eje X muestra las categorías.
# - El color representa el clúster.
# - Muestra porcentajes dentro de cada clúster.
#
# DEVUELVE:
# - figura
# - tabla_frecuencias
#=========================================================

def f_visualizar_clusters_categoricos(
    datos,
    variable_cluster,
    variables=None,
    ncol=2,
    titulo="Distribución porcentual de variables categóricas por clúster",
    decimales=1,
    posicion="dodge",
    mostrar_etiquetas=True,
    rotar_etiquetas=True,
    figsize_base=(7, 4)
):
    
    #-----------------------------------------------------
    # Validaciones básicas
    #-----------------------------------------------------
    
    if not isinstance(datos, pd.DataFrame):
        raise TypeError("El objeto 'datos' debe ser un DataFrame de pandas.")
    
    if variable_cluster not in datos.columns:
        raise ValueError("La variable de clúster no existe en los datos.")
    
    if posicion not in ["dodge", "fill"]:
        raise ValueError("El argumento 'posicion' debe ser 'dodge' o 'fill'.")
    
    #-----------------------------------------------------
    # Seleccionar variables categóricas
    #-----------------------------------------------------
    
    if variables is None:
        
        variables = datos.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
        
        variables = [
            variable for variable in variables
            if variable != variable_cluster
        ]
        
        if len(variables) == 0:
            raise ValueError("No existen variables categóricas para graficar.")
    
    else:
        
        for variable in variables:
            
            if variable not in datos.columns:
                raise ValueError(f"La variable '{variable}' no existe en los datos.")
            
            if not (
                pd.api.types.is_object_dtype(datos[variable]) or
                pd.api.types.is_categorical_dtype(datos[variable]) or
                pd.api.types.is_bool_dtype(datos[variable])
            ):
                raise TypeError(f"La variable '{variable}' no es categórica.")
    
    #-----------------------------------------------------
    # Preparar datos en formato largo
    #-----------------------------------------------------
    
    datos_largos = (
        datos[[variable_cluster] + variables]
        .copy()
    )
    
    datos_largos[variable_cluster] = datos_largos[variable_cluster].astype(str)
    
    for variable in variables:
        datos_largos[variable] = datos_largos[variable].astype(str)
    
    datos_largos = datos_largos.melt(
        id_vars=variable_cluster,
        value_vars=variables,
        var_name="variable",
        value_name="categoria"
    )
    
    datos_largos = datos_largos[
        datos_largos["categoria"].notna()
    ]
    
    #-----------------------------------------------------
    # Calcular frecuencias y porcentajes
    # Porcentaje dentro de cada clúster y variable
    #-----------------------------------------------------
    
    tabla_frecuencias = (
        datos_largos
        .groupby([variable_cluster, "variable", "categoria"])
        .size()
        .reset_index(name="frecuencia")
    )
    
    tabla_frecuencias["total_cluster_variable"] = (
        tabla_frecuencias
        .groupby([variable_cluster, "variable"])["frecuencia"]
        .transform("sum")
    )
    
    tabla_frecuencias["porcentaje"] = (
        tabla_frecuencias["frecuencia"] /
        tabla_frecuencias["total_cluster_variable"]
    )
    
    tabla_frecuencias["etiqueta"] = (
        (tabla_frecuencias["porcentaje"] * 100)
        .round(decimales)
        .astype(str) + "%"
    )
    
    #-----------------------------------------------------
    # Configurar figura
    #-----------------------------------------------------
    
    n_variables = len(variables)
    nfilas = math.ceil(n_variables / ncol)
    
    fig, axes = plt.subplots(
        nfilas,
        ncol,
        figsize=(figsize_base[0] * ncol, figsize_base[1] * nfilas)
    )
    
    axes = np.array(axes).reshape(-1)
    
    clusters = sorted(
        tabla_frecuencias[variable_cluster].unique()
    )
    
    colores = plt.cm.tab10(
        np.linspace(0, 1, len(clusters))
    )
    
    mapa_colores = {
        cluster: colores[i]
        for i, cluster in enumerate(clusters)
    }
    
    #-----------------------------------------------------
    # Construir gráficos por variable
    #-----------------------------------------------------
    
    for i, variable in enumerate(variables):
        
        ax = axes[i]
        
        datos_var = tabla_frecuencias[
            tabla_frecuencias["variable"] == variable
        ].copy()
        
        categorias = sorted(datos_var["categoria"].unique())
        x = np.arange(len(categorias))
        
        #-------------------------------------------------
        # Barras agrupadas
        #-------------------------------------------------
        
        if posicion == "dodge":
            
            ancho = 0.80 / len(clusters)
            
            for j, cluster in enumerate(clusters):
                
                datos_cluster = datos_var[
                    datos_var[variable_cluster] == cluster
                ]
                
                valores = []
                etiquetas = []
                
                for categoria in categorias:
                    
                    fila = datos_cluster[
                        datos_cluster["categoria"] == categoria
                    ]
                    
                    if fila.empty:
                        valores.append(0)
                        etiquetas.append("")
                    else:
                        valores.append(float(fila["porcentaje"].iloc[0]))
                        etiquetas.append(fila["etiqueta"].iloc[0])
                
                posiciones = x - 0.40 + ancho / 2 + j * ancho
                
                barras = ax.bar(
                    posiciones,
                    valores,
                    width=ancho,
                    label=str(cluster),
                    color=mapa_colores[cluster],
                    alpha=0.85
                )
                
                if mostrar_etiquetas:
                    
                    for barra, etiqueta, valor in zip(barras, etiquetas, valores):
                        
                        if valor > 0:
                            ax.text(
                                barra.get_x() + barra.get_width() / 2,
                                barra.get_height() + 0.015,
                                etiqueta,
                                ha="center",
                                va="bottom",
                                fontsize=8
                            )
            
            limite_superior = max(datos_var["porcentaje"]) + 0.20
            
            ax.set_ylim(
                0,
                min(1.05, limite_superior)
            )
            
            ax.set_xticks(x)
            ax.set_xticklabels(categorias)
        
        #-------------------------------------------------
        # Barras apiladas
        #-------------------------------------------------
        
        else:
            
            acumulado = np.zeros(len(categorias))
            
            for cluster in clusters:
                
                datos_cluster = datos_var[
                    datos_var[variable_cluster] == cluster
                ]
                
                valores = []
                etiquetas = []
                
                for categoria in categorias:
                    
                    fila = datos_cluster[
                        datos_cluster["categoria"] == categoria
                    ]
                    
                    if fila.empty:
                        valores.append(0)
                        etiquetas.append("")
                    else:
                        valores.append(float(fila["porcentaje"].iloc[0]))
                        etiquetas.append(fila["etiqueta"].iloc[0])
                
                barras = ax.bar(
                    x,
                    valores,
                    bottom=acumulado,
                    label=str(cluster),
                    color=mapa_colores[cluster],
                    alpha=0.85
                )
                
                if mostrar_etiquetas:
                    
                    for idx, (valor, etiqueta) in enumerate(zip(valores, etiquetas)):
                        
                        if valor >= 0.08:
                            ax.text(
                                x[idx],
                                acumulado[idx] + valor / 2,
                                etiqueta,
                                ha="center",
                                va="center",
                                fontsize=8
                            )
                
                acumulado += np.array(valores)
            
            ax.set_ylim(0, 1)
            ax.set_xticks(x)
            ax.set_xticklabels(categorias)
        
        #-------------------------------------------------
        # Formato del panel
        #-------------------------------------------------
        
        ax.set_title(
            variable,
            fontweight="bold"
        )
        
        ax.set_ylabel(
            "Porcentaje dentro del clúster"
        )
        
        ax.grid(
            axis="y",
            alpha=0.25
        )
        
        #-------------------------------------------------
        # Corrección del warning:
        # usar PercentFormatter en lugar de set_yticklabels
        #-------------------------------------------------
        
        ax.yaxis.set_major_formatter(
            PercentFormatter(xmax=1.0, decimals=0)
        )
        
        if rotar_etiquetas:
            
            ax.tick_params(
                axis="x",
                rotation=45
            )
            
            for label in ax.get_xticklabels():
                label.set_ha("right")
    
    #-----------------------------------------------------
    # Ocultar paneles vacíos
    #-----------------------------------------------------
    
    for j in range(n_variables, len(axes)):
        axes[j].axis("off")
    
    #-----------------------------------------------------
    # Leyenda general
    #-----------------------------------------------------
    
    handles, labels = axes[0].get_legend_handles_labels()
    
    fig.legend(
        handles,
        labels,
        title="Clúster",
        loc="lower center",
        ncol=len(clusters),
        bbox_to_anchor=(0.5, -0.01)
    )
    
    fig.suptitle(
        titulo,
        fontsize=16,
        fontweight="bold"
    )
    
    plt.tight_layout(
        rect=[0, 0.04, 1, 0.96]
    )
    
    plt.show()
    
    #-----------------------------------------------------
    # Devolver resultados
    #-----------------------------------------------------
    
    return {
        "figura": fig,
        "tabla_frecuencias": tabla_frecuencias
    }


#=========================================================
# FUNCIÓN
# f_evaluar_costo_KPrototypes()
#
# OBJETIVO:
# - Evaluar el costo de modelos K-Prototypes.
# - Comparar modelos con diferente número de clústeres.
# - Recibe un modelo o una lista de modelos.
# - Calcula costo total, costo promedio y reducción porcentual.
#
# ACEPTA:
# - Modelos creados directamente con KPrototypes.
# - Resultados creados con una función tipo f_crear_KPrototypes()
#   que devuelve:
#   "modelo", "cluster", "prototipos_finales",
#   "costo", "costo_total", "costo_promedio", etc.
#=========================================================

def f_evaluar_costo_KPrototypes(
    modelos,
    nombres_modelos=None,
    graficar=True,
    titulo="Evaluación del costo en K-Prototypes",
    decimales=4,
    reemplazar_NA=False
):
    
    #-----------------------------------------------------
    # Convertir a lista si se recibe un solo modelo
    #-----------------------------------------------------
    
    if not isinstance(modelos, list):
        modelos = [modelos]
    
    if len(modelos) == 0:
        raise ValueError("Debe proporcionar al menos un modelo.")
    
    #-----------------------------------------------------
    # Definir nombres de modelos
    #-----------------------------------------------------
    
    if nombres_modelos is None:
        nombres_modelos = [
            f"Modelo_{i+1}" for i in range(len(modelos))
        ]
    
    if len(nombres_modelos) != len(modelos):
        raise ValueError(
            "La longitud de 'nombres_modelos' debe coincidir "
            "con el número de modelos."
        )
    
    #-----------------------------------------------------
    # Función auxiliar para extraer costo
    #-----------------------------------------------------
    
    def f_extraer_costo(modelo):
        
        #-------------------------------------------------
        # Caso 1: resultado personalizado tipo diccionario
        #-------------------------------------------------
        
        if isinstance(modelo, dict):
            
            if "costo_total" in modelo:
                return float(modelo["costo_total"])
            
            if "costo" in modelo:
                return float(modelo["costo"])
            
            if "cost" in modelo:
                return float(modelo["cost"])
            
            if "cost_" in modelo:
                return float(modelo["cost_"])
            
            if "modelo" in modelo:
                
                modelo_interno = modelo["modelo"]
                
                if hasattr(modelo_interno, "cost_"):
                    return float(modelo_interno.cost_)
                
                if hasattr(modelo_interno, "cost"):
                    return float(modelo_interno.cost)
                
                if hasattr(modelo_interno, "costo"):
                    return float(modelo_interno.costo)
        
        #-------------------------------------------------
        # Caso 2: modelo directo KPrototypes
        #-------------------------------------------------
        
        if hasattr(modelo, "cost_"):
            return float(modelo.cost_)
        
        if hasattr(modelo, "cost"):
            return float(modelo.cost)
        
        if hasattr(modelo, "costo"):
            return float(modelo.costo)
        
        raise ValueError(
            "No se pudo extraer el costo del modelo. "
            "Revise si el modelo contiene 'costo_total', 'costo', "
            "'cost', 'cost_' o si es un objeto KPrototypes "
            "con atributo cost_."
        )
    
    #-----------------------------------------------------
    # Función auxiliar para extraer K
    #-----------------------------------------------------
    
    def f_extraer_k(modelo):
        
        #-------------------------------------------------
        # Caso 1: resultado personalizado tipo diccionario
        #-------------------------------------------------
        
        if isinstance(modelo, dict):
            
            if "n_clusters" in modelo:
                return int(modelo["n_clusters"])
            
            if "k" in modelo:
                return int(modelo["k"])
            
            if "prototipos_finales" in modelo:
                return int(modelo["prototipos_finales"].shape[0])
            
            if "prototipos_modelo" in modelo:
                return int(modelo["prototipos_modelo"].shape[0])
            
            if "modelo" in modelo:
                
                modelo_interno = modelo["modelo"]
                
                if hasattr(modelo_interno, "n_clusters"):
                    return int(modelo_interno.n_clusters)
                
                if hasattr(modelo_interno, "cluster_centroids_"):
                    return int(modelo_interno.cluster_centroids_.shape[0])
        
        #-------------------------------------------------
        # Caso 2: modelo directo KPrototypes
        #-------------------------------------------------
        
        if hasattr(modelo, "n_clusters"):
            return int(modelo.n_clusters)
        
        if hasattr(modelo, "cluster_centroids_"):
            return int(modelo.cluster_centroids_.shape[0])
        
        if hasattr(modelo, "labels_"):
            return int(len(np.unique(modelo.labels_)))
        
        return np.nan
    
    #-----------------------------------------------------
    # Función auxiliar para extraer n
    #-----------------------------------------------------
    
    def f_extraer_n(modelo):
        
        #-------------------------------------------------
        # Caso 1: resultado personalizado tipo diccionario
        #-------------------------------------------------
        
        if isinstance(modelo, dict):
            
            if "cluster" in modelo:
                return len(modelo["cluster"])
            
            if "datos_cluster" in modelo:
                return modelo["datos_cluster"].shape[0]
            
            if "modelo" in modelo:
                
                modelo_interno = modelo["modelo"]
                
                if hasattr(modelo_interno, "labels_"):
                    return len(modelo_interno.labels_)
        
        #-------------------------------------------------
        # Caso 2: modelo directo KPrototypes
        #-------------------------------------------------
        
        if hasattr(modelo, "labels_"):
            return len(modelo.labels_)
        
        return np.nan
    
    #-----------------------------------------------------
    # Función auxiliar para extraer gamma
    #-----------------------------------------------------
    
    def f_extraer_gamma(modelo):
        
        if isinstance(modelo, dict):
            
            if "gamma" in modelo:
                try:
                    return float(modelo["gamma"])
                except:
                    return np.nan
            
            if "modelo" in modelo:
                
                modelo_interno = modelo["modelo"]
                
                if hasattr(modelo_interno, "gamma"):
                    try:
                        return float(modelo_interno.gamma)
                    except:
                        return np.nan
        
        if hasattr(modelo, "gamma"):
            try:
                return float(modelo.gamma)
            except:
                return np.nan
        
        return np.nan
    
    #-----------------------------------------------------
    # Calcular costos
    #-----------------------------------------------------
    
    registros = []
    
    for i, modelo_actual in enumerate(modelos):
        
        costo_total = f_extraer_costo(modelo_actual)
        k = f_extraer_k(modelo_actual)
        n = f_extraer_n(modelo_actual)
        gamma = f_extraer_gamma(modelo_actual)
        
        if pd.isna(n):
            costo_promedio = np.nan
        else:
            costo_promedio = costo_total / n
        
        registros.append({
            "modelo": nombres_modelos[i],
            "k": k,
            "n": n,
            "gamma": gamma,
            "costo_total": costo_total,
            "costo_promedio": costo_promedio
        })
    
    tabla_costos = pd.DataFrame(registros)
    
    #-----------------------------------------------------
    # Ordenar por K
    #-----------------------------------------------------
    
    tabla_costos = tabla_costos.sort_values(
        by="k"
    ).reset_index(drop=True)
    
    #-----------------------------------------------------
    # Calcular reducción del costo
    #-----------------------------------------------------
    
    tabla_costos["reduccion_absoluta"] = (
        tabla_costos["costo_total"].shift(1) -
        tabla_costos["costo_total"]
    )
    
    tabla_costos["reduccion_porcentual"] = (
        tabla_costos["reduccion_absoluta"] /
        tabla_costos["costo_total"].shift(1)
    ) * 100
    
    #-----------------------------------------------------
    # Reemplazar NaN de primera fila si se solicita
    #-----------------------------------------------------
    
    if reemplazar_NA:
        tabla_costos["reduccion_absoluta"] = (
            tabla_costos["reduccion_absoluta"].fillna(0)
        )
        
        tabla_costos["reduccion_porcentual"] = (
            tabla_costos["reduccion_porcentual"].fillna(0)
        )
    
    #-----------------------------------------------------
    # Interpretación
    #-----------------------------------------------------
    
    tabla_costos["interpretacion"] = np.where(
        tabla_costos["reduccion_absoluta"].isna() |
        (tabla_costos["reduccion_absoluta"] == 0),
        "Modelo base",
        "Mejora respecto al modelo anterior"
    )
    
    #-----------------------------------------------------
    # Redondear
    #-----------------------------------------------------
    
    columnas_redondear = [
        "gamma",
        "costo_total",
        "costo_promedio",
        "reduccion_absoluta",
        "reduccion_porcentual"
    ]
    
    for columna in columnas_redondear:
        tabla_costos[columna] = tabla_costos[columna].round(decimales)
    
    #-----------------------------------------------------
    # Gráfico
    #-----------------------------------------------------
    
    figura = None
    
    if graficar:
        
        figura, ax = plt.subplots(
            figsize=(8, 5)
        )
        
        ax.plot(
            tabla_costos["k"],
            tabla_costos["costo_total"],
            marker="o",
            linewidth=2
        )
        
        for _, fila in tabla_costos.iterrows():
            
            ax.text(
                fila["k"],
                fila["costo_total"],
                str(fila["costo_total"]),
                ha="center",
                va="bottom",
                fontsize=10
            )
        
        ax.set_title(
            titulo,
            fontsize=14,
            fontweight="bold"
        )
        
        ax.set_xlabel(
            "Número de clústeres K"
        )
        
        ax.set_ylabel(
            "Costo total"
        )
        
        ax.set_xticks(
            tabla_costos["k"]
        )
        
        ax.grid(
            alpha=0.25
        )
        
        plt.tight_layout()
        plt.show()
    
    #-----------------------------------------------------
    # Devolver resultados
    #-----------------------------------------------------
    
    return {
        "tabla_costos": tabla_costos,
        "figura": figura
    }