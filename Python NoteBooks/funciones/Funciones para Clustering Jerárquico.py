# Funciones para implementar Clustering Jerárquico
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
# f_preparar_datos_modelo()
#
# OBJETIVO:
# - Preparar los datos para clustering jerárquico.
# - Separar variables numéricas y categóricas.
# - Construir el conjunto de datos del modelo.
# - Validar variables existentes.
# - Convertir categóricas a texto.
# - Revisar valores perdidos.
#=========================================================

def f_preparar_datos_modelo(
    datos,
    variables_numericas=None,
    variables_categoricas=None,
    variable_id=None,
    mostrar_resumen=True
):
    
    #-----------------------------------------------------
    # Librerías
    #-----------------------------------------------------
    
    import pandas as pd
    import numpy as np
    
    #-----------------------------------------------------
    # Validar que datos sea DataFrame
    #-----------------------------------------------------
    
    if not isinstance(datos, pd.DataFrame):
        raise TypeError("El objeto 'datos' debe ser un DataFrame de pandas.")
    
    #-----------------------------------------------------
    # Variables numéricas por defecto
    #-----------------------------------------------------
    
    if variables_numericas is None:
        
        variables_numericas = [
            "edad",
            "ingreso_mensual",
            "visitas_mes",
            "gasto_promedio"
        ]
    
    #-----------------------------------------------------
    # Variables categóricas por defecto
    #-----------------------------------------------------
    
    if variables_categoricas is None:
        
        variables_categoricas = [
            "canal_preferido",
            "region",
            "nivel_satisfaccion",
            "respuesta_promocion"
        ]
    
    #-----------------------------------------------------
    # Construir variables del modelo
    #-----------------------------------------------------
    
    variables_modelo = variables_numericas + variables_categoricas
    
    #-----------------------------------------------------
    # Validar que las variables existan
    #-----------------------------------------------------
    
    variables_no_existen = [
        variable for variable in variables_modelo
        if variable not in datos.columns
    ]
    
    if len(variables_no_existen) > 0:
        raise ValueError(
            "Las siguientes variables no existen en los datos: "
            + ", ".join(variables_no_existen)
        )
    
    #-----------------------------------------------------
    # Validar variable ID si se indica
    #-----------------------------------------------------
    
    if variable_id is not None:
        
        if variable_id not in datos.columns:
            raise ValueError(
                f"La variable_id '{variable_id}' no existe en los datos."
            )
    
    #-----------------------------------------------------
    # Crear datos del modelo
    #-----------------------------------------------------
    
    datos_modelo = datos[variables_modelo].copy()
    
    #-----------------------------------------------------
    # Validar variables numéricas
    #-----------------------------------------------------
    
    for variable in variables_numericas:
        
        if not pd.api.types.is_numeric_dtype(datos_modelo[variable]):
            raise TypeError(
                f"La variable '{variable}' fue indicada como numérica, "
                "pero no tiene tipo numérico."
            )
    
    #-----------------------------------------------------
    # Convertir variables categóricas a texto
    #-----------------------------------------------------
    
    for variable in variables_categoricas:
        
        datos_modelo[variable] = datos_modelo[variable].astype(str)
    
    #-----------------------------------------------------
    # Revisar valores perdidos
    #-----------------------------------------------------
    
    valores_perdidos = datos_modelo.isna().sum()
    
    total_valores_perdidos = valores_perdidos.sum()
    
    if total_valores_perdidos > 0:
        
        print("Advertencia: existen valores perdidos en datos_modelo.")
        print(valores_perdidos[valores_perdidos > 0])
    
    #-----------------------------------------------------
    # Crear datos con identificador si existe
    #-----------------------------------------------------
    
    if variable_id is not None:
        
        datos_identificacion = datos[[variable_id]].copy()
        
    else:
        
        datos_identificacion = pd.DataFrame(
            {
                "id_registro": range(1, len(datos) + 1)
            }
        )
        
        variable_id = "id_registro"
    
    #-----------------------------------------------------
    # Mostrar resumen
    #-----------------------------------------------------
    
    if mostrar_resumen:
        
        print("\n========================================")
        print("PREPARACIÓN DE DATOS PARA EL MODELO")
        print("========================================")
        print("Registros:", datos_modelo.shape[0])
        print("Variables del modelo:", datos_modelo.shape[1])
        print("----------------------------------------")
        print("Variables numéricas:")
        print(variables_numericas)
        print("----------------------------------------")
        print("Variables categóricas:")
        print(variables_categoricas)
        print("----------------------------------------")
        print("Variable ID:", variable_id)
        print("Valores perdidos totales:", total_valores_perdidos)
        print("========================================\n")
    
    #-----------------------------------------------------
    # Salida
    #-----------------------------------------------------
    
    return {
        "datos_modelo": datos_modelo,
        "datos_identificacion": datos_identificacion,
        "variables_numericas": variables_numericas,
        "variables_categoricas": variables_categoricas,
        "variables_modelo": variables_modelo,
        "variable_id": variable_id,
        "valores_perdidos": valores_perdidos,
        "total_valores_perdidos": total_valores_perdidos
    }




#=========================================================
# FUNCIÓN
# f_crear_modelo_clustering_jerarquico()
#
# OBJETIVO:
# - Crear un modelo de clustering jerárquico en Python.
# - Usar distancia de Gower para datos mixtos.
# - Construir el modelo con scipy.cluster.hierarchy.linkage().
# - Asignar clústeres con fcluster().
# - Devolver tablas de pertenencia y frecuencia.
#
# REQUIERE:
# - pip install gower
#=========================================================

def f_crear_modelo_clustering_jerarquico(
    datos,
    datos_modelo=None,
    datos_identificacion=None,
    variable_id=None,
    k=4,
    metodo_enlace="complete",
    graficar=False,
    etiquetas=True,
    figsize=(12, 6),
    titulo="Dendrograma del clustering jerárquico",
    decimales=4,
    mostrar_resumen=True
):
    
    #-----------------------------------------------------
    # Librerías
    #-----------------------------------------------------
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    
    try:
        import gower
    except ImportError:
        raise ImportError(
            "No está instalado el paquete 'gower'. "
            "Instálalo con: pip install gower"
        )
    
    #-----------------------------------------------------
    # Validaciones básicas
    #-----------------------------------------------------
    
    if not isinstance(datos, pd.DataFrame):
        raise TypeError("El objeto 'datos' debe ser un DataFrame de pandas.")
    
    if datos_modelo is None:
        datos_modelo = datos.copy()
    else:
        if not isinstance(datos_modelo, pd.DataFrame):
            raise TypeError("El objeto 'datos_modelo' debe ser un DataFrame de pandas.")
        datos_modelo = datos_modelo.copy()
    
    if len(datos_modelo) != len(datos):
        raise ValueError(
            "'datos' y 'datos_modelo' deben tener el mismo número de registros."
        )
    
    n = datos_modelo.shape[0]
    
    if n < 3:
        raise ValueError("Se requieren al menos 3 registros para clustering jerárquico.")
    
    if k < 2:
        raise ValueError("El número de clústeres k debe ser al menos 2.")
    
    if k >= n:
        raise ValueError("El número de clústeres k debe ser menor que el número de registros.")
    
    metodos_validos = [
        "single",
        "complete",
        "average",
        "weighted",
        "centroid",
        "median",
        "ward"
    ]
    
    if metodo_enlace not in metodos_validos:
        raise ValueError(
            "El método de enlace debe ser uno de: "
            + ", ".join(metodos_validos)
        )
    
    if metodo_enlace == "ward":
        raise ValueError(
            "El método 'ward' no es recomendable con distancia de Gower. "
            "Usa preferentemente 'complete' o 'average'."
        )
    
    #-----------------------------------------------------
    # Identificación de registros
    #-----------------------------------------------------
    
    if datos_identificacion is not None:
        
        if not isinstance(datos_identificacion, pd.DataFrame):
            raise TypeError("'datos_identificacion' debe ser un DataFrame.")
        
        if len(datos_identificacion) != n:
            raise ValueError(
                "'datos_identificacion' debe tener el mismo número de registros que datos_modelo."
            )
        
        datos_id = datos_identificacion.copy()
        
        if variable_id is None:
            variable_id = datos_id.columns[0]
        
    else:
        
        if variable_id is not None and variable_id in datos.columns:
            datos_id = datos[[variable_id]].copy()
        else:
            variable_id = "id_registro"
            datos_id = pd.DataFrame({
                variable_id: range(1, n + 1)
            })
    
    #-----------------------------------------------------
    # Validar valores perdidos
    #-----------------------------------------------------
    
    valores_perdidos = datos_modelo.isna().sum()
    total_valores_perdidos = int(valores_perdidos.sum())
    
    if total_valores_perdidos > 0:
        raise ValueError(
            "Existen valores perdidos en datos_modelo. "
            "Debes imputarlos o eliminarlos antes de crear el modelo."
        )
    
    #-----------------------------------------------------
    # Convertir variables categóricas a texto
    # Gower identifica tipos numéricos y no numéricos
    #-----------------------------------------------------
    
    datos_gower = datos_modelo.copy()
    
    for columna in datos_gower.columns:
        if not pd.api.types.is_numeric_dtype(datos_gower[columna]):
            datos_gower[columna] = datos_gower[columna].astype(str)
    
    #-----------------------------------------------------
    # Calcular distancia de Gower
    #-----------------------------------------------------
    
    matriz_gower = gower.gower_matrix(datos_gower)
    
    matriz_gower = np.asarray(matriz_gower, dtype=float)
    
    # Por seguridad, eliminar posibles valores nan
    if np.isnan(matriz_gower).any():
        raise ValueError(
            "La matriz de Gower contiene valores NaN. "
            "Revisa valores perdidos o variables problemáticas."
        )
    
    #-----------------------------------------------------
    # Convertir matriz cuadrada a formato condensado
    #-----------------------------------------------------
    
    distancias_condensadas = squareform(
        matriz_gower,
        checks=False
    )
    
    #-----------------------------------------------------
    # Crear modelo jerárquico
    #-----------------------------------------------------
    
    modelo_hc = linkage(
        distancias_condensadas,
        method=metodo_enlace
    )
    
    #-----------------------------------------------------
    # Asignar clústeres
    #-----------------------------------------------------
    
    cluster = fcluster(
        modelo_hc,
        t=k,
        criterion="maxclust"
    )
    
    #-----------------------------------------------------
    # Crear datos con clúster
    #-----------------------------------------------------
    
    datos_cluster = datos.copy()
    datos_cluster["cluster"] = cluster
    
    #-----------------------------------------------------
    # Tabla de pertenencia
    #-----------------------------------------------------
    
    tabla_pertenencia = pd.concat(
        [
            datos_id.reset_index(drop=True),
            pd.DataFrame({"cluster": cluster}),
            datos_modelo.reset_index(drop=True)
        ],
        axis=1
    )
    
    tabla_pertenencia = tabla_pertenencia.sort_values(
        by="cluster"
    ).reset_index(drop=True)
    
    #-----------------------------------------------------
    # Frecuencia por clúster
    #-----------------------------------------------------
    
    frecuencia_cluster = (
        pd.Series(cluster)
        .value_counts()
        .sort_index()
        .reset_index()
    )
    
    frecuencia_cluster.columns = [
        "cluster",
        "frecuencia"
    ]
    
    frecuencia_cluster["porcentaje"] = (
        frecuencia_cluster["frecuencia"] / n * 100
    ).round(2)
    
    #-----------------------------------------------------
    # Matriz Gower como DataFrame
    #-----------------------------------------------------
    
    etiquetas_id = datos_id[variable_id].astype(str).values
    
    matriz_gower_df = pd.DataFrame(
        np.round(matriz_gower, decimales),
        index=etiquetas_id,
        columns=etiquetas_id
    )
    
    #-----------------------------------------------------
    # Graficar dendrograma opcional
    #-----------------------------------------------------
    
    figura = None
    
    if graficar:
        
        figura = plt.figure(figsize=figsize)
        
        if etiquetas:
            labels = etiquetas_id
        else:
            labels = None
        
        dendrogram(
            modelo_hc,
            labels=labels
        )
        
        plt.title(titulo)
        plt.xlabel("Registros")
        plt.ylabel("Distancia de Gower")
        plt.tight_layout()
        plt.show()
    
    #-----------------------------------------------------
    # Resumen en consola
    #-----------------------------------------------------
    
    if mostrar_resumen:
        
        print("\n========================================")
        print("MODELO DE CLUSTERING JERÁRQUICO")
        print("========================================")
        print("Registros:", n)
        print("Variables del modelo:", datos_modelo.shape[1])
        print("Distancia utilizada: Gower")
        print("Método de enlace:", metodo_enlace)
        print("Número de clústeres K:", k)
        print("----------------------------------------")
        print("Frecuencia por clúster:")
        print(frecuencia_cluster.to_string(index=False))
        print("========================================\n")
    
    #-----------------------------------------------------
    # Salida
    #-----------------------------------------------------
    
    return {
        "modelo_hc": modelo_hc,
        "datos_cluster": datos_cluster,
        "datos_modelo": datos_modelo,
        "datos_gower": datos_gower,
        "datos_identificacion": datos_id,
        "variable_id": variable_id,
        "matriz_gower": matriz_gower,
        "matriz_gower_df": matriz_gower_df,
        "distancias_condensadas": distancias_condensadas,
        "cluster": cluster,
        "tabla_pertenencia": tabla_pertenencia,
        "frecuencia_cluster": frecuencia_cluster,
        "k": k,
        "metodo_enlace": metodo_enlace,
        "tipo_distancia": "gower",
        "figura": figura
    }

#=========================================================
# FUNCIÓN
# f_perfiles()
#
# OBJETIVO:
# - Crear perfiles numéricos por clúster.
# - Crear perfiles categóricos por clúster.
# - Integrar ambos perfiles en una tabla general.
# - Calcular frecuencia y porcentaje por clúster.
#
# ENTRADAS:
# - modelo: diccionario generado por f_crear_modelo_clustering_jerarquico()
# - variables_numericas: lista de variables numéricas
# - variables_categoricas: lista de variables categóricas
# - variable_cluster: nombre de la variable de clúster
#=========================================================

def f_perfiles(
    modelo,
    variables_numericas,
    variables_categoricas,
    variable_cluster="cluster",
    decimales=2,
    mostrar_resumen=True
):
    
    #-----------------------------------------------------
    # Librerías
    #-----------------------------------------------------
    
    import pandas as pd
    import numpy as np
    
    #-----------------------------------------------------
    # Validaciones básicas
    #-----------------------------------------------------
    
    if not isinstance(modelo, dict):
        raise TypeError("El objeto 'modelo' debe ser un diccionario.")
    
    if "datos_cluster" not in modelo:
        raise ValueError("El modelo no contiene el elemento 'datos_cluster'.")
    
    datos_cluster = modelo["datos_cluster"].copy()
    
    if not isinstance(datos_cluster, pd.DataFrame):
        raise TypeError("'modelo[datos_cluster]' debe ser un DataFrame de pandas.")
    
    if variable_cluster not in datos_cluster.columns:
        raise ValueError(
            f"La variable de clúster '{variable_cluster}' no existe en datos_cluster."
        )
    
    #-----------------------------------------------------
    # Validar variables numéricas y categóricas
    #-----------------------------------------------------
    
    variables_no_existen_num = [
        variable for variable in variables_numericas
        if variable not in datos_cluster.columns
    ]
    
    variables_no_existen_cat = [
        variable for variable in variables_categoricas
        if variable not in datos_cluster.columns
    ]
    
    if len(variables_no_existen_num) > 0:
        raise ValueError(
            "Estas variables numéricas no existen en datos_cluster: "
            + ", ".join(variables_no_existen_num)
        )
    
    if len(variables_no_existen_cat) > 0:
        raise ValueError(
            "Estas variables categóricas no existen en datos_cluster: "
            + ", ".join(variables_no_existen_cat)
        )
    
    for variable in variables_numericas:
        if not pd.api.types.is_numeric_dtype(datos_cluster[variable]):
            raise TypeError(
                f"La variable '{variable}' fue indicada como numérica, "
                "pero no tiene tipo numérico."
            )
    
    #-----------------------------------------------------
    # Función auxiliar: moda
    #-----------------------------------------------------
    
    def f_moda(serie):
        
        serie = serie.dropna()
        
        if len(serie) == 0:
            return np.nan
        
        return serie.mode().iloc[0]
    
    #-----------------------------------------------------
    # Frecuencia por clúster
    #-----------------------------------------------------
    
    frecuencia_cluster = (
        datos_cluster[variable_cluster]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    
    frecuencia_cluster.columns = [
        variable_cluster,
        "frecuencia"
    ]
    
    frecuencia_cluster["porcentaje"] = (
        frecuencia_cluster["frecuencia"] / len(datos_cluster) * 100
    ).round(decimales)
    
    #-----------------------------------------------------
    # Perfil numérico por clúster
    #-----------------------------------------------------
    
    perfil_numerico = (
        datos_cluster
        .groupby(variable_cluster)[variables_numericas]
        .mean()
        .round(decimales)
        .reset_index()
    )
    
    # Agregar n después de cluster
    tamanios_cluster = (
        datos_cluster
        .groupby(variable_cluster)
        .size()
        .reset_index(name="n")
    )
    
    perfil_numerico = tamanios_cluster.merge(
        perfil_numerico,
        on=variable_cluster,
        how="left"
    )
    
    #-----------------------------------------------------
    # Perfil categórico por clúster
    #-----------------------------------------------------
    
    perfil_categorico = (
        datos_cluster
        .groupby(variable_cluster)[variables_categoricas]
        .agg(f_moda)
        .reset_index()
    )
    
    # Renombrar columnas categóricas
    nuevos_nombres = {
        variable: variable + "_dominante"
        for variable in variables_categoricas
    }
    
    perfil_categorico = perfil_categorico.rename(
        columns=nuevos_nombres
    )
    
    #-----------------------------------------------------
    # Perfil general
    #-----------------------------------------------------
    
    perfil_general = perfil_numerico.merge(
        perfil_categorico,
        on=variable_cluster,
        how="left"
    )
    
    #-----------------------------------------------------
    # Resumen en consola
    #-----------------------------------------------------
    
    if mostrar_resumen:
        
        print("\n========================================")
        print("PERFILES DE CLÚSTERES")
        print("========================================")
        print("Número de registros:", len(datos_cluster))
        print("Variable de clúster:", variable_cluster)
        print("----------------------------------------")
        print("Variables numéricas:")
        print(variables_numericas)
        print("----------------------------------------")
        print("Variables categóricas:")
        print(variables_categoricas)
        print("----------------------------------------")
        print("Frecuencia por clúster:")
        print(frecuencia_cluster.to_string(index=False))
        print("========================================\n")
    
    #-----------------------------------------------------
    # Salida
    #-----------------------------------------------------
    
    return {
        "frecuencia_cluster": frecuencia_cluster,
        "perfil_numerico": perfil_numerico,
        "perfil_categorico": perfil_categorico,
        "perfil_general": perfil_general,
        "variables_numericas": variables_numericas,
        "variables_categoricas": variables_categoricas,
        "variable_cluster": variable_cluster
    }

#=========================================================
# FUNCIÓN
# f_evaluacion_clustering_jerarquico()
#
# OBJETIVO:
# - Evaluar un modelo de clustering jerárquico.
# - Calcular Silhouette promedio para distintos valores de K.
# - Calcular correlación cofenética.
# - Presentar tabla de evaluación.
# - Generar resumen interpretativo.
#
# REQUIERE:
# - Modelo generado con f_crear_modelo_clustering_jerarquico()
#=========================================================

def f_evaluacion_clustering_jerarquico(
    modelo,
    k_min=2,
    k_max=8,
    variable_cluster="cluster",
    decimales=4,
    graficar=True,
    mostrar_resumen=True
):
    
    #-----------------------------------------------------
    # Librerías
    #-----------------------------------------------------
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    from scipy.cluster.hierarchy import fcluster, cophenet
    from sklearn.metrics import silhouette_score
    
    #-----------------------------------------------------
    # Validaciones básicas
    #-----------------------------------------------------
    
    if not isinstance(modelo, dict):
        raise TypeError("El objeto 'modelo' debe ser un diccionario.")
    
    elementos_requeridos = [
        "modelo_hc",
        "matriz_gower",
        "distancias_condensadas"
    ]
    
    for elemento in elementos_requeridos:
        if elemento not in modelo:
            raise ValueError(
                f"El modelo no contiene el elemento requerido: '{elemento}'."
            )
    
    modelo_hc = modelo["modelo_hc"]
    matriz_gower = modelo["matriz_gower"]
    distancias_condensadas = modelo["distancias_condensadas"]
    
    matriz_gower = np.asarray(matriz_gower, dtype=float)
    
    n = matriz_gower.shape[0]
    
    if matriz_gower.shape[0] != matriz_gower.shape[1]:
        raise ValueError("'matriz_gower' debe ser una matriz cuadrada.")
    
    if n < 3:
        raise ValueError("Se requieren al menos 3 registros para evaluar Silhouette.")
    
    if k_min < 2:
        raise ValueError("k_min debe ser al menos 2.")
    
    if k_max >= n:
        k_max = n - 1
        print(
            f"Advertencia: k_max se ajustó automáticamente a {k_max}, "
            "porque debe ser menor que el número de registros."
        )
    
    if k_max < k_min:
        raise ValueError("k_max debe ser mayor o igual que k_min.")
    
    if np.isnan(matriz_gower).any():
        raise ValueError("La matriz de distancias contiene valores NaN.")
    
    #-----------------------------------------------------
    # Funciones auxiliares de interpretación
    #-----------------------------------------------------
    
    def interpretar_silhouette(valor):
        
        if valor >= 0.70:
            return "Estructura fuerte"
        elif valor >= 0.50:
            return "Estructura razonable"
        elif valor >= 0.25:
            return "Estructura débil o moderada"
        else:
            return "Estructura poco definida"
    
    def interpretar_cofenetica(valor):
        
        if valor >= 0.90:
            return "Muy alta: el dendrograma representa muy bien las distancias originales"
        elif valor >= 0.80:
            return "Alta: el dendrograma representa adecuadamente las distancias originales"
        elif valor >= 0.70:
            return "Aceptable: el dendrograma conserva parte importante de la estructura"
        else:
            return "Baja: el dendrograma representa débilmente las distancias originales"
    
    #-----------------------------------------------------
    # Evaluación con Silhouette para varios K
    #-----------------------------------------------------
    
    resultados = []
    
    for k in range(k_min, k_max + 1):
        
        clusters_k = fcluster(
            modelo_hc,
            t=k,
            criterion="maxclust"
        )
        
        numero_clusters = len(np.unique(clusters_k))
        
        # Silhouette requiere al menos 2 grupos
        # y menos grupos que registros
        if numero_clusters < 2 or numero_clusters >= n:
            continue
        
        silhouette_promedio = silhouette_score(
            matriz_gower,
            clusters_k,
            metric="precomputed"
        )
        
        frecuencias = pd.Series(clusters_k).value_counts().sort_index()
        
        frecuencia_texto = ", ".join(
            [str(valor) for valor in frecuencias.values]
        )
        
        resultados.append({
            "k": k,
            "clusters_obtenidos": numero_clusters,
            "silhouette_promedio": round(silhouette_promedio, decimales),
            "interpretacion_silhouette": interpretar_silhouette(silhouette_promedio),
            "cluster_menor": int(frecuencias.min()),
            "cluster_mayor": int(frecuencias.max()),
            "frecuencia_cluster": frecuencia_texto
        })
    
    tabla_evaluacion = pd.DataFrame(resultados)
    
    if tabla_evaluacion.empty:
        raise ValueError(
            "No fue posible calcular Silhouette para los valores de K indicados."
        )
    
    #-----------------------------------------------------
    # Mejor K según Silhouette
    #-----------------------------------------------------
    
    indice_mejor = tabla_evaluacion["silhouette_promedio"].idxmax()
    
    mejor_k = int(tabla_evaluacion.loc[indice_mejor, "k"])
    
    mejor_silhouette = float(
        tabla_evaluacion.loc[indice_mejor, "silhouette_promedio"]
    )
    
    interpretacion_silhouette = tabla_evaluacion.loc[
        indice_mejor,
        "interpretacion_silhouette"
    ]
    
    #-----------------------------------------------------
    # Correlación cofenética
    #-----------------------------------------------------
    
    correlacion_cofenetica, distancias_cofeneticas = cophenet(
        modelo_hc,
        distancias_condensadas
    )
    
    correlacion_cofenetica = round(
        float(correlacion_cofenetica),
        decimales
    )
    
    interpretacion_cofenetica = interpretar_cofenetica(
        correlacion_cofenetica
    )
    
    #-----------------------------------------------------
    # Tabla resumen
    #-----------------------------------------------------
    
    tabla_resumen = pd.DataFrame({
        "metrica": [
            "Mejor K según Silhouette",
            "Silhouette promedio del mejor K",
            "Interpretación Silhouette",
            "Correlación cofenética",
            "Interpretación cofenética"
        ],
        "valor": [
            mejor_k,
            mejor_silhouette,
            interpretacion_silhouette,
            correlacion_cofenetica,
            interpretacion_cofenetica
        ]
    })
    
    #-----------------------------------------------------
    # Gráfico de Silhouette
    #-----------------------------------------------------
    
    figura = None
    
    if graficar:
        
        figura = plt.figure(figsize=(8, 5))
        
        plt.plot(
            tabla_evaluacion["k"],
            tabla_evaluacion["silhouette_promedio"],
            marker="o"
        )
        
        plt.axvline(
            x=mejor_k,
            linestyle="--"
        )
        
        plt.title("Evaluación del clustering jerárquico")
        plt.xlabel("Número de clústeres K")
        plt.ylabel("Silhouette promedio")
        plt.xticks(tabla_evaluacion["k"])
        plt.grid(alpha=0.3)
        plt.show()
    
    #-----------------------------------------------------
    # Interpretación textual
    #-----------------------------------------------------
    
    interpretacion_texto = (
        "La evaluación del clustering jerárquico se realizó mediante el "
        "estadístico Silhouette y la correlación cofenética. De acuerdo "
        f"con Silhouette, el mejor valor de K fue {mejor_k}, con un "
        f"promedio de {mejor_silhouette}, lo que indica una "
        f"{interpretacion_silhouette.lower()}. La correlación cofenética "
        f"fue de {correlacion_cofenetica}, por lo que se interpreta como: "
        f"{interpretacion_cofenetica.lower()}."
    )
    
    #-----------------------------------------------------
    # Resumen en consola
    #-----------------------------------------------------
    
    if mostrar_resumen:
        
        print("\n========================================")
        print("EVALUACIÓN DEL CLUSTERING JERÁRQUICO")
        print("========================================")
        print("Número de registros:", n)
        
        if "tipo_distancia" in modelo:
            print("Tipo de distancia:", modelo["tipo_distancia"])
        
        if "metodo_enlace" in modelo:
            print("Método de enlace:", modelo["metodo_enlace"])
        
        print("Valores de K evaluados:", k_min, "a", k_max)
        print("----------------------------------------")
        print("Mejor K según Silhouette:", mejor_k)
        print("Silhouette promedio:", mejor_silhouette)
        print("Interpretación:", interpretacion_silhouette)
        print("----------------------------------------")
        print("Correlación cofenética:", correlacion_cofenetica)
        print("Interpretación:", interpretacion_cofenetica)
        print("========================================\n")
        
        print("TABLA DE EVALUACIÓN:")
        print(tabla_evaluacion.to_string(index=False))
        
        print("\nINTERPRETACIÓN PARA REPORTE:")
        print(interpretacion_texto)
    
    #-----------------------------------------------------
    # Salida
    #-----------------------------------------------------
    
    return {
        "tabla_evaluacion": tabla_evaluacion,
        "tabla_resumen": tabla_resumen,
        "mejor_k": mejor_k,
        "mejor_silhouette": mejor_silhouette,
        "correlacion_cofenetica": correlacion_cofenetica,
        "interpretacion_silhouette": interpretacion_silhouette,
        "interpretacion_cofenetica": interpretacion_cofenetica,
        "interpretacion_texto": interpretacion_texto,
        "figura": figura
    }