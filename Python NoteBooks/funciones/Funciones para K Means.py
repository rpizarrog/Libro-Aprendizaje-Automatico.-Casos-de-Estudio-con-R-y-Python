# Funciones para implementar # K-Means

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
