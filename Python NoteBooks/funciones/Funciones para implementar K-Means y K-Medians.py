# Funciones para implementar # K-Means y K Medians

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


#=========================================================
# FUNCIÓN
# f_crear_KMeans()
#
# OBJETIVO:
# - Crear un modelo K-Means de forma robusta.
# - Permitir trabajar con datos estandarizados o no.
# - Devolver centroides, clústeres, SSE y frecuencias.
#=========================================================

def f_crear_KMeans(
    datos,
    variables=None,
    n_clusters=3,
    n_init=25,
    max_iter=300,
    random_state=2026,
    estandarizar=True,
    decimales=4,
    nombre_cluster="cluster_KMeans"
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
    
    #-----------------------------------------------------
    # Seleccionar variables numéricas
    #-----------------------------------------------------
    
    if variables is None:
        
        variables = datos.select_dtypes(include=np.number).columns.tolist()
        
        if len(variables) == 0:
            raise ValueError("No existen variables numéricas para aplicar K-Means.")
    
    else:
        
        for variable in variables:
            
            if variable not in datos.columns:
                raise ValueError(f"La variable '{variable}' no existe en los datos.")
            
            if not pd.api.types.is_numeric_dtype(datos[variable]):
                raise TypeError(f"La variable '{variable}' no es numérica.")
    
    #-----------------------------------------------------
    # Matriz de datos original
    #-----------------------------------------------------
    
    X_original = datos[variables].to_numpy(dtype=float)
    
    if np.isnan(X_original).any():
        raise ValueError(
            "Existen valores perdidos en las variables seleccionadas. "
            "Deben tratarse antes de aplicar K-Means."
        )
    
    if n_clusters > X_original.shape[0]:
        raise ValueError(
            "El número de clústeres no puede ser mayor que el número de registros."
        )
    
    #-----------------------------------------------------
    # Estandarización opcional
    #-----------------------------------------------------
    
    if estandarizar:
        
        escalador = StandardScaler()
        
        X_modelo = escalador.fit_transform(X_original)
        
    else:
        
        escalador = None
        
        X_modelo = X_original.copy()
    
    #-----------------------------------------------------
    # Crear modelo K-Means
    #-----------------------------------------------------
    
    modelo_KMeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state
    )
    
    modelo_KMeans.fit(X_modelo)
    
    #-----------------------------------------------------
    # Obtener clústeres
    #-----------------------------------------------------
    
    cluster = modelo_KMeans.labels_ + 1
    
    #-----------------------------------------------------
    # Centroides en escala del modelo
    #-----------------------------------------------------
    
    centroides_modelo = pd.DataFrame(
        modelo_KMeans.cluster_centers_,
        columns=variables
    )
    
    centroides_modelo.insert(
        0,
        "cluster",
        range(1, n_clusters + 1)
    )
    
    centroides_modelo[variables] = centroides_modelo[variables].round(decimales)
    
    #-----------------------------------------------------
    # Centroides en escala original
    #-----------------------------------------------------
    
    if estandarizar:
        
        centroides_originales_array = escalador.inverse_transform(
            modelo_KMeans.cluster_centers_
        )
        
    else:
        
        centroides_originales_array = modelo_KMeans.cluster_centers_
    
    centroides_originales = pd.DataFrame(
        centroides_originales_array,
        columns=variables
    )
    
    centroides_originales.insert(
        0,
        "cluster",
        range(1, n_clusters + 1)
    )
    
    centroides_originales[variables] = centroides_originales[variables].round(decimales)
    
    #-----------------------------------------------------
    # Datos con clúster asignado
    #-----------------------------------------------------
    
    datos_cluster = datos.copy()
    
    datos_cluster[nombre_cluster] = cluster
    
    #-----------------------------------------------------
    # Frecuencia por clúster
    #-----------------------------------------------------
    
    frecuencia_cluster = (
        datos_cluster[nombre_cluster]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    
    frecuencia_cluster.columns = ["cluster", "n"]
    
    #-----------------------------------------------------
    # SSE total
    # En scikit-learn se conoce como inertia_
    #-----------------------------------------------------
    
    SSE_total = round(modelo_KMeans.inertia_, decimales)
    
    #-----------------------------------------------------
    # SSE por clúster
    #-----------------------------------------------------
    
    distancias_cuadradas = np.sum(
        (X_modelo - modelo_KMeans.cluster_centers_[modelo_KMeans.labels_]) ** 2,
        axis=1
    )
    
    SSE_por_cluster = pd.DataFrame({
        "cluster": cluster,
        "SSE_registro": distancias_cuadradas
    })
    
    SSE_por_cluster = (
        SSE_por_cluster
        .groupby("cluster")["SSE_registro"]
        .sum()
        .reset_index()
        .rename(columns={"SSE_registro": "SSE"})
    )
    
    SSE_por_cluster["SSE"] = SSE_por_cluster["SSE"].round(decimales)
    
    #-----------------------------------------------------
    # Resultado
    #-----------------------------------------------------
    
    resultado = {
        "modelo": modelo_KMeans,
        "cluster": cluster,
        "datos_cluster": datos_cluster,
        "centroides_modelo": centroides_modelo,
        "centroides_originales": centroides_originales,
        "SSE_total": SSE_total,
        "SSE_por_cluster": SSE_por_cluster,
        "frecuencia_cluster": frecuencia_cluster,
        "datos_modelo": pd.DataFrame(X_modelo, columns=variables),
        "escalador": escalador,
        "variables": variables,
        "n_clusters": n_clusters,
        "n_init": n_init,
        "max_iter": max_iter,
        "random_state": random_state,
        "estandarizar": estandarizar
    }
    
    return resultado

#=========================================================
# FUNCIÓN
# f_crear_KMedians()
#
# OBJETIVO:
# - Crear modelo K-Medians en Python.
# - Usar distancia Manhattan.
# - Actualizar centros con medianas.
# - Permitir datos estandarizados o no estandarizados.
# - Probar varias inicializaciones con nstart.
# - Conservar la solución con menor SAD.
#
# DEVUELVE:
# - cluster: grupo asignado a cada registro.
# - medianas_modelo: medianas en la escala usada por el modelo.
# - medianas_originales: medianas en escala original.
# - SAD_total: suma total de desviaciones absolutas.
# - SAD_por_cluster.
# - SAD_por_registro.
# - frecuencia_cluster.
#=========================================================

def f_crear_KMedians(
    datos,
    variables,
    centers=3,
    nstart=25,
    max_iter=100,
    semilla=123,
    estandarizar=False,
    decimales=4
):
    
    #-----------------------------------------------------
    # Validaciones
    #-----------------------------------------------------
    
    if not isinstance(datos, pd.DataFrame):
        raise TypeError("El objeto 'datos' debe ser un DataFrame de pandas.")
    
    if variables is None or len(variables) == 0:
        raise ValueError("Debe indicar las variables a utilizar.")
    
    for variable in variables:
        if variable not in datos.columns:
            raise ValueError(f"La variable '{variable}' no existe en los datos.")
        
        if not pd.api.types.is_numeric_dtype(datos[variable]):
            raise TypeError(f"La variable '{variable}' no es numérica.")
    
    if centers < 2:
        raise ValueError("El número de clústeres debe ser al menos 2.")
    
    if nstart < 1:
        raise ValueError("nstart debe ser al menos 1.")
    
    if max_iter < 1:
        raise ValueError("max_iter debe ser al menos 1.")
    
    #-----------------------------------------------------
    # Datos originales
    #-----------------------------------------------------
    
    X_original = datos[variables].to_numpy(dtype=float)
    
    if np.isnan(X_original).any():
        raise ValueError(
            "Existen valores perdidos. Deben tratarse antes de aplicar K-Medians."
        )
    
    #-----------------------------------------------------
    # Estandarizar si se solicita
    #-----------------------------------------------------
    
    if estandarizar:
        escalador = StandardScaler()
        X_modelo = escalador.fit_transform(X_original)
    else:
        escalador = None
        X_modelo = X_original.copy()
    
    n = X_modelo.shape[0]
    
    #-----------------------------------------------------
    # Objetos para guardar la mejor solución
    #-----------------------------------------------------
    
    mejor_SAD = np.inf
    mejor_cluster = None
    mejores_medianas = None
    mejor_iteracion = None
    mejor_inicio = None
    
    resumen_inicios = []
    
    rng = np.random.default_rng(semilla)
    
    #-----------------------------------------------------
    # Varias inicializaciones
    #-----------------------------------------------------
    
    for inicio in range(1, nstart + 1):
        
        #-------------------------------------------------
        # Seleccionar centros iniciales aleatorios
        #-------------------------------------------------
        
        indices_iniciales = rng.choice(
            n,
            size=centers,
            replace=False
        )
        
        medianas = X_modelo[indices_iniciales, :].copy()
        
        cluster_anterior = np.full(n, -1)
        
        #-------------------------------------------------
        # Iteraciones
        #-------------------------------------------------
        
        for iteracion in range(1, max_iter + 1):
            
            #---------------------------------------------
            # Distancia Manhattan a cada mediana
            #---------------------------------------------
            
            distancias = np.zeros((n, centers))
            
            for k in range(centers):
                distancias[:, k] = np.sum(
                    np.abs(X_modelo - medianas[k, :]),
                    axis=1
                )
            
            #---------------------------------------------
            # Asignar al centro más cercano
            #---------------------------------------------
            
            cluster = np.argmin(distancias, axis=1)
            
            #---------------------------------------------
            # Detener si ya no cambia
            #---------------------------------------------
            
            if np.array_equal(cluster, cluster_anterior):
                break
            
            cluster_anterior = cluster.copy()
            
            #---------------------------------------------
            # Actualizar medianas
            #---------------------------------------------
            
            for k in range(centers):
                
                if np.sum(cluster == k) > 0:
                    medianas[k, :] = np.median(
                        X_modelo[cluster == k, :],
                        axis=0
                    )
                else:
                    # Si un clúster queda vacío, se reinicia
                    indice_aleatorio = rng.integers(0, n)
                    medianas[k, :] = X_modelo[indice_aleatorio, :]
        
        #-------------------------------------------------
        # Calcular SAD de esta inicialización
        #-------------------------------------------------
        
        SAD_por_registro = np.zeros(n)
        
        for i in range(n):
            k = cluster[i]
            SAD_por_registro[i] = np.sum(
                np.abs(X_modelo[i, :] - medianas[k, :])
            )
        
        SAD_total = np.sum(SAD_por_registro)
        
        resumen_inicios.append({
            "inicio": inicio,
            "SAD_total": round(SAD_total, decimales),
            "iteraciones": iteracion
        })
        
        #-------------------------------------------------
        # Guardar mejor solución
        #-------------------------------------------------
        
        if SAD_total < mejor_SAD:
            mejor_SAD = SAD_total
            mejor_cluster = cluster.copy()
            mejores_medianas = medianas.copy()
            mejor_iteracion = iteracion
            mejor_inicio = inicio
    
    #-----------------------------------------------------
    # Calcular SAD final de la mejor solución
    #-----------------------------------------------------
    
    SAD_por_registro = np.zeros(n)
    
    for i in range(n):
        k = mejor_cluster[i]
        SAD_por_registro[i] = np.sum(
            np.abs(X_modelo[i, :] - mejores_medianas[k, :])
        )
    
    SAD_por_cluster = []
    
    for k in range(centers):
        SAD_por_cluster.append(
            np.sum(SAD_por_registro[mejor_cluster == k])
        )
    
    #-----------------------------------------------------
    # Medianas en la escala del modelo
    #-----------------------------------------------------
    
    medianas_modelo = pd.DataFrame(
        mejores_medianas,
        columns=variables
    )
    
    medianas_modelo.insert(
        0,
        "cluster",
        np.arange(1, centers + 1)
    )
    
    medianas_modelo[variables] = medianas_modelo[variables].round(decimales)
    
    #-----------------------------------------------------
    # Medianas en escala original
    # Se recalculan directamente desde los datos originales
    # usando los clústeres finales.
    #-----------------------------------------------------
    
    datos_temp = datos.copy()
    datos_temp["cluster_KMedians"] = mejor_cluster + 1
    
    medianas_originales = (
        datos_temp
        .groupby("cluster_KMedians")[variables]
        .median()
        .reset_index()
        .rename(columns={"cluster_KMedians": "cluster"})
    )
    
    medianas_originales[variables] = medianas_originales[variables].round(decimales)
    
    #-----------------------------------------------------
    # Frecuencia por clúster
    #-----------------------------------------------------
    
    frecuencia_cluster = (
        datos_temp["cluster_KMedians"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    
    frecuencia_cluster.columns = ["cluster", "n"]
    
    #-----------------------------------------------------
    # Resultado
    #-----------------------------------------------------
    
    resultado = {
        "cluster": mejor_cluster + 1,
        "medianas_modelo": medianas_modelo,
        "medianas_originales": medianas_originales,
        "SAD_total": round(mejor_SAD, decimales),
        "SAD_por_cluster": np.round(SAD_por_cluster, decimales),
        "SAD_por_registro": np.round(SAD_por_registro, decimales),
        "frecuencia_cluster": frecuencia_cluster,
        "mejor_inicio": mejor_inicio,
        "iteraciones": mejor_iteracion,
        "resumen_inicios": pd.DataFrame(resumen_inicios),
        "datos_modelo": pd.DataFrame(X_modelo, columns=variables),
        "escalador": escalador,
        "variables": variables,
        "centers": centers,
        "nstart": nstart,
        "max_iter": max_iter,
        "estandarizar": estandarizar
    }
    
    return resultado

#=========================================================
# FUNCIÓN
# f_atipicos()
#
# OBJETIVO:
# - Detectar visualmente valores atípicos mediante
#   diagramas de caja.
# - Si no se indican variables, usa todas las numéricas.
#=========================================================

def f_atipicos(
    datos,
    variables=None,
    ncol=3,
    titulo="Detección visual de valores atípicos",
    color_caja="lightblue",
    color_atipicos="red"
):
    
    #-----------------------------------------------------
    # Validaciones
    #-----------------------------------------------------
    
    if not isinstance(datos, pd.DataFrame):
        raise TypeError("El objeto 'datos' debe ser un DataFrame de pandas.")
    
    #-----------------------------------------------------
    # Si no se indican variables, tomar todas las numéricas
    #-----------------------------------------------------
    
    if variables is None:
        
        variables = datos.select_dtypes(include=np.number).columns.tolist()
        
    else:
        
        for variable in variables:
            
            if variable not in datos.columns:
                raise ValueError(f"La variable '{variable}' no existe en los datos.")
            
            if not pd.api.types.is_numeric_dtype(datos[variable]):
                raise TypeError(f"La variable '{variable}' no es numérica.")
    
    if len(variables) == 0:
        raise ValueError("No hay variables numéricas para graficar.")
    
    #-----------------------------------------------------
    # Definir número de filas y columnas
    #-----------------------------------------------------
    
    n_variables = len(variables)
    
    nfilas = math.ceil(n_variables / ncol)
    
    #-----------------------------------------------------
    # Crear figura
    #-----------------------------------------------------
    
    fig, axes = plt.subplots(
        nfilas,
        ncol,
        figsize=(5 * ncol, 4 * nfilas)
    )
    
    # Convertir axes a arreglo plano
    if n_variables == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()
    
    #-----------------------------------------------------
    # Construir boxplot por variable
    #-----------------------------------------------------
    
    for i, variable in enumerate(variables):
        
        ax = axes[i]
        
        ax.boxplot(
            datos[variable].dropna(),
            patch_artist=True,
            boxprops=dict(
                facecolor=color_caja,
                color="gray"
            ),
            medianprops=dict(
                color="black"
            ),
            whiskerprops=dict(
                color="gray"
            ),
            capprops=dict(
                color="gray"
            ),
            flierprops=dict(
                marker="o",
                markerfacecolor=color_atipicos,
                markeredgecolor=color_atipicos,
                markersize=4,
                alpha=0.70
            )
        )
        
        ax.set_title(variable, fontweight="bold")
        ax.set_ylabel("Valor")
        ax.set_xticks([])
        ax.grid(alpha=0.25)
    
    #-----------------------------------------------------
    # Ocultar espacios vacíos si sobran subgráficos
    #-----------------------------------------------------
    
    for j in range(n_variables, len(axes)):
        axes[j].axis("off")
    
    #-----------------------------------------------------
    # Título general
    #-----------------------------------------------------
    
    fig.suptitle(
        titulo,
        fontsize=16,
        fontweight="bold"
    )
    
    plt.tight_layout()
    plt.show()
    
    return fig

#=========================================================
# FUNCIÓN
# f_transformar_atipicos()
#
# OBJETIVO:
# - Detectar valores atípicos mediante el criterio IQR.
# - Reemplazar valores atípicos por la mediana o la media.
# - Devolver los datos transformados y una tabla resumen.
#=========================================================

def f_transformar_atipicos(
    datos,
    variables=None,
    metodo="mediana",
    factor_iqr=1.5,
    decimales=4,
    mostrar_resumen=True
):
    
    #-----------------------------------------------------
    # Validaciones básicas
    #-----------------------------------------------------
    
    if not isinstance(datos, pd.DataFrame):
        raise TypeError("El objeto 'datos' debe ser un DataFrame de pandas.")
    
    if metodo not in ["mediana", "media"]:
        raise ValueError("El argumento 'metodo' debe ser 'mediana' o 'media'.")
    
    #-----------------------------------------------------
    # Copia de datos
    #-----------------------------------------------------
    
    datos_limpios = datos.copy()
    
    #-----------------------------------------------------
    # Seleccionar variables numéricas
    #-----------------------------------------------------
    
    if variables is None:
        
        variables = datos_limpios.select_dtypes(
            include=np.number
        ).columns.tolist()
        
    else:
        
        for variable in variables:
            
            if variable not in datos_limpios.columns:
                raise ValueError(f"La variable '{variable}' no existe en los datos.")
            
            if not pd.api.types.is_numeric_dtype(datos_limpios[variable]):
                raise TypeError(f"La variable '{variable}' no es numérica.")
    
    if len(variables) == 0:
        raise ValueError("No hay variables numéricas para transformar.")
    
    #-----------------------------------------------------
    # Crear tabla resumen
    #-----------------------------------------------------
    
    resumen = []
    
    #-----------------------------------------------------
    # Transformar valores atípicos
    #-----------------------------------------------------
    
    for variable in variables:
        
        serie = datos_limpios[variable]
        
        #-------------------------------------------------
        # Si la variable está completamente vacía
        #-------------------------------------------------
        
        if serie.isna().all():
            
            resumen.append({
                "variable": variable,
                "limite_inferior": np.nan,
                "limite_superior": np.nan,
                "n_atipicos": 0,
                "valor_reemplazo": np.nan,
                "metodo": metodo
            })
            
            continue
        
        #-------------------------------------------------
        # Calcular Q1, Q3 e IQR
        #-------------------------------------------------
        
        Q1 = serie.quantile(0.25)
        Q3 = serie.quantile(0.75)
        
        IQR_valor = Q3 - Q1
        
        limite_inferior = Q1 - factor_iqr * IQR_valor
        limite_superior = Q3 + factor_iqr * IQR_valor
        
        #-------------------------------------------------
        # Identificar valores atípicos
        #-------------------------------------------------
        
        condicion_atipicos = (
            (serie < limite_inferior) |
            (serie > limite_superior)
        )
        
        condicion_atipicos = condicion_atipicos.fillna(False)
        
        n_atipicos = int(condicion_atipicos.sum())
        
        #-------------------------------------------------
        # Calcular valor de reemplazo
        #-------------------------------------------------
        
        if metodo == "mediana":
            
            valor_reemplazo = serie.median(skipna=True)
            
        else:
            
            valor_reemplazo = serie.mean(skipna=True)
        
        valor_reemplazo = round(valor_reemplazo, decimales)
        
        #-------------------------------------------------
        # Reemplazar valores atípicos
        #-------------------------------------------------
        
        datos_limpios.loc[
            condicion_atipicos,
            variable
        ] = valor_reemplazo
        
        #-------------------------------------------------
        # Guardar resumen
        #-------------------------------------------------
        
        resumen.append({
            "variable": variable,
            "limite_inferior": round(limite_inferior, decimales),
            "limite_superior": round(limite_superior, decimales),
            "n_atipicos": n_atipicos,
            "valor_reemplazo": valor_reemplazo,
            "metodo": metodo
        })
    
    #-----------------------------------------------------
    # Convertir resumen a DataFrame
    #-----------------------------------------------------
    
    resumen_atipicos = pd.DataFrame(resumen)
    
    #-----------------------------------------------------
    # Mostrar resumen
    #-----------------------------------------------------
    
    if mostrar_resumen:
        print("Resumen de transformación de valores atípicos:")
        display(resumen_atipicos)
    
    #-----------------------------------------------------
    # Devolver resultados
    #-----------------------------------------------------
    
    return {
        "datos_limpios": datos_limpios,
        "resumen_atipicos": resumen_atipicos
    }

  #=========================================================
# FUNCIÓN
# f_correlaciones()
#
# OBJETIVO:
# - Calcular la matriz de correlación de variables numéricas.
# - Visualizar la matriz mediante un mapa de calor.
#=========================================================

def f_correlaciones(
    datos,
    variables=None,
    metodo="pearson",
    figsize=(12, 10),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    titulo="Matriz de Correlación de Variables"
):
    
    #-----------------------------------------------------
    # Validaciones básicas
    #-----------------------------------------------------
    
    if not isinstance(datos, pd.DataFrame):
        raise TypeError("El objeto 'datos' debe ser un DataFrame de pandas.")
    
    if metodo not in ["pearson", "spearman", "kendall"]:
        raise ValueError("El argumento 'metodo' debe ser 'pearson', 'spearman' o 'kendall'.")
    
    #-----------------------------------------------------
    # Seleccionar variables numéricas
    #-----------------------------------------------------
    
    if variables is None:
        
        datos_correlacion = datos.select_dtypes(include=np.number)
        
    else:
        
        for variable in variables:
            
            if variable not in datos.columns:
                raise ValueError(f"La variable '{variable}' no existe en los datos.")
            
            if not pd.api.types.is_numeric_dtype(datos[variable]):
                raise TypeError(f"La variable '{variable}' no es numérica.")
        
        datos_correlacion = datos[variables]
    
    if datos_correlacion.shape[1] < 2:
        raise ValueError("Se requieren al menos dos variables numéricas para calcular correlaciones.")
    
    #-----------------------------------------------------
    # Calcular matriz de correlación
    #-----------------------------------------------------
    
    matriz_correlacion = datos_correlacion.corr(method=metodo)
    
    #-----------------------------------------------------
    # Visualizar matriz de correlación
    #-----------------------------------------------------
    
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        matriz_correlacion,
        annot=annot,
        cmap=cmap,
        fmt=fmt,
        linewidths=linewidths
    )
    
    plt.title(titulo, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    
    #-----------------------------------------------------
    # Devolver matriz
    #-----------------------------------------------------
    
    return matriz_correlacion