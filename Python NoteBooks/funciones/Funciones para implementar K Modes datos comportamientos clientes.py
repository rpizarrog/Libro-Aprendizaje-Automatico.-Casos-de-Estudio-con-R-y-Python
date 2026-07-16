# Funciones para implementar # K-Modes

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

#=========================================================
# FUNCIÓN
# f_crear_KModes()
#
# OBJETIVO:
# - Crear un modelo K-Modes para variables categóricas.
# - Usar modas como centros de los clústeres.
# - Devolver clústeres, modas finales, frecuencia y costo.
#=========================================================

def f_crear_KModes(
    datos,
    variables=None,
    n_clusters=2,
    init="Huang",
    n_init=25,
    max_iter=100,
    random_state=2026,
    nombre_cluster="cluster_KModes"
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
    # Seleccionar variables categóricas
    #-----------------------------------------------------
    
    if variables is None:
        
        variables = datos.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        
        if len(variables) == 0:
            raise ValueError("No existen variables categóricas para aplicar K-Modes.")
    
    else:
        
        for variable in variables:
            
            if variable not in datos.columns:
                raise ValueError(f"La variable '{variable}' no existe en los datos.")
            
            if not (
                pd.api.types.is_object_dtype(datos[variable]) or
                pd.api.types.is_categorical_dtype(datos[variable])
            ):
                raise TypeError(f"La variable '{variable}' no es categórica.")
    
    #-----------------------------------------------------
    # Datos para el modelo
    #-----------------------------------------------------
    
    datos_modelo = datos[variables].copy()
    
    if datos_modelo.isna().any().any():
        raise ValueError(
            "Existen valores perdidos. Deben tratarse antes de aplicar K-Modes."
        )
    
    if n_clusters > datos_modelo.shape[0]:
        raise ValueError(
            "El número de clústeres no puede ser mayor que el número de registros."
        )
    
    # Convertir a texto para evitar problemas con categorías
    datos_modelo = datos_modelo.astype(str)
    
    #-----------------------------------------------------
    # Crear modelo K-Modes
    #-----------------------------------------------------
    
    modelo_KModes = KModes(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state
    )
    
    modelo_KModes.fit(datos_modelo)
    
    #-----------------------------------------------------
    # Clúster asignado
    #-----------------------------------------------------
    
    cluster = modelo_KModes.labels_ + 1
    
    #-----------------------------------------------------
    # Datos con clúster
    #-----------------------------------------------------
    
    datos_cluster = datos.copy()
    datos_cluster[nombre_cluster] = cluster
    
    #-----------------------------------------------------
    # Modas finales
    #-----------------------------------------------------
    
    modas_finales = pd.DataFrame(
        modelo_KModes.cluster_centroids_,
        columns=variables
    )
    
    modas_finales.insert(
        0,
        "cluster",
        range(1, n_clusters + 1)
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
    
    frecuencia_cluster.columns = ["cluster", "n"]
    
    #-----------------------------------------------------
    # Resultado
    #-----------------------------------------------------
    
    resultado = {
        "modelo": modelo_KModes,
        "cluster": cluster,
        "datos_cluster": datos_cluster,
        "modas_finales": modas_finales,
        "frecuencia_cluster": frecuencia_cluster,
        "costo": modelo_KModes.cost_,
        "variables": variables,
        "n_clusters": n_clusters,
        "init": init,
        "n_init": n_init,
        "max_iter": max_iter,
        "random_state": random_state
    }
    
    return resultado

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
# f_evaluar_costo_KModes()
#
# OBJETIVO:
# - Evaluar el costo de modelos K-Modes.
# - Comparar modelos con diferente número de clústeres.
# - Recibe un modelo o una lista de modelos.
# - Calcula costo total, costo promedio y reducción porcentual.
#
# ACEPTA:
# - Modelos creados directamente con KModes.
# - Resultados creados con una función tipo f_crear_KModes()
#   que devuelve:
#   "modelo", "cluster", "modas_finales", "costo", etc.
#=========================================================

def f_evaluar_costo_KModes(
    modelos,
    nombres_modelos=None,
    graficar=True,
    titulo="Evaluación del costo en K-Modes",
    decimales=4
):
    
    #-----------------------------------------------------
    # Convertir a lista si se recibe un solo modelo
    #-----------------------------------------------------
    
    if not isinstance(modelos, list):
        modelos = [modelos]
    
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
        
        # Caso 1: resultado personalizado tipo diccionario
        if isinstance(modelo, dict):
            
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
        
        # Caso 2: modelo directo KModes
        if hasattr(modelo, "cost_"):
            return float(modelo.cost_)
        
        if hasattr(modelo, "cost"):
            return float(modelo.cost)
        
        if hasattr(modelo, "costo"):
            return float(modelo.costo)
        
        raise ValueError(
            "No se pudo extraer el costo del modelo. "
            "Revise si el modelo contiene 'costo', 'cost', 'cost_' "
            "o si es un objeto KModes con atributo cost_."
        )
    
    #-----------------------------------------------------
    # Función auxiliar para extraer K
    #-----------------------------------------------------
    
    def f_extraer_k(modelo):
        
        # Caso 1: resultado personalizado tipo diccionario
        if isinstance(modelo, dict):
            
            if "n_clusters" in modelo:
                return int(modelo["n_clusters"])
            
            if "k" in modelo:
                return int(modelo["k"])
            
            if "modas_finales" in modelo:
                return int(modelo["modas_finales"].shape[0])
            
            if "modelo" in modelo:
                
                modelo_interno = modelo["modelo"]
                
                if hasattr(modelo_interno, "n_clusters"):
                    return int(modelo_interno.n_clusters)
                
                if hasattr(modelo_interno, "cluster_centroids_"):
                    return int(modelo_interno.cluster_centroids_.shape[0])
        
        # Caso 2: modelo directo KModes
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
        
        # Caso 1: resultado personalizado tipo diccionario
        if isinstance(modelo, dict):
            
            if "cluster" in modelo:
                return len(modelo["cluster"])
            
            if "datos_cluster" in modelo:
                return modelo["datos_cluster"].shape[0]
            
            if "modelo" in modelo:
                
                modelo_interno = modelo["modelo"]
                
                if hasattr(modelo_interno, "labels_"):
                    return len(modelo_interno.labels_)
        
        # Caso 2: modelo directo KModes
        if hasattr(modelo, "labels_"):
            return len(modelo.labels_)
        
        return np.nan
    
    #-----------------------------------------------------
    # Calcular costos
    #-----------------------------------------------------
    
    registros = []
    
    for i, modelo_actual in enumerate(modelos):
        
        costo_total = f_extraer_costo(modelo_actual)
        k = f_extraer_k(modelo_actual)
        n = f_extraer_n(modelo_actual)
        
        if pd.isna(n):
            costo_promedio = np.nan
        else:
            costo_promedio = costo_total / n
        
        registros.append({
            "modelo": nombres_modelos[i],
            "k": k,
            "n": n,
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
    # Redondear
    #-----------------------------------------------------
    
    columnas_redondear = [
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
