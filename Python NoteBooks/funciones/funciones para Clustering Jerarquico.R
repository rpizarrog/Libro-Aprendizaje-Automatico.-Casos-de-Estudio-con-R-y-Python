# Por. Rubén Pizarro Gurrola
# Julio 2026
# Funcione para Clustering Jerarquico
# Datos de comportamiento de clientes

f_cargar_datos <- function(ruta_archivo) {
  #------------------------------------------------------------
  #   Importar datos desde un archivo CSV.
  # Argumentos:
  #   ruta_archivo: ruta del archivo a cargar.
  # Retorna:
  #   Un data.frame listo para análisis.
  #------------------------------------------------------------
  
  datos <- read_csv(ruta_archivo)
  datos <- as.data.frame(datos)
  return(datos)
}



f_visualizar_head_tail_reducido_word <- function(datos, n = 6) {
  #------------------------------------------------------------
  # Objetivo:
  #   Mostrar primeros n y últimos n registros con:
  #     - Primeras 4 columnas
  #     - Columna separadora "..."
  #     - Últimas 4 columnas
  #------------------------------------------------------------
  
  library(dplyr)
  library(flextable)
  
  total_columnas <- ncol(datos)
  
  if (total_columnas >= 8) {
    # Índices
    idx_prim <- 1:min(4, total_columnas)
    idx_ult  <- max(total_columnas - 3, 1):total_columnas
    
    # Evitar duplicados si hay pocas columnas
    idx_ult <- setdiff(idx_ult, idx_prim)
    
    datos_prim <- datos[, idx_prim, drop = FALSE]
    datos_ult  <- datos[, idx_ult, drop = FALSE]
    
    # Head y tail
    head_prim <- head(datos_prim, n)
    tail_prim <- tail(datos_prim, n)
    
    head_ult <- head(datos_ult, n)
    tail_ult <- tail(datos_ult, n)
    
    # Convertir a character
    head_prim <- as.data.frame(lapply(head_prim, as.character))
    tail_prim <- as.data.frame(lapply(tail_prim, as.character))
    
    head_ult <- as.data.frame(lapply(head_ult, as.character))
    tail_ult <- as.data.frame(lapply(tail_ult, as.character))
    
    # Columna separadora
    sep_head <- data.frame("..." = rep("...", n), check.names = FALSE)
    sep_tail <- data.frame("..." = rep("...", n), check.names = FALSE)
    
    # Combinar columnas
    head_comb <- cbind(head_prim, sep_head, head_ult)
    tail_comb <- cbind(tail_prim, sep_tail, tail_ult)
    
    # Fila separadora horizontal
    fila_puntos <- as.data.frame(
      matrix("...", nrow = 1, ncol = ncol(head_comb))
    )
    colnames(fila_puntos) <- colnames(head_comb)
    
    # Tabla final
    tabla_final <- bind_rows(head_comb, fila_puntos, tail_comb)
    colnames(tabla_final) <- colnames(head_comb)
    # Flextable
    tabla <- flextable(tabla_final)
    tabla <- autofit(tabla)
    
  } else {
    #--------------------------------------------------
    # convertir todo a character temporalmente
    #--------------------------------------------------
    head_datos <- head(datos, n)
    tail_datos <- tail(datos, n)
    head_datos_chr <- data.frame(lapply(head_datos, as.character), stringsAsFactors = FALSE)
    
    tail_datos_chr <- data.frame(lapply(tail_datos, as.character), stringsAsFactors = FALSE)
    
    #--------------------------------------------------
    # fila ...
    #--------------------------------------------------
    
    fila_puntos <- as.data.frame(matrix("...", nrow = 1, ncol = ncol(head_datos_chr) ), stringsAsFactors = FALSE)
    
    colnames(fila_puntos) <- colnames(head_datos_chr)
    # unir
    tabla_final <- bind_rows(head_datos_chr, fila_puntos, tail_datos_chr )
    
    #--------------------------------------------------
    # flextable
    #--------------------------------------------------
    
    tabla <- flextable(tabla_final)
    tabla <- autofit(tabla)
  }
  return(tabla)
}


f_convertir_factor <- function(datos) {
  #------------------------------------------------------------
  # Convierte variables character → factor
  # Convierte variables lógicas → numéricas (0/1)
  #------------------------------------------------------------
  
  datos_mod <- datos
  
  # Convertir character → factor
  idx_char <- sapply(datos_mod, is.character)
  datos_mod[idx_char] <- lapply(datos_mod[idx_char], as.factor)
  
  # Convertir logical → numeric
  idx_logical <- sapply(datos_mod, is.logical)
  datos_mod[idx_logical] <- lapply(datos_mod[idx_logical], function(x) as.numeric(x))
  
  return(datos_mod)
}

# Solo muestr resumnen de los atributos tipo factor
f_summary_factores <- function(datos){
  
  datos_factor <- datos[, sapply(datos, is.factor)]
  
  summary(datos_factor)
  
}

# Describ edatos estadísticos

f_describir_datos <- function(datos) {
  #------------------------------------------------------------
  # f_describir_datos()
  # Objetivo:
  #   Generar estadísticas descriptivas básicas.
  # Uso:
  #   res <- f_describir_datos(datos)
  #   res$summary  # resumen
  #   res$structure # estructura
  #------------------------------------------------------------
  
  res_describe <- describe(datos) # de la librería psych
  # Capturar la estructura como texto (sin imprimir)
  res_str <- paste(capture.output(str(datos)), collapse = "\n")
  
  
  # Devolver ambos para reutilización
  return(list(describe = res_describe, structure = res_str))
}


#=========================================================
# FUNCIÓN
# f_frecuencias()
#
# OBJETIVO:
# - Visualizar frecuencias de variables categóricas.
# - Mostrar porcentaje sobre cada barra.
# - Si no se indican variables, usa todas las categóricas.
#=========================================================

f_frecuencias <- function(
    datos,
    variables = NULL,
    ncol = 3,
    titulo = "Distribución de frecuencias de variables categóricas",
    color_barras = "steelblue",
    color_texto = "black",
    decimales = 1) {
  
  #-------------------------------------------------------
  # Librerías
  #-------------------------------------------------------
  
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(scales)
  
  #-------------------------------------------------------
  # Validaciones básicas
  #-------------------------------------------------------
  
  if (!is.data.frame(datos)) {
    stop("El objeto 'datos' debe ser un data.frame.")
  }
  
  #-------------------------------------------------------
  # Seleccionar variables categóricas
  #-------------------------------------------------------
  
  if (is.null(variables)) {
    
    variables <- names(datos)[
      sapply(datos, function(x) is.factor(x) || is.character(x) || is.logical(x))
    ]
    
    if (length(variables) == 0) {
      stop("No existen variables categóricas en los datos.")
    }
    
  } else {
    
    for (variable in variables) {
      
      if (!(variable %in% names(datos))) {
        stop(paste("La variable", variable, "no existe en los datos."))
      }
      
      if (!(is.factor(datos[[variable]]) ||
            is.character(datos[[variable]]) ||
            is.logical(datos[[variable]]))) {
        
        stop(paste("La variable", variable, "no es categórica."))
      }
    }
  }
  
  #-------------------------------------------------------
  # Transformar datos a formato largo
  #-------------------------------------------------------
  
  datos_largos <- datos %>%
    select(all_of(variables)) %>%
    mutate(across(everything(), as.character)) %>%
    pivot_longer(
      cols = everything(),
      names_to = "variable",
      values_to = "categoria"
    ) %>%
    filter(!is.na(categoria))
  
  #-------------------------------------------------------
  # Calcular frecuencias y porcentajes
  #-------------------------------------------------------
  
  tabla_frecuencias <- datos_largos %>%
    group_by(variable, categoria) %>%
    summarise(
      frecuencia = n(),
      .groups = "drop"
    ) %>%
    group_by(variable) %>%
    mutate(
      porcentaje = frecuencia / sum(frecuencia),
      etiqueta = paste0(
        round(porcentaje * 100, decimales),
        "%"
      )
    ) %>%
    ungroup()
  
  #-------------------------------------------------------
  # Construir gráfico
  #-------------------------------------------------------
  
  grafico <- ggplot(
    tabla_frecuencias,
    aes(
      x = reorder(categoria, -frecuencia),
      y = frecuencia
    )
  ) +
    geom_col(
      fill = color_barras,
      alpha = 0.85
    ) +
    geom_text(
      aes(label = etiqueta),
      vjust = -0.3,
      color = color_texto,
      size = 2.2
    ) +
    facet_wrap(
      ~ variable,
      scales = "free_x",
      ncol = ncol
    ) +
    labs(
      title = titulo,
      x = "",
      y = "Frecuencia"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(
        face = "bold",
        hjust = 0.5
      ),
      strip.text = element_text(
        face = "bold"
      ),
      axis.text.x = element_text(
        angle = 45,
        hjust = 1
      )
    ) +
    expand_limits(
      y = max(tabla_frecuencias$frecuencia) * 1.10
    )
  
  #-------------------------------------------------------
  # Devolver resultados
  #-------------------------------------------------------
  
  return(
    list(
      grafico = grafico,
      tabla_frecuencias = tabla_frecuencias
    )
  )
}

#=========================================================
# FUNCIÓN
# f_frecuencia_clase()
#=========================================================

f_frecuencia_clase <- function(
    datos,
    variable_dependiente){
  
  #-------------------------------------------------------
  # Librerías
  #-------------------------------------------------------
  
  library(ggplot2)
  
  #-------------------------------------------------------
  # Frecuencias
  #-------------------------------------------------------
  
  frecuencia <- as.data.frame(
    table(datos[[variable_dependiente]])
  )
  
  names(frecuencia) <- c("Clase","Frecuencia")
  
  frecuencia$Porcentaje <-
    round(
      frecuencia$Frecuencia /
        sum(frecuencia$Frecuencia) * 100,
      2
    )
  
  #-------------------------------------------------------
  # Gráfico
  #-------------------------------------------------------
  
  grafica <- ggplot(
    frecuencia,
    aes(
      x = Clase,
      y = Frecuencia,
      fill = Clase
    )
  ) +
    
    geom_col(
      width = 0.7
    ) +
    
    geom_text(
      aes(
        label = paste0(
          Frecuencia,
          "\n(",
          Porcentaje,
          "%)"
        )
      ),
      vjust = -0.3,
      size = 4
    ) +
    
    labs(
      title = paste(
        "Frecuencia de clases:",
        variable_dependiente
      ),
      x = "Clase",
      y = "Frecuencia"
    ) +
    
    theme_minimal(base_size = 12) +
    
    theme(
      legend.position = "none",
      plot.title = element_text(
        hjust = 0.5,
        face = "bold"
      )
    )
  
  print(grafica)
  
  return(frecuencia)
  
}


#=========================================================
# FUNCIÓN
# f_redondear_datos()
#=========================================================

f_redondear_datos <- function(
    datos,
    decimales = 2){
  
  #-------------------------------------------------------
  # COLUMNAS NUMÉRICAS
  #-------------------------------------------------------
  
  columnas_numericas <- sapply(
    datos,
    is.numeric
  )
  
  #-------------------------------------------------------
  # REDONDEAR
  #-------------------------------------------------------
  
  datos[columnas_numericas] <-
    lapply(
      datos[columnas_numericas],
      round,
      digits = decimales
    )
  
  return(datos)
}


# Estandarizaar datos
f_estandarizar <- function(
    datos){
  
  #-------------------------------------------------------
  # VARIABLES NUMÉRICAS
  #-------------------------------------------------------
  
  variables_numericas <- names(datos)[
    sapply(datos, is.numeric)
  ]
  
  
  #-------------------------------------------------------
  # MEDIAS
  #-------------------------------------------------------
  
  medias <- sapply(
    datos[variables_numericas],
    mean,
    na.rm = TRUE
  )
  
  #-------------------------------------------------------
  # DESVIACIONES
  #-------------------------------------------------------
  
  desviaciones <- sapply(
    datos[variables_numericas],
    sd,
    na.rm = TRUE
  )
  
  #-------------------------------------------------------
  # EVITAR DIVISIÓN ENTRE CERO
  #-------------------------------------------------------
  
  desviaciones[
    desviaciones == 0
  ] <- 1
  
  #-------------------------------------------------------
  # COPIA
  #-------------------------------------------------------
  
  datos_estandarizados <- datos
  
  #-------------------------------------------------------
  # ESTANDARIZAR
  #-------------------------------------------------------
  
  datos_estandarizados[
    variables_numericas
  ] <- scale(
    
    datos[
      variables_numericas
    ],
    
    center = medias,
    
    scale = desviaciones
    
  )
  
  #-------------------------------------------------------
  # RESULTADO
  #-------------------------------------------------------
  
  resultado <- list(
    
    datos_estandarizados =
      as.data.frame(
        datos_estandarizados
      ),
    
    medias =
      medias,
    
    desviaciones =
      desviaciones,
    
    variables_estandarizadas =
      variables_numericas
    
  )
  
  return(resultado)
}

#=========================================================
# FUNCIÓN
# f_crear_clustering_jerarquico()
#
# OBJETIVO:
# - Crear clustering jerárquico aglomerativo.
# - Detectar automáticamente si los datos son numéricos
#   o mixtos.
# - Usar distancia euclidiana para datos numéricos.
# - Usar distancia de Gower para datos mixtos.
# - Generar dendrograma.
# - Obtener clústeres para un valor de K.
#=========================================================

f_crear_clustering_jerarquico <- function(
    datos,
    variable_nombre = NULL,
    variables_modelo = NULL,
    k = 3,
    metodo_enlace = "single",
    metodo_distancia = "euclidean",
    titulo = "Dendrograma del clustering jerárquico",
    graficar = TRUE,
    usar_factoextra = TRUE,
    decimales = 4) {
  
  #-------------------------------------------------------
  # Validaciones básicas
  #-------------------------------------------------------
  
  if (!is.data.frame(datos)) {
    stop("El objeto 'datos' debe ser un data.frame.")
  }
  
  if (nrow(datos) < 2) {
    stop("Se requieren al menos dos registros para clustering jerárquico.")
  }
  
  if (k < 2) {
    stop("El número de clústeres k debe ser al menos 2.")
  }
  
  if (k > nrow(datos)) {
    stop("El número de clústeres k no puede ser mayor que el número de registros.")
  }
  
  metodos_enlace_validos <- c(
    "single",
    "complete",
    "average",
    "mcquitty",
    "ward.D",
    "ward.D2",
    "centroid",
    "median"
  )
  
  if (!(metodo_enlace %in% metodos_enlace_validos)) {
    stop(
      paste(
        "El método de enlace debe ser uno de:",
        paste(metodos_enlace_validos, collapse = ", ")
      )
    )
  }
  
  #-------------------------------------------------------
  # Seleccionar variables del modelo
  #-------------------------------------------------------
  
  if (is.null(variables_modelo)) {
    
    variables_modelo <- names(datos)
    
    if (!is.null(variable_nombre)) {
      variables_modelo <- setdiff(
        variables_modelo,
        variable_nombre
      )
    }
  }
  
  if (!all(variables_modelo %in% names(datos))) {
    stop("Una o más variables indicadas no existen en el data.frame.")
  }
  
  if (!is.null(variable_nombre)) {
    
    if (!(variable_nombre %in% names(datos))) {
      stop("La variable indicada en 'variable_nombre' no existe en los datos.")
    }
  }
  
  datos_modelo <- datos[, variables_modelo, drop = FALSE]
  
  #-------------------------------------------------------
  # Quitar variables no recomendadas automáticamente
  # si no se especificaron variables_modelo
  #-------------------------------------------------------
  
  # Convertir character a factor para detectar variables categóricas
  for (variable in names(datos_modelo)) {
    
    if (is.character(datos_modelo[[variable]])) {
      datos_modelo[[variable]] <- as.factor(datos_modelo[[variable]])
    }
  }
  
  #-------------------------------------------------------
  # Validar valores perdidos
  #-------------------------------------------------------
  
  if (any(is.na(datos_modelo))) {
    stop("Los datos contienen valores NA. Debes imputarlos o eliminarlos antes del clustering.")
  }
  
  #-------------------------------------------------------
  # Identificar tipos de variables
  #-------------------------------------------------------
  
  variables_numericas <- names(datos_modelo)[
    sapply(datos_modelo, is.numeric)
  ]
  
  variables_categoricas <- names(datos_modelo)[
    sapply(datos_modelo, function(x) {
      is.factor(x) || is.character(x) || is.logical(x)
    })
  ]
  
  todas_numericas <- length(variables_numericas) == ncol(datos_modelo)
  
  datos_mixtos <- length(variables_numericas) > 0 &&
    length(variables_categoricas) > 0
  
  solo_categoricas <- length(variables_categoricas) == ncol(datos_modelo)
  
  #-------------------------------------------------------
  # Asignar nombres de registros
  #-------------------------------------------------------
  
  etiquetas <- NULL
  
  if (!is.null(variable_nombre)) {
    
    etiquetas <- as.character(datos[[variable_nombre]])
    
    rownames(datos_modelo) <- etiquetas
  }
  
  #-------------------------------------------------------
  # Calcular matriz de distancias
  #-------------------------------------------------------
  
  tipo_distancia <- NULL
  
  if (todas_numericas) {
    
    distancias <- dist(
      datos_modelo,
      method = metodo_distancia
    )
    
    tipo_distancia <- metodo_distancia
    
  } else {
    
    if (!requireNamespace("cluster", quietly = TRUE)) {
      install.packages("cluster")
    }
    
    distancias <- cluster::daisy(
      datos_modelo,
      metric = "gower"
    )
    
    tipo_distancia <- "gower"
  }
  
  matriz_distancias <- as.matrix(distancias)
  
  if (!is.null(variable_nombre)) {
    
    rownames(matriz_distancias) <- etiquetas
    colnames(matriz_distancias) <- etiquetas
  }
  
  matriz_distancias <- round(
    matriz_distancias,
    decimales
  )
  
  #-------------------------------------------------------
  # Crear modelo de clustering jerárquico
  #-------------------------------------------------------
  
  modelo_hc <- hclust(
    d = distancias,
    method = metodo_enlace
  )
  
  #-------------------------------------------------------
  # Obtener clústeres
  #-------------------------------------------------------
  
  cluster <- cutree(
    modelo_hc,
    k = k
  )
  
  datos_cluster <- datos
  datos_cluster$cluster <- as.factor(cluster)
  
  #-------------------------------------------------------
  # Tabla de frecuencias por clúster
  #-------------------------------------------------------
  
  frecuencia_cluster <- as.data.frame(
    table(datos_cluster$cluster)
  )
  
  names(frecuencia_cluster) <- c(
    "cluster",
    "frecuencia"
  )
  
  #-------------------------------------------------------
  # Proceso de agrupamiento
  #-------------------------------------------------------
  
  proceso_agrupamiento <- data.frame(
    paso = 1:length(modelo_hc$height),
    union_1 = modelo_hc$merge[, 1],
    union_2 = modelo_hc$merge[, 2],
    altura = round(modelo_hc$height, decimales)
  )
  
  #-------------------------------------------------------
  # Gráfico del dendrograma
  #-------------------------------------------------------
  
  grafico <- NULL
  
  if (graficar) {
    
    if (usar_factoextra) {
      
      if (!requireNamespace("factoextra", quietly = TRUE)) {
        install.packages("factoextra")
      }
      
      grafico <- factoextra::fviz_dend(
        modelo_hc,
        cex = 1.1,
        k = k,
        rect = TRUE,
        rect_fill = TRUE,
        rect_border = "jco",
        color_labels_by_k = TRUE,
        main = titulo
      )
      
      print(grafico)
      
    } else {
      
      plot(
        modelo_hc,
        labels = etiquetas,
        main = titulo,
        xlab = "",
        ylab = paste("Distancia", tipo_distancia),
        sub = "",
        cex = 1.1
      )
      
      rect.hclust(
        modelo_hc,
        k = k,
        border = "red"
      )
    }
  }
  
  #-------------------------------------------------------
  # Mensaje resumen
  #-------------------------------------------------------
  
  cat("\n========================================\n")
  cat("CLUSTERING JERÁRQUICO CREADO\n")
  cat("========================================\n")
  cat("Registros:", nrow(datos), "\n")
  cat("Variables del modelo:", paste(variables_modelo, collapse = ", "), "\n")
  cat("Variables numéricas:", paste(variables_numericas, collapse = ", "), "\n")
  cat("Variables categóricas:", paste(variables_categoricas, collapse = ", "), "\n")
  cat("Tipo de distancia:", tipo_distancia, "\n")
  cat("Método de enlace:", metodo_enlace, "\n")
  cat("Número de clústeres:", k, "\n")
  cat("========================================\n")
  
  #-------------------------------------------------------
  # Salida
  #-------------------------------------------------------
  
  return(
    list(
      datos_cluster = datos_cluster,
      datos_modelo = datos_modelo,
      distancias = distancias,
      matriz_distancias = matriz_distancias,
      modelo_hc = modelo_hc,
      cluster = cluster,
      frecuencia_cluster = frecuencia_cluster,
      proceso_agrupamiento = proceso_agrupamiento,
      variables_modelo = variables_modelo,
      variables_numericas = variables_numericas,
      variables_categoricas = variables_categoricas,
      tipo_distancia = tipo_distancia,
      metodo_enlace = metodo_enlace,
      k = k,
      grafico = grafico
    )
  )
}

#=========================================================
# FUNCIÓN
# f_evaluar_clustering_jerarquico()
#
# OBJETIVO:
# - Evaluar clustering jerárquico para diferentes valores de K.
# - Calcular Silhouette promedio.
# - Calcular correlación cofenética.
# - Mostrar frecuencias por clúster.
#=========================================================

f_evaluar_clustering_jerarquico <- function(
    modelo_hc,
    distancias,
    k_min = 2,
    k_max = 6,
    graficar = TRUE,
    decimales = 4) {
  
  #-------------------------------------------------------
  # Librerías
  #-------------------------------------------------------
  
  if (!requireNamespace("cluster", quietly = TRUE)) {
    install.packages("cluster")
  }
  
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    install.packages("ggplot2")
  }
  
  library(cluster)
  library(ggplot2)
  
  #-------------------------------------------------------
  # Validaciones
  #-------------------------------------------------------
  
  if (!inherits(modelo_hc, "hclust")) {
    stop("El objeto 'modelo_hc' debe ser de clase hclust.")
  }
  
  if (!inherits(distancias, "dist")) {
    stop("El objeto 'distancias' debe ser de clase dist.")
  }
  
  if (k_min < 2) {
    stop("k_min debe ser al menos 2.")
  }
  
  if (k_max <= k_min) {
    stop("k_max debe ser mayor que k_min.")
  }
  
  #-------------------------------------------------------
  # Evaluar Silhouette para varios K
  #-------------------------------------------------------
  
  tabla_evaluacion <- data.frame()
  
  for (k in k_min:k_max) {
    
    cluster_k <- cutree(
      modelo_hc,
      k = k
    )
    
    sil <- silhouette(
      cluster_k,
      distancias
    )
    
    silhouette_promedio <- mean(
      sil[, 3]
    )
    
    frecuencia <- paste(
      table(cluster_k),
      collapse = ", "
    )
    
    tabla_evaluacion <- rbind(
      tabla_evaluacion,
      data.frame(
        k = k,
        silhouette_promedio = round(
          silhouette_promedio,
          decimales
        ),
        frecuencia_cluster = frecuencia
      )
    )
  }
  
  #-------------------------------------------------------
  # Correlación cofenética
  #-------------------------------------------------------
  
  distancia_cofenetica <- cophenetic(
    modelo_hc
  )
  
  correlacion_cofenetica <- cor(
    distancias,
    distancia_cofenetica
  )
  
  correlacion_cofenetica <- round(
    correlacion_cofenetica,
    decimales
  )
  
  #-------------------------------------------------------
  # Gráfico
  #-------------------------------------------------------
  
  grafico <- NULL
  
  if (graficar) {
    
    grafico <- ggplot(
      tabla_evaluacion,
      aes(
        x = k,
        y = silhouette_promedio
      )
    ) +
      geom_line() +
      geom_point(size = 3) +
      scale_x_continuous(
        breaks = tabla_evaluacion$k
      ) +
      labs(
        title = "Evaluación del clustering jerárquico",
        subtitle = paste(
          "Correlación cofenética =",
          correlacion_cofenetica
        ),
        x = "Número de clústeres",
        y = "Silhouette promedio"
      ) +
      theme_minimal()
    
    print(grafico)
  }
  
  #-------------------------------------------------------
  # Resultado
  #-------------------------------------------------------
  
  return(
    list(
      tabla_evaluacion = tabla_evaluacion,
      correlacion_cofenetica = correlacion_cofenetica,
      grafico = grafico
    )
  )
}


#=========================================================
# FUNCIÓN
# f_registros_por_cluster()
#
# OBJETIVO:
# - Crear tabla de pertenencia a clústeres.
# - Ordenar registros por clúster.
# - Visualizar la tabla general.
# - Visualizar registros por cada clúster.
#
# REQUIERE:
# - Un modelo generado con f_crear_clustering_jerarquico()
# - Que el modelo contenga modelo$datos_cluster
# - Que exista la función f_visualizar_head_tail_reducido_word()
#=========================================================

f_registros_por_cluster <- function(
    modelo,
    variables,
    variable_cluster = "cluster",
    visualizar = TRUE,
    funcion_visualizar = f_visualizar_head_tail_reducido_word) {
  
  #-------------------------------------------------------
  # Validaciones básicas
  #-------------------------------------------------------
  
  if (!is.list(modelo)) {
    stop("El objeto 'modelo' debe ser una lista generada por la función de clustering.")
  }
  
  if (!("datos_cluster" %in% names(modelo))) {
    stop("El modelo no contiene el elemento 'datos_cluster'.")
  }
  
  datos_cluster <- modelo$datos_cluster
  
  if (!is.data.frame(datos_cluster)) {
    stop("'modelo$datos_cluster' debe ser un data.frame.")
  }
  
  if (!(variable_cluster %in% names(datos_cluster))) {
    stop(
      paste0(
        "La variable de clúster '",
        variable_cluster,
        "' no existe en modelo$datos_cluster."
      )
    )
  }
  
  if (missing(variables) || is.null(variables)) {
    stop("Debes indicar el vector de variables a incluir en la tabla.")
  }
  
  variables_no_existen <- setdiff(
    variables,
    names(datos_cluster)
  )
  
  if (length(variables_no_existen) > 0) {
    stop(
      paste(
        "Estas variables no existen en datos_cluster:",
        paste(variables_no_existen, collapse = ", ")
      )
    )
  }
  
  #-------------------------------------------------------
  # Construir tabla de pertenencia
  #-------------------------------------------------------
  
  columnas_tabla <- unique(
    c(
      variables,
      variable_cluster
    )
  )
  
  tabla_pertenencia <- datos_cluster[, columnas_tabla, drop = FALSE]
  
  #-------------------------------------------------------
  # Ordenar por clúster
  #-------------------------------------------------------
  
  tabla_pertenencia <- tabla_pertenencia[
    order(tabla_pertenencia[[variable_cluster]]),
    ,
    drop = FALSE
  ]
  
  #-------------------------------------------------------
  # Identificar clústeres
  #-------------------------------------------------------
  
  clusters <- sort(
    unique(tabla_pertenencia[[variable_cluster]])
  )
  
  #-------------------------------------------------------
  # Crear lista de tablas por clúster
  #-------------------------------------------------------
  
  tablas_por_cluster <- list()
  
  for (cl in clusters) {
    
    tabla_cl <- tabla_pertenencia[
      tabla_pertenencia[[variable_cluster]] == cl,
      ,
      drop = FALSE
    ]
    
    tablas_por_cluster[[paste0("cluster_", cl)]] <- tabla_cl
  }
  
  #-------------------------------------------------------
  # Frecuencia por clúster
  #-------------------------------------------------------
  
  frecuencia_cluster <- as.data.frame(
    table(tabla_pertenencia[[variable_cluster]])
  )
  
  names(frecuencia_cluster) <- c(
    variable_cluster,
    "frecuencia"
  )
  
  frecuencia_cluster$porcentaje <- round(
    frecuencia_cluster$frecuencia / sum(frecuencia_cluster$frecuencia) * 100,
    2
  )
  
  #-------------------------------------------------------
  # Visualizaciones
  #-------------------------------------------------------
  
  if (visualizar) {
    
    cat("TABLA GENERAL DE PERTENENCIA A CLÚSTERES\n")
    
    funcion_visualizar(tabla_pertenencia)
    cat("FRECUENCIA POR CLÚSTER\n")
    
    print(frecuencia_cluster)

  }
  
  #-------------------------------------------------------
  # Salida
  #-------------------------------------------------------
  
  return(
    list(
      tabla_pertenencia = tabla_pertenencia,
      tablas_por_cluster = tablas_por_cluster,
      frecuencia_cluster = frecuencia_cluster,
      variables = variables,
      variable_cluster = variable_cluster
    )
  )
}

#=========================================================
# FUNCIÓN
# f_perfil_cluster()
#
# OBJETIVO:
# - Construir perfiles numéricos por clúster.
# - Construir perfiles categóricos por clúster.
# - Unir ambos perfiles en una tabla general.
# - Calcular frecuencia y porcentaje por clúster.
#
# REQUIERE:
# - Un modelo generado previamente con f_crear_clustering_jerarquico()
# - Que el modelo contenga modelo$datos_cluster
# - Que exista una variable llamada cluster en datos_cluster
#=========================================================

f_perfil_cluster <- function(
    modelo,
    variables_numericas,
    variables_categoricas,
    variable_cluster = "cluster",
    decimales = 2) {
  
  #-------------------------------------------------------
  # Librería
  #-------------------------------------------------------
  
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    install.packages("dplyr")
  }
  
  library(dplyr)
  
  #-------------------------------------------------------
  # Validaciones básicas
  #-------------------------------------------------------
  
  if (!is.list(modelo)) {
    stop("El objeto 'modelo' debe ser una lista generada por la función de clustering.")
  }
  
  if (!("datos_cluster" %in% names(modelo))) {
    stop("El modelo no contiene el elemento 'datos_cluster'.")
  }
  
  datos_cluster <- modelo$datos_cluster
  
  if (!is.data.frame(datos_cluster)) {
    stop("'modelo$datos_cluster' debe ser un data.frame.")
  }
  
  if (!(variable_cluster %in% names(datos_cluster))) {
    stop(
      paste0(
        "La variable de clúster '",
        variable_cluster,
        "' no existe en modelo$datos_cluster."
      )
    )
  }
  
  if (missing(variables_numericas) || is.null(variables_numericas)) {
    stop("Debes indicar el vector 'variables_numericas'.")
  }
  
  if (missing(variables_categoricas) || is.null(variables_categoricas)) {
    stop("Debes indicar el vector 'variables_categoricas'.")
  }
  
  variables_no_existen_num <- setdiff(
    variables_numericas,
    names(datos_cluster)
  )
  
  variables_no_existen_cat <- setdiff(
    variables_categoricas,
    names(datos_cluster)
  )
  
  if (length(variables_no_existen_num) > 0) {
    stop(
      paste(
        "Estas variables numéricas no existen en datos_cluster:",
        paste(variables_no_existen_num, collapse = ", ")
      )
    )
  }
  
  if (length(variables_no_existen_cat) > 0) {
    stop(
      paste(
        "Estas variables categóricas no existen en datos_cluster:",
        paste(variables_no_existen_cat, collapse = ", ")
      )
    )
  }
  
  #-------------------------------------------------------
  # Validar que las variables numéricas realmente sean numéricas
  #-------------------------------------------------------
  
  variables_no_numericas <- variables_numericas[
    !sapply(datos_cluster[variables_numericas], is.numeric)
  ]
  
  if (length(variables_no_numericas) > 0) {
    stop(
      paste(
        "Estas variables fueron indicadas como numéricas, pero no son numéricas:",
        paste(variables_no_numericas, collapse = ", ")
      )
    )
  }
  
  #-------------------------------------------------------
  # Función auxiliar para calcular moda
  #-------------------------------------------------------
  
  f_moda <- function(x) {
    
    x <- x[!is.na(x)]
    
    if (length(x) == 0) {
      return(NA)
    }
    
    tabla <- table(x)
    
    names(tabla)[which.max(tabla)]
  }
  
  #-------------------------------------------------------
  # Frecuencia por clúster
  #-------------------------------------------------------
  
  frecuencia_cluster <- datos_cluster %>%
    group_by(.data[[variable_cluster]]) %>%
    summarise(
      frecuencia = n(),
      .groups = "drop"
    ) %>%
    mutate(
      porcentaje = round(
        frecuencia / sum(frecuencia) * 100,
        decimales
      )
    )
  
  #-------------------------------------------------------
  # Perfil numérico por clúster
  #-------------------------------------------------------
  
  perfil_numerico <- datos_cluster %>%
    group_by(.data[[variable_cluster]]) %>%
    summarise(
      n = n(),
      across(
        all_of(variables_numericas),
        ~ round(mean(.x, na.rm = TRUE), decimales)
      ),
      .groups = "drop"
    )
  
  #-------------------------------------------------------
  # Perfil categórico dinámico por clúster
  #-------------------------------------------------------
  
  perfil_categorico <- datos_cluster %>%
    group_by(.data[[variable_cluster]]) %>%
    summarise(
      across(
        all_of(variables_categoricas),
        ~ f_moda(.x),
        .names = "{.col}_dominante"
      ),
      .groups = "drop"
    )
  
  #-------------------------------------------------------
  # Perfil general
  #-------------------------------------------------------
  
  perfil_general <- perfil_numerico %>%
    left_join(
      perfil_categorico,
      by = variable_cluster
    )
  
  #-------------------------------------------------------
  # Mensaje resumen
  #-------------------------------------------------------
  
  cat("\n========================================\n")
  cat("PERFIL DE CLÚSTERES\n")
  cat("========================================\n")
  cat("Número de registros:", nrow(datos_cluster), "\n")
  cat("Variable de clúster:", variable_cluster, "\n")
  cat("Variables numéricas:", paste(variables_numericas, collapse = ", "), "\n")
  cat("Variables categóricas:", paste(variables_categoricas, collapse = ", "), "\n")
  cat("========================================\n")
  
  #-------------------------------------------------------
  # Salida
  #-------------------------------------------------------
  
  return(
    list(
      frecuencia_cluster = frecuencia_cluster,
      perfil_numerico = perfil_numerico,
      perfil_categorico = perfil_categorico,
      perfil_general = perfil_general,
      variables_numericas = variables_numericas,
      variables_categoricas = variables_categoricas,
      variable_cluster = variable_cluster
    )
  )
}



#=========================================================
# FUNCIÓN
# f_evaluar_clustering()
#
# OBJETIVO:
# - Evaluar un modelo de clustering jerárquico.
# - Calcular Silhouette promedio para varios valores de K.
# - Calcular correlación cofenética.
# - Presentar una tabla amigable.
# - Mostrar resumen e interpretación en consola.
#
# REQUIERE:
# - Un modelo creado con f_crear_clustering_jerarquico()
# - Que el modelo contenga:
#     modelo$modelo_hc
#     modelo$distancias
#=========================================================

f_evaluar_clustering <- function(
    modelo,
    k_min = 2,
    k_max = NULL,
    decimales = 4,
    graficar = TRUE) {
  
  #-------------------------------------------------------
  # Librerías
  #-------------------------------------------------------
  
  if (!requireNamespace("cluster", quietly = TRUE)) {
    install.packages("cluster")
  }
  
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    install.packages("ggplot2")
  }
  
  library(cluster)
  library(ggplot2)
  
  #-------------------------------------------------------
  # Validaciones básicas
  #-------------------------------------------------------
  
  if (!is.list(modelo)) {
    stop("El objeto 'modelo' debe ser una lista generada por la función de clustering.")
  }
  
  if (!("modelo_hc" %in% names(modelo))) {
    stop("El modelo no contiene el elemento 'modelo_hc'.")
  }
  
  if (!("distancias" %in% names(modelo))) {
    stop("El modelo no contiene el elemento 'distancias'.")
  }
  
  if (!inherits(modelo$modelo_hc, "hclust")) {
    stop("modelo$modelo_hc debe ser un objeto de clase 'hclust'.")
  }
  
  if (!inherits(modelo$distancias, "dist")) {
    stop("modelo$distancias debe ser un objeto de clase 'dist'.")
  }
  
  modelo_hc <- modelo$modelo_hc
  distancias <- modelo$distancias
  
  n <- length(modelo_hc$order)
  
  if (n < 3) {
    stop("Se requieren al menos 3 registros para evaluar Silhouette.")
  }
  
  if (k_min < 2) {
    stop("k_min debe ser al menos 2.")
  }
  
  #-------------------------------------------------------
  # Ajuste automático de k_max
  #-------------------------------------------------------
  
  if (is.null(k_max)) {
    k_max <- min(10, n - 1)
  }
  
  if (k_max >= n) {
    
    warning(
      paste0(
        "k_max = ", k_max,
        " no es válido para n = ", n,
        ". Se ajusta automáticamente a k_max = ", n - 1, "."
      )
    )
    
    k_max <- n - 1
  }
  
  if (k_max < k_min) {
    stop("k_max debe ser mayor o igual que k_min y menor que el número de registros.")
  }
  
  #-------------------------------------------------------
  # Función auxiliar para interpretar Silhouette
  #-------------------------------------------------------
  
  f_interpretar_silhouette <- function(valor) {
    
    if (is.na(valor)) {
      return("No calculable")
    } else if (valor >= 0.70) {
      return("Estructura fuerte")
    } else if (valor >= 0.50) {
      return("Estructura razonable")
    } else if (valor >= 0.25) {
      return("Estructura débil o moderada")
    } else {
      return("Estructura poco definida")
    }
  }
  
  #-------------------------------------------------------
  # Evaluar Silhouette para diferentes K
  #-------------------------------------------------------
  
  tabla_evaluacion <- data.frame()
  
  for (k in k_min:k_max) {
    
    cluster_k <- cutree(
      modelo_hc,
      k = k
    )
    
    num_clusters <- length(unique(cluster_k))
    
    # Silhouette no es útil si todos son clústeres individuales
    if (num_clusters < 2 || num_clusters >= n) {
      next
    }
    
    sil <- cluster::silhouette(
      cluster_k,
      distancias
    )
    
    silhouette_promedio <- mean(
      sil[, "sil_width"],
      na.rm = TRUE
    )
    
    frecuencia <- table(cluster_k)
    
    frecuencia_texto <- paste(
      frecuencia,
      collapse = ", "
    )
    
    cluster_mayor <- max(frecuencia)
    cluster_menor <- min(frecuencia)
    
    tabla_evaluacion <- rbind(
      tabla_evaluacion,
      data.frame(
        k = k,
        silhouette_promedio = round(silhouette_promedio, decimales),
        interpretacion_silhouette = f_interpretar_silhouette(silhouette_promedio),
        cluster_menor = cluster_menor,
        cluster_mayor = cluster_mayor,
        frecuencia_cluster = frecuencia_texto
      )
    )
  }
  
  if (nrow(tabla_evaluacion) == 0) {
    stop("No fue posible calcular Silhouette para los valores de K indicados.")
  }
  
  #-------------------------------------------------------
  # Mejor K según Silhouette
  #-------------------------------------------------------
  
  mejor_indice <- which.max(
    tabla_evaluacion$silhouette_promedio
  )
  
  mejor_k <- tabla_evaluacion$k[mejor_indice]
  
  mejor_silhouette <- tabla_evaluacion$silhouette_promedio[mejor_indice]
  
  mejor_interpretacion <- tabla_evaluacion$interpretacion_silhouette[mejor_indice]
  
  #-------------------------------------------------------
  # Correlación cofenética
  #-------------------------------------------------------
  
  distancia_cofenetica <- cophenetic(
    modelo_hc
  )
  
  correlacion_cofenetica <- cor(
    distancias,
    distancia_cofenetica
  )
  
  correlacion_cofenetica <- round(
    correlacion_cofenetica,
    decimales
  )
  
  #-------------------------------------------------------
  # Interpretación cofenética
  #-------------------------------------------------------
  
  interpretacion_cofenetica <- ifelse(
    correlacion_cofenetica >= 0.90,
    "Muy alta: el dendrograma representa muy bien las distancias originales",
    ifelse(
      correlacion_cofenetica >= 0.80,
      "Alta: el dendrograma representa adecuadamente las distancias originales",
      ifelse(
        correlacion_cofenetica >= 0.70,
        "Aceptable: el dendrograma conserva parte importante de la estructura",
        "Baja: el dendrograma representa débilmente las distancias originales"
      )
    )
  )
  
  #-------------------------------------------------------
  # Tabla resumen general
  #-------------------------------------------------------
  
  tabla_resumen <- data.frame(
    metrica = c(
      "Mejor K según Silhouette",
      "Silhouette promedio del mejor K",
      "Interpretación Silhouette",
      "Correlación cofenética",
      "Interpretación cofenética"
    ),
    valor = c(
      mejor_k,
      mejor_silhouette,
      mejor_interpretacion,
      correlacion_cofenetica,
      interpretacion_cofenetica
    )
  )
  
  #-------------------------------------------------------
  # Gráfico de Silhouette promedio
  #-------------------------------------------------------
  
  grafico <- NULL
  
  if (graficar) {
    
    grafico <- ggplot(
      tabla_evaluacion,
      aes(
        x = k,
        y = silhouette_promedio
      )
    ) +
      geom_line() +
      geom_point(size = 3) +
      scale_x_continuous(
        breaks = tabla_evaluacion$k
      ) +
      labs(
        title = "Evaluación del clustering jerárquico",
        subtitle = paste(
          "Correlación cofenética =",
          correlacion_cofenetica
        ),
        x = "Número de clústeres",
        y = "Silhouette promedio"
      ) +
      theme_minimal()
    
    print(grafico)
  }
  
  #-------------------------------------------------------
  # Resumen en consola
  #-------------------------------------------------------
  
  cat("\n========================================\n")
  cat("EVALUACIÓN DEL CLUSTERING JERÁRQUICO\n")
  cat("========================================\n")
  cat("Número de registros:", n, "\n")
  
  if ("tipo_distancia" %in% names(modelo)) {
    cat("Tipo de distancia:", modelo$tipo_distancia, "\n")
  }
  
  if ("metodo_enlace" %in% names(modelo)) {
    cat("Método de enlace:", modelo$metodo_enlace, "\n")
  }
  
  cat("Valores de K evaluados:", k_min, "a", k_max, "\n")
  cat("----------------------------------------\n")
  cat("Mejor K según Silhouette:", mejor_k, "\n")
  cat("Silhouette promedio:", mejor_silhouette, "\n")
  cat("Interpretación:", mejor_interpretacion, "\n")
  cat("----------------------------------------\n")
  cat("Correlación cofenética:", correlacion_cofenetica, "\n")
  cat("Interpretación:", interpretacion_cofenetica, "\n")
  cat("========================================\n\n")
  
  #-------------------------------------------------------
  # Interpretación textual para reporte
  #-------------------------------------------------------
  
  interpretacion_texto <- paste0(
    "La evaluación del clustering jerárquico se realizó mediante el estadístico ",
    "Silhouette y la correlación cofenética. De acuerdo con Silhouette, el mejor ",
    "valor de K fue ", mejor_k, ", con un promedio de ",
    mejor_silhouette, ", lo que indica una ",
    tolower(mejor_interpretacion), ". La correlación cofenética fue de ",
    correlacion_cofenetica, ", por lo que se considera que ",
    tolower(interpretacion_cofenetica), "."
  )
  
  cat("INTERPRETACIÓN PARA EL REPORTE:\n")
  cat(interpretacion_texto, "\n")
  
  #-------------------------------------------------------
  # Salida
  #-------------------------------------------------------
  
  return(
    list(
      tabla_evaluacion = tabla_evaluacion,
      tabla_resumen = tabla_resumen,
      mejor_k = mejor_k,
      mejor_silhouette = mejor_silhouette,
      correlacion_cofenetica = correlacion_cofenetica,
      interpretacion_cofenetica = interpretacion_cofenetica,
      interpretacion_texto = interpretacion_texto,
      grafico = grafico
    )
  )
}

