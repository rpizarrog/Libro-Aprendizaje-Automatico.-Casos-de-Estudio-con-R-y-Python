# Por. Rubén Pizarro Gurrola
# Julio 2026
# Funcione para K Modes
# DATOS DE 

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
# f_crear_KModes()
#
# OBJETIVO:
# - Crear un modelo K-Modes para variables categóricas.
# - Usar la función kmodes() del paquete klaR.
# - Devolver clústeres, modas finales, frecuencias y datos con clúster.
#=========================================================

f_crear_KModes <- function(
    datos,
    variables = NULL,
    k = 2,
    iter_max = 100,
    semilla = 2026,
    weighted = FALSE,
    nombre_cluster = "cluster_KModes") {
  
  #-------------------------------------------------------
  # Librería
  #-------------------------------------------------------
  
  if (!require(klaR)) {
    install.packages("klaR")
    library(klaR)
  }
  
  #-------------------------------------------------------
  # Validaciones básicas
  #-------------------------------------------------------
  
  if (!is.data.frame(datos)) {
    stop("El objeto 'datos' debe ser un data.frame.")
  }
  
  if (k < 2) {
    stop("El número de clústeres debe ser al menos 2.")
  }
  
  #-------------------------------------------------------
  # Seleccionar variables categóricas
  #-------------------------------------------------------
  
  if (is.null(variables)) {
    
    variables <- names(datos)[
      sapply(datos, function(x) is.factor(x) || is.character(x))
    ]
    
    if (length(variables) == 0) {
      stop("No existen variables categóricas para aplicar K-Modes.")
    }
    
  } else {
    
    for (variable in variables) {
      
      if (!(variable %in% names(datos))) {
        stop(paste("La variable", variable, "no existe en los datos."))
      }
      
      if (!(is.factor(datos[[variable]]) || is.character(datos[[variable]]))) {
        stop(paste("La variable", variable, "no es categórica."))
      }
    }
  }
  
  #-------------------------------------------------------
  # Datos para el modelo
  #-------------------------------------------------------
  
  datos_modelo <- datos[, variables]
  
  # Convertir variables a factor
  datos_modelo <- as.data.frame(
    lapply(datos_modelo, as.factor)
  )
  
  if (any(is.na(datos_modelo))) {
    stop("Existen valores perdidos. Deben tratarse antes de aplicar K-Modes.")
  }
  
  if (k > nrow(datos_modelo)) {
    stop("El número de clústeres no puede ser mayor que el número de registros.")
  }
  
  #-------------------------------------------------------
  # Crear modelo K-Modes
  #-------------------------------------------------------
  
  set.seed(semilla)
  
  modelo_KModes <- klaR::kmodes(
    data = datos_modelo,
    modes = k,
    iter.max = iter_max,
    weighted = weighted
  )
  
  #-------------------------------------------------------
  # Datos con clúster
  #-------------------------------------------------------
  
  datos_cluster <- datos
  
  datos_cluster[[nombre_cluster]] <- modelo_KModes$cluster
  
  #-------------------------------------------------------
  # Frecuencia por clúster
  #-------------------------------------------------------
  
  frecuencia_cluster <- as.data.frame(
    table(modelo_KModes$cluster)
  )
  
  names(frecuencia_cluster) <- c("cluster", "n")
  
  frecuencia_cluster$cluster <- as.integer(
    as.character(frecuencia_cluster$cluster)
  )
  
  #-------------------------------------------------------
  # Modas finales
  #-------------------------------------------------------
  
  modas_finales <- as.data.frame(modelo_KModes$modes)
  
  modas_finales$cluster <- 1:k
  
  modas_finales <- modas_finales[, c("cluster", variables)]
  
  #-------------------------------------------------------
  # Resultado
  #-------------------------------------------------------
  
  resultado <- list(
    modelo = modelo_KModes,
    cluster = modelo_KModes$cluster,
    datos_cluster = datos_cluster,
    modas_finales = modas_finales,
    frecuencia_cluster = frecuencia_cluster,
    variables = variables,
    k = k,
    iter_max = iter_max,
    semilla = semilla
  )
  
  return(resultado)
}


#=========================================================
# FUNCIÓN
# f_visualizar_clusters_categoricos()
#
# OBJETIVO:
# - Visualizar clústeres construidos con variables categóricas.
# - Útil para K-Modes.
# - Muestra la distribución porcentual de categorías
#   dentro de cada clúster.
#=========================================================

#=========================================================
# FUNCIÓN
# f_visualizar_clusters_categoricos()
#
# OBJETIVO:
# - Visualizar variables categóricas por clúster.
# - Más amigable para K-Modes.
# - Cada panel corresponde a una variable.
# - El eje X muestra las categorías.
# - El color representa el clúster.
# - Muestra porcentajes dentro de cada clúster.
#=========================================================

f_visualizar_clusters_categoricos <- function(
    datos,
    variable_cluster,
    variables = NULL,
    ncol = 2,
    titulo = "Distribución porcentual de variables categóricas por clúster",
    decimales = 1,
    posicion = "dodge",
    mostrar_etiquetas = TRUE,
    rotar_etiquetas = TRUE) {
  
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
  
  if (!(variable_cluster %in% names(datos))) {
    stop("La variable de clúster no existe en los datos.")
  }
  
  if (!(posicion %in% c("dodge", "fill"))) {
    stop("El argumento 'posicion' debe ser 'dodge' o 'fill'.")
  }
  
  #-------------------------------------------------------
  # Seleccionar variables categóricas
  #-------------------------------------------------------
  
  if (is.null(variables)) {
    
    variables <- names(datos)[
      sapply(datos, function(x) is.factor(x) || is.character(x) || is.logical(x))
    ]
    
    variables <- setdiff(variables, variable_cluster)
    
    if (length(variables) == 0) {
      stop("No existen variables categóricas para graficar.")
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
  # Preparar datos en formato largo
  #-------------------------------------------------------
  
  datos_largos <- datos %>%
    select(
      all_of(variable_cluster),
      all_of(variables)
    ) %>%
    mutate(
      across(everything(), as.character)
    ) %>%
    pivot_longer(
      cols = all_of(variables),
      names_to = "variable",
      values_to = "categoria"
    ) %>%
    filter(!is.na(categoria))
  
  #-------------------------------------------------------
  # Calcular frecuencias y porcentajes
  # Porcentaje dentro de cada clúster y variable
  #-------------------------------------------------------
  
  tabla_frecuencias <- datos_largos %>%
    group_by(
      .data[[variable_cluster]],
      variable,
      categoria
    ) %>%
    summarise(
      frecuencia = n(),
      .groups = "drop"
    ) %>%
    group_by(
      .data[[variable_cluster]],
      variable
    ) %>%
    mutate(
      porcentaje = frecuencia / sum(frecuencia),
      etiqueta = paste0(
        round(porcentaje * 100, decimales),
        "%"
      )
    ) %>%
    ungroup()
  
  #-------------------------------------------------------
  # Definir posición de barras
  #-------------------------------------------------------
  
  if (posicion == "dodge") {
    
    posicion_barras <- position_dodge(width = 0.80)
    eje_y <- scale_y_continuous(
      labels = percent_format(accuracy = 1),
      limits = c(0, NA)
    )
    
  } else {
    
    posicion_barras <- "fill"
    eje_y <- scale_y_continuous(
      labels = percent_format(accuracy = 1)
    )
  }
  
  #-------------------------------------------------------
  # Construir gráfico
  #-------------------------------------------------------
  
  grafico <- ggplot(
    tabla_frecuencias,
    aes(
      x = categoria,
      y = porcentaje,
      fill = factor(.data[[variable_cluster]])
    )
  ) +
    geom_col(
      position = posicion_barras,
      alpha = 0.85
    ) +
    eje_y +
    facet_wrap(
      ~ variable,
      scales = "free_x",
      ncol = ncol
    ) +
    labs(
      title = titulo,
      x = "Categoría",
      y = "Porcentaje dentro del clúster",
      fill = "Clúster"
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
      legend.position = "bottom"
    )
  
  #-------------------------------------------------------
  # Agregar etiquetas si se solicita
  #-------------------------------------------------------
  
  if (mostrar_etiquetas) {
    
    if (posicion == "dodge") {
      
      grafico <- grafico +
        geom_text(
          aes(label = etiqueta),
          position = position_dodge(width = 0.80),
          vjust = -0.3,
          size = 3
        )
      
    } else {
      
      grafico <- grafico +
        geom_text(
          aes(label = ifelse(porcentaje >= 0.08, etiqueta, "")),
          position = position_fill(vjust = 0.5),
          size = 3,
          color = "black"
        )
    }
  }
  
  #-------------------------------------------------------
  # Rotar etiquetas del eje X
  #-------------------------------------------------------
  
  if (rotar_etiquetas) {
    
    grafico <- grafico +
      theme(
        axis.text.x = element_text(
          angle = 45,
          hjust = 1
        )
      )
  }
  
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
# f_evaluar_costo()
#
# OBJETIVO:
# - Evaluar el costo de modelos K-Modes.
# - Comparar modelos con diferente número de clústeres.
# - Recibe una lista de modelos.
# - Calcula costo total, costo promedio y reducción porcentual.
#
# ACEPTA:
# - Modelos creados directamente con klaR::kmodes()
# - Resultados creados con una función tipo f_crear_KModes()
#   que devuelve $modelo, $cluster, $modas_finales, $costo, etc.
#=========================================================

f_evaluar_costo_KModes <- function(
    modelos,
    nombres_modelos = NULL,
    graficar = TRUE,
    titulo = "Evaluación del costo en K-Modes",
    decimales = 4) {
  
  #-------------------------------------------------------
  # Librerías
  #-------------------------------------------------------
  
  library(ggplot2)
  library(dplyr)
  
  #-------------------------------------------------------
  # Validaciones
  #-------------------------------------------------------
  
  if (!is.list(modelos)) {
    modelos <- list(modelos)
  }
  
  if (is.null(nombres_modelos)) {
    
    if (!is.null(names(modelos)) && all(names(modelos) != "")) {
      nombres_modelos <- names(modelos)
    } else {
      nombres_modelos <- paste0("Modelo_", seq_along(modelos))
    }
  }
  
  if (length(nombres_modelos) != length(modelos)) {
    stop("La longitud de 'nombres_modelos' debe coincidir con el número de modelos.")
  }
  
  #-------------------------------------------------------
  # Función auxiliar para extraer costo
  #-------------------------------------------------------
  
  f_extraer_costo <- function(modelo) {
    
    # Caso 1: resultado personalizado con $costo
    if (!is.null(modelo$costo)) {
      return(as.numeric(modelo$costo))
    }
    
    # Caso 2: resultado personalizado con $cost
    if (!is.null(modelo$cost)) {
      return(as.numeric(modelo$cost))
    }
    
    # Caso 3: modelo Python vía reticulate con $cost_
    if (!is.null(modelo$cost_)) {
      return(as.numeric(modelo$cost_))
    }
    
    # Caso 4: resultado personalizado con $modelo interno
    if (!is.null(modelo$modelo)) {
      
      if (!is.null(modelo$modelo$costo)) {
        return(as.numeric(modelo$modelo$costo))
      }
      
      if (!is.null(modelo$modelo$cost)) {
        return(as.numeric(modelo$modelo$cost))
      }
      
      if (!is.null(modelo$modelo$cost_)) {
        return(as.numeric(modelo$modelo$cost_))
      }
      
      # klaR::kmodes suele guardar disimilitud interna en withindiff
      if (!is.null(modelo$modelo$withindiff)) {
        return(sum(as.numeric(modelo$modelo$withindiff)))
      }
    }
    
    # Caso 5: objeto directo de klaR::kmodes()
    if (!is.null(modelo$withindiff)) {
      return(sum(as.numeric(modelo$withindiff)))
    }
    
    stop("No se pudo extraer el costo del modelo. Revise si el modelo contiene $costo, $cost, $cost_, $withindiff o $modelo$withindiff.")
  }
  
  #-------------------------------------------------------
  # Función auxiliar para extraer K
  #-------------------------------------------------------
  
  f_extraer_k <- function(modelo) {
    
    if (!is.null(modelo$k)) {
      return(as.integer(modelo$k))
    }
    
    if (!is.null(modelo$n_clusters)) {
      return(as.integer(modelo$n_clusters))
    }
    
    if (!is.null(modelo$n_cluster)) {
      return(as.integer(modelo$n_cluster))
    }
    
    if (!is.null(modelo$modas_finales)) {
      return(nrow(modelo$modas_finales))
    }
    
    if (!is.null(modelo$modes)) {
      return(nrow(modelo$modes))
    }
    
    if (!is.null(modelo$modelo)) {
      
      if (!is.null(modelo$modelo$modes)) {
        return(nrow(modelo$modelo$modes))
      }
      
      if (!is.null(modelo$modelo$cluster)) {
        return(length(unique(modelo$modelo$cluster)))
      }
    }
    
    if (!is.null(modelo$cluster)) {
      return(length(unique(modelo$cluster)))
    }
    
    return(NA_integer_)
  }
  
  #-------------------------------------------------------
  # Función auxiliar para extraer n
  #-------------------------------------------------------
  
  f_extraer_n <- function(modelo) {
    
    if (!is.null(modelo$cluster)) {
      return(length(modelo$cluster))
    }
    
    if (!is.null(modelo$modelo)) {
      if (!is.null(modelo$modelo$cluster)) {
        return(length(modelo$modelo$cluster))
      }
    }
    
    return(NA_integer_)
  }
  
  #-------------------------------------------------------
  # Calcular costos
  #-------------------------------------------------------
  
  tabla_costos <- data.frame()
  
  for (i in seq_along(modelos)) {
    
    modelo_actual <- modelos[[i]]
    
    costo_total <- f_extraer_costo(modelo_actual)
    k <- f_extraer_k(modelo_actual)
    n <- f_extraer_n(modelo_actual)
    
    costo_promedio <- ifelse(
      is.na(n),
      NA,
      costo_total / n
    )
    
    tabla_costos <- rbind(
      tabla_costos,
      data.frame(
        modelo = nombres_modelos[i],
        k = k,
        n = n,
        costo_total = costo_total,
        costo_promedio = costo_promedio
      )
    )
  }
  
  #-------------------------------------------------------
  # Ordenar por K
  #-------------------------------------------------------
  
  tabla_costos <- tabla_costos %>%
    arrange(k)
  
  #-------------------------------------------------------
  # Calcular reducción del costo
  #-------------------------------------------------------
  
  tabla_costos <- tabla_costos %>%
    mutate(
      reduccion_absoluta = lag(costo_total) - costo_total,
      reduccion_porcentual = (reduccion_absoluta / lag(costo_total)) * 100
    )
  
  #-------------------------------------------------------
  # Redondear
  #-------------------------------------------------------
  
  tabla_costos$costo_total <- round(tabla_costos$costo_total, decimales)
  tabla_costos$costo_promedio <- round(tabla_costos$costo_promedio, decimales)
  tabla_costos$reduccion_absoluta <- round(tabla_costos$reduccion_absoluta, decimales)
  tabla_costos$reduccion_porcentual <- round(tabla_costos$reduccion_porcentual, decimales)
  
  #-------------------------------------------------------
  # Gráfico
  #-------------------------------------------------------
  
  grafico <- NULL
  
  if (graficar) {
    
    grafico <- ggplot(
      tabla_costos,
      aes(
        x = k,
        y = costo_total
      )
    ) +
      geom_line(linewidth = 1) +
      geom_point(size = 3) +
      geom_text(
        aes(label = costo_total),
        vjust = -0.8,
        size = 4
      ) +
      scale_x_continuous(
        breaks = tabla_costos$k
      ) +
      labs(
        title = titulo,
        x = "Número de clústeres K",
        y = "Costo total"
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(
          face = "bold",
          hjust = 0.5
        )
      )
  }
  
  #-------------------------------------------------------
  # Devolver resultados
  #-------------------------------------------------------
  
  return(
    list(
      tabla_costos = tabla_costos,
      grafico = grafico
    )
  )
}






