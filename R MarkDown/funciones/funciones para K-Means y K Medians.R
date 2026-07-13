# Por. Rubén Pizarro Gurrola
# Julio 2026
# Funcione para K Means
# DATOS DE Estudiantes para variables estudio, promedio participacion

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


# Estandarizaar datos
f_estandarizar_entrenamiento <- function(
    datos,
    variable_dependiente){
  
  #-------------------------------------------------------
  # VARIABLES NUMÉRICAS
  #-------------------------------------------------------
  
  variables_numericas <- names(datos)[
    sapply(datos, is.numeric)
  ]
  
  #-------------------------------------------------------
  # EXCLUIR VARIABLE DEPENDIENTE
  #-------------------------------------------------------
  
  variables_numericas <- setdiff(
    variables_numericas,
    variable_dependiente
  )
  
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
# f_convertir_dummys()
#=========================================================

f_convertir_dummys <- function(
    datos,
    variable_dependiente){
  
  #-------------------------------------------------------
  # VALIDACIONES
  #-------------------------------------------------------
  
  if(!variable_dependiente %in% names(datos)){
    
    stop(
      paste(
        "La variable",
        variable_dependiente,
        "no existe en los datos."
      )
    )
    
  }
  
  #-------------------------------------------------------
  # SEPARAR VARIABLE DEPENDIENTE
  #-------------------------------------------------------
  
  y <- datos[[variable_dependiente]]
  
  X <- datos[
    ,
    names(datos) != variable_dependiente,
    drop = FALSE
  ]
  
  #-------------------------------------------------------
  # VARIABLES LÓGICAS A FACTOR
  #-------------------------------------------------------
  
  X[] <- lapply(
    
    X,
    
    function(x){
      
      if(is.logical(x)){
        
        factor(
          ifelse(
            x,
            "TRUE",
            "FALSE"
          )
        )
        
      }else{
        
        x
        
      }
      
    }
    
  )
  
  #-------------------------------------------------------
  # CREAR DUMMIES
  #-------------------------------------------------------
  
  matriz_dummys <- model.matrix(
    ~ .,
    data = X
  )
  
  #-------------------------------------------------------
  # ELIMINAR INTERCEPTO
  #-------------------------------------------------------
  
  matriz_dummys <-
    matriz_dummys[
      ,
      colnames(matriz_dummys) != "(Intercept)",
      drop = FALSE
    ]
  
  #-------------------------------------------------------
  # DATA FRAME
  #-------------------------------------------------------
  
  datos_dummys <-
    as.data.frame(
      matriz_dummys
    )
  
  #-------------------------------------------------------
  # NORMALIZAR NOMBRES
  #-------------------------------------------------------
  
  colnames(datos_dummys) <-
    make.names(
      colnames(datos_dummys),
      unique = TRUE
    )
  
  #-------------------------------------------------------
  # VARIABLE DEPENDIENTE AL FINAL
  #-------------------------------------------------------
  
  datos_dummys[[variable_dependiente]] <- y
  
  #-------------------------------------------------------
  # INFORMACIÓN
  #-------------------------------------------------------
  
  cat("\n")
  cat("====================================\n")
  cat(" CONVERSIÓN A VARIABLES DUMMY\n")
  cat("====================================\n")
  cat("Variables originales :", ncol(datos), "\n")
  cat("Variables finales    :", ncol(datos_dummys), "\n")
  cat("Observaciones        :", nrow(datos), "\n")
  cat("====================================\n")
  
  return(datos_dummys)
  
}



# Funciones para K-Means
# Rubén Pizarro Gurrola
# Julio 2026

#=========================================================
# FUNCIÓN:
# f_dispersion_variables_clusters()
#
# OBJETIVO:
# - Genera diagramas de dispersión por pares de variables.
# - Colorea los puntos según el clúster.
# - Opcionalmente agrega los centroides.
#
# REQUIERE:
# install.packages("ggplot2")
# install.packages("patchwork")
#=========================================================

f_dispersion_variables_clusters <- function(
    datos,
    variable_cluster = "cluster",
    variables = NULL,
    centroides = NULL,
    titulo = "Diagramas de dispersión por clúster",
    ncol = 2) {
  
  #-------------------------------------------------------
  # Librerías
  #-------------------------------------------------------
  
  library(ggplot2)
  library(patchwork)
  
  #-------------------------------------------------------
  # Validaciones
  #-------------------------------------------------------
  
  if (!is.data.frame(datos)) {
    stop("El objeto 'datos' debe ser un data.frame.")
  }
  
  if (!(variable_cluster %in% names(datos))) {
    stop(paste("La variable de clúster", variable_cluster, "no existe en los datos."))
  }
  
  #-------------------------------------------------------
  # Seleccionar variables numéricas
  #-------------------------------------------------------
  
  if (is.null(variables)) {
    
    variables <- names(datos)[sapply(datos, is.numeric)]
    variables <- setdiff(variables, variable_cluster)
  }
  
  if (length(variables) < 2) {
    stop("Se requieren al menos dos variables numéricas para generar diagramas de dispersión.")
  }
  
  #-------------------------------------------------------
  # Convertir cluster a factor
  #-------------------------------------------------------
  
  datos[[variable_cluster]] <- as.factor(datos[[variable_cluster]])
  
  #-------------------------------------------------------
  # Preparar centroides si se proporcionan
  #-------------------------------------------------------
  
  if (!is.null(centroides)) {
    
    centroides <- as.data.frame(centroides)
    
    if (!all(variables %in% names(centroides))) {
      stop("Los centroides deben contener las mismas variables numéricas seleccionadas.")
    }
    
    centroides[[variable_cluster]] <- as.factor(rownames(centroides))
    
    # Si los centroides no tienen nombres de fila adecuados,
    # se asignan números consecutivos.
    if (any(is.na(centroides[[variable_cluster]])) ||
        all(centroides[[variable_cluster]] == "")) {
      
      centroides[[variable_cluster]] <- as.factor(1:nrow(centroides))
    }
  }
  
  #-------------------------------------------------------
  # Crear combinaciones de pares de variables
  #-------------------------------------------------------
  
  combinaciones <- combn(variables, 2, simplify = FALSE)
  
  lista_graficos <- list()
  
  #-------------------------------------------------------
  # Crear un gráfico por cada combinación
  #-------------------------------------------------------
  
  for (i in seq_along(combinaciones)) {
    
    var_x <- combinaciones[[i]][1]
    var_y <- combinaciones[[i]][2]
    
    grafico <- ggplot(
      datos,
      aes(
        x = .data[[var_x]],
        y = .data[[var_y]],
        color = .data[[variable_cluster]]
      )
    ) +
      geom_point(size = 1, alpha = 0.50) +
      labs(
        title = paste(var_x, "vs", var_y),
        x = var_x,
        y = var_y,
        color = "Clúster"
      ) +
      theme_minimal()
    
    #-----------------------------------------------------
    # Agregar centroides si existen
    #-----------------------------------------------------
    
    if (!is.null(centroides)) {
      
      grafico <- grafico +
        geom_point(
          data = centroides,
          aes(
            x = .data[[var_x]],
            y = .data[[var_y]]
          ),
          color = "black",
          shape = 19,
          size = 4
        )
    }
    
    lista_graficos[[i]] <- grafico
  }
  
  #-------------------------------------------------------
  # Unir gráficos
  #-------------------------------------------------------
  
  grafico_final <- wrap_plots(lista_graficos, ncol = ncol) +
    plot_annotation(title = titulo)
  
  return(grafico_final)
}



#=========================================================
# FUNCIÓN:
# f_diagramas_cajas()
#
# OBJETIVO:
# - Genera diagramas de caja para comparar variables
#   numéricas dentro de cada clúster.
#
# REQUIERE:
# install.packages("ggplot2")
# install.packages("patchwork")
#=========================================================

f_diagramas_cajas <- function(
    datos,
    variable_cluster = "cluster",
    variables = NULL,
    titulo = "Diagramas de caja por clúster",
    ncol = 2) {
  
  #-------------------------------------------------------
  # Librerías
  #-------------------------------------------------------
  
  library(ggplot2)
  library(patchwork)
  
  #-------------------------------------------------------
  # Validaciones
  #-------------------------------------------------------
  
  if (!is.data.frame(datos)) {
    stop("El objeto 'datos' debe ser un data.frame.")
  }
  
  if (!(variable_cluster %in% names(datos))) {
    stop(paste("La variable de clúster", variable_cluster, "no existe en los datos."))
  }
  
  #-------------------------------------------------------
  # Seleccionar variables numéricas
  #-------------------------------------------------------
  
  if (is.null(variables)) {
    
    variables <- names(datos)[sapply(datos, is.numeric)]
    variables <- setdiff(variables, variable_cluster)
  }
  
  if (length(variables) < 1) {
    stop("Se requiere al menos una variable numérica para generar diagramas de caja.")
  }
  
  #-------------------------------------------------------
  # Convertir cluster a factor
  #-------------------------------------------------------
  
  datos[[variable_cluster]] <- as.factor(datos[[variable_cluster]])
  
  #-------------------------------------------------------
  # Crear diagramas de caja
  #-------------------------------------------------------
  
  lista_graficos <- list()
  
  for (i in seq_along(variables)) {
    
    var_y <- variables[i]
    
    grafico <- ggplot(
      datos,
      aes(
        x = .data[[variable_cluster]],
        y = .data[[var_y]],
        fill = .data[[variable_cluster]]
      )
    ) +
      geom_boxplot(alpha = 0.5, outlier.shape = 16) +
      geom_jitter(
        aes(color = .data[[variable_cluster]]),
        width = 0.15,
        size = 1,
        alpha = 0.5,
        show.legend = FALSE
      ) +
      labs(
        title = paste("Distribución de", var_y),
        x = "Clúster",
        y = var_y,
        fill = "Clúster"
      ) +
      theme_minimal()
    
    lista_graficos[[i]] <- grafico
  }
  
  #-------------------------------------------------------
  # Unir gráficos
  #-------------------------------------------------------
  
  grafico_final <- wrap_plots(lista_graficos, ncol = ncol) +
    plot_annotation(title = titulo)
  
  return(grafico_final)
}



#=========================================================
# FUNCIÓN
# f_atipicos()
#
# OBJETIVO:
# - Generar boxplots por variable numérica.
# - Mostrar varios gráficos en columnas.
# - Útil para detectar valores atípicos.
#
# REQUIERE:
# install.packages("ggplot2")
# install.packages("tidyr")
# install.packages("dplyr")
#=========================================================

f_atipicos <- function(
    datos,
    variables = NULL,
    ncol = 3,
    titulo = "Detección visual de valores atípicos",
    color_caja = "lightblue",
    color_atipicos = "red") {
  
  #-------------------------------------------------------
  # Librerías
  #-------------------------------------------------------
  
  library(ggplot2)
  library(tidyr)
  library(dplyr)
  
  #-------------------------------------------------------
  # Validaciones
  #-------------------------------------------------------
  
  if (!is.data.frame(datos)) {
    stop("El objeto 'datos' debe ser un data.frame.")
  }
  
  #-------------------------------------------------------
  # Si no se indican variables, toma todas las numéricas
  #-------------------------------------------------------
  
  if (is.null(variables)) {
    
    variables <- names(datos)[sapply(datos, is.numeric)]
    
  } else {
    
    for (variable in variables) {
      
      if (!(variable %in% names(datos))) {
        stop(paste("La variable", variable, "no existe en los datos."))
      }
      
      if (!is.numeric(datos[[variable]])) {
        stop(paste("La variable", variable, "no es numérica."))
      }
    }
  }
  
  #-------------------------------------------------------
  # Transformar datos a formato largo
  #-------------------------------------------------------
  
  datos_largos <- datos %>%
    select(all_of(variables)) %>%
    pivot_longer(
      cols = everything(),
      names_to = "variable",
      values_to = "valor"
    )
  
  #-------------------------------------------------------
  # Construir gráfico
  #-------------------------------------------------------
  
  grafico <- ggplot(
    datos_largos,
    aes(
      x = variable,
      y = valor
    )
  ) +
    geom_boxplot(
      fill = color_caja,
      color = "gray30",
      outlier.color = color_atipicos,
      outlier.shape = 16,
      outlier.size = 1.8,
      alpha = 0.70
    ) +
    facet_wrap(
      ~ variable,
      scales = "free_y",
      ncol = ncol
    ) +
    labs(
      title = titulo,
      x = "",
      y = "Valor"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      strip.text = element_text(face = "bold"),
      plot.title = element_text(face = "bold", hjust = 0.5)
    )
  
  return(grafico)
}


#=========================================================
# FUNCIÓN
# f_transformar_atipicos()
#
# OBJETIVO:
# - Detectar valores atípicos usando IQR.
# - Reemplazar valores atípicos por la mediana o la media.
# - Devolver datos limpios y resumen de atípicos.
#=========================================================

f_transformar_atipicos <- function(
    datos,
    variables = NULL,
    metodo = "mediana",
    factor_iqr = 1.5,
    decimales = 4,
    mostrar_resumen = TRUE) {
  
  #-------------------------------------------------------
  # Validaciones básicas
  #-------------------------------------------------------
  
  if (!is.data.frame(datos)) {
    stop("El objeto 'datos' debe ser un data.frame.")
  }
  
  if (!(metodo %in% c("mediana", "media"))) {
    stop("El argumento 'metodo' debe ser 'mediana' o 'media'.")
  }
  
  #-------------------------------------------------------
  # Copia de datos
  #-------------------------------------------------------
  
  datos_limpios <- datos
  
  #-------------------------------------------------------
  # Seleccionar variables numéricas
  #-------------------------------------------------------
  
  if (is.null(variables)) {
    
    variables <- names(datos_limpios)[sapply(datos_limpios, is.numeric)]
    
  } else {
    
    for (variable in variables) {
      
      if (!(variable %in% names(datos_limpios))) {
        stop(paste("La variable", variable, "no existe en los datos."))
      }
      
      if (!is.numeric(datos_limpios[[variable]])) {
        stop(paste("La variable", variable, "no es numérica."))
      }
    }
  }
  
  #-------------------------------------------------------
  # Crear tabla resumen
  #-------------------------------------------------------
  
  resumen_atipicos <- data.frame(
    variable = character(),
    limite_inferior = numeric(),
    limite_superior = numeric(),
    n_atipicos = integer(),
    valor_reemplazo = numeric(),
    metodo = character(),
    stringsAsFactors = FALSE
  )
  
  #-------------------------------------------------------
  # Transformar valores atípicos
  #-------------------------------------------------------
  
  for (variable in variables) {
    
    serie <- datos_limpios[[variable]]
    
    #-----------------------------------------------------
    # Si la variable está completamente vacía
    #-----------------------------------------------------
    
    if (all(is.na(serie))) {
      
      resumen_atipicos <- rbind(
        resumen_atipicos,
        data.frame(
          variable = variable,
          limite_inferior = NA,
          limite_superior = NA,
          n_atipicos = 0,
          valor_reemplazo = NA,
          metodo = metodo,
          stringsAsFactors = FALSE
        )
      )
      
      next
    }
    
    #-----------------------------------------------------
    # Calcular Q1, Q3 e IQR
    #-----------------------------------------------------
    
    Q1 <- quantile(serie, 0.25, na.rm = TRUE)
    Q3 <- quantile(serie, 0.75, na.rm = TRUE)
    
    IQR_valor <- Q3 - Q1
    
    limite_inferior <- Q1 - factor_iqr * IQR_valor
    limite_superior <- Q3 + factor_iqr * IQR_valor
    
    #-----------------------------------------------------
    # Identificar valores atípicos
    #-----------------------------------------------------
    
    condicion_atipicos <- serie < limite_inferior | serie > limite_superior
    
    condicion_atipicos[is.na(condicion_atipicos)] <- FALSE
    
    n_atipicos <- sum(condicion_atipicos)
    
    #-----------------------------------------------------
    # Calcular valor de reemplazo
    #-----------------------------------------------------
    
    if (metodo == "mediana") {
      
      valor_reemplazo <- median(serie, na.rm = TRUE)
      
    } else {
      
      valor_reemplazo <- mean(serie, na.rm = TRUE)
    }
    
    valor_reemplazo <- round(valor_reemplazo, decimales)
    
    #-----------------------------------------------------
    # Reemplazar valores atípicos
    #-----------------------------------------------------
    
    datos_limpios[[variable]][condicion_atipicos] <- valor_reemplazo
    
    #-----------------------------------------------------
    # Guardar resumen
    #-----------------------------------------------------
    
    resumen_atipicos <- rbind(
      resumen_atipicos,
      data.frame(
        variable = variable,
        limite_inferior = round(limite_inferior, decimales),
        limite_superior = round(limite_superior, decimales),
        n_atipicos = n_atipicos,
        valor_reemplazo = valor_reemplazo,
        metodo = metodo,
        stringsAsFactors = FALSE
      )
    )
  }
  
  #-------------------------------------------------------
  # Mostrar resumen
  #-------------------------------------------------------
  
  if (mostrar_resumen) {
    
    cat("Resumen de transformación de valores atípicos:\n")
    print(resumen_atipicos)
  }
  
  #-------------------------------------------------------
  # Devolver resultados
  #-------------------------------------------------------
  
  return(
    list(
      datos_limpios = datos_limpios,
      resumen_atipicos = resumen_atipicos
    )
  )
}


#=========================================================
# FUNCIÓN
# f_crear_KMeans()
#
# OBJETIVO:
# - Crear un modelo K-Means a partir de un conjunto de datos.
# - Recibe los datos y las variables que se usarán para clustering.
# - Devuelve el modelo K-Means.
#=========================================================

f_crear_KMeans <- function(
    datos,
    variables,
    centers = 3,
    nstart = 25,
    semilla = 2026) {
  
  #-------------------------------------------------------
  # Validaciones
  #-------------------------------------------------------
  
  if (!is.data.frame(datos)) {
    stop("El objeto 'datos' debe ser un data.frame.")
  }
  
  if (missing(variables)) {
    stop("Debe indicar el vector de variables numéricas a utilizar.")
  }
  
  for (variable in variables) {
    
    if (!(variable %in% names(datos))) {
      stop(paste("La variable", variable, "no existe en los datos."))
    }
    
    if (!is.numeric(datos[[variable]])) {
      stop(paste("La variable", variable, "no es numérica."))
    }
  }
  
  if (centers < 2) {
    stop("El número de clústeres debe ser al menos 2.")
  }
  
  #-------------------------------------------------------
  # Seleccionar datos para clustering
  #-------------------------------------------------------
  
  datos_modelo <- datos[, variables]
  
  #-------------------------------------------------------
  # Validar valores perdidos
  #-------------------------------------------------------
  
  if (any(is.na(datos_modelo))) {
    stop("Existen valores perdidos en las variables seleccionadas. Deben tratarse antes de aplicar K-Means.")
  }
  
  #-------------------------------------------------------
  # Crear modelo K-Means
  #-------------------------------------------------------
  
  set.seed(semilla)
  
  modelo_KMeans <- kmeans(
    x = datos_modelo,
    centers = centers,
    nstart = nstart
  )
  
  #-------------------------------------------------------
  # Devolver modelo
  #-------------------------------------------------------
  
  return(modelo_KMeans)
}

#=========================================================
# FUNCIÓN
# f_crear_KMedians()
#
# OBJETIVO:
# - Crear un modelo K-Medians.
# - Usa distancia Manhattan.
# - Actualiza centros con medianas.
#=========================================================

#=========================================================
# FUNCIÓN
# f_crear_KMedians()
#
# OBJETIVO:
# - Crear modelo K-Medians.
# - Usar distancia Manhattan.
# - Actualizar centros con medianas.
# - Permitir datos estandarizados o no estandarizados.
# - Probar varias inicializaciones con nstart.
# - Conservar la solución con menor SAD.
#
# DEVUELVE:
# - cluster: grupo asignado a cada registro.
# - medianas_modelo: medianas en la escala usada para modelar.
# - medianas_originales: medianas en la escala original.
# - SAD_total: suma total de distancias absolutas.
# - SAD_por_cluster.
# - SAD_por_registro.
# - datos_modelo: datos usados por el algoritmo.
# - objeto_escala: medias y desviaciones si se estandarizó.
#=========================================================

f_crear_KMedians <- function(
    datos,
    variables,
    centers = 3,
    nstart = 25,
    max_iter = 100,
    semilla = 123,
    estandarizar = TRUE,
    decimales = 4) {
  
  #-------------------------------------------------------
  # Validaciones básicas
  #-------------------------------------------------------
  
  if (!is.data.frame(datos)) {
    stop("El objeto 'datos' debe ser un data.frame.")
  }
  
  if (missing(variables)) {
    stop("Debe indicar el vector de variables a utilizar.")
  }
  
  for (variable in variables) {
    
    if (!(variable %in% names(datos))) {
      stop(paste("La variable", variable, "no existe en los datos."))
    }
    
    if (!is.numeric(datos[[variable]])) {
      stop(paste("La variable", variable, "no es numérica."))
    }
  }
  
  if (centers < 2) {
    stop("El número de clústeres debe ser al menos 2.")
  }
  
  if (nstart < 1) {
    stop("nstart debe ser al menos 1.")
  }
  
  if (max_iter < 1) {
    stop("max_iter debe ser al menos 1.")
  }
  
  #-------------------------------------------------------
  # Datos originales
  #-------------------------------------------------------
  
  X_original <- as.matrix(datos[, variables])
  
  if (any(is.na(X_original))) {
    stop("Existen valores perdidos. Deben tratarse antes de aplicar K-Medians.")
  }
  
  #-------------------------------------------------------
  # Estandarizar si se solicita
  #-------------------------------------------------------
  
  if (estandarizar) {
    
    X_modelo <- scale(X_original)
    
    objeto_escala <- list(
      center = attr(X_modelo, "scaled:center"),
      scale = attr(X_modelo, "scaled:scale")
    )
    
    X_modelo <- as.matrix(X_modelo)
    
  } else {
    
    X_modelo <- X_original
    
    objeto_escala <- NULL
  }
  
  n <- nrow(X_modelo)
  
  #-------------------------------------------------------
  # Objetos para guardar la mejor solución
  #-------------------------------------------------------
  
  mejor_SAD <- Inf
  mejor_cluster <- NULL
  mejores_medianas <- NULL
  mejor_iteracion <- NULL
  mejor_inicio <- NULL
  
  resumen_inicios <- data.frame()
  
  set.seed(semilla)
  
  #-------------------------------------------------------
  # Repetir varias inicializaciones
  #-------------------------------------------------------
  
  for (inicio in 1:nstart) {
    
    #-----------------------------------------------------
    # Seleccionar centros iniciales aleatorios
    #-----------------------------------------------------
    
    indices_iniciales <- sample(1:n, centers)
    
    medianas <- X_modelo[indices_iniciales, , drop = FALSE]
    
    cluster_anterior <- rep(NA, n)
    
    #-----------------------------------------------------
    # Iteraciones
    #-----------------------------------------------------
    
    for (iteracion in 1:max_iter) {
      
      #---------------------------------------------------
      # Calcular distancia Manhattan
      #---------------------------------------------------
      
      distancias <- matrix(0, nrow = n, ncol = centers)
      
      for (k in 1:centers) {
        
        distancias[, k] <- rowSums(
          abs(
            sweep(
              X_modelo,
              2,
              medianas[k, ],
              FUN = "-"
            )
          )
        )
      }
      
      #---------------------------------------------------
      # Asignar cada registro al centro más cercano
      #---------------------------------------------------
      
      cluster <- apply(distancias, 1, which.min)
      
      #---------------------------------------------------
      # Detener si ya no cambia
      #---------------------------------------------------
      
      if (all(cluster == cluster_anterior, na.rm = TRUE)) {
        break
      }
      
      cluster_anterior <- cluster
      
      #---------------------------------------------------
      # Actualizar medianas
      #---------------------------------------------------
      
      for (k in 1:centers) {
        
        if (sum(cluster == k) > 0) {
          
          medianas[k, ] <- apply(
            X_modelo[cluster == k, , drop = FALSE],
            2,
            median
          )
          
        } else {
          
          # Si un clúster queda vacío, se reinicia
          medianas[k, ] <- X_modelo[sample(1:n, 1), ]
        }
      }
    }
    
    #-----------------------------------------------------
    # Calcular SAD de esta inicialización
    #-----------------------------------------------------
    
    SAD_por_registro <- numeric(n)
    
    for (i in 1:n) {
      
      k <- cluster[i]
      
      SAD_por_registro[i] <- sum(
        abs(X_modelo[i, ] - medianas[k, ])
      )
    }
    
    SAD_total <- sum(SAD_por_registro)
    
    resumen_inicios <- rbind(
      resumen_inicios,
      data.frame(
        inicio = inicio,
        SAD_total = round(SAD_total, decimales),
        iteraciones = iteracion
      )
    )
    
    #-----------------------------------------------------
    # Guardar mejor solución
    #-----------------------------------------------------
    
    if (SAD_total < mejor_SAD) {
      
      mejor_SAD <- SAD_total
      mejor_cluster <- cluster
      mejores_medianas <- medianas
      mejor_iteracion <- iteracion
      mejor_inicio <- inicio
    }
  }
  
  #-------------------------------------------------------
  # Calcular SAD final de la mejor solución
  #-------------------------------------------------------
  
  SAD_por_registro <- numeric(n)
  
  for (i in 1:n) {
    
    k <- mejor_cluster[i]
    
    SAD_por_registro[i] <- sum(
      abs(X_modelo[i, ] - mejores_medianas[k, ])
    )
  }
  
  SAD_por_cluster <- tapply(
    SAD_por_registro,
    mejor_cluster,
    sum
  )
  
  #-------------------------------------------------------
  # Medianas en la escala del modelo
  #-------------------------------------------------------
  
  colnames(mejores_medianas) <- variables
  
  medianas_modelo <- as.data.frame(mejores_medianas)
  
  medianas_modelo$cluster <- 1:centers
  
  medianas_modelo <- medianas_modelo[, c("cluster", variables)]
  
  medianas_modelo[, variables] <- round(
    medianas_modelo[, variables],
    decimales
  )
  
  #-------------------------------------------------------
  # Medianas en escala original
  # Se recalculan directamente sobre los datos originales
  # según los clústeres finales.
  #-------------------------------------------------------
  
  datos_temp <- datos
  
  datos_temp$cluster_KMedians <- mejor_cluster
  
  medianas_originales <- aggregate(
    datos_temp[, variables],
    by = list(cluster = datos_temp$cluster_KMedians),
    FUN = median
  )
  
  medianas_originales[, variables] <- round(
    medianas_originales[, variables],
    decimales
  )
  
  #-------------------------------------------------------
  # Frecuencia por clúster
  #-------------------------------------------------------
  
  frecuencia_cluster <- as.data.frame(
    table(mejor_cluster)
  )
  
  names(frecuencia_cluster) <- c("cluster", "n")
  
  frecuencia_cluster$cluster <- as.integer(
    as.character(frecuencia_cluster$cluster)
  )
  
  #-------------------------------------------------------
  # Resultado
  #-------------------------------------------------------
  
  resultado <- list(
    cluster = mejor_cluster,
    medianas_modelo = medianas_modelo,
    medianas_originales = medianas_originales,
    SAD_total = round(mejor_SAD, decimales),
    SAD_por_cluster = round(SAD_por_cluster, decimales),
    SAD_por_registro = round(SAD_por_registro, decimales),
    frecuencia_cluster = frecuencia_cluster,
    mejor_inicio = mejor_inicio,
    iteraciones = mejor_iteracion,
    resumen_inicios = resumen_inicios,
    datos_modelo = as.data.frame(X_modelo),
    objeto_escala = objeto_escala,
    variables = variables,
    centers = centers,
    nstart = nstart,
    max_iter = max_iter,
    estandarizar = estandarizar
  )
  
  return(resultado)
}