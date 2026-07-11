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

