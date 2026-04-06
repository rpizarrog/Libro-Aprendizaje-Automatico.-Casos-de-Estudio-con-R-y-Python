# Funciones para implementar y evaluar modelos exponencial, logarítmico y polinomiales en R
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

f_visualizar_head_tail_reducido_word <- function(datos, n = 10) {
  #------------------------------------------------------------
  # f_visualizar_head_tail_reducido_word()
  # Objetivo:
  #   Mostrar primeros n y últimos n registros en una misma tabla,
  #   visualizando únicamente:
  #     - Los primeros 4 atributos
  #     - Los últimos 3 atributos
  #   Insertando una fila con "..." como separador.
  #
  # Nota:
  #   Para evitar conflictos de tipos (numérico vs texto),
  #   se convierte a character SOLO para la tabla de visualización.
  #
  # Argumentos:
  #   datos : data.frame
  #   n     : número de registros a mostrar (default = 10)
  #
  # Retorna:
  #   Objeto flextable compatible con Word.
  #
  # Requiere:
  #   library(dplyr)
  #   library(flextable)
  #------------------------------------------------------------
  
  
  total_columnas <- ncol(datos)
  
  # Índices: primeras 4 y últimas 3 (sin duplicar si hay pocas columnas)
  idx_prim <- 1:min(4, total_columnas)
  idx_ult  <- max(total_columnas - 2, 1):total_columnas
  columnas_seleccionadas <- unique(c(idx_prim, idx_ult))
  
  # Subconjunto reducido
  datos_reducidos <- datos[, columnas_seleccionadas, drop = FALSE]
  
  # Head y tail
  head_datos <- head(datos_reducidos, n)
  tail_datos <- tail(datos_reducidos, n)
  
  # Convertir a character SOLO para evitar choque de tipos en bind_rows()
  head_chr <- as.data.frame(lapply(head_datos, as.character), stringsAsFactors = FALSE)
  tail_chr <- as.data.frame(lapply(tail_datos, as.character), stringsAsFactors = FALSE)
  
  # Fila separadora "..."
  fila_puntos <- as.data.frame(
    matrix("...", nrow = 1, ncol = ncol(head_chr)),
    stringsAsFactors = FALSE
  )
  colnames(fila_puntos) <- colnames(head_chr)
  
  # Concatenar
  tabla_final <- bind_rows(head_chr, fila_puntos, tail_chr)
  
  # Flextable para Word
  tabla <- flextable(tabla_final)
  tabla <- autofit(tabla)
  
  return(tabla)
}

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

f_particionar_datos <- function(datos, proporcion_entrenamiento = 0.7) {
  #------------------------------------------------------------
  # f_particionar_datos()
  #
  # Objetivo:
  #   Dividir un conjunto de datos previamente preparado
  #   en dos subconjuntos: entrenamiento y validación.
  #
  # Descripción:
  #   La función realiza una partición aleatoria del dataset.
  #   El subconjunto de entrenamiento se utiliza para ajustar
  #   el modelo, mientras que el subconjunto de validación
  #   permite evaluar el desempeño del modelo en datos no
  #   utilizados durante el entrenamiento.
  #
  # Argumentos:
  #   datos : data.frame con los datos preparados
  #   proporcion_entrenamiento : proporción destinada al
  #                              entrenamiento (default = 0.70)
  #
  # Retorna:
  #   Lista con:
  #     $datos_entrenamiento
  #     $datos_validacion
  #
  # Reproducibilidad:
  #   Se fija la semilla en 2026, correspondiente al año de
  #   edición del libro, garantizando resultados replicables.
  #------------------------------------------------------------
  
  # Semilla para reproducibilidad
  set.seed(2026)
  
  # Número total de observaciones
  n <- nrow(datos)
  
  # Número de observaciones para entrenamiento
  n_train <- floor(proporcion_entrenamiento * n)
  
  # Selección aleatoria de índices
  indices_train <- sample(seq_len(n), size = n_train)
  
  # Generar subconjuntos
  datos_entrenamiento <- datos[indices_train, ]
  datos_validacion <- datos[-indices_train, ]
  
  # Devolver lista con ambos datasets
  return(list(
    datos_entrenamiento = datos_entrenamiento,
    datos_validacion = datos_validacion
  ))
}

f_construir_modelo_log <- function(datos, x, y){
  # Construye un modelo de regresión logarítmica (lin-log)
  # datos: data frame
  # x: variable independiente (texto)
  # y: variable dependiente (texto)
  
  formula_texto <- paste0(y, " ~ log(", x, ")")
  
  modelo <- lm(as.formula(formula_texto), data = datos)
  
  return(modelo)
}

f_construir_modelo_exp <- function(datos, var_x, var_y) {
  
  #------------------------------------------------------------
  # Objetivo:
  #   Construir un modelo de regresión exponencial
  #   mediante transformación logarítmica
  #
  # Retorna:
  #   Modelo lineal (lm) sobre log(y)
  #------------------------------------------------------------
  
  # Validación de variables
  if (!all(c(var_x, var_y) %in% names(datos))) {
    stop("Las variables no existen en el dataset")
  }
  
  # Extraer variables
  x <- datos[[var_x]]
  y <- datos[[var_y]]
  
  # Validación: log requiere valores positivos
  if (any(y <= 0)) {
    stop("La variable dependiente debe ser positiva")
  }
  
  # Transformación logarítmica
  log_y <- log(y)
  
  # Construcción del modelo
  modelo <- lm(log_y ~ x)
  
  return(modelo)
}

f_construir_modelo <- function(datos, x, y, grado = 1){
  # Construye un modelo polinomial
  # recibe los datos y los nombres de las variables independiente y dependiente así como el 
  # el grado o orden en la ecuación
  # devuelvel el modelo creado
  
  formula_texto <- paste0(y, " ~ poly(", x, ", ", grado, ", raw = TRUE)")
  
  modelo <- lm(as.formula(formula_texto), data = datos)
  
  return(modelo)
}


f_diagrama_dispersion_tendencia <- function(modelo, datos, x, y){
  # La función recibe el modelo los datos de entrenamiento 
  # y construye las tendencias de los modelos polinomiales
  r <- cor(datos[[x]], datos[[y]])
  r2 <- summary(modelo)$r.squared
  
  ggplot(datos, aes_string(x = x, y = y)) +
    geom_point(color = "black") +
    geom_smooth(method = "lm", formula = y ~ poly(x, modelo$rank-1, raw=TRUE),
                color = "red", se = FALSE) +
    ggtitle("Dispersión y tendencia",
            subtitle = paste(x, "vs", y,
                             "; r =", round(r,3),
                             "; R² =", round(r2,3))) +
    theme_minimal()
}






f_matriz_dispersion_modelos_tendencia <- function(modelos, datos, x, y, nombres = NULL){
  # La función recibe modelos los datos de entrenamiento
  # las variables independiente 'x' y dependiente 'y' así como el nombre de modelo
  # y visualiza las lineas de tendencia que ofrecen un panorama visual de postulado de linealidad
  
  if(length(modelos) != 4){
    stop("Debes proporcionar exactamente 4 modelos")
  }
  
  if(is.null(nombres)){
    nombres <- paste("Modelo", 1:4)
  }
  
  graficos <- list()
  
  for(i in 1:4){
    
    modelo <- modelos[[i]]
    
    r <- cor(datos[[x]], datos[[y]])
    r2 <- summary(modelo)$r.squared
    
    # Secuencia suave
    x_seq <- seq(min(datos[[x]]), max(datos[[x]]), length.out = 200)
    
    # Detectar variable del modelo
    vars_modelo <- all.vars(formula(modelo))
    var_x_modelo <- vars_modelo[2]
    
    new_data <- data.frame(x_seq)
    colnames(new_data) <- var_x_modelo
    
    # Predicción
    y_pred <- predict(modelo, newdata = new_data)
    
    # 🔥 Detectar si es modelo exponencial
    es_exponencial <- grepl("log", as.character(formula(modelo))[2])
    
    if(es_exponencial){
      y_pred <- exp(y_pred)
    }
    
    df_linea <- data.frame(x = x_seq, y = y_pred)
    
    g <- ggplot(datos, aes_string(x = x, y = y)) +
      geom_point(color = "black") +
      geom_line(data = df_linea, aes(x = x, y = y),
                color = "red", linewidth = 1.2) +
      ggtitle(nombres[i],
              subtitle = paste("r =", round(r,3),
                               "; R² =", round(r2,3))) +
      theme_minimal()
    
    graficos[[i]] <- g
  }
  
  (graficos[[1]] | graficos[[2]]) /
    (graficos[[3]] | graficos[[4]])
}

f_matriz_verificar_homocedasticidad <- function(modelos, datos, x, y, nombres = NULL){
  #------------------------------------------------------------
  # Verifica homocedasticidad de 4 modelos
  # Soporta modelos lineales, polinomiales y exponenciales
  #------------------------------------------------------------
  # La función implementada permite analizar la homocedasticidad mediante gráficos de residuos 
  # contra valores ajustados para diferentes tipos de modelos, 
  # tales como polinomiales, logarítmicos y exponenciales. 
  # Para los modelos estimados mediante transformación logarítmica, se realiza 
  # una retransformación de las predicciones a la escala original, 
  # asegurando que los residuos reflejen adecuadamente la variabilidad del modelo 
  # en su dominio natural.
  
  if(length(modelos) != 4){
    stop("Debes proporcionar exactamente 4 modelos")
  }
  
  if(is.null(nombres)){
    nombres <- paste("Modelo", 1:4)
  }
  
  graficos <- list()
  
  for(i in 1:4){
    
    modelo <- modelos[[i]]
    
    #--------------------------------------------------------
    # 🔥 DETECTAR SI ES MODELO EXPONENCIAL
    #--------------------------------------------------------
    es_exponencial <- FALSE
    
    if(inherits(modelo, "lm")){
      formula_modelo <- as.character(formula(modelo))
      es_exponencial <- grepl("log", formula_modelo[2])
    }
    
    #--------------------------------------------------------
    # PREDICCIONES
    #--------------------------------------------------------
    y_pred <- predict(modelo, newdata = datos)
    
    # 🔥 Corrección para modelo exponencial
    if(es_exponencial){
      y_pred <- exp(y_pred)
    }
    
    #--------------------------------------------------------
    # RESIDUOS CORRECTOS
    #--------------------------------------------------------
    residuos <- datos[[y]] - y_pred
    
    df_plot <- data.frame(
      y_pred = y_pred,
      residuos = residuos
    )
    
    #--------------------------------------------------------
    # GRÁFICA
    #--------------------------------------------------------
    g <- ggplot(df_plot, aes(x = y_pred, y = residuos)) +
      
      geom_point(alpha = 0.5, color = "black") +
      
      # línea en cero
      geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
      
      # suavizado LOESS
      geom_smooth(method = "loess", se = FALSE,
                  color = "blue", linewidth = 1, alpha = 0.5) +
      
      # bandas de dispersión
      geom_hline(yintercept = c(-2*sd(residuos), 2*sd(residuos)),
                 linetype = "dotted", alpha = 0.3) +
      
      ggtitle(nombres[i]) +
      xlab("Valores ajustados") +
      ylab("Residuos") +
      theme_minimal()
    
    graficos[[i]] <- g
  }
  
  #------------------------------------------------------------
  # MATRIZ 2x2
  #------------------------------------------------------------
  (graficos[[1]] | graficos[[2]]) /
    (graficos[[3]] | graficos[[4]])
}

f_matriz_verificar_normalidad <- function(modelos, datos, x, y, nombres = NULL){
  #------------------------------------------------------------
  # Verifica NORMALIDAD de residuos (visual + Shapiro)
  # Soporta:
  #   - Polinomiales
  #   - Logarítmicos
  #   - Exponenciales (log(y) ~ x)
  #------------------------------------------------------------
  
  if(length(modelos) != 4){
    stop("Debes proporcionar exactamente 4 modelos")
  }
  
  if(is.null(nombres)){
    nombres <- paste("Modelo", 1:4)
  }
  
  resultados <- data.frame()
  graficos <- list()
  
  for(i in 1:4){
    
    modelo <- modelos[[i]]
    
    #--------------------------------------------------------
    # 🔥 DETECTAR MODELO EXPONENCIAL
    #--------------------------------------------------------
    es_exponencial <- FALSE
    
    if(inherits(modelo, "lm")){
      formula_modelo <- as.character(formula(modelo))
      es_exponencial <- grepl("log", formula_modelo[2])
    }
    
    #--------------------------------------------------------
    # PREDICCIONES
    #--------------------------------------------------------
    y_pred <- predict(modelo, newdata = datos)
    
    # 🔥 corrección exponencial
    if(es_exponencial){
      y_pred <- exp(y_pred)
    }
    
    #--------------------------------------------------------
    # RESIDUOS CORRECTOS
    #--------------------------------------------------------
    residuos <- datos[[y]] - y_pred
    
    #--------------------------------------------------------
    # SHAPIRO (limitado a 5000)
    #--------------------------------------------------------
    if(length(residuos) > 5000){
      set.seed(123)
      residuos_test <- sample(residuos, 5000)
    } else {
      residuos_test <- residuos
    }
    
    sh <- shapiro.test(residuos_test)
    
    W <- sh$statistic
    p <- sh$p.value
    
    interpretacion <- ifelse(p > 0.05, "Normal", "No normal")
    
    resultados <- rbind(resultados, data.frame(
      Modelo = nombres[i],
      W = round(W,4),
      p_value = round(p,4),
      Normalidad = interpretacion
    ))
    
    df <- data.frame(residuos = residuos)
    
    #--------------------------------------------------------
    # HISTOGRAMA
    #--------------------------------------------------------
    g1 <- ggplot(df, aes(x = residuos)) +
      geom_histogram(bins = 30,
                     fill = "gray80",
                     color = "black") +
      geom_density(aes(y = ..count..),
                   color = "blue", linewidth = 1) +
      labs(
        title = nombres[i],
        subtitle = paste("Histograma | W =", round(W,3),
                         "| p =", round(p,3),
                         "|", interpretacion),
        x = "Residuos",
        y = "Frecuencia"
      ) +
      theme_minimal(base_size = 11)
    
    #--------------------------------------------------------
    # QQ-PLOT
    #--------------------------------------------------------
    g2 <- ggplot(df, aes(sample = residuos)) +
      stat_qq(size = 1.2, color = "blue") +
      stat_qq_line(color = "red", linewidth = 1) +
      labs(
        title = nombres[i],
        subtitle = paste("Q-Q Plot | W =", round(W,3),
                         "| p =", round(p,3),
                         "|", interpretacion),
        x = "Cuantiles teóricos",
        y = "Cuantiles observados"
      ) +
      theme_minimal(base_size = 11)
    
    graficos[[i]] <- g1 | g2
  }
  
  #------------------------------------------------------------
  # MATRIZ VISUAL 2x2
  #------------------------------------------------------------
  layout <- (graficos[[1]] / graficos[[2]]) |
    (graficos[[3]] / graficos[[4]])
  
  print(layout)
  
  #------------------------------------------------------------
  # RANKING (mejor p-value arriba)
  #------------------------------------------------------------
  resultados$Ranking <- rank(-resultados$p_value)
  resultados <- resultados[order(resultados$Ranking), ]
  
  return(resultados)
}

f_matriz_verificar_independencia_residuos <- function(modelos, datos, x, y, nombres = NULL, graficar = TRUE){
  
  library(ggplot2)
  library(lmtest)
  library(patchwork)
  
  if(length(modelos) != 4){
    stop("Debes proporcionar exactamente 4 modelos")
  }
  
  if(is.null(nombres)){
    nombres <- paste("Modelo", 1:4)
  }
  
  resultados <- data.frame()
  graficos <- list()
  
  for(i in 1:4){
    
    modelo <- modelos[[i]]
    
    #--------------------------------------------------------
    # 🔥 TRY para evitar fallos silenciosos
    #--------------------------------------------------------
    try({
      
      prueba <- lmtest::dwtest(modelo)
      
      dw <- as.numeric(prueba$statistic)
      p_value <- prueba$p.value
      
      # interpretación
      if(dw >= 1.5 & dw <= 2.5){
        interpretacion <- "Independencia"
      } else if(dw < 1.5){
        interpretacion <- "Autocorrelación positiva"
      } else {
        interpretacion <- "Autocorrelación negativa"
      }
      
      decision <- ifelse(p_value > 0.05,
                         "No se rechaza H0",
                         "Se rechaza H0")
      
      resultados <- rbind(resultados, data.frame(
        Modelo = nombres[i],
        DW = round(dw,4),
        p_value = round(p_value,4),
        Interpretacion = interpretacion,
        Decision = decision
      ))
      
      # residuos
      residuos <- residuals(modelo)
      
      df_plot <- data.frame(
        orden = 1:length(residuos),
        residuos = residuos
      )
      
      g <- ggplot(df_plot, aes(x = orden, y = residuos)) +
        geom_line(color = "black", linewidth = 0.5) +
        geom_point(color = "blue", size = 1) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
        labs(
          title = nombres[i],
          subtitle = paste("DW =", round(dw,3),
                           "| p =", round(p_value,3),
                           "\n", interpretacion,
                           "|", decision),
          x = "Orden",
          y = "Residuo"
        ) +
        theme_minimal()
      
      graficos[[i]] <- g
      
    }, silent = TRUE)
  }
  
  #------------------------------------------------------------
  # 🔥 VERIFICAR QUE EXISTEN LOS 4 GRÁFICOS
  #------------------------------------------------------------
  if(graficar){
    
    if(length(graficos) < 4){
      warning("No se generaron los 4 gráficos. Revisa modelos.")
    }
    
    layout <- (graficos[[1]] | graficos[[2]]) /
      (graficos[[3]] | graficos[[4]])
    
    print(layout)
  }
  
  #------------------------------------------------------------
  # RANKING
  #------------------------------------------------------------
  if(nrow(resultados) > 0){
    resultados$Distancia_2 <- abs(resultados$DW - 2)
    resultados$Ranking <- rank(resultados$Distancia_2)
    resultados <- resultados[order(resultados$Ranking), ]
  }
  
  return(resultados)
}

f_ecuaciones_modelos <- function(modelos, nombres = NULL){
  
  if(is.null(nombres)){
    nombres <- paste("Modelo", 1:length(modelos))
  }
  
  for(i in 1:length(modelos)){
    
    modelo <- modelos[[i]]
    
    cat("\n============================\n")
    cat(nombres[i], "\n")
    cat("============================\n")
    
    coefs_raw <- coef(modelo)
    
    # eliminar NA
    coefs <- coefs_raw[!is.na(coefs_raw)]
    
    # asegurar formato numérico
    coefs <- as.numeric(coefs)
    names(coefs) <- names(coefs_raw)[!is.na(coefs_raw)]
    
    nombres_vars <- names(coefs)
    
    cat("\nCoeficientes:\n")
    print(round(coefs,4))
    
    #------------------------------------------------------------
    # DETECTAR SI ES MODELO EXPONENCIAL
    #------------------------------------------------------------
    formula_modelo <- as.character(formula(modelo))
    
    es_exponencial <- grepl("log", formula_modelo[2])
    
    #------------------------------------------------------------
    # CONSTRUIR ECUACIÓN
    #------------------------------------------------------------
    
    if(es_exponencial){
      
      # modelo: log(y) = b0 + b1 x
      b0 <- coefs[1]
      b1 <- coefs[2]
      
      a <- exp(b0)
      b <- b1
      
      ecuacion <- paste0("ŷ = ", round(a,4), " * e^(", round(b,4), " * x)")
      
    } else {
      
      # modelo normal (lineal, polinomial, log, etc.)
      ecuacion <- paste0("ŷ = ", round(coefs[1],4))
      
      for(j in 2:length(coefs)){
        
        signo <- ifelse(coefs[j] >= 0, "+", "-")
        
        termino <- nombres_vars[j]
        termino <- gsub("log", "ln", termino)
        
        ecuacion <- paste(ecuacion,
                          signo,
                          abs(round(coefs[j],4)),
                          "*", termino)
      }
    }
    
    cat("\nEcuación:\n", ecuacion, "\n")
  }
}


f_evaluar_modelo <- function(modelo, datos_validacion, variable_dependiente){
  
  #----------------------------------------------------------
  # La función recibe como argumentos el modelo, los datos de validación y la variable dependiente
  # que los utiliza para precisamente evaluar el modelo con la comparación entre datos reales y      # los datos de predicción. 
  # Se generan las predicciones del modelo usando los datos de validación.
  # Se calcula el error cuadrático medio (*MSE*) o promedio de los errores al cuadrado
  # Se calcula la raíz del error cuadrático medio (*RMSE*) como medida interpretable del error
  # Se calcula el coeficiente de determinación R Square r2
  # Luego el coeficiente coeficiente de determinación *R Square ajustad* 
  # Se construye y se devuelve como valor de retorno una tabla que resume todos los estadísticos calculados  
  #------------------------------------------------------------
  
  y_real <- datos_validacion[[variable_dependiente]]
  
  
  pred <- predict(modelo, newdata = datos_validacion)
  
  
  mse <- mean((y_real - pred)^2)
  
  
  rmse <- sqrt(mse)
  
  
  r2 <- summary(modelo)$r.squared
  
  
  r2_adj <- summary(modelo)$adj.r.squared
  
  
  resultado <- data.frame(
    R_square = round(r2,4),
    R_square_ajustado = round(r2_adj,4),
    MSE = round(mse,4),
    RMSE = round(rmse,4)
  )
  
  
  return(resultado)
  
}

f_evaluar_modelos_varios <- function(modelos, datos, y, x, nombres){
  # Evalúa modelos calculando r square y RMSE 
  resultados <- lapply(modelos, function(m){
    f_evaluar_modelo(m, datos, y)
  })
  
  df <- bind_rows(resultados)
  
  df <- cbind(df, nombres)
  
  df <- df[order(df$RMSE), ]
  
  return(df)
}