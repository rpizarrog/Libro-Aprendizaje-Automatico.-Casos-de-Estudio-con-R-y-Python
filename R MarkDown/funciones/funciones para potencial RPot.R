# Funciones para implementar y evaluar modelos potencial, exponencial, logarítmico y polinomiales en R
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
  
  formula_modelo <- paste0(y, " ~ log(", x, ")")
  
  modelo <- lm(formula_modelo, data = datos)
  # 🔥 limpiar el call (esto es la clave)
  modelo$call$formula <- formula_modelo
  
  return(modelo)
}

f_construir_modelo_pot <- function(datos, var_x, var_y) {
  
  if (!all(c(var_x, var_y) %in% names(datos))) {
    stop("Las variables no existen en el dataset")
  }
  
  if (any(datos[[var_x]] <= 0) | any(datos[[var_y]] <= 0)) {
    stop("Variables deben ser positivas")
  }
  
  # construir fórmula
  formula_modelo <- as.formula(
    paste0("log(", var_y, ") ~ log(", var_x, ")")
  )
  
  modelo <- lm(formula_modelo, data = datos)
  
  # 🔥 limpiar el call (esto es la clave)
  modelo$call$formula <- formula_modelo
  
  return(modelo)
}



f_construir_modelo_exp <- function(datos, x, y) {
  
  #------------------------------------------------------------
  # Construye modelo exponencial: log(y) ~ x
  #------------------------------------------------------------
  
  # Validación
  if (!all(c(x, y) %in% names(datos))) {
    stop("Las variables no existen en el dataset")
  }
  
  if (any(datos[[y]] <= 0)) {
    stop("La variable dependiente debe ser positiva")
  }
  
  #------------------------------------------------------------
  # Fórmula dinámica (🔥 clave)
  #------------------------------------------------------------
  formula_modelo <- paste0("log(", y, ") ~ ", x)
  
  modelo <- lm(formula_modelo, data = datos)
  # 🔥 limpiar el call (esto es la clave)
  modelo$call$formula <- formula_modelo
  
  return(modelo)
}

f_construir_modelo <- function(datos, x, y, grado = 1){
  # Construye un modelo polinomial
  # recibe los datos y los nombres de las variables independiente y dependiente así como el 
  # el grado o orden en la ecuación
  # devuelvel el modelo creado
  
  formula_modelo <- paste0(y, " ~ poly(", x, ", ", grado, ", raw = TRUE)")
  
  modelo <- lm(formula_modelo, data = datos)
  # 🔥 limpiar el call (esto es la clave)
  modelo$call$formula <- formula_modelo
  
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
  
  # Función que recibe una lista de seis modelo y sus nombres y visualiza la tendencia de cada uno
  n_modelos <- length(modelos)
  
  if(is.null(nombres)){
    nombres <- paste("Modelo", 1:n_modelos)
  }
  
  graficos <- list()
  
  for(i in 1:n_modelos){
    
    modelo <- modelos[[i]]
    
    r <- cor(datos[[x]], datos[[y]])
    r2 <- summary(modelo)$r.squared
    
    # Secuencia suave
    x_seq <- seq(min(datos[[x]]), max(datos[[x]]), length.out = 200)
    
    # Detectar variables del modelo
    vars_modelo <- all.vars(formula(modelo))
    var_x_modelo <- vars_modelo[length(vars_modelo)]
    
    new_data <- data.frame(x_seq)
    colnames(new_data) <- var_x_modelo
    
    #------------------------------------------------------------
    # 🔥 DETECCIÓN ROBUSTA DE MODELO
    #------------------------------------------------------------
    lhs <- as.character(formula(modelo))[2]
    rhs <- as.character(formula(modelo))[3]
    
    es_potencial   <- grepl("log", lhs) & grepl("log", rhs)
    es_exponencial <- grepl("log", lhs) & !grepl("log", rhs)
    
    #------------------------------------------------------------
    # PREDICCIÓN
    #------------------------------------------------------------
    y_pred <- predict(modelo, newdata = new_data)
    
    if(es_exponencial){
      y_pred <- exp(y_pred)
    }
    
    if(es_potencial){
      y_pred <- exp(y_pred)
    }
    
    df_linea <- data.frame(x = x_seq, y = y_pred)
    
    #------------------------------------------------------------
    # GRÁFICA
    #------------------------------------------------------------
    g <- ggplot(datos, aes(x = .data[[x]], y = .data[[y]])) +
      geom_point(color = "black", alpha = 0.5) +
      geom_line(data = df_linea,
                aes(x = x, y = y),
                color = "red",
                linewidth = 1.2) +
      ggtitle(nombres[i],
              subtitle = paste("r =", round(r,3),
                               "; R² =", round(r2,3))) +
      theme_minimal()
    
    graficos[[i]] <- g
  }
  
  #------------------------------------------------------------
  # 🔥 PANEL DINÁMICO
  #------------------------------------------------------------
  wrap_plots(graficos, ncol = 3)
}


f_linealidad_residuos <- function(modelos, nombres = NULL){
# Funci[on que recibe modelos y sus nombres
# Hace la prueba de turkey de cada uno de los modelos
# Visualzia la linealidad o la curvatura de los residuos con los valores ajustados
# PResenta la prueba de Tukey en cada modelo
# Interpreta si existte linealidad grafica y con la prueba de Tukey
  library(car)
  library(ggplot2)
  library(patchwork)
  
  n <- length(modelos)
  
  if(is.null(nombres)){
    nombres <- paste("Modelo", 1:n)
  }
  
  graficos <- list()
  resultados <- data.frame()
  
  for(i in 1:n){
    
    modelo <- modelos[[i]]
    
    #------------------------------------------
    # 1. Prueba de Tukey
    #------------------------------------------
    rp <- residualPlots(modelo, plot = FALSE)
    p_value <- rp["Tukey test", "Pr(>|Test stat|)"]
    
    #------------------------------------------
    # 2. Interpretación
    #------------------------------------------
    if (p_value > 0.05){
      interpretacion <- "✔ Linealidad (sin curvatura)"
    } else {
      interpretacion <- "✖ No linealidad (curvatura)"
    }
    
    #------------------------------------------
    # 3. Datos para gráfica
    #------------------------------------------
    df <- data.frame(
      ajustados = fitted(modelo),
      residuos  = residuals(modelo)
    )
    
    #------------------------------------------
    # 4. Gráfica ggplot
    #------------------------------------------
    g <- ggplot(df, aes(x = ajustados, y = residuos)) +
      geom_point(color = "black", alpha = 0.6) +
      geom_hline(yintercept = 0, color = "red", linewidth = 1) +
      geom_smooth(method = "loess", se = FALSE, color = "blue", linewidth = 1) +
      ggtitle(nombres[i],
              subtitle = paste0(
                "Tukey p = ", round(p_value,4),
                " → ", interpretacion
              )) +
      xlab("Valores ajustados") +
      ylab("Residuos") +
      theme_minimal()
    
    graficos[[i]] <- g
    
    #------------------------------------------
    # 5. Guardar resultados
    #------------------------------------------
    resultados <- rbind(resultados, data.frame(
      Modelo = nombres[i],
      Tukey_p = round(p_value, 6),
      Interpretacion = interpretacion
    ))
  }
  
  #------------------------------------------
  # 6. Panel dinámico
  #------------------------------------------
  panel <- wrap_plots(graficos, ncol = 2)
  
  print(panel)
  
  #------------------------------------------
  # 7. Retornar tabla
  #------------------------------------------
  return(resultados)
}

f_matriz_verificar_homocedasticidad <- function(modelos, datos, x, y, nombres = NULL){
  #------------------------------------------------------------
  # Verifica homocedasticidad de 4 modelos
  # Soporta modelos lineales, polinomiales y exponenciales
  #------------------------------------------------------------
  # La función implementada permite analizar la homocedasticidad mediante gráficos de residuos 
  # contra valores ajustados para diferentes tipos de modelos, 
  # tales como polinomiales, logarítmicos y exponenciales y potenciales 
  # Para los modelos estimados mediante transformación logarítmica, se realiza 
  # una retransformación de las predicciones a la escala original, 
  # asegurando que los residuos reflejen adecuadamente la variabilidad del modelo 
  # en su dominio natural.
 
    n <- length(modelos)
    
    if(is.null(nombres)){
      nombres <- paste("Modelo", 1:n)
    }
    
    graficos <- list()
    
    for(i in 1:n){
      
      modelo <- modelos[[i]]
      
      #--------------------------------------------------------
      # DETECTAR TIPO DE MODELO
      #--------------------------------------------------------
      tipo <- "lineal"
      
      if(inherits(modelo, "lm")){
        
        formula_modelo <- paste(as.character(formula(modelo)), collapse = " ")
        
        es_log_y <- grepl("log\\(", formula_modelo) && grepl("~", formula_modelo)
        es_log_x <- grepl("~ log\\(", formula_modelo)
        
        if(es_log_y & es_log_x){
          tipo <- "potencial"
        } else if(es_log_y){
          tipo <- "exponencial"
        } else if(es_log_x){
          tipo <- "logaritmico"
        } else {
          tipo <- "polinomial"
        }
      }
      
      #--------------------------------------------------------
      # PREDICCIONES
      #--------------------------------------------------------
      y_pred <- predict(modelo, newdata = datos)
      
      #--------------------------------------------------------
      # CORRECCIÓN SEGÚN MODELO
      #--------------------------------------------------------
      if(tipo == "exponencial"){
        y_pred <- exp(y_pred)
      }
      
      if(tipo == "potencial"){
        y_pred <- exp(y_pred)
      }
      
      # logarítmico y polinomial no requieren ajuste
      
      #--------------------------------------------------------
      # RESIDUOS
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
        
        geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
        
        geom_smooth(method = "loess", se = FALSE,
                    color = "blue", linewidth = 1, alpha = 0.5) +
        
        geom_hline(yintercept = c(-2*sd(residuos), 2*sd(residuos)),
                   linetype = "dotted", alpha = 0.3) +
        
        ggtitle(nombres[i],
                subtitle = paste("Tipo:", tipo)) +
        
        xlab("Valores ajustados") +
        ylab("Residuos") +
        theme_minimal()
      
      graficos[[i]] <- g
    }
    
    #------------------------------------------------------------
    # PANEL DINÁMICO
    #------------------------------------------------------------
    ncol_panel <- ifelse(n <= 4, 2, 3)
    
    panel <- wrap_plots(graficos, ncol = ncol_panel) +
      plot_annotation(title = "Evaluación de homocedasticidad de modelos")
    
    print(panel)
}


f_pruebas_homocedasticidad <- function(modelos, nombres = NULL){
  # La función recibe los modelos y los nombres
  # Verifica la homocedasticidad de los residuos
  # Mediante la función bptest() de la librería "lmtest" se hace la prueba de Breusch–Pagan
  # Con la misma función  bptest() modificando argumentos se hace la prueba de White

    n <- length(modelos)
    
    if(is.null(nombres)){
      nombres <- paste("Modelo", 1:n)
    }
    
    resultados <- data.frame()
    
    for(i in 1:n){
      
      modelo <- modelos[[i]]
      
      #------------------------------------------
      # Breusch-Pagan
      #------------------------------------------
      bp <- bptest(modelo)
      
      #------------------------------------------
      # White (corregido)
      #------------------------------------------
      y_hat <- fitted(modelo)
      
      df_aux <- data.frame(y_hat = y_hat)
      
      white <- bptest(modelo, ~ y_hat + I(y_hat^2), data = df_aux)
      
      #------------------------------------------
      # Guardar resultados
      #------------------------------------------
      resultados <- rbind(resultados, data.frame(
        Modelo = nombres[i],
        BP_p_value = round(bp$p.value, 6),
        White_p_value = round(white$p.value, 6),
        BP_resultado = ifelse(bp$p.value > 0.05, "✔ Homo", "✖ Hetero"),
        White_resultado = ifelse(white$p.value > 0.05, "✔ Homo", "✖ Hetero")
      ))
    }
    
    return(resultados)
}

f_matriz_verificar_normalidad <- function(modelos, datos, x, y, nombres = NULL){
  
  library(ggplot2)
  library(patchwork)
  
  n <- length(modelos)
  
  if(is.null(nombres)){
    nombres <- paste("Modelo", 1:n)
  }
  
  resultados <- data.frame()
  graficos <- list()
  
  for(i in 1:n){
    
    modelo <- modelos[[i]]
    
    #--------------------------------------------------------
    # 🔥 DETECCIÓN DE TIPO (TU LÓGICA MEJORADA)
    #--------------------------------------------------------
    lhs <- as.character(formula(modelo))[2]
    rhs <- as.character(formula(modelo))[3]
    
    es_log_y <- grepl("^log\\(", lhs)
    es_log_x <- grepl("log\\(", rhs)
    
    if(es_log_y & es_log_x){
      tipo <- "potencial"
    } else if(es_log_y){
      tipo <- "exponencial"
    } else if(es_log_x){
      tipo <- "logaritmico"
    } else if(grepl("poly", rhs)){
      tipo <- "polinomial"
    } else {
      tipo <- "lineal"
    }
    
    #--------------------------------------------------------
    # 🔥 RESIDUOS CORRECTOS (CLAVE)
    #--------------------------------------------------------
    residuos <- residuals(modelo)
    
    # eliminar valores problemáticos
    residuos <- residuos[is.finite(residuos)]
    
    # estandarizar (evita escalas absurdas)
    residuos <- scale(residuos)[,1]
    
    #--------------------------------------------------------
    # SHAPIRO
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
    
    interpretacion <- ifelse(p > 0.05, "✔ Normal", "✖ No normal")
    
    resultados <- rbind(resultados, data.frame(
      Modelo = nombres[i],
      Tipo = tipo,
      W = round(W,4),
      p_value = round(p,4),
      Normalidad = interpretacion
    ))
    
    df <- data.frame(residuos = residuos)
    
    #--------------------------------------------------------
    # HISTOGRAMA
    #--------------------------------------------------------
    g1 <- ggplot(df, aes(x = residuos)) +
      geom_histogram(bins = 30, fill = "gray80", color = "black") +
      geom_density(aes(y = ..count..), color = "blue", linewidth = 1) +
      labs(
        title = nombres[i],
        subtitle = paste("Tipo:", tipo,
                         "| W =", round(W,3),
                         "| p =", round(p,3),
                         "|", interpretacion),
        x = "Residuos estandarizados",
        y = "Frecuencia"
      ) +
      theme_minimal(base_size = 11) +
      theme(
        plot.subtitle = element_text(size = 8)
      )
    
    #--------------------------------------------------------
    # QQ-PLOT
    #--------------------------------------------------------
    g2 <- ggplot(df, aes(sample = residuos)) +
      stat_qq(size = 1.2, color = "blue") +
      stat_qq_line(color = "red", linewidth = 1) +
      labs(
        title = nombres[i],
        subtitle = paste("Q-Q Plot |", interpretacion),
        x = "Cuantiles teóricos",
        y = "Cuantiles observados"
      ) +
      theme_minimal(base_size = 11) +
      theme(
        plot.subtitle = element_text(size = 8)
      )
    
    graficos[[i]] <- g1 | g2
  }
  
  #------------------------------------------------------------
  # PANEL DINÁMICO
  #------------------------------------------------------------
  panel <- wrap_plots(graficos, ncol = ifelse(n <= 4, 2, 3)) +
    plot_annotation(title = "Evaluación de normalidad de residuos")
  
  print(panel)
  
  #------------------------------------------------------------
  # RANKING
  #------------------------------------------------------------
  resultados$Ranking <- rank(-resultados$p_value)
  resultados <- resultados[order(resultados$Ranking), ]
  
  rownames(resultados) <- NULL
  
  return(resultados)
}


f_shapiro_residuos_modelos <- function(modelos, datos, x, y, nombres = NULL){
  
  #------------------------------------------------------------
  # Evalúa normalidad de residuos con Shapiro-Wilk
  # Soporta:
  #   - Lineal
  #   - Polinomial
  #   - Logarítmico
  #   - Exponencial
  #   - Potencial
  #------------------------------------------------------------
  
  n <- length(modelos)
  
  if(is.null(nombres)){
    nombres <- paste("Modelo", 1:n)
  }
  
  resultados <- data.frame()
  
  for(i in 1:n){
    
    modelo <- modelos[[i]]
    
    #--------------------------------------------------------
    # 🔥 DETECTAR TIPO DE MODELO
    #--------------------------------------------------------
    tipo <- "lineal"
    
    if(inherits(modelo, "lm")){
      
      formula_modelo <- paste(as.character(formula(modelo)), collapse = " ")
      
      es_log_y <- grepl("log\\(", formula_modelo) && grepl("~", formula_modelo)
      es_log_x <- grepl("~ log\\(", formula_modelo)
      
      if(es_log_y & es_log_x){
        tipo <- "potencial"
      } else if(es_log_y){
        tipo <- "exponencial"
      } else if(es_log_x){
        tipo <- "logaritmico"
      } else {
        tipo <- "polinomial"
      }
    }
    
    #--------------------------------------------------------
    # 🔥 PREDICCIONES
    #--------------------------------------------------------
    y_pred <- predict(modelo, newdata = datos)
    
    #--------------------------------------------------------
    # 🔥 CORRECCIÓN DE ESCALA
    #--------------------------------------------------------
    if(tipo == "exponencial"){
      y_pred <- exp(y_pred)
    }
    
    if(tipo == "potencial"){
      y_pred <- exp(y_pred)
    }
    
    #--------------------------------------------------------
    # 🔥 RESIDUOS CORRECTOS
    #--------------------------------------------------------
    residuos <- datos[[y]] - y_pred
    
    #--------------------------------------------------------
    # 🔥 SHAPIRO (máx 5000)
    #--------------------------------------------------------
    if(length(residuos) > 5000){
      set.seed(123)
      residuos_test <- sample(residuos, 5000)
    } else {
      residuos_test <- residuos
    }
    
    sh <- shapiro.test(residuos_test)
    
    W <- as.numeric(sh$statistic)
    p <- sh$p.value
    
    interpretacion <- ifelse(p > 0.05,
                             "✔ Normal",
                             "✖ No normal")
    
    resultados <- rbind(resultados, data.frame(
      Modelo = nombres[i],
      Tipo = tipo,
      W = round(W,4),
      p_value = round(p,6),
      Decision = interpretacion
    ))
  }
  
  #------------------------------------------------------------
  # 🔥 ORDENAR POR MEJOR NORMALIDAD
  #------------------------------------------------------------
  resultados$Ranking <- rank(-resultados$p_value)
  resultados <- resultados[order(resultados$Ranking), ]
  
  return(resultados)
}


f_anderson_residuos_modelos <- function(modelos, datos, x, y, nombres = NULL){
  
  #------------------------------------------------------------
  # Evalúa normalidad de residuos con Anderson-Darling (nortest)
  # Soporta:
  #   - Lineal
  #   - Polinomial
  #   - Logarítmico
  #   - Exponencial (log(y) ~ x)
  #   - Potencial (log(y) ~ log(x))
  #------------------------------------------------------------
  
  if(!require(nortest)) stop("Instala el paquete 'nortest'")
  
  n <- length(modelos)
  
  if(is.null(nombres)){
    nombres <- paste("Modelo", 1:n)
  }
  
  resultados <- data.frame()
  
  for(i in 1:n){
    
    modelo <- modelos[[i]]
    
    #--------------------------------------------------------
    # 🔥 DETECTAR TIPO DE MODELO
    #--------------------------------------------------------
    tipo <- "lineal"
    
    if(inherits(modelo, "lm")){
      
      formula_modelo <- paste(as.character(formula(modelo)), collapse = " ")
      
      es_log_y <- grepl("log\\(", formula_modelo) && grepl("~", formula_modelo)
      es_log_x <- grepl("~ log\\(", formula_modelo)
      
      if(es_log_y & es_log_x){
        tipo <- "potencial"
      } else if(es_log_y){
        tipo <- "exponencial"
      } else if(es_log_x){
        tipo <- "logaritmico"
      } else {
        tipo <- "polinomial"
      }
    }
    
    #--------------------------------------------------------
    # 🔥 PREDICCIONES
    #--------------------------------------------------------
    y_pred <- predict(modelo, newdata = datos)
    
    #--------------------------------------------------------
    # 🔥 CORRECCIÓN DE ESCALA
    #--------------------------------------------------------
    if(tipo == "exponencial"){
      y_pred <- exp(y_pred)
    }
    
    if(tipo == "potencial"){
      y_pred <- exp(y_pred)
    }
    
    #--------------------------------------------------------
    # 🔥 RESIDUOS
    #--------------------------------------------------------
    residuos <- datos[[y]] - y_pred
    
    #--------------------------------------------------------
    # 🔥 ANDERSON-DARLING
    #--------------------------------------------------------
    ad <- nortest::ad.test(residuos)
    
    A <- as.numeric(ad$statistic)
    p <- ad$p.value
    
    interpretacion <- ifelse(p > 0.05,
                             "✔ Normal",
                             "✖ No normal")
    
    resultados <- rbind(resultados, data.frame(
      Modelo = nombres[i],
      Tipo = tipo,
      A2 = round(A,4),
      p_value = round(p,6),
      Decision = interpretacion
    ))
  }
  
  #------------------------------------------------------------
  # 🔥 RANKING (mejor normalidad arriba)
  #------------------------------------------------------------
  resultados$Ranking <- rank(-resultados$p_value)
  resultados <- resultados[order(resultados$Ranking), ]
  
  return(resultados)
}


f_kolmogorov_residuos_modelos <- function(modelos, datos, x, y, nombres = NULL){
  
  #------------------------------------------------------------
  # Evalúa normalidad de residuos con Kolmogorov-Smirnov
  # IMPORTANTE: requiere estandarización
  #------------------------------------------------------------
  
  n <- length(modelos)
  
  if(is.null(nombres)){
    nombres <- paste("Modelo", 1:n)
  }
  
  resultados <- data.frame()
  
  for(i in 1:n){
    
    modelo <- modelos[[i]]
    
    #--------------------------------------------------------
    # 🔥 DETECTAR TIPO DE MODELO
    #--------------------------------------------------------
    tipo <- "lineal"
    
    if(inherits(modelo, "lm")){
      
      formula_modelo <- paste(as.character(formula(modelo)), collapse = " ")
      
      es_log_y <- grepl("log\\(", formula_modelo) && grepl("~", formula_modelo)
      es_log_x <- grepl("~ log\\(", formula_modelo)
      
      if(es_log_y & es_log_x){
        tipo <- "potencial"
      } else if(es_log_y){
        tipo <- "exponencial"
      } else if(es_log_x){
        tipo <- "logaritmico"
      } else {
        tipo <- "polinomial"
      }
    }
    
    #--------------------------------------------------------
    # 🔥 PREDICCIONES
    #--------------------------------------------------------
    y_pred <- predict(modelo, newdata = datos)
    
    #--------------------------------------------------------
    # 🔥 CORRECCIÓN DE ESCALA
    #--------------------------------------------------------
    if(tipo == "exponencial"){
      y_pred <- exp(y_pred)
    }
    
    if(tipo == "potencial"){
      y_pred <- exp(y_pred)
    }
    
    #--------------------------------------------------------
    # 🔥 RESIDUOS
    #--------------------------------------------------------
    residuos <- datos[[y]] - y_pred
    
    #--------------------------------------------------------
    # 🔥 ESTANDARIZACIÓN (CLAVE)
    #--------------------------------------------------------
    media <- mean(residuos)
    desv  <- sd(residuos)
    
    if(desv == 0){
      next
    }
    
    z <- (residuos - media) / desv
    
    #--------------------------------------------------------
    # 🔥 PRUEBA KS
    #--------------------------------------------------------
    ks <- ks.test(z, "pnorm")
    
    D <- as.numeric(ks$statistic)
    p <- ks$p.value
    
    interpretacion <- ifelse(p > 0.05,
                             "✔ Normal",
                             "✖ No normal")
    
    resultados <- rbind(resultados, data.frame(
      Modelo = nombres[i],
      Tipo = tipo,
      D = round(D,4),
      p_value = round(p,6),
      Decision = interpretacion
    ))
  }
  
  #------------------------------------------------------------
  # 🔥 RANKING
  #------------------------------------------------------------
  resultados$Ranking <- rank(-resultados$p_value)
  resultados <- resultados[order(resultados$Ranking), ]
  
  return(resultados)
}



f_normalidad_residuos_plot <- function(modelo){
  
  #----------------------------------------------------------
  # Evaluación de normalidad de residuos:
  # Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov
  #----------------------------------------------------------
  # Función que hace pruebas de normalidad en un modelo
  # Recibe modelo y regresa las pruebas y los graficos
  # Utiliza funciones de paquetes base para visualizar
  # Utiliza funcioness para pruebas de Shapiro-Wilks; Anderson Darling y Kolmogorov-Smirnov
  
  residuos <- residuals(modelo)
  
  media <- mean(residuos)
  desv  <- sd(residuos)
  
  # 🔥 estandarización (clave para KS)
  z <- (residuos - media) / desv
  
  library(nortest)
  
  #----------------------------------------------------------
  # PRUEBAS
  #----------------------------------------------------------
  shapiro <- shapiro.test(residuos)
  ad      <- ad.test(residuos)
  ks      <- ks.test(z, "pnorm")   # 🔥 corregido
  
  #----------------------------------------------------------
  # INTERPRETACIÓN
  #----------------------------------------------------------
  interpretar <- function(p){
    if(p > 0.05) return("✔ Normal")
    else return("✖ No normal")
  }
  
  #----------------------------------------------------------
  # TEXTO SUBTÍTULO (con saltos de línea 🔥)
  #----------------------------------------------------------
  subtitulo <- paste0(
    "Shapiro p=", round(shapiro$p.value,4), " ", interpretar(shapiro$p.value),
    "\nAD p=", round(ad$p.value,4), " ", interpretar(ad$p.value),
    "\nKS p=", round(ks$p.value,4), " ", interpretar(ks$p.value)
  )
  
  #----------------------------------------------------------
  # GRÁFICOS
  #----------------------------------------------------------
  par(mfrow = c(1, 2), cex.sub = 0.7)  # 🔥 reduce tamaño del subtítulo
  
 par(mfrow = c(1, 2), cex.sub = 0.7)

# HISTOGRAMA
hist(residuos,
     breaks = 20,
     col = "gray80",
     border = "black",
     main = "Histograma de residuos",
     xlab = "Residuos")

lines(density(residuos), col = "blue", lwd = 2)

mtext(subtitulo, side = 1, line = 3, cex = 0.7)  # 🔥 control total

# QQ-PLOT
qqnorm(residuos,
       main = "Q-Q Plot",
       xlab = "Cuantiles teóricos")

qqline(residuos, col = "red", lwd = 2)

mtext(subtitulo, side = 1, line = 3, cex = 0.7)

par(mfrow = c(1, 1))
  
  #----------------------------------------------------------
  # TABLA RESULTADOS
  #----------------------------------------------------------
  resultados <- data.frame(
    Prueba = c("Shapiro-Wilk", "Anderson-Darling", "Kolmogorov-Smirnov"),
    Estadistico = c(
      as.numeric(shapiro$statistic),
      as.numeric(ad$statistic),
      as.numeric(ks$statistic)
    ),
    p_value = c(
      shapiro$p.value,
      ad$p.value,
      ks$p.value
    ),
    Decision = c(
      interpretar(shapiro$p.value),
      interpretar(ad$p.value),
      interpretar(ks$p.value)
    )
  )
  
  return(resultados)
}




f_normalidad_residuos_modelos_plot <- function(modelos, nombres = NULL){
  
  # funci[on que recibe lista de modelos y nombre
  # A su vez manda llamar la función f_normalidad_residuos_plot()
  # que valora las prueba de normalidad ybcluyendo histograma y qqplot() de todos los modelos
  if(is.null(nombres)){
    nombres <- paste("Modelo", 1:length(modelos))
  }
  
  resultados <- list()
  
  for(i in 1:length(modelos)){
    
    cat("\n=============================\n")
    cat(nombres[i], "\n")
    cat("=============================\n")
    
    res <- f_normalidad_residuos_plot(modelos[[i]])
    
    resultados[[i]] <- res
  }
  
  return(resultados)
}



f_matriz_verificar_independencia_residuos <- function(
    # Valida la independencia de residuos a los modelos de regresi[on que
    # se reciben en la funci[on
    # Calcula el estadistico Durbin Watosn
  modelos, datos = NULL, x = NULL, y = NULL,
    nombres = NULL, graficar = TRUE){
  
  library(ggplot2)
  library(lmtest)
  library(patchwork)
  
  n <- length(modelos)
  
  if(is.null(nombres)){
    nombres <- paste("Modelo", 1:n)
  }
  
  resultados <- data.frame()
  graficos <- list()
  
  for(i in 1:n){
    
    modelo <- modelos[[i]]
    
    try({
      
      #--------------------------------------------------------
      # 🔥 DURBIN-WATSON
      #--------------------------------------------------------
      prueba <- lmtest::dwtest(modelo)
      
      dw <- as.numeric(prueba$statistic)
      p_value <- prueba$p.value
      
      #--------------------------------------------------------
      # 🔥 INTERPRETACIÓN
      #--------------------------------------------------------
      if(dw >= 1.5 & dw <= 2.5){
        interpretacion <- "✔ Independencia"
      } else if(dw < 1.5){
        interpretacion <- "✖ Autocorrelación positiva"
      } else {
        interpretacion <- "✖ Autocorrelación negativa"
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
      
      #--------------------------------------------------------
      # 🔥 RESIDUOS
      #--------------------------------------------------------
      residuos <- residuals(modelo)
      
      df_plot <- data.frame(
        orden = 1:length(residuos),
        residuos = residuos
      )
      
      #--------------------------------------------------------
      # 🔥 GRÁFICO
      #--------------------------------------------------------
      g <- ggplot(df_plot, aes(x = orden, y = residuos)) +
        geom_line(color = "black", linewidth = 0.5) +
        geom_point(color = "blue", size = 1) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
        labs(
          title = nombres[i],
          subtitle = paste(
            "DW =", round(dw,3),
            "| p =", round(p_value,3),
            "\n", interpretacion,
            "|", decision
          ),
          x = "Orden",
          y = "Residuo"
        ) +
        theme_minimal(base_size = 10) +
        theme(
          plot.subtitle = element_text(size = 8)
        )
      
      graficos[[i]] <- g
      
    }, silent = TRUE)
  }
  
  #------------------------------------------------------------
  # 🔥 PANEL DINÁMICO
  #------------------------------------------------------------
  if(graficar && length(graficos) > 0){
    
    ncol_panel <- ifelse(n <= 4, 2, 3)
    
    panel <- wrap_plots(graficos, ncol = ncol_panel) +
      plot_annotation(title = "Evaluación de independencia de residuos")
    
    print(panel)
  }
  
  #------------------------------------------------------------
  # 🔥 RANKING (cercanía a 2)
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
    
    formula_modelo <- paste(as.character(formula(modelo)), collapse = " ")
    
    #------------------------------------------------------------
    # DETECCIÓN CORRECTA 
    #------------------------------------------------------------
    lhs <- as.character(formula(modelo))[2]
    rhs <- as.character(formula(modelo))[3]
    
    es_potencial   <- grepl("log", lhs) & grepl("log", rhs)
    es_exponencial <- grepl("log", lhs) & !grepl("log", rhs)
    
    #------------------------------------------------------------
    # CONSTRUIR ECUACIÓN
    #------------------------------------------------------------
    
    if(es_potencial){
      
      b0 <- coefs[1]
      b1 <- coefs[2]
      
      a <- exp(b0)
      b <- b1
      
      # 🔥 usar nombre real de variable
      var <- gsub("log\\(|\\)", "", nombres_vars[2])
      
      ecuacion <- paste0("ŷ = ", round(a,4), " * ", var, "^(", round(b,4), ")")
      
    } else if(es_exponencial){
      
      b0 <- coefs[1]
      b1 <- coefs[2]
      
      a <- exp(b0)
      b <- b1
      
      var <- nombres_vars[2]
      
      ecuacion <- paste0("ŷ = ", round(a,4), " * e^(", round(b,4), " * ", var, ")")
      
    } else {
      
      ecuacion <- paste0("ŷ = ", round(coefs[1],4))
      
      for(j in 2:length(coefs)){
        
        signo <- ifelse(coefs[j] >= 0, "+", "-")
        
        termino <- nombres_vars[j]
        termino <- gsub("log", "ln", termino)
        termino <- gsub("\\(", "", termino)
        termino <- gsub("\\)", "", termino)
        
        ecuacion <- paste(ecuacion,
                          signo,
                          abs(round(coefs[j],4)),
                          "*", termino)
      }
    }
    
    cat("\nEcuación:\n", ecuacion, "\n")
  }
}





f_evaluar_modelos_varios <- function(modelos, datos, y, nombres){
  
  resultados <- lapply(1:length(modelos), function(i){
    
    res <- f_evaluar_modelo(modelos[[i]], datos, y)
    
    if(!is.null(res)){
      res$Modelo <- nombres[i]
      return(res)
    }
  })
  
  df <- dplyr::bind_rows(resultados)
  
  #--------------------------------------------------------
  # 🔥 FILTRAR MODELOS VÁLIDOS
  #--------------------------------------------------------
  df_validos <- df[df$Estado == "OK", ]
  
  #--------------------------------------------------------
  # 🔥 ORDENAR
  #--------------------------------------------------------
  df_validos <- df_validos[order(df_validos$RMSE), ]
  
  rownames(df_validos) <- NULL
  
  return(df_validos)
}



f_evaluar_modelo <- function(modelo, datos_validacion, variable_dependiente){
  # Evalúa los modelos recibidos 
  # R-Square y RMSE
    #--------------------------------------------------------
    # VALIDACIÓN INICIAL
    #--------------------------------------------------------
    if(!inherits(modelo, "lm")){
      stop("El modelo debe ser de tipo lm")
    }
    
    y_real <- datos_validacion[[variable_dependiente]]
    
    #--------------------------------------------------------
    # 🔥 DETECCIÓN CORRECTA (TU LÓGICA)
    #--------------------------------------------------------
    lhs <- as.character(formula(modelo))[2]
    rhs <- as.character(formula(modelo))[3]
    
    es_log_y <- grepl("^log\\(", lhs)
    es_log_x <- grepl("log\\(", rhs)
    
    if(es_log_y & es_log_x){
      tipo <- "potencial"
    } else if(es_log_y){
      tipo <- "exponencial"
    } else if(es_log_x){
      tipo <- "logaritmico"
    } else if(grepl("poly", rhs)){
      tipo <- "polinomial"
    } else {
      tipo <- "lineal"
    }
    
    #--------------------------------------------------------
    # 🔥 PREDICCIONES
    #--------------------------------------------------------
    pred <- tryCatch(
      predict(modelo, newdata = datos_validacion),
      error = function(e) return(NULL)
    )
    
    if(is.null(pred)){
      warning("Error en predicción")
      return(NULL)
    }
    
    #--------------------------------------------------------
    # 🔥 CORRECCIÓN DE ESCALA
    #--------------------------------------------------------
    if(tipo %in% c("exponencial", "potencial")){
      pred <- exp(pred)
    }
    
    #--------------------------------------------------------
    # 🔥 CONTROL NUMÉRICO (MODELOS INESTABLES)
    #--------------------------------------------------------
    if(any(!is.finite(pred)) || max(abs(pred), na.rm = TRUE) > 1e10){
      warning("Modelo inestable detectado")
      return(data.frame(
        Tipo = tipo,
        R_square = NA,
        MSE = NA,
        RMSE = NA,
        MAE = NA,
        MAPE = NA,
        Estado = "Modelo inestable"
      ))
    }
    
    #--------------------------------------------------------
    # VALIDACIÓN LONGITUD
    #--------------------------------------------------------
    if(length(pred) != length(y_real)){
      stop("Predicciones y valores reales no coinciden")
    }
    
    #--------------------------------------------------------
    # 🔥 ERRORES
    #--------------------------------------------------------
    errores <- y_real - pred
    
    mse  <- mean(errores^2, na.rm = TRUE)
    rmse <- sqrt(mse)
    mae  <- mean(abs(errores), na.rm = TRUE)
    
    mape <- mean(abs(errores / ifelse(y_real == 0, NA, y_real)), na.rm = TRUE) * 100
    
    #--------------------------------------------------------
    # 🔥 R²
    #--------------------------------------------------------
    sst <- sum((y_real - mean(y_real))^2)
    sse <- sum((y_real - pred)^2)
    r2 <- 1 - (sse / sst)
    
    #--------------------------------------------------------
    # RESULTADO FINAL
    #--------------------------------------------------------
    resultado <- data.frame(
      Tipo = tipo,
      R_square = round(r2,4),
      MSE  = round(mse,4),
      RMSE = round(rmse,4),
      MAE  = round(mae,4),
      MAPE = round(mape,2),
      Estado = "OK"
    )
    
    return(resultado)
  }





