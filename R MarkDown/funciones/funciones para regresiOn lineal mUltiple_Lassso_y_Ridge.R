
# Funciones para implementar y evaluar modelos de regresión multiple en R
# Se adecúan para que funcione Lasso y Ridge
# Rubén Pizarro Gurrola
# Abril 2025
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


f_redondear_numericas <- function(datos, decimales = 2) {
  #------------------------------------------------------------
  # f_redondear_numericas()
  #
  # Objetivo:
  #   Redondear variables numéricas de un data.frame
  #
  # Argumentos:
  #   datos      : data.frame
  #   decimales  : número de decimales (default = 2)
  #
  # Retorna:
  #   data.frame con variables numéricas redondeadas
  #------------------------------------------------------------
  
  datos_out <- datos
  
  # Identificar columnas numéricas
  idx_num <- sapply(datos_out, is.numeric)
  
  # Aplicar redondeo solo a numéricas
  datos_out[idx_num] <- lapply(datos_out[idx_num], function(x) {
    round(x, decimales)
  })
  
  return(datos_out)
}

f_construir_modelo_RLM <- function(datos, variable_dependiente, ver_resumen = TRUE) {
  #------------------------------------------------------------
  # f_construir_modelo_RLM()
  #
  # Objetivo:
  #   Construir un modelo de regresión lineal múltiple (RLM)
  #
  # Argumentos:
  #   datos                 : data.frame
  #   variable_dependiente  : nombre de la variable Y (string)
  #   ver_resumen           : TRUE/FALSE (mostrar summary)
  #
  # Retorna:
  #   Modelo lm()
  #------------------------------------------------------------
  
  #------------------------------------------
  # 1. Validación
  #------------------------------------------
  if (!variable_dependiente %in% names(datos)) {
    stop("La variable dependiente no existe en el dataset")
  }
  
  #------------------------------------------
  # 2. Copia de datos
  #------------------------------------------
  datos_modelo <- datos
  
  #------------------------------------------
  # 3. Conversión de variables
  #------------------------------------------
  
  # character → factor
  idx_char <- sapply(datos_modelo, is.character)
  datos_modelo[idx_char] <- lapply(datos_modelo[idx_char], as.factor)
  
  # logical → numeric (0/1)
  idx_logical <- sapply(datos_modelo, is.logical)
  datos_modelo[idx_logical] <- lapply(datos_modelo[idx_logical], function(x) as.numeric(x))
  
  #------------------------------------------
  # 4. Construcción de fórmula
  #------------------------------------------
  formula_modelo <- as.formula(paste(variable_dependiente, "~ ."))
  
  #------------------------------------------
  # 5. Ajuste del modelo
  #------------------------------------------
  modelo <- lm(formula_modelo, data = datos_modelo)
  
  #------------------------------------------
  # 6. Información del modelo
  #------------------------------------------
  cat("\n============================\n")
  cat("Modelo de Regresión Lineal Múltiple\n")
  cat("============================\n")
  cat("Variable dependiente:", variable_dependiente, "\n")
  cat("Número de observaciones:", nrow(datos_modelo), "\n")
  cat("Número de variables independientes:", ncol(datos_modelo) - 1, "\n")
  
  #------------------------------------------
  # 7. Resumen opcional
  #------------------------------------------
  if (ver_resumen) {
    print(summary(modelo))
  }
  
  return(modelo)
}

f_estandarizar_escalar <- function(datos, decimales = 4) {
  #------------------------------------------------------------
  # f_estandarizar_escalar()
  #
  # Objetivo:
  #   Generar versiones estandarizadas y escaladas del dataset
  #   con redondeo en variables numéricas
  #
  # Argumentos:
  #   datos      : data.frame
  #   decimales  : número de decimales (default = 4)
  #
  # Retorna:
  #   Lista con:
  #     - datos_estandarizados
  #     - datos_escalados
  #------------------------------------------------------------
  
  datos_est <- datos
  datos_esc <- datos
  
  #------------------------------------------
  # Identificar variables numéricas
  #------------------------------------------
  idx_num <- sapply(datos, is.numeric)
  
  #------------------------------------------
  # ESTANDARIZACIÓN (Z-score)
  #------------------------------------------
  datos_est[idx_num] <- lapply(datos[idx_num], function(x) {
    if (sd(x) == 0) {
      return(round(rep(0, length(x)), decimales))
    }
    round((x - mean(x)) / sd(x), decimales)
  })
  
  #------------------------------------------
  # ESCALADO (Min-Max)
  #------------------------------------------
  datos_esc[idx_num] <- lapply(datos[idx_num], function(x) {
    rango <- max(x) - min(x)
    if (rango == 0) {
      return(round(rep(0, length(x)), decimales))
    }
    round((x - min(x)) / rango, decimales)
  })
  
  #------------------------------------------
  # Mensaje informativo
  #------------------------------------------
  cat("\n============================\n")
  cat("Transformación de datos\n")
  cat("============================\n")
  cat("Variables numéricas transformadas:", sum(idx_num), "\n")
  cat("Variables no numéricas preservadas:", sum(!idx_num), "\n")
  cat("Decimales aplicados:", decimales, "\n")
  
  #------------------------------------------
  # Retorno
  #------------------------------------------
  return(list(
    datos_estandarizados = datos_est,
    datos_escalados = datos_esc
  ))
}

f_construir_modelo_lasso <- function(datos, variable_dependiente, ver_resumen = TRUE) {
  #------------------------------------------------------------
  # f_construir_modelo_lasso()
  #------------------------------------------------------------
  
  if (!variable_dependiente %in% names(datos)) {
    stop("La variable dependiente no existe en el dataset")
  }
  
  if (!require(glmnet, quietly = TRUE)) {
    stop("Debes instalar la librería glmnet")
  }
  
  datos_modelo <- datos
  
  #------------------------------------------
  # Conversión de variables
  #------------------------------------------
  idx_char <- sapply(datos_modelo, is.character)
  datos_modelo[idx_char] <- lapply(datos_modelo[idx_char], as.factor)
  
  idx_logical <- sapply(datos_modelo, is.logical)
  datos_modelo[idx_logical] <- lapply(datos_modelo[idx_logical], as.numeric)
  
  #------------------------------------------
  # Matriz X e y
  #------------------------------------------
  formula_modelo <- as.formula(paste(variable_dependiente, "~ ."))
  
  y <- datos_modelo[[variable_dependiente]]
  X <- model.matrix(formula_modelo, data = datos_modelo)[, -1]
  
  #------------------------------------------
  # Validación cruzada
  #------------------------------------------
  set.seed(2026)
  
  cv_lasso <- glmnet::cv.glmnet(
    X, y,
    alpha = 1,
    nfolds = 10
  )
  
  lambda_min <- cv_lasso$lambda.min
  lambda_1se <- cv_lasso$lambda.1se
  
  #------------------------------------------
  # Modelo final (con lambda.min)
  #------------------------------------------
  modelo <- glmnet::glmnet(
    X, y,
    alpha = 1,
    lambda = lambda_min
  )
  
  #------------------------------------------
  # Información
  #------------------------------------------
  cat("\n============================\n")
  cat("Modelo LASSO (Regresión L1)\n")
  cat("============================\n")
  cat("Variable dependiente:", variable_dependiente, "\n")
  cat("Observaciones:", nrow(datos_modelo), "\n")
  cat("Variables independientes:", ncol(X), "\n")
  cat("Lambda.min:", round(lambda_min, 6), "\n")
  cat("Lambda.1se:", round(lambda_1se, 6), "\n")
  
  #------------------------------------------
  # Resumen
  #------------------------------------------
  if (ver_resumen) {
    cat("\nCoeficientes (lambda.min):\n")
    print(coef(modelo, s = lambda_min))
    
    cat("\nCoeficientes (lambda.1se):\n")
    print(coef(modelo, s = lambda_1se))
  }
  
  #------------------------------------------
  # RETORNO COMPLETO
  #------------------------------------------
  return(list(
    modelo = modelo,
    cv = cv_lasso,
    lambda_min = lambda_min,
    lambda_1se = lambda_1se
  ))
}


f_construir_modelo_ridge <- function(datos, variable_dependiente, ver_resumen = TRUE) {
  #------------------------------------------------------------
  # f_construir_modelo_ridge()
  #
  # Objetivo:
  #   Construir modelo RIDGE (L2) con validación cruzada
  #
  # Nota:
  #   Se asume que los datos ya están estandarizados
  #------------------------------------------------------------
  
  #------------------------------------------
  # 1. Validación
  #------------------------------------------
  if (!variable_dependiente %in% names(datos)) {
    stop("La variable dependiente no existe en el dataset")
  }
  
  if (!require(glmnet, quietly = TRUE)) {
    stop("Debes instalar la librería glmnet")
  }
  
  datos_modelo <- datos
  
  #------------------------------------------
  # 2. Conversión de variables
  #------------------------------------------
  idx_char <- sapply(datos_modelo, is.character)
  datos_modelo[idx_char] <- lapply(datos_modelo[idx_char], as.factor)
  
  idx_logical <- sapply(datos_modelo, is.logical)
  datos_modelo[idx_logical] <- lapply(datos_modelo[idx_logical], as.numeric)
  
  #------------------------------------------
  # 3. Matriz X e y
  #------------------------------------------
  formula_modelo <- as.formula(paste(variable_dependiente, "~ ."))
  
  y <- datos_modelo[[variable_dependiente]]
  X <- model.matrix(formula_modelo, data = datos_modelo)[, -1]
  
  #------------------------------------------
  # 4. Validación cruzada (RIDGE)
  #------------------------------------------
  set.seed(2026)
  
  cv_ridge <- glmnet::cv.glmnet(
    X, y,
    alpha = 0,   # 🔥 RIDGE
    nfolds = 10
  )
  
  lambda_min <- cv_ridge$lambda.min
  lambda_1se <- cv_ridge$lambda.1se
  
  #------------------------------------------
  # 5. Modelo final
  #------------------------------------------
  modelo <- glmnet::glmnet(
    X, y,
    alpha = 0,
    lambda = lambda_min
  )
  
  #------------------------------------------
  # 6. Información
  #------------------------------------------
  cat("\n============================\n")
  cat("Modelo RIDGE (Regresión L2)\n")
  cat("============================\n")
  cat("Variable dependiente:", variable_dependiente, "\n")
  cat("Observaciones:", nrow(datos_modelo), "\n")
  cat("Variables independientes:", ncol(X), "\n")
  cat("Lambda.min:", round(lambda_min, 6), "\n")
  cat("Lambda.1se:", round(lambda_1se, 6), "\n")
  
  #------------------------------------------
  # 7. Resumen
  #------------------------------------------
  if (ver_resumen) {
    cat("\nCoeficientes (lambda.min):\n")
    print(coef(modelo, s = lambda_min))
    
    cat("\nCoeficientes (lambda.1se):\n")
    print(coef(modelo, s = lambda_1se))
  }
  
  #------------------------------------------
  # 8. Retorno
  #------------------------------------------
  return(list(
    modelo = modelo,
    cv = cv_ridge,
    lambda_min = lambda_min,
    lambda_1se = lambda_1se
  ))
}


f_multicolinealidad <- function(modelo) {
  #------------------------------------------------------------
  # f_multicolinealidad()
  #
  # Objetivo:
  #   Evaluar multicolinealidad mediante VIF
  #
  # Argumentos:
  #   modelo : objeto lm()
  #
  # Retorna:
  #   data.frame con VIF e interpretación
  #------------------------------------------------------------
  
  if (!inherits(modelo, "lm")) {
    stop("El objeto debe ser un modelo de tipo lm")
  }
  
  # Requiere car
  if (!require(car, quietly = TRUE)) {
    stop("Instala la librería 'car'")
  }
  
  vif_valores <- car::vif(modelo)
  
  #------------------------------------------
  # Interpretación
  #------------------------------------------
  interpretar_vif <- function(vif) {
    if (vif == 1) {
      return("Sin multicolinealidad")
    } else if (vif > 1 & vif < 5) {
      return("Baja (aceptable)")
    } else if (vif >= 5 & vif < 10) {
      return("Moderada (precaución)")
    } else {
      return("Alta (problema serio)")
    }
  }
  
  resultado <- data.frame(
    Variable = names(vif_valores),
    VIF = round(vif_valores, 2),
    Interpretacion = sapply(vif_valores, interpretar_vif)
  )
  
  cat("\n============================\n")
  cat("Diagnóstico de Multicolinealidad (VIF)\n")
  cat("============================\n")
  
  print(resultado)
  
  return(resultado)
}


f_linealidad <- function(modelo) {
  #------------------------------------------------------------
  # f_linealidad()
  #
  # Objetivo:
  #   Evaluar el supuesto de linealidad en RLM
  #
  # Argumentos:
  #   modelo : objeto lm()
  #
  # Retorna:
  #   Gráficos de diagnóstico
  #------------------------------------------------------------
  
  if (!inherits(modelo, "lm")) {
    stop("El objeto debe ser un modelo de tipo lm")
  }
  
  #------------------------------------------
  # Librerías
  #------------------------------------------
  if (!require(ggplot2, quietly = TRUE)) {
    stop("Instala la librería ggplot2")
  }
  
  datos <- modelo$model
  
  residuos <- resid(modelo)
  ajustados <- fitted(modelo)
  
  #------------------------------------------
  # 1. Residuos vs valores ajustados
  #------------------------------------------
  df_plot <- data.frame(ajustados, residuos)
  
  p1 <- ggplot(df_plot, aes(x = ajustados, y = residuos)) +
    geom_point(color = "blue") +
    geom_smooth(method = "loess", color = "red", se = FALSE) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    labs(title = "Residuos vs Valores Ajustados",
         x = "Valores ajustados",
         y = "Residuos") +
    theme_minimal()
  
  print(p1)
  
  #------------------------------------------
  # 2. Residuos vs cada variable independiente
  #------------------------------------------
  variables <- names(datos)[names(datos) != all.vars(formula(modelo))[1]]
  
  for (var in variables) {
    
    df_temp <- data.frame(
      x = datos[[var]],
      residuos = residuos
    )
    
    p <- ggplot(df_temp, aes(x = x, y = residuos)) +
      geom_point(color = "darkgreen") +
      geom_smooth(method = "lm", color = "red", se = FALSE) +
      geom_hline(yintercept = 0, linetype = "dashed") +
      labs(title = paste("Residuos vs", var),
           x = var,
           y = "Residuos") +
      theme_minimal()
    
    print(p)
  }
  
  #------------------------------------------
  # Interpretación
  #------------------------------------------
  cat("\n============================\n")
  cat("Diagnóstico de Linealidad\n")
  cat("============================\n")
  cat("Interpretación:\n")
  cat("- Los residuos deben distribuirse aleatoriamente alrededor de 0.\n")
  cat("- No deben observarse patrones curvos o tendencias.\n")
  cat("- Si hay curvatura, el modelo no es lineal.\n")
}


f_linealidad_grid <- function(modelo, ncol = 4) {
  #------------------------------------------------------------
  # f_linealidad_grid()
  #
  # Objetivo:
  #   Evaluar linealidad mostrando múltiples gráficos en grid
  #
  # Argumentos:
  #   modelo : objeto lm()
  #   ncol   : número de columnas del grid
  #
  # Requiere:
  #   ggplot2, patchwork
  #------------------------------------------------------------
  
  if (!inherits(modelo, "lm")) {
    stop("El objeto debe ser un modelo lm")
  }
  
  if (!require(ggplot2, quietly = TRUE)) {
    stop("Instala ggplot2")
  }
  
  if (!require(patchwork, quietly = TRUE)) {
    stop("Instala patchwork")
  }
  
  datos <- modelo$model
  residuos <- resid(modelo)
  ajustados <- fitted(modelo)
  
  lista_graficos <- list()
  
  #------------------------------------------
  # 1. Residuos vs ajustados
  #------------------------------------------
  df_base <- data.frame(ajustados, residuos)
  
  p0 <- ggplot(df_base, aes(x = ajustados, y = residuos)) +
    geom_point(color = "blue") +
    geom_smooth(method = "loess", color = "red", se = FALSE) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    labs(title = "Residuos vs Ajustados",
         x = "Ajustados",
         y = "Residuos") +
    theme_minimal()
  
  lista_graficos[[1]] <- p0
  
  #------------------------------------------
  # 2. Residuos vs variables
  #------------------------------------------
  y_var <- all.vars(formula(modelo))[1]
  variables <- setdiff(names(datos), y_var)
  
  for (i in seq_along(variables)) {
    
    var <- variables[i]
    
    df_temp <- data.frame(
      x = datos[[var]],
      residuos = residuos
    )
    
    p <- ggplot(df_temp, aes(x = x, y = residuos)) +
      geom_point(color = "darkgreen") +
      geom_smooth(method = "loess", color = "red", se = FALSE) +
      geom_hline(yintercept = 0, linetype = "dashed") +
      labs(title = paste("Residuos vs", var),
           x = var,
           y = "Residuos") +
      theme_minimal()
    
    lista_graficos[[i + 1]] <- p
  }
  
  #------------------------------------------
  # 3. Mostrar en grid
  #------------------------------------------
  grid <- wrap_plots(lista_graficos, ncol = ncol)
  
  print(grid)
  
  #------------------------------------------
  # Interpretación
  #------------------------------------------
  cat("\n============================\n")
  cat("Diagnóstico de Linealidad (Grid)\n")
  cat("============================\n")
  cat("- Los residuos deben verse aleatorios.\n")
  cat("- La línea roja (loess) debe ser aproximadamente horizontal.\n")
  cat("- Patrones curvos indican no linealidad.\n")
}


f_linealidad_test <- function(modelo) {
  #------------------------------------------------------------
  # Test de linealidad usando Ramsey RESET
  #------------------------------------------------------------
  
  if (!require(lmtest, quietly = TRUE)) {
    stop("Instala la librería lmtest")
  }
  
  resultado <- lmtest::resettest(modelo, power = 2:3, type = "fitted")
  
  cat("\n============================\n")
  cat("Test de Linealidad (Ramsey RESET)\n")
  cat("============================\n")
  
  print(resultado)
  
  # Interpretación
  p_valor <- resultado$p.value
  
  cat("\nInterpretación:\n")
  
  if (p_valor > 0.05) {
    cat("✔ No se rechaza H0 → El modelo es lineal (no hay evidencia de curvatura)\n")
  } else {
    cat("❌ Se rechaza H0 → Existe evidencia de no linealidad\n")
  }
  
  return(resultado)
}


f_homocedasticidad <- function(modelo) {
  
  if (!inherits(modelo, "lm")) {
    stop("Debe ser un modelo lm")
  }
  
  library(ggplot2)
  library(lmtest)
  
  residuos <- resid(modelo)
  ajustados <- fitted(modelo)
  
  df_plot <- na.omit(data.frame(ajustados, residuos))
  
  #------------------------------------------
  # Gráfico (estable)
  #------------------------------------------
  p <- ggplot(df_plot, aes(x = ajustados, y = residuos)) +
    geom_point(color = "blue") +
    geom_smooth(method = "lm", color = "red", se = FALSE) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    theme_minimal()
  
  print(p)
  
  #------------------------------------------
  # Breusch–Pagan
  #------------------------------------------
  bp <- lmtest::bptest(modelo)
  
  #------------------------------------------
  # White (manual, robusto)
  #------------------------------------------
  y_aux <- residuos^2
  x_aux <- ajustados
  
  df_aux <- data.frame(y_aux, x_aux)
  
  modelo_white <- lm(y_aux ~ x_aux + I(x_aux^2), data = df_aux)
  
  white_stat <- summary(modelo_white)$r.squared * nrow(df_aux)
  white_p <- pchisq(white_stat, df = 2, lower.tail = FALSE)
  
  #------------------------------------------
  # Resultados
  #------------------------------------------
  cat("\n============================\n")
  cat("Diagnóstico de Homocedasticidad\n")
  cat("============================\n")
  
  cat("\nBreusch–Pagan:\n")
  print(bp)
  
  cat("\nWhite Test (manual):\n")
  cat("Estadístico:", round(white_stat, 4), "\n")
  cat("p-value:", round(white_p, 4), "\n")
  
  #------------------------------------------
  # Interpretación
  #------------------------------------------
  cat("\nInterpretación:\n")
  
  if (bp$p.value > 0.05) {
    cat("✔ BP: No hay heterocedasticidad\n")
  } else {
    cat("❌ BP: Existe heterocedasticidad\n")
  }
  
  if (white_p > 0.05) {
    cat("✔ White: No hay heterocedasticidad\n")
  } else {
    cat("❌ White: Existe heterocedasticidad\n")
  }
  
  return(list(
    Breusch_Pagan = bp,
    White_stat = white_stat,
    White_p = white_p
  ))
}


f_normalidad <- function(modelo) {
  #------------------------------------------------------------
  # f_normalidad()
  #
  # Objetivo:
  #   Evaluar normalidad de residuos y devolver resultados
  #   en un data.frame listo para reporte
  #
  #------------------------------------------------------------
  
  if (!inherits(modelo, "lm")) {
    stop("El objeto debe ser un modelo de tipo lm")
  }
  
  # Librerías
  if (!require(nortest, quietly = TRUE)) stop("Instala 'nortest'")
  if (!require(ggplot2, quietly = TRUE)) stop("Instala 'ggplot2'")
  
  #------------------------------------------
  # 1. Residuos estandarizados
  #------------------------------------------
  residuos <- rstandard(modelo)
  residuos <- na.omit(residuos)
  
  #------------------------------------------
  # 2. Pruebas
  #------------------------------------------
  shapiro <- shapiro.test(residuos)
  ad <- nortest::ad.test(residuos)
  
  ks <- ks.test(
    residuos,
    "pnorm",
    mean = mean(residuos),
    sd = sd(residuos)
  )
  
  #------------------------------------------
  # 3. Interpretación
  #------------------------------------------
  interpretar <- function(p) {
    if (p > 0.05) {
      return("Normalidad")
    } else {
      return("No normalidad")
    }
  }
  
  #------------------------------------------
  # 4. Data frame resultado
  #------------------------------------------
  resultado <- data.frame(
    Prueba = c("Shapiro-Wilk", "Anderson-Darling", "Kolmogorov-Smirnov"),
    p_value = c(shapiro$p.value, ad$p.value, ks$p.value),
    Interpretacion = c(
      interpretar(shapiro$p.value),
      interpretar(ad$p.value),
      interpretar(ks$p.value)
    )
  )
  
  # Redondeo opcional
  resultado$p_value <- round(resultado$p_value, 4)
  
  #------------------------------------------
  # 5. Gráficos
  #------------------------------------------
  df_plot <- data.frame(residuos = residuos)
  
  # Histograma
  p1 <- ggplot(df_plot, aes(x = residuos)) +
    geom_histogram(aes(y = ..density..), bins = 10,
                   fill = "lightblue", color = "black") +
    stat_function(fun = dnorm,
                  args = list(mean = mean(residuos), sd = sd(residuos)),
                  color = "red", size = 1) +
    theme_minimal() +
    labs(title = "Histograma de residuos")
  
  print(p1)
  
  # QQ-Plot
  p2 <- ggplot(df_plot, aes(sample = residuos)) +
    stat_qq() +
    stat_qq_line(color = "red") +
    theme_minimal() +
    labs(title = "QQ-Plot de residuos")
  
  print(p2)
  
  #------------------------------------------
  # 6. Mostrar tabla
  #------------------------------------------
  cat("\n============================\n")
  cat("Diagnóstico de Normalidad\n")
  cat("============================\n")
  print(resultado)
  
  return(resultado)
}


f_independencia <- function(modelo, nombre_modelo = "Modelo") {
  #------------------------------------------------------------
  # Evaluar independencia de residuos (Durbin–Watson + gráfico)
  #------------------------------------------------------------
  
  if (!inherits(modelo, "lm")) {
    stop("El objeto debe ser un modelo lm")
  }
  
  library(ggplot2)
  library(lmtest)
  
  #------------------------------------------
  # Residuos
  #------------------------------------------
  residuos <- resid(modelo)
  residuos <- na.omit(residuos)
  
  df_plot <- data.frame(
    orden = 1:length(residuos),
    residuos = residuos
  )
  
  #------------------------------------------
  # Durbin–Watson
  #------------------------------------------
  dw_test <- lmtest::dwtest(modelo)
  dw <- as.numeric(dw_test$statistic)
  p_value <- dw_test$p.value
  
  #------------------------------------------
  # Interpretación
  #------------------------------------------
  interpretacion <- if (p_value > 0.05) {
    "Sin autocorrelación"
  } else if (dw < 2) {
    "Autocorrelación positiva"
  } else {
    "Autocorrelación negativa"
  }
  
  decision <- if (p_value > 0.05) {
    "✔ No se rechaza H0"
  } else {
    "❌ Se rechaza H0"
  }
  
  #------------------------------------------
  # Gráfico (TU FORMATO 🔥)
  #------------------------------------------
  g <- ggplot(df_plot, aes(x = orden, y = residuos)) +
    geom_line(color = "black", linewidth = 0.5) +
    geom_point(color = "blue", size = 1) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    labs(
      title = nombre_modelo,
      subtitle = paste(
        "DW =", round(dw, 3),
        "| p =", round(p_value, 3),
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
  
  print(g)
  
  #------------------------------------------
  # Data frame de salida
  #------------------------------------------
  resultado <- data.frame(
    Prueba = "Durbin-Watson",
    Estadistico = round(dw, 4),
    p_value = round(p_value, 4),
    Interpretacion = interpretacion,
    Decision = decision
  )
  
  cat("\n============================\n")
  cat("Diagnóstico de Independencia\n")
  cat("============================\n")
  print(resultado)
  
  return(resultado)
}


f_ecuacion_modelo <- function(modelo, redondeo = 4) {
  #------------------------------------------------------------
  # f_ecuacion_modelo()
  #
  # Objetivo:
  #   Mostrar la ecuación del modelo de regresión múltiple
  #
  # Argumentos:
  #   modelo    : objeto lm()
  #   redondeo  : número de decimales
  #
  #------------------------------------------------------------
  
  if (!inherits(modelo, "lm")) {
    stop("El objeto debe ser un modelo lm")
  }
  
  coefs <- coef(modelo)
  nombres <- names(coefs)
  
  # Redondear coeficientes
  coefs <- round(coefs, redondeo)
  
  #------------------------------------------
  # Intercepto
  #------------------------------------------
  ecuacion <- paste0("ŷ = ", coefs[1])
  
  #------------------------------------------
  # Construcción del resto
  #------------------------------------------
  for (i in 2:length(coefs)) {
    
    signo <- ifelse(coefs[i] >= 0, " + ", " - ")
    
    valor <- abs(coefs[i])
    
    variable <- nombres[i]
    
    termino <- paste0(signo, valor, "*", variable)
    
    ecuacion <- paste0(ecuacion, termino)
  }
  
  #------------------------------------------
  # Mostrar resultado
  #------------------------------------------
  cat("\n============================\n")
  cat("Ecuación del Modelo de Regresión\n")
  cat("============================\n\n")
  
  cat(ecuacion, "\n\n")
  
  return(ecuacion)
}


f_evaluacion <- function(modelos, datos_validacion_list, variable_dependiente, 
                         nombres = NULL, lambdas = NULL) {
  
  #------------------------------------------
  # Normalizar entrada
  #------------------------------------------
  if (!is.list(modelos)) {
    modelos <- list(modelos)
  }
  
  if (!is.list(datos_validacion_list)) {
    datos_validacion_list <- list(datos_validacion_list)
  }
  
  if (length(modelos) != length(datos_validacion_list)) {
    stop("El número de modelos y datasets debe coincidir")
  }
  
  if (is.null(nombres)) {
    nombres <- paste0("Modelo_", seq_along(modelos))
  }
  
  resultados <- data.frame()
  
  #------------------------------------------
  # Loop
  #------------------------------------------
  for (i in seq_along(modelos)) {
    
    modelo <- modelos[[i]]
    datos_validacion <- datos_validacion_list[[i]]
    
    y_real <- datos_validacion[[variable_dependiente]]
    
    #------------------------------------------
    # Modelo lm
    #------------------------------------------
    if (inherits(modelo, "lm")) {
      
      pred <- predict(modelo, newdata = datos_validacion)
      p <- length(coef(modelo)) - 1
      
    }
    
    #------------------------------------------
    # Modelo glmnet
    #------------------------------------------
    else if ("glmnet" %in% class(modelo)) {
      
      if (is.null(lambdas)) {
        stop("Debes proporcionar lambdas para modelos glmnet")
      }
      
      X_val <- model.matrix(
        as.formula(paste(variable_dependiente, "~ .")),
        data = datos_validacion
      )[, -1]
      
      pred <- predict(modelo, s = lambdas[i], newx = X_val)
      
      coefs <- coef(modelo, s = lambdas[i])
      p <- sum(coefs != 0) - 1
      
    }
    
    else {
      stop(paste("Tipo de modelo no soportado:", paste(class(modelo), collapse = ", ")))
    }
    
    #------------------------------------------
    # Métricas
    #------------------------------------------
    mse  <- mean((y_real - pred)^2)
    rmse <- sqrt(mse)
    mae  <- mean(abs(y_real - pred))
    
    #------------------------------------------
    # R²
    #------------------------------------------
    ss_res <- sum((y_real - pred)^2)
    ss_tot <- sum((y_real - mean(y_real))^2)
    
    r2 <- 1 - (ss_res / ss_tot)
    
    #------------------------------------------
    # R² ajustado
    #------------------------------------------
    n <- length(y_real)
    r2_adj <- 1 - (1 - r2) * ((n - 1) / (n - p - 1))
    
    #------------------------------------------
    # Guardar
    #------------------------------------------
    resultados <- rbind(resultados, data.frame(
      Modelo = nombres[i],
      R_square = round(r2, 4),
      R_square_ajustado = round(r2_adj, 4),
      MSE = round(mse, 4),
      RMSE = round(rmse, 4),
      MAE = round(mae, 4)
    ))
  }
  
  cat("\n============================\n")
  cat("Evaluación de Modelos\n")
  cat("============================\n")
  
  print(resultados)
  
  return(resultados)
}




f_validar_postulados_modelos <- function(modelos, datos_list, variable_dependiente, nombres = NULL) {
  
  if (!is.list(modelos)) modelos <- list(modelos)
  if (!is.list(datos_list)) datos_list <- list(datos_list)
  
  if (length(modelos) != length(datos_list)) {
    stop("Modelos y datasets deben tener la misma longitud")
  }
  
  if (is.null(nombres)) {
    nombres <- paste0("Modelo_", seq_along(modelos))
  }
  
  resultados <- data.frame()
  
  for (i in seq_along(modelos)) {
    
    modelo <- modelos[[i]]
    datos  <- datos_list[[i]]
    
    y_real <- datos[[variable_dependiente]]
    
    #------------------------------------------
    # Predicciones
    #------------------------------------------
    if (inherits(modelo, "lm")) {
      pred <- predict(modelo, newdata = datos)
      residuos <- y_real - pred
    } else if ("glmnet" %in% class(modelo)) {
      X <- model.matrix(
        as.formula(paste(variable_dependiente, "~ .")),
        data = datos
      )[, -1]
      
      pred <- predict(modelo, newx = X)
      residuos <- y_real - pred
    } else {
      stop("Tipo de modelo no soportado")
    }
    
    #------------------------------------------
    # 1. Multicolinealidad (solo lm)
    #------------------------------------------
    if (inherits(modelo, "lm")) {
      vif_val <- tryCatch({
        max(car::vif(modelo))
      }, error = function(e) NA)
    } else {
      vif_val <- NA
    }
    
    #------------------------------------------
    # 2. Linealidad (Ramsey RESET)
    #------------------------------------------
    p_lineal <- tryCatch({
      lmtest::resettest(modelo)$p.value
    }, error = function(e) NA)
    
    #------------------------------------------
    # 3. Homocedasticidad (Breusch-Pagan)
    #------------------------------------------
    p_homo <- tryCatch({
      lmtest::bptest(modelo)$p.value
    }, error = function(e) NA)
    
    #------------------------------------------
    # 4. Normalidad (Shapiro)
    #------------------------------------------
    p_norm <- tryCatch({
      shapiro.test(residuos)$p.value
    }, error = function(e) NA)
    
    #------------------------------------------
    # 5. Independencia (Durbin-Watson)
    #------------------------------------------
    p_dw <- tryCatch({
      lmtest::dwtest(modelo)$p.value
    }, error = function(e) NA)
    
    #------------------------------------------
    # Interpretación
    #------------------------------------------
    interpret <- function(p) {
      if (is.na(p)) return("NA")
      if (p > 0.05) return("Cumple")
      return("No cumple")
    }
    
    resultados <- rbind(resultados, data.frame(
      Modelo = nombres[i],
      VIF_Max = round(vif_val, 2),
      Linealidad = interpret(p_lineal),
      Homocedasticidad = interpret(p_homo),
      Normalidad = interpret(p_norm),
      Independencia = interpret(p_dw)
    ))
  }
  
  cat("\n============================\n")
  cat("Validación de Postulados\n")
  cat("============================\n")
  
  print(resultados)
  
  return(resultados)
}