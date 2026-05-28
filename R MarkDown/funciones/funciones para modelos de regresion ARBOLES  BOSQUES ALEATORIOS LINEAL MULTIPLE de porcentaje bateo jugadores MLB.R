# Funciones para implementar y evaluar modelos de regresión múltiple en R
# Se adecúan para que funcione Lasso y Ridge. No se usan en este caso
# Se adecúan para Múltiple Polinomial. No se usan en este caso
# No se incluye para modelos SVR varios kernels. No se usan en este caso
# Se adecúa para arboles de regresión y bosques aleatorios para regresión
# Rubén Pizarro Gurrola
# Mayo 2025

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

#========================================================
# FUNCIÓN
# f_crear_modelo_SVR_lineal()
#========================================================

f_crear_modelo_SVR_lineal <- function(
    
  datos_entrenamiento,
  
  variables_independientes,
  
  variable_dependiente,
  
  epsilon = 0.1,
  
  cost = 1,
  
  gamma = NULL
  
){
  
  #------------------------------------------------------
  # LIBRERÍA
  #------------------------------------------------------
  
  library(e1071)
  
  #------------------------------------------------------
  # FÓRMULA DINÁMICA
  #------------------------------------------------------
  
  formula_modelo <- as.formula(
    
    paste(
      
      variable_dependiente,
      
      "~",
      
      paste(
        
        variables_independientes,
        
        collapse = " + "
      )
    )
  )
  
  #------------------------------------------------------
  # SI gamma ES NULL
  # e1071 usa 1 / número_variables
  #------------------------------------------------------
  
  if(is.null(gamma)){
    
    gamma <- 1 / length(
      
      variables_independientes
    )
  }
  
  #------------------------------------------------------
  # CONSTRUIR MODELO
  #------------------------------------------------------
  
  modelo <- svm(
    
    formula_modelo,
    
    data = datos_entrenamiento,
    
    type = "eps-regression",
    
    kernel = "linear",
    
    epsilon = epsilon,
    
    cost = cost,
    
    gamma = gamma,
    
    scale = FALSE
  )
  
  #======================================================
  # RESUMEN
  #======================================================
  
  cat("\n")
  cat("=====================================\n")
  cat("MODELO SVR KERNEL LINEAL\n")
  cat("=====================================\n")
  
  #------------------------------------------------------
  # Fórmula
  #------------------------------------------------------
  cat("\nFÓRMULA DEL MODELO:\n")
  print(formula_modelo)
  
  #------------------------------------------------------
  # Gamma
  #------------------------------------------------------
  cat("\nGAMMA:\n")
  print(modelo$gamma)
  
  #------------------------------------------------------
  # Epsilon
  #------------------------------------------------------
  cat("\nEPSILON:\n")
  print(modelo$epsilon)
  
  #------------------------------------------------------
  # Cost
  #------------------------------------------------------
  cat("\nCOST:\n")
  print(modelo$cost)
  
  #------------------------------------------------------
  # Rho / Intercepto
  #------------------------------------------------------
  
  cat("\nRHO (INTERCEPTO):\n")
  print(modelo$rho)
  b <- -(modelo$rho)
  print("Valor de b:")
  print(b)
  
  #------------------------------------------------------
  # Número vectores soporte
  #------------------------------------------------------
  cat("\nNÚMERO VECTORES SOPORTE:\n")
  print(modelo$tot.nSV)
  
  #------------------------------------------------------
  # Retornar modelo
  #------------------------------------------------------
  
  return(modelo)
}


#========================================================
# FUNCIÓN
# f_crear_modelo_SVR_polinomial()
#========================================================

f_crear_modelo_SVR_polinomial <- function(
    datos_entrenamiento,
    variables_independientes,
    variable_dependiente,
    grado = 2,
    epsilon = 0.1,
    cost = 1,
    gamma = NULL
){
  
  #------------------------------------------------------
  # LIBRERÍA
  #------------------------------------------------------
  
  library(e1071)
  #------------------------------------------------------
  # FÓRMULA DINÁMICA
  #------------------------------------------------------
  
  formula_modelo <- as.formula(
    paste(
      variable_dependiente,
      "~",
      paste(
        variables_independientes,
        collapse = " + "
      )
    )
  )
  
  #------------------------------------------------------
  # SI gamma ES NULL
  # e1071 usa 1 / número_variables
  #------------------------------------------------------
  
  if(is.null(gamma)){
    gamma <- 1 / length(
      variables_independientes
    )
  }
  
  #------------------------------------------------------
  # CONSTRUIR MODELO
  #------------------------------------------------------
  
  modelo <- svm(
    formula_modelo,
    data = datos_entrenamiento,
    type = "eps-regression",
    kernel = "polynomial",
    degree = grado,
    epsilon = epsilon,
    cost = cost,
    gamma = gamma,
    scale = FALSE
  )
  
  #======================================================
  # RESUMEN
  #======================================================
  cat("\n")
  
  cat(
    paste0(
      "MODELO SVR KERNEL POLINOMIAL GRADO ",
      grado,
      "\n"
    )
  )
  
  #------------------------------------------------------
  # Fórmula
  #------------------------------------------------------
  cat("\nFÓRMULA DEL MODELO:\n")
  print(formula_modelo)
  
  #------------------------------------------------------
  # Kernel
  #------------------------------------------------------
  cat("\nKERNEL:\n")
  print(modelo$kernel)
  
  #------------------------------------------------------
  # Degree
  #------------------------------------------------------
  cat("\nGRADO POLINOMIAL:\n")
  print(modelo$degree)
  
  #------------------------------------------------------
  # Gamma
  #------------------------------------------------------
  cat("\nGAMMA:\n")
  print(modelo$gamma)
  
  #------------------------------------------------------
  # Epsilon
  #------------------------------------------------------
  cat("\nEPSILON:\n")
  print(modelo$epsilon)
  
  #------------------------------------------------------
  # Cost
  #------------------------------------------------------
  cat("\nCOST:\n")
  print(modelo$cost)
  
  #------------------------------------------------------
  # Rho
  #------------------------------------------------------
  cat("\nRHO:\n")
  print(modelo$rho)
  
  #------------------------------------------------------
  # Intercepto real
  #------------------------------------------------------
  b <- -(modelo$rho)
  cat("\nINTERCEPTO REAL b:\n")
  print(b)
  
  #------------------------------------------------------
  # Número vectores soporte
  #------------------------------------------------------
  cat("\nNÚMERO VECTORES SOPORTE:\n")
  print(modelo$tot.nSV)
  
  
  #------------------------------------------------------
  # Retornar modelo
  #------------------------------------------------------
  
  return(modelo)
}





#========================================================
# FUNCIÓN
# f_crear_modelo_SVR_radial()
#========================================================

f_crear_modelo_SVR_radial <- function(
    datos_entrenamiento,
    variables_independientes,
    variable_dependiente,
    epsilon = 0.1,
    cost = 1,
    gamma = NULL
    
){
  
  #------------------------------------------------------
  # LIBRERÍA
  #------------------------------------------------------
  
  library(e1071)
  
  #------------------------------------------------------
  # FÓRMULA DINÁMICA
  #------------------------------------------------------
  
  formula_modelo <- as.formula(
    paste(
      variable_dependiente,
      "~",
      paste(
        variables_independientes,
        collapse = " + "
      )
    )
  )
  
  #------------------------------------------------------
  # SI gamma ES NULL
  # e1071 usa 1 / número_variables
  #------------------------------------------------------
  if(is.null(gamma)){
    gamma <- 1 / length(
      variables_independientes
    )
  }
  
  #------------------------------------------------------
  # CONSTRUIR MODELO
  #------------------------------------------------------
  
  modelo <- svm(
    formula_modelo,
    data = datos_entrenamiento,
    type = "eps-regression",
    kernel = "radial",
    epsilon = epsilon,
    cost = cost,
    gamma = gamma,
    scale = FALSE
  )
  
  #======================================================
  # RESUMEN
  #======================================================
  
  cat("\n")
  
  cat("MODELO SVR KERNEL RADIAL\n")
  
  #------------------------------------------------------
  # Fórmula
  #------------------------------------------------------
  cat("\nFÓRMULA DEL MODELO:\n")
  print(formula_modelo)
  
  #------------------------------------------------------
  # Kernel
  #------------------------------------------------------
  cat("\nKERNEL:\n")
  print(modelo$kernel)
  
  #------------------------------------------------------
  # Gamma
  #------------------------------------------------------
  cat("\nGAMMA:\n")
  print(modelo$gamma)
  
  #------------------------------------------------------
  # Epsilon
  #------------------------------------------------------
  cat("\nEPSILON:\n")
  print(modelo$epsilon)
  
  #------------------------------------------------------
  # Cost
  #------------------------------------------------------
  cat("\nCOST:\n")
  print(modelo$cost)
  
  #------------------------------------------------------
  # Rho
  #------------------------------------------------------
  cat("\nRHO:\n")
  print(modelo$rho)
  
  #------------------------------------------------------
  # Intercepto real
  #------------------------------------------------------
  b <- -(modelo$rho)
  cat("\nINTERCEPTO REAL b:\n")
  print(b)
  
  #------------------------------------------------------
  # Número vectores soporte
  #------------------------------------------------------
  cat("\nNÚMERO VECTORES SOPORTE:\n")
  print(modelo$tot.nSV)
  
  #------------------------------------------------------
  # Retornar modelo
  #------------------------------------------------------
  
  return(modelo)
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



#========================================================
# FUNCIÓN
# f_crear_modelo_AR()
#
# OBJETIVO:
# Construir un modelo de Árbol de Regresión
#
# ARGUMENTOS:
# datos                 -> dataframe
# variable_dependiente  -> nombre de la variable Y
#
# VALOR DE RETORNO:
# Modelo de árbol de regresión
#
# REQUIERE:
# rpart
#
#========================================================

f_crear_modelo_AR <- function(
    
  datos,
  variable_dependiente){
  
  #------------------------------------------------------
  # LIBRERÍAS
  #------------------------------------------------------
  
  library(rpart)
  
  #------------------------------------------------------
  # VALIDACIÓN
  #------------------------------------------------------
  
  if(!is.data.frame(datos)){
    
    stop("El argumento datos debe ser un data.frame")
  }
  
  if(!(variable_dependiente %in% names(datos))){
    
    stop("La variable dependiente no existe en el dataframe")
  }
  
  #------------------------------------------------------
  # CONSTRUIR FÓRMULA
  #------------------------------------------------------
  
  formula_modelo <- as.formula(
    
    paste(variable_dependiente, "~ .")
  )
  
  #------------------------------------------------------
  # CONSTRUIR MODELO
  #------------------------------------------------------
  
  modelo <- rpart(
    formula = formula_modelo,
    data = datos,
    method = "anova"
  )
  
  #------------------------------------------------------
  # RETORNAR MODELO
  #------------------------------------------------------
  
  return(modelo)
}


#========================================================
# FUNCIÓN
# f_random_forest()
#
# OBJETIVO:
# Construir un modelo Random Forest
#
# ARGUMENTOS:
# datos                 -> dataframe
# variable_dependiente  -> variable dependiente
#
# VALOR DE RETORNO:
# Modelo Random Forest
#
# REQUIERE:
# randomForest
#
#========================================================

f_crear_modelo_RF <- function(
    datos,
    variable_dependiente, 
    arboles=100) {
  
  #------------------------------------------------------
  # LIBRERÍAS
  #------------------------------------------------------
  
  library(randomForest)
  
  #------------------------------------------------------
  # VALIDACIONES
  #------------------------------------------------------
  
  if(!is.data.frame(datos)){
    
    stop("El argumento datos debe ser un data.frame")
  }
  
  if(!(variable_dependiente %in% names(datos))){
    
    stop("La variable dependiente no existe en el dataframe")
  }
  
  #------------------------------------------------------
  # CONSTRUIR FÓRMULA
  #------------------------------------------------------
  
  formula_modelo <- as.formula(
    
    paste(variable_dependiente, "~ .")
  )
  
  #------------------------------------------------------
  # CONSTRUIR MODELO RANDOM FOREST
  #------------------------------------------------------
  
  modelo <- randomForest(
    
    formula = formula_modelo,
    
    data = datos,
    
    ntree = arboles,
    
    importance = TRUE
  )
  
  #------------------------------------------------------
  # RETORNAR MODELO
  #------------------------------------------------------
  
  return(modelo)
}

#========================================================
# FUNCIÓN
# f_visualizar_AR()
#
# OBJETIVO:
# Visualizar un Árbol de Regresión
#
# ARGUMENTOS:
# modelo -> modelo de árbol de regresión
#
# REQUIERE:
# rpart.plot
#
#========================================================

f_visualizar_AR <- function(modelo){
  
  #------------------------------------------------------
  # LIBRERÍAS
  #------------------------------------------------------
  
  library(rpart.plot)
  
  #------------------------------------------------------
  # VALIDACIÓN
  #------------------------------------------------------
  
  if(is.null(modelo)){
    
    stop("Debe proporcionar un modelo")
  }
  
  #------------------------------------------------------
  # VISUALIZACIÓN
  #------------------------------------------------------
  
  rpart.plot(
    
    modelo,
    
    type = 2,
    
    extra = 101,
    
    fallen.leaves = TRUE,
    
    box.palette = "Blues",
    
    branch.lty = 1,
    
    shadow.col = "gray",
    
    nn = TRUE,
    
    main = "Árbol de Regresión"
  )
}



#========================================================
# FUNCIÓN
# f_validar_postulados_modelos()
#
# ROBUSTA PARA:
# - lm
# - glmnet
# - svm e1071
# - rpart
# - randomForest
#
# OBJETIVO:
# Validar postulados clásicos de regresión
# y reconocer modelos no paramétricos
#
#========================================================

f_validar_postulados_modelos <- function(
    
  modelos,
  
  datos_list,
  
  variable_dependiente,
  
  nombres = NULL
  
){
  
  #------------------------------------------------------
  # LIBRERÍAS
  #------------------------------------------------------
  
  library(car)
  
  library(lmtest)
  
  #------------------------------------------------------
  # Convertir a lista
  #------------------------------------------------------
  
  if(!is.list(modelos)){
    
    modelos <- list(modelos)
  }
  
  if(!is.list(datos_list)){
    
    datos_list <- list(datos_list)
  }
  
  #------------------------------------------------------
  # Validación
  #------------------------------------------------------
  
  if(length(modelos) != length(datos_list)){
    
    stop(
      "Modelos y datasets deben tener la misma longitud"
    )
  }
  
  #------------------------------------------------------
  # Nombres
  #------------------------------------------------------
  
  if(is.null(nombres)){
    
    nombres <- paste0(
      
      "Modelo_",
      
      seq_along(modelos)
    )
  }
  
  #------------------------------------------------------
  # Resultados
  #------------------------------------------------------
  
  resultados <- data.frame()
  
  #======================================================
  # RECORRER MODELOS
  #======================================================
  
  for(i in seq_along(modelos)){
    
    modelo <- modelos[[i]]
    
    datos <- datos_list[[i]]
    
    y_real <- datos[[variable_dependiente]]
    
    #----------------------------------------------------
    # Variables independientes
    #----------------------------------------------------
    
    X <- datos[
      
      ,
      
      names(datos) != variable_dependiente,
      
      drop = FALSE
    ]
    
    #====================================================
    # PREDICCIONES
    #====================================================
    
    #----------------------------------------------------
    # MODELO LM
    #----------------------------------------------------
    
    if(inherits(modelo, "lm")){
      
      pred <- predict(
        
        modelo,
        
        newdata = datos
      )
      
      residuos <- y_real - pred
      
      tipo_modelo <- "lm"
    }
    
    #----------------------------------------------------
    # MODELO GLMNET
    #----------------------------------------------------
    
    else if("glmnet" %in% class(modelo)){
      
      X_matrix <- model.matrix(
        
        as.formula(
          
          paste(
            variable_dependiente,
            "~ ."
          )
        ),
        
        data = datos
      )[ , -1]
      
      pred <- predict(
        
        modelo,
        
        newx = X_matrix
      )
      
      residuos <- y_real - pred
      
      tipo_modelo <- "glmnet"
    }
    
    #----------------------------------------------------
    # MODELO SVM e1071
    #----------------------------------------------------
    
    else if("svm" %in% class(modelo)){
      
      pred <- predict(
        
        modelo,
        
        newdata = datos
      )
      
      residuos <- y_real - pred
      
      tipo_modelo <- "svm"
    }
    
    #----------------------------------------------------
    # MODELO ÁRBOL DE REGRESIÓN
    #----------------------------------------------------
    
    else if("rpart" %in% class(modelo)){
      
      pred <- predict(
        
        modelo,
        
        newdata = datos
      )
      
      residuos <- y_real - pred
      
      tipo_modelo <- "arbol_regresion"
    }
    
    #----------------------------------------------------
    # MODELO RANDOM FOREST
    #----------------------------------------------------
    
    else if("randomForest" %in% class(modelo)){
      
      pred <- predict(
        
        modelo,
        
        newdata = datos
      )
      
      residuos <- y_real - pred
      
      tipo_modelo <- "random_forest"
    }
    
    #----------------------------------------------------
    # NO SOPORTADO
    #----------------------------------------------------
    
    else{
      
      stop(
        paste(
          "Tipo de modelo no soportado:",
          class(modelo)
        )
      )
    }
    
    #====================================================
    # 1. MULTICOLINEALIDAD
    #====================================================
    
    if(tipo_modelo == "lm"){
      
      vif_val <- tryCatch({
        
        max(
          
          car::vif(modelo)
        )
        
      }, error = function(e) NA)
      
    } else {
      
      vif_val <- NA
    }
    
    #====================================================
    # 2. LINEALIDAD
    #====================================================
    
    if(tipo_modelo == "lm"){
      
      p_lineal <- tryCatch({
        
        lmtest::resettest(
          
          modelo
        )$p.value
        
      }, error = function(e) NA)
      
    } else {
      
      p_lineal <- NA
    }
    
    #====================================================
    # 3. HOMOCEDASTICIDAD
    #====================================================
    
    if(tipo_modelo == "lm"){
      
      p_homo <- tryCatch({
        
        lmtest::bptest(
          
          modelo
        )$p.value
        
      }, error = function(e) NA)
      
    } else {
      
      p_homo <- NA
    }
    
    #====================================================
    # 4. NORMALIDAD
    #====================================================
    
    if(tipo_modelo == "lm"){
      
      p_norm <- tryCatch({
        
        shapiro.test(
          
          residuos
        )$p.value
        
      }, error = function(e) NA)
      
    } else {
      
      p_norm <- NA
    }
    
    #====================================================
    # 5. INDEPENDENCIA
    #====================================================
    
    if(tipo_modelo == "lm"){
      
      p_dw <- tryCatch({
        
        lmtest::dwtest(
          
          modelo
        )$p.value
        
      }, error = function(e) NA)
      
    } else {
      
      p_dw <- NA
    }
    
    #====================================================
    # INTERPRETACIÓN
    #====================================================
    
    interpretar <- function(p){
      
      if(is.na(p)){
        
        return("NA")
      }
      
      if(p > 0.05){
        
        return("Cumple")
      }
      
      return("No cumple")
    }
    
    #====================================================
    # RESULTADO
    #====================================================
    
    resultados <- rbind(
      
      resultados,
      
      data.frame(
        
        Modelo = nombres[i],
        
        Tipo = tipo_modelo,
        
        VIF_Max = round(
          
          vif_val,
          
          4
        ),
        
        Linealidad = interpretar(
          
          p_lineal
        ),
        
        Homocedasticidad = interpretar(
          
          p_homo
        ),
        
        Normalidad = interpretar(
          
          p_norm
        ),
        
        Independencia = interpretar(
          
          p_dw
        )
      )
    )
  }
  
  #======================================================
  # MOSTRAR
  #======================================================
  
  cat("\n")
  
  cat("============================\n")
  
  cat("Validación de Postulados\n")
  
  cat("============================\n")
  
  print(resultados)
  
  #------------------------------------------------------
  # RETORNO
  #------------------------------------------------------
  
  return(resultados)
}

#========================================================
# FUNCIÓN
# f_evaluacion()
#
# ROBUSTA PARA:
# - lm
# - glmnet
# - svm e1071
# - rpart
# - randomForest
#
# OBJETIVO:
# Evaluar modelos de regresión y
# aprendizaje automático
#
#========================================================

f_evaluacion <- function(
    
  modelos,
  
  datos_validacion_list,
  
  variable_dependiente,
  
  nombres = NULL,
  
  lambdas = NULL
  
){
  
  #------------------------------------------------------
  # Convertir a lista
  #------------------------------------------------------
  
  if(!is.list(modelos)){
    
    modelos <- list(modelos)
  }
  
  if(!is.list(datos_validacion_list)){
    
    datos_validacion_list <- list(
      
      datos_validacion_list
    )
  }
  
  #------------------------------------------------------
  # Validar longitud
  #------------------------------------------------------
  
  if(length(modelos) != length(datos_validacion_list)){
    
    stop(
      "El número de modelos y datasets debe coincidir"
    )
  }
  
  #------------------------------------------------------
  # Nombres
  #------------------------------------------------------
  
  if(is.null(nombres)){
    
    nombres <- paste0(
      
      "Modelo_",
      
      seq_along(modelos)
    )
  }
  
  #------------------------------------------------------
  # Tabla resultados
  #------------------------------------------------------
  
  resultados <- data.frame()
  
  #======================================================
  # RECORRER MODELOS
  #======================================================
  
  for(i in seq_along(modelos)){
    
    modelo <- modelos[[i]]
    
    datos_validacion <- datos_validacion_list[[i]]
    
    y_real <- datos_validacion[[variable_dependiente]]
    
    #====================================================
    # MODELOS LM
    #====================================================
    
    if(inherits(modelo, "lm")){
      
      #----------------------------------------------
      # Predicciones robustas
      #----------------------------------------------
      
      X_val <- model.matrix(
        
        formula(modelo),
        
        datos_validacion
      )
      
      pred <- X_val %*% coef(modelo)
      
      pred <- as.vector(pred)
      
      #----------------------------------------------
      # Parámetros
      #----------------------------------------------
      
      p <- length(
        
        coef(modelo)
        
      ) - 1
      
      tipo_modelo <- "lm"
    }
    
    #====================================================
    # MODELOS GLMNET
    #====================================================
    
    else if("glmnet" %in% class(modelo)){
      
      #----------------------------------------------
      # Lambda requerido
      #----------------------------------------------
      
      if(is.null(lambdas) || is.na(lambdas[i])){
        
        stop(
          
          paste(
            "Falta lambda para el modelo:",
            nombres[i]
          )
        )
      }
      
      #----------------------------------------------
      # Matriz
      #----------------------------------------------
      
      X_val <- model.matrix(
        
        as.formula(
          
          paste(
            variable_dependiente,
            "~ ."
          )
        ),
        
        data = datos_validacion
        
      )[ , -1]
      
      #----------------------------------------------
      # Predicción
      #----------------------------------------------
      
      pred <- predict(
        
        modelo,
        
        s = lambdas[i],
        
        newx = X_val
      )
      
      pred <- as.vector(pred)
      
      #----------------------------------------------
      # Parámetros
      #----------------------------------------------
      
      coefs <- coef(
        
        modelo,
        
        s = lambdas[i]
      )
      
      p <- sum(
        
        coefs != 0
        
      ) - 1
      
      tipo_modelo <- "glmnet"
    }
    
    #====================================================
    # MODELOS SVM / SVR
    #====================================================
    
    else if("svm" %in% class(modelo)){
      
      #----------------------------------------------
      # Predicción
      #----------------------------------------------
      
      pred <- predict(
        
        modelo,
        
        newdata = datos_validacion
      )
      
      pred <- as.vector(pred)
      
      #----------------------------------------------
      # Número aproximado parámetros
      # vectores soporte
      #----------------------------------------------
      
      p <- modelo$tot.nSV
      
      tipo_modelo <- "svm"
    }
    
    #====================================================
    # MODELOS ÁRBOLES DE REGRESIÓN
    #====================================================
    
    else if("rpart" %in% class(modelo)){
      
      #----------------------------------------------
      # Predicción
      #----------------------------------------------
      
      pred <- predict(
        
        modelo,
        
        newdata = datos_validacion
      )
      
      pred <- as.vector(pred)
      
      #----------------------------------------------
      # Número aproximado parámetros
      # regiones terminales
      #----------------------------------------------
      
      p <- length(
        
        unique(pred)
      )
      
      tipo_modelo <- "arbol_regresion"
    }
    
    #====================================================
    # MODELOS RANDOM FOREST
    #====================================================
    
    else if("randomForest" %in% class(modelo)){
      
      #----------------------------------------------
      # Predicción
      #----------------------------------------------
      
      pred <- predict(
        
        modelo,
        
        newdata = datos_validacion
      )
      
      pred <- as.vector(pred)
      
      #----------------------------------------------
      # Número aproximado parámetros
      # árboles construidos
      #----------------------------------------------
      
      p <- modelo$ntree
      
      tipo_modelo <- "random_forest"
    }
    
    #====================================================
    # NO SOPORTADO
    #====================================================
    
    else{
      
      stop(
        
        paste(
          
          "Tipo de modelo no soportado:",
          
          paste(
            class(modelo),
            collapse = ", "
          )
        )
      )
    }
    
    #====================================================
    # MÉTRICAS
    #====================================================
    
    mse <- mean(
      
      (y_real - pred)^2
    )
    
    rmse <- sqrt(mse)
    
    mae <- mean(
      
      abs(y_real - pred)
    )
    
    #====================================================
    # R SQUARE
    #====================================================
    
    ss_res <- sum(
      
      (y_real - pred)^2
    )
    
    ss_tot <- sum(
      
      (y_real - mean(y_real))^2
    )
    
    r2 <- 1 - (
      
      ss_res / ss_tot
    )
    
    #====================================================
    # R SQUARE AJUSTADO
    #====================================================
    
    n <- length(y_real)
    
    #----------------------------------------------------
    # Evitar división inválida
    #----------------------------------------------------
    
    if((n - p - 1) <= 0){
      
      r2_adj <- NA
      
    } else {
      
      r2_adj <- 1 - (
        
        (1 - r2)
        
        *
          
          (
            (n - 1)
            /
              (n - p - 1)
          )
      )
    }
    
    #====================================================
    # RESULTADO
    #====================================================
    
    resultados <- rbind(
      
      resultados,
      
      data.frame(
        
        Modelo = nombres[i],
        
        Tipo = tipo_modelo,
        
        Parametros = p,
        
        R_square = round(
          
          r2,
          
          4
        ),
        
        R_square_ajustado = round(
          
          r2_adj,
          
          4
        ),
        
        MSE = round(
          
          mse,
          
          4
        ),
        
        RMSE = round(
          
          rmse,
          
          4
        ),
        
        MAE = round(
          
          mae,
          
          4
        )
      )
    )
  }
  
  #======================================================
  # MOSTRAR
  #======================================================
  
  cat("\n")
  
  cat("============================\n")
  
  cat("EVALUACIÓN DE MODELOS\n")
  
  cat("============================\n")
  
  print(resultados)
  
  #------------------------------------------------------
  # RETORNO
  #------------------------------------------------------
  
  return(resultados)
}


#========================================================
# FUNCIÓN
# f_visualizar_RMSE()
# ACEPTA:
# - 1 o varios modelos
# - genera GRID AUTOMÁTICO
#========================================================

f_visualizar_RMSE <- function(
    
  modelos,
  datos,
  variable_dependiente,
  nombres_modelos = NULL,
  ncol = 3) {
  
  #------------------------------------------------------
  # LIBRERÍAS
  #------------------------------------------------------
  
  library(ggplot2)
  
  library(patchwork)
  
  #------------------------------------------------------
  # CONVERTIR A LISTA
  #------------------------------------------------------
  
  if(!is.list(modelos)){
    
    modelos <- list(modelos)
  }
  
  #------------------------------------------------------
  # NOMBRES
  #------------------------------------------------------
  
  if(is.null(nombres_modelos)){
    
    nombres_modelos <- paste0(
      
      "Modelo_",
      
      seq_along(modelos)
    )
  }
  
  #------------------------------------------------------
  # LISTA GRÁFICAS
  #------------------------------------------------------
  
  lista_graficas <- list()
  
  #======================================================
  # RECORRER MODELOS
  #======================================================
  
  for(i in seq_along(modelos)){
    
    modelo <- modelos[[i]]
    
    nombre_modelo <- nombres_modelos[i]
    
    #----------------------------------------------------
    # DATOS REALES
    #----------------------------------------------------
    
    y_real <- datos[[variable_dependiente]]
    
    #----------------------------------------------------
    # PREDICCIONES
    #----------------------------------------------------
    
    pred <- predict(
      
      modelo,
      
      newdata = datos
    )
    
    pred <- as.vector(pred)
    
    #----------------------------------------------------
    # RMSE
    #----------------------------------------------------
    
    rmse <- sqrt(mean((y_real - pred)^2))
    
    ss_res <- sum((y_real - pred)^2)
    ss_tot <- sum((y_real - mean(y_real))^2)
    r2 <- 1 - (ss_res / ss_tot)
    
    #----------------------------------------------------
    # R SQUARE
    #----------------------------------------------------
    
    #----------------------------------------------------
    # DATAFRAME
    #----------------------------------------------------
    
    df_plot <- data.frame(
      
      Observacion = 1:length(y_real),
      
      Real = y_real,
      
      Prediccion = pred
    )
    
    #====================================================
    # GRÁFICA
    #====================================================
    
    g <- ggplot(
      
      df_plot,
      
      aes(
        x = Observacion
      )
    ) +
      
      #--------------------------------------------------
    # REALES
    #--------------------------------------------------
    
    geom_line(
      
      aes(
        y = Real,
        color = "Valores reales"
      ),
      
      linewidth = 0.5
    ) +
      
      geom_point(
        
        aes(
          y = Real,
          color = "Valores reales"
        ),
        
        size = 1
      ) +
      
      #--------------------------------------------------
    # PREDICCIONES
    #--------------------------------------------------
    
    geom_line(
      
      aes(y = Prediccion, color = "Predicciones"), linewidth = 0.5) +
      
      geom_point(aes(y = Prediccion, color = "Predicciones"), size = 1) +
      
      #--------------------------------------------------
    # COLORES
    #--------------------------------------------------
    
    scale_color_manual(values = c("Valores reales" = "yellow", "Predicciones" = "blue") ) +
      
      #--------------------------------------------------
    # TÍTULOS
    #--------------------------------------------------
    
    labs(title = nombre_modelo, subtitle = paste("RMSE =", round(rmse, 4), 
                                                 "| R² =",round(r2,4)),
         x = "Observación",
         y = variable_dependiente,
         color = "Leyenda" ) +
      
      theme_minimal(
        
        base_size = 12
      )
    
    #----------------------------------------------------
    # GUARDAR
    #----------------------------------------------------
    
    lista_graficas[[i]] <- g
  }
  
  #======================================================
  # GRID FINAL
  #======================================================
  
  grid_final <- wrap_plots(
    
    lista_graficas,
    
    ncol = ncol
  )
  
  #------------------------------------------------------
  # MOSTRAR
  #------------------------------------------------------
  
  print(grid_final)
  
  #------------------------------------------------------
  # RETORNO
  #------------------------------------------------------
  
  return(grid_final)
}
