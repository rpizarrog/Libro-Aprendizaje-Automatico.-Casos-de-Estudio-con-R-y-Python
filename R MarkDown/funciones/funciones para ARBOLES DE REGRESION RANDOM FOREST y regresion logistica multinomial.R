# Por. Rubén Pizarro Gurrola
# Junio 2026
# Funcione para arboles de regresión y random florest
# Funciones para regresión logística multinomial

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



#=========================================================
# FUNCIÓN
# f_construir_arbol_clasificacion()
#=========================================================

f_construir_arbol_clasificacion <- function(
    datos,
    variable_dependiente,
    criterio = "gini",
    cp = 0.01,
    maxdepth = 30,
    minsplit = 20,
    minbucket = 7){
  
  #-------------------------------------------------------
  # LIBRERÍAS
  #-------------------------------------------------------
  
  library(rpart)
  
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
  # VARIABLE DEPENDIENTE
  #-------------------------------------------------------
  
  datos[[variable_dependiente]] <-
    as.factor(
      datos[[variable_dependiente]]
    )
  
  n_clases <-
    nlevels(
      datos[[variable_dependiente]]
    )
  
  #-------------------------------------------------------
  # FÓRMULA
  #-------------------------------------------------------
  
  formula_modelo <-
    as.formula(
      paste(
        variable_dependiente,
        "~ ."
      )
    )
  
  #-------------------------------------------------------
  # CRITERIO
  #-------------------------------------------------------
  
  criterio <- tolower(criterio)
  
  if(!criterio %in% c("gini","information")){
    
    stop(
      "criterio debe ser 'gini' o 'information'"
    )
  }
  
  #-------------------------------------------------------
  # MODELO
  #-------------------------------------------------------
  
  modelo <- rpart(
    
    formula = formula_modelo,
    
    data = datos,
    
    method = "class",
    
    parms = list(
      split = criterio
    ),
    
    control = rpart.control(
      
      cp = cp,
      
      maxdepth = maxdepth,
      
      minsplit = minsplit,
      
      minbucket = minbucket
      
    )
  )
  
  #-------------------------------------------------------
  # METADATOS
  #-------------------------------------------------------
  
  modelo$variable_dependiente <- variable_dependiente
  
  modelo$tipo_modelo <- "Arbol_Clasificacion"
  
  modelo$criterio <- criterio
  
  modelo$n_clases <- n_clases
  
  modelo$clases <-
    levels(
      datos[[variable_dependiente]]
    )
  
  modelo$frecuencias_clases <-
    table(
      datos[[variable_dependiente]]
    )
  
  #-------------------------------------------------------
  # RESUMEN
  #-------------------------------------------------------
  
  cat("\n")
  cat("====================================\n")
  cat(" ÁRBOL DE CLASIFICACIÓN\n")
  cat("====================================\n")
  
  cat("Variable objetivo :",
      variable_dependiente,
      "\n")
  
  cat("Criterio          :",
      criterio,
      "\n")
  
  cat("Número clases     :",
      n_clases,
      "\n")
  
  cat("Observaciones     :",
      nrow(datos),
      "\n")
  
  cat("\nFrecuencia de clases:\n")
  
  print(
    modelo$frecuencias_clases
  )
  
  cat("====================================\n")
  
  return(modelo)
}



#=========================================================
# FUNCIÓN
# f_construir_random_forest()
#=========================================================

f_construir_random_forest <- function(
    datos,
    variable_dependiente,
    ntree = 500,
    mtry = NULL,
    nodesize = 1,
    importance = TRUE,
    semilla = 123){
  
  #-------------------------------------------------------
  # LIBRERÍA
  #-------------------------------------------------------
  
  library(randomForest)
  
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
  # VARIABLE DEPENDIENTE
  #-------------------------------------------------------
  
  datos[[variable_dependiente]] <-
    as.factor(
      datos[[variable_dependiente]]
    )
  
  clases <-
    levels(
      datos[[variable_dependiente]]
    )
  
  n_clases <-
    length(clases)
  
  #-------------------------------------------------------
  # NÚMERO DE VARIABLES
  #-------------------------------------------------------
  
  p <- ncol(datos) - 1
  
  #-------------------------------------------------------
  # mtry AUTOMÁTICO
  #-------------------------------------------------------
  
  if(is.null(mtry)){
    
    mtry <- floor(
      sqrt(p)
    )
    
  }
  
  #-------------------------------------------------------
  # FÓRMULA
  #-------------------------------------------------------
  
  formula_modelo <-
    as.formula(
      paste(
        variable_dependiente,
        "~ ."
      )
    )
  
  #-------------------------------------------------------
  # SEMILLA
  #-------------------------------------------------------
  
  set.seed(semilla)
  
  #-------------------------------------------------------
  # MODELO
  #-------------------------------------------------------
  
  modelo <- randomForest(
    
    formula = formula_modelo,
    
    data = datos,
    
    ntree = ntree,
    
    mtry = mtry,
    
    nodesize = nodesize,
    
    importance = importance
  )
  
  #-------------------------------------------------------
  # METADATOS
  #-------------------------------------------------------
  
  modelo$variable_dependiente <-
    variable_dependiente
  
  modelo$tipo_modelo <-
    "Random_Forest"
  
  modelo$n_clases <-
    n_clases
  
  modelo$clases <-
    clases
  
  modelo$frecuencias_clases <-
    table(
      datos[[variable_dependiente]]
    )
  
  #-------------------------------------------------------
  # RESUMEN
  #-------------------------------------------------------
  
  cat("\n")
  
  cat("====================================\n")
  cat(" RANDOM FOREST\n")
  cat("====================================\n")
  
  cat(
    "Variable objetivo :",
    variable_dependiente,
    "\n"
  )
  
  cat(
    "Número clases     :",
    n_clases,
    "\n"
  )
  
  cat(
    "Variables predictoras :",
    p,
    "\n"
  )
  
  cat(
    "Número árboles    :",
    ntree,
    "\n"
  )
  
  cat(
    "mtry              :",
    mtry,
    "\n"
  )
  
  cat(
    "nodesize          :",
    nodesize,
    "\n"
  )
  
  cat(
    "Observaciones     :",
    nrow(datos),
    "\n"
  )
  
  cat("\nFrecuencia de clases:\n")
  
  print(
    modelo$frecuencias_clases
  )
  
  cat("====================================\n")
  
  return(modelo)
  
}


#=========================================================
# FUNCIÓN
# f_crear_modelo_regresion_logistica()
#=========================================================

f_crear_modelo_regresion_logistica <- function(
    datos,
    variable_dependiente,
    tipo = "binomial",
    balanceo = "ninguno",
    semilla = 123){
  
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
  
  set.seed(semilla)
  
  #-------------------------------------------------------
  # VARIABLE DEPENDIENTE
  #-------------------------------------------------------
  
  datos[[variable_dependiente]] <-
    as.factor(
      datos[[variable_dependiente]]
    )
  
  n_clases <-
    nlevels(
      datos[[variable_dependiente]]
    )
  
  #-------------------------------------------------------
  # BALANCEO
  #-------------------------------------------------------
  
  pesos <- NULL
  
  if(balanceo == "undersampling"){
    
    frecuencias <- table(
      datos[[variable_dependiente]]
    )
    
    n_min <- min(frecuencias)
    
    clases <- names(frecuencias)
    
    datos <- do.call(
      rbind,
      lapply(
        clases,
        function(clase){
          
          datos_clase <-
            datos[
              datos[[variable_dependiente]] == clase,
            ]
          
          if(nrow(datos_clase) > n_min){
            
            datos_clase <-
              datos_clase[
                sample(
                  nrow(datos_clase),
                  n_min
                ),
              ]
            
          }
          
          datos_clase
          
        }
      )
    )
    
  }
  
  #-------------------------------------------------------
  
  else if(balanceo == "oversampling"){
    
    frecuencias <- table(
      datos[[variable_dependiente]]
    )
    
    n_max <- max(frecuencias)
    
    clases <- names(frecuencias)
    
    datos <- do.call(
      rbind,
      lapply(
        clases,
        function(clase){
          
          datos_clase <-
            datos[
              datos[[variable_dependiente]] == clase,
            ]
          
          if(nrow(datos_clase) < n_max){
            
            datos_clase <-
              datos_clase[
                sample(
                  nrow(datos_clase),
                  n_max,
                  replace = TRUE
                ),
              ]
            
          }
          
          datos_clase
          
        }
      )
    )
    
  }
  
  #-------------------------------------------------------
  
  else if(balanceo == "ponderacion"){
    
    frecuencias <- table(
      datos[[variable_dependiente]]
    )
    
    frecuencia_max <- max(frecuencias)
    
    pesos <- sapply(
      
      datos[[variable_dependiente]],
      
      function(x){
        
        frecuencia_max /
          frecuencias[x]
        
      }
      
    )
    
  }
  
  #-------------------------------------------------------
  
  else if(balanceo == "SMOTE"){
    
    if(!requireNamespace(
      "smotefamily",
      quietly = TRUE
    )){
      
      stop(
        "Instale el paquete smotefamily."
      )
      
    }
    
    y <- datos[[variable_dependiente]]
    
    X <- datos[
      ,
      names(datos) != variable_dependiente
    ]
    
    X <- data.frame(
      lapply(
        X,
        function(x){
          
          if(is.factor(x))
            as.numeric(x)
          else
            x
          
        }
      )
    )
    
    sm <- smotefamily::SMOTE(
      X = X,
      target = y,
      K = 5
    )
    
    datos <- data.frame(
      sm$data
    )
    
    colnames(datos)[ncol(datos)] <-
      variable_dependiente
    
    datos[[variable_dependiente]] <-
      as.factor(
        datos[[variable_dependiente]]
      )
    
  }
  
  #-------------------------------------------------------
  
  else if(balanceo != "ninguno"){
    
    stop(
      "Balanceo inválido."
    )
    
  }
  
  #-------------------------------------------------------
  # FÓRMULA
  #-------------------------------------------------------
  
  formula_modelo <-
    as.formula(
      paste(
        variable_dependiente,
        "~ ."
      )
    )
  
  #-------------------------------------------------------
  # BINOMIAL
  #-------------------------------------------------------
  
  if(tipo == "binomial"){
    
    if(n_clases != 2){
      
      stop(
        paste(
          "La regresión logística binomial requiere exactamente 2 clases."
        )
      )
      
    }
    
    if(is.null(pesos)){
      
      modelo <- glm(
        formula_modelo,
        data = datos,
        family = binomial("logit")
      )
      
    }else{
      
      modelo <- glm(
        formula_modelo,
        data = datos,
        family = binomial("logit"),
        weights = pesos
      )
      
    }
    
  }
  
  #-------------------------------------------------------
  # MULTINOMIAL
  #-------------------------------------------------------
  
  else if(tipo == "multinomial"){
    
    if(!requireNamespace(
      "nnet",
      quietly = TRUE
    )){
      
      stop(
        "Instale el paquete nnet."
      )
      
    }
    
    if(is.null(pesos)){
      
      modelo <- nnet::multinom(
        formula_modelo,
        data = datos,
        trace = FALSE
      )
      
    }else{
      
      modelo <- nnet::multinom(
        formula_modelo,
        data = datos,
        weights = pesos,
        trace = FALSE
      )
      
    }
    
  }
  
  else{
    
    stop(
      "tipo debe ser 'binomial' o 'multinomial'"
    )
    
  }
  
  #-------------------------------------------------------
  # METADATOS
  #-------------------------------------------------------
  
  modelo$variable_dependiente <- variable_dependiente
  
  modelo$tipo_modelo <- tipo
  
  modelo$balanceo <- balanceo
  
  modelo$n_clases <- n_clases
  
  #-------------------------------------------------------
  # RESUMEN
  #-------------------------------------------------------
  
  cat("\n====================================\n")
  cat(" REGRESIÓN LOGÍSTICA\n")
  cat("====================================\n")
  cat("Tipo              :", tipo, "\n")
  cat("Balanceo          :", balanceo, "\n")
  cat("Variable objetivo :", variable_dependiente, "\n")
  cat("Clases            :", n_clases, "\n")
  cat("Observaciones     :", nrow(datos), "\n")
  cat("====================================\n")
  
  return(modelo)
  
}


#=========================================================
# FUNCIÓN
# f_predicciones()
#=========================================================

f_predicciones <- function(
    modelo,
    datos_validacion,
    variable_dependiente){
  
  #-------------------------------------------------------
  # VALIDACIONES
  #-------------------------------------------------------
  
  if(!variable_dependiente %in% names(datos_validacion)){
    
    stop(
      paste(
        "La variable",
        variable_dependiente,
        "no existe en los datos."
      )
    )
    
  }
  
  #-------------------------------------------------------
  # VARIABLE REAL
  #-------------------------------------------------------
  
  y_real <- as.factor(
    datos_validacion[[variable_dependiente]]
  )
  
  clases <- levels(y_real)
  
  #-------------------------------------------------------
  # REGRESIÓN LOGÍSTICA BINOMIAL
  #-------------------------------------------------------
  
  if(inherits(modelo,"glm")){
    
    prob <- predict(
      modelo,
      newdata = datos_validacion,
      type = "response"
    )
    
    pred <- factor(
      ifelse(
        prob >= 0.50,
        clases[2],
        clases[1]
      ),
      levels = clases
    )
    
    probabilidad <- prob
    
  }
  
  #-------------------------------------------------------
  # REGRESIÓN LOGÍSTICA MULTINOMIAL
  #-------------------------------------------------------
  
  else if(inherits(modelo,"multinom")){
    
    probs <- predict(
      modelo,
      newdata = datos_validacion,
      type = "probs"
    )
    
    pred <- predict(
      modelo,
      newdata = datos_validacion,
      type = "class"
    )
    
    probabilidad <- apply(
      probs,
      1,
      max
    )
    
  }
  
  #-------------------------------------------------------
  # ÁRBOL DE CLASIFICACIÓN
  #-------------------------------------------------------
  
  else if(inherits(modelo,"rpart")){
    
    probs <- predict(
      modelo,
      newdata = datos_validacion,
      type = "prob"
    )
    
    pred <- predict(
      modelo,
      newdata = datos_validacion,
      type = "class"
    )
    
    probabilidad <- apply(
      probs,
      1,
      max
    )
    
  }
  
  #-------------------------------------------------------
  # RANDOM FOREST
  #-------------------------------------------------------
  
  else if(inherits(modelo,"randomForest")){
    
    probs <- predict(
      modelo,
      newdata = datos_validacion,
      type = "prob"
    )
    
    pred <- predict(
      modelo,
      newdata = datos_validacion,
      type = "response"
    )
    
    probabilidad <- apply(
      probs,
      1,
      max
    )
    
  }
  
  #-------------------------------------------------------
  # CARET
  #-------------------------------------------------------
  
  else if(inherits(modelo,"train")){
    
    pred <- predict(
      modelo,
      newdata = datos_validacion
    )
    
    probabilidad <- tryCatch(
      
      {
        
        probs <- predict(
          modelo,
          newdata = datos_validacion,
          type = "prob"
        )
        
        apply(
          probs,
          1,
          max
        )
        
      },
      
      error = function(e){
        
        rep(
          NA,
          nrow(datos_validacion)
        )
        
      }
    )
    
  }
  
  #-------------------------------------------------------
  # OTROS MODELOS
  #-------------------------------------------------------
  
  else{
    
    pred <- predict(
      modelo,
      newdata = datos_validacion
    )
    
    pred <- factor(
      pred,
      levels = clases
    )
    
    probabilidad <- rep(
      NA,
      length(pred)
    )
    
  }
  
  #-------------------------------------------------------
  # RESULTADO
  #-------------------------------------------------------
  
  resultado <- data.frame(
    
    Real =
      y_real,
    
    Prediccion =
      as.factor(pred),
    
    Probabilidad =
      round(
        probabilidad,
        4
      ),
    
    Porcentual =
      ifelse(
        is.na(probabilidad),
        NA,
        paste0(
          round(
            probabilidad * 100,
            2
          ),
          " %"
        )
      )
    
  )
  
  return(resultado)
}


#=========================================================
# FUNCIÓN
# f_matriz_confusion()
#=========================================================

f_matriz_confusion <- function(
    modelo,
    datos_validacion,
    variable_dependiente,
    clase_interes = NULL){
  
  #-------------------------------------------------------
  # LIBRERÍA
  #-------------------------------------------------------
  
  library(caret)
  
  #-------------------------------------------------------
  # VALIDACIONES
  #-------------------------------------------------------
  
  if(!variable_dependiente %in% names(datos_validacion)){
    
    stop(
      paste(
        "La variable",
        variable_dependiente,
        "no existe en los datos."
      )
    )
    
  }
  
  #-------------------------------------------------------
  # VARIABLE REAL
  #-------------------------------------------------------
  
  y_real <- factor(
    datos_validacion[[variable_dependiente]]
  )
  
  clases <- levels(y_real)
  
  #-------------------------------------------------------
  # REGRESIÓN LOGÍSTICA BINOMIAL
  #-------------------------------------------------------
  
  if(inherits(modelo,"glm")){
    
    prob <- predict(
      modelo,
      newdata = datos_validacion,
      type = "response"
    )
    
    pred <- factor(
      ifelse(
        prob >= 0.50,
        clases[2],
        clases[1]
      ),
      levels = clases
    )
    
  }
  
  #-------------------------------------------------------
  # REGRESIÓN LOGÍSTICA MULTINOMIAL
  #-------------------------------------------------------
  
  else if(inherits(modelo,"multinom")){
    
    pred <- predict(
      modelo,
      newdata = datos_validacion,
      type = "class"
    )
    
    pred <- factor(
      pred,
      levels = clases
    )
    
  }
  
  #-------------------------------------------------------
  # ÁRBOL DE CLASIFICACIÓN
  #-------------------------------------------------------
  
  else if(inherits(modelo,"rpart")){
    
    pred <- predict(
      modelo,
      newdata = datos_validacion,
      type = "class"
    )
    
    pred <- factor(
      pred,
      levels = clases
    )
    
  }
  
  #-------------------------------------------------------
  # RANDOM FOREST
  #-------------------------------------------------------
  
  else if(inherits(modelo,"randomForest")){
    
    pred <- predict(
      modelo,
      newdata = datos_validacion,
      type = "response"
    )
    
    pred <- factor(
      pred,
      levels = clases
    )
    
  }
  
  #-------------------------------------------------------
  # CARET
  #-------------------------------------------------------
  
  else if(inherits(modelo,"train")){
    
    pred <- predict(
      modelo,
      newdata = datos_validacion
    )
    
    pred <- factor(
      pred,
      levels = clases
    )
    
  }
  
  #-------------------------------------------------------
  # OTROS MODELOS
  #-------------------------------------------------------
  
  else{
    
    pred <- predict(
      modelo,
      newdata = datos_validacion
    )
    
    pred <- factor(
      pred,
      levels = clases
    )
    
  }
  
  #-------------------------------------------------------
  # MATRIZ DE CONFUSIÓN BINARIA
  #-------------------------------------------------------
  
  if(length(clases) == 2){
    
    if(is.null(clase_interes)){
      
      clase_interes <- clases[2]
      
    }
    
    mc <- confusionMatrix(
      data = pred,
      reference = y_real,
      positive = clase_interes
    )
    
  }
  
  #-------------------------------------------------------
  # MATRIZ DE CONFUSIÓN MULTICLASE
  #-------------------------------------------------------
  
  else{
    
    mc <- confusionMatrix(
      data = pred,
      reference = y_real
    )
    
  }
  
  #-------------------------------------------------------
  # RETORNO
  #-------------------------------------------------------
  
  return(mc)
}


#=========================================================
# FUNCIÓN
# f_visualizar_arbol()
#=========================================================

f_visualizar_arbol <- function(
    modelo,
    arbol_rf = 1){
  
  library(rpart.plot)
  
  #-------------------------------------------------------
  # ÁRBOL DE CLASIFICACIÓN
  #-------------------------------------------------------
  
  if(inherits(modelo,"rpart")){
    
    rpart.plot(
      modelo,
      type = 3,
      extra = 104,
      fallen.leaves = TRUE,
      shadow.col = "gray",
      nn = TRUE,
      cex = 0.8
    )
    

  }
  
  #-------------------------------------------------------
  # RANDOM FOREST
  #-------------------------------------------------------
  
  else if(inherits(modelo,"randomForest")){
    
    cat("\n")
    cat("====================================\n")
    cat(" RANDOM FOREST\n")
    cat("====================================\n")
    cat(
      "No es posible visualizar simultáneamente",
      modelo$ntree,
      "árboles.\n"
    )
    
    cat(
      "Se mostrará el árbol:",
      arbol_rf,
      "\n"
    )
    
    cat("====================================\n")
    
    randomForest::getTree(
      modelo,
      k = arbol_rf,
      labelVar = TRUE
    )
    
  }
  
  else{
    
    stop(
      "Modelo no soportado."
    )
    
  }
  
}

#=========================================================
# FUNCIÓN
# f_variables_importantes()
#=========================================================

f_variables_importantes <- function(
    modelo,
    top = NULL){
  
  library(ggplot2)
  
  #-------------------------------------------------------
  # ÁRBOL DE CLASIFICACIÓN
  #-------------------------------------------------------
  
  if(inherits(modelo,"rpart")){
    
    importancia <- data.frame(
      
      Variable =
        names(
          modelo$variable.importance
        ),
      
      Importancia =
        as.numeric(
          modelo$variable.importance
        )
    )
    
  }
  
  #-------------------------------------------------------
  # RANDOM FOREST
  #-------------------------------------------------------
  
  else if(inherits(modelo,"randomForest")){
    
    imp <- importance(modelo)
    
    if(is.matrix(imp)){
      
      importancia <- data.frame(
        
        Variable = rownames(imp),
        
        Importancia =
          imp[,1]
      )
      
    } else{
      
      importancia <- data.frame(
        
        Variable = names(imp),
        
        Importancia = imp
      )
      
    }
    
  }
  
  else{
    
    stop(
      "Modelo no soportado."
    )
    
  }
  
  #-------------------------------------------------------
  # ORDENAR
  #-------------------------------------------------------
  
  importancia <-
    importancia[
      order(
        importancia$Importancia,
        decreasing = TRUE
      ),
    ]
  
  #-------------------------------------------------------
  # TOP
  #-------------------------------------------------------
  
  if(!is.null(top)){
    
    importancia <-
      head(
        importancia,
        top
      )
    
  }
  
  #-------------------------------------------------------
  # GRÁFICO
  #-------------------------------------------------------
  
  g <- ggplot(
    
    importancia,
    
    aes(
      x =
        reorder(
          Variable,
          Importancia
        ),
      
      y = Importancia
    )
    
  ) +
    
    geom_col() +
    
    coord_flip() +
    
    labs(
      x = "",
      y = "Importancia",
      title = "Variables importantes"
    ) +
    
    theme_minimal()
  
  print(g)
  
  return(importancia)
  
}



#=========================================================
# FUNCIÓN
# f_evaluar_modelos()
#=========================================================

f_evaluar_modelos <- function(
    modelos,
    datos_validacion,
    variable_dependiente,
    clase_interes = NULL,
    nombres_modelos = NULL){
  
  #-------------------------------------------------------
  # MODELO INDIVIDUAL -> LISTA
  #-------------------------------------------------------
  
  if(
    inherits(modelos,"glm") ||
    inherits(modelos,"multinom") ||
    inherits(modelos,"rpart") ||
    inherits(modelos,"randomForest") ||
    inherits(modelos,"train")
  ){
    
    modelos <- list(modelos)
    
  }
  
  #-------------------------------------------------------
  # DATAFRAME -> LISTA
  #-------------------------------------------------------
  
  if(inherits(datos_validacion,"data.frame")){
    
    datos_validacion <-
      rep(
        list(datos_validacion),
        length(modelos)
      )
    
  }
  
  #-------------------------------------------------------
  # VALIDACIÓN
  #-------------------------------------------------------
  
  if(length(modelos) != length(datos_validacion)){
    
    stop(
      "El número de modelos y conjuntos de validación debe coincidir."
    )
    
  }
  
  #-------------------------------------------------------
  # NOMBRES
  #-------------------------------------------------------
  
  if(is.null(nombres_modelos)){
    
    nombres_modelos <-
      paste(
        "Modelo",
        seq_along(modelos)
      )
    
  }
  
  #-------------------------------------------------------
  # EVALUAR
  #-------------------------------------------------------
  
  resultados <- lapply(
    
    seq_along(modelos),
    
    function(i){
      
      modelo <- modelos[[i]]
      
      datos_val <- datos_validacion[[i]]
      
      nombre_modelo <- nombres_modelos[i]
      
      mc <- f_matriz_confusion(
        modelo,
        datos_val,
        variable_dependiente,
        clase_interes
      )
      
      accuracy <-
        as.numeric(
          mc$overall["Accuracy"]
        )
      
      kappa <-
        as.numeric(
          mc$overall["Kappa"]
        )
      
      n_clases <-
        length(
          levels(
            factor(
              datos_val[[variable_dependiente]]
            )
          )
        )
      
      #===================================================
      # BINOMIAL
      #===================================================
      
      if(n_clases == 2){
        
        precision <-
          as.numeric(
            mc$byClass["Pos Pred Value"]
          )
        
        recall <-
          as.numeric(
            mc$byClass["Sensitivity"]
          )
        
        sensitivity <- recall
        
        specificity <-
          as.numeric(
            mc$byClass["Specificity"]
          )
        
        f1 <- ifelse(
          
          is.na(precision) |
            is.na(recall) |
            (precision + recall == 0),
          
          NA,
          
          2 * precision * recall /
            (precision + recall)
        )
        
        balanced_accuracy <-
          as.numeric(
            mc$byClass["Balanced Accuracy"]
          )
        
      }
      
      #===================================================
      # MULTINOMIAL
      #===================================================
      
      else{
        
        precision <-
          mean(
            mc$byClass[,"Pos Pred Value"],
            na.rm = TRUE
          )
        
        recall <-
          mean(
            mc$byClass[,"Sensitivity"],
            na.rm = TRUE
          )
        
        sensitivity <- recall
        
        specificity <-
          mean(
            mc$byClass[,"Specificity"],
            na.rm = TRUE
          )
        
        f1 <-
          mean(
            2 *
              mc$byClass[,"Pos Pred Value"] *
              mc$byClass[,"Sensitivity"] /
              (
                mc$byClass[,"Pos Pred Value"] +
                  mc$byClass[,"Sensitivity"]
              ),
            na.rm = TRUE
          )
        
        balanced_accuracy <-
          mean(
            mc$byClass[,"Balanced Accuracy"],
            na.rm = TRUE
          )
        
      }
      
      data.frame(
        
        Modelo = nombre_modelo,
        
        Accuracy =
          round(accuracy,4),
        
        Kappa =
          round(kappa,4),
        
        Precision =
          round(precision,4),
        
        Recall =
          round(recall,4),
        
        Sensitivity =
          round(sensitivity,4),
        
        Specificity =
          round(specificity,4),
        
        F1 =
          round(f1,4),
        
        Balanced_Accuracy =
          round(
            balanced_accuracy,
            4
          )
      )
      
    }
    
  )
  
  resultados <-
    do.call(
      rbind,
      resultados
    )
  
  rownames(resultados) <- NULL
  
  return(resultados)
  
}



#=========================================================
# FUNCIÓN
# f_undersampling()
#=========================================================

f_undersampling <- function(datos, variable_dependiente, semilla = 123){
  
  #-------------------------------------------------------
  # VALIDACIONES
  #-------------------------------------------------------
  
  if(!variable_dependiente %in% names(datos)){
    
    stop(paste("La variable", variable_dependiente, "no existe en los datos."))
  }
  
  #-------------------------------------------------------
  # SEMILLA
  #-------------------------------------------------------
  
  set.seed(semilla)
  
  #-------------------------------------------------------
  # FRECUENCIAS
  #-------------------------------------------------------
  
  frecuencias <- table(datos[[variable_dependiente]])
  
  #-------------------------------------------------------
  # CLASE MINORITARIA
  #-------------------------------------------------------
  
  clase_minoritaria <- names(frecuencias)[which.min(frecuencias)]
  
  n_minoritaria <- min(frecuencias)
  
  #-------------------------------------------------------
  # BALANCEAR CLASES
  #-------------------------------------------------------
  
  clases <- names(frecuencias)
  
  datos_balanceados <- lapply(clases, function(clase){
      
      datos_clase <- datos[datos[[variable_dependiente]] == clase,]
      
      #-----------------------------------------------
      # Reducir clases mayores
      #-----------------------------------------------
      
      if(nrow(datos_clase) > n_minoritaria){
        
        datos_clase <- datos_clase[sample(nrow(datos_clase), n_minoritaria, replace = FALSE),]
        }
      return(datos_clase)
    }
  )
  
  #-------------------------------------------------------
  # UNIR
  #-------------------------------------------------------
  
  datos_balanceados <- do.call(rbind, datos_balanceados)
  
  datos_balanceados <- as.data.frame(datos_balanceados)
  
  #-------------------------------------------------------
  # MEZCLAR REGISTROS
  #-------------------------------------------------------
  
  datos_balanceados <- datos_balanceados[sample(nrow(datos_balanceados)),]
  
  rownames(datos_balanceados) <- NULL
  
  #-------------------------------------------------------
  # RESUMEN
  #-------------------------------------------------------
  
  cat("\n")
  cat("====================================\n")
  cat(" UNDERSAMPLING\n")
  cat("====================================\n")
  
  cat("\nFrecuencias originales:\n")
  print(frecuencias)
  
  cat("\nFrecuencias balanceadas:\n")
  print(table(datos_balanceados[[variable_dependiente]]))
  
  cat("\nClase minoritaria:\n")
  cat(clase_minoritaria, "\n")
  
  cat("\nTotal registros originales:\n")
  cat(nrow(datos), "\n")
  
  cat("\nTotal registros balanceados:\n")
  cat(nrow(datos_balanceados), "\n")
  
  cat("====================================\n")
  
  return(datos_balanceados)
  
}


#=========================================================
# FUNCIÓN
# f_convertir_dummys()
#=========================================================

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



#=========================================================
# FUNCIÓN
# f_crear_modelo_RL_balanceada()
#=========================================================

f_crear_modelo_RL_balanceada <- function(
    datos,
    variable_dependiente,
    tecnica = "undersampling",
    semilla = 123){
  
  #-------------------------------------------------------
  # LIBRERÍAS
  #-------------------------------------------------------
  
  library(caret)
  
  #-------------------------------------------------------
  # VALIDACIONES
  #-------------------------------------------------------
  
  if(!variable_dependiente %in% names(datos)){
    
    stop(
      paste(
        "La variable",
        variable_dependiente,
        "no existe."
      )
    )
    
  }
  
  #-------------------------------------------------------
  # VARIABLE DEPENDIENTE
  #-------------------------------------------------------
  
  datos[[variable_dependiente]] <-
    as.factor(
      datos[[variable_dependiente]]
    )
  
  if(nlevels(datos[[variable_dependiente]]) != 2){
    
    stop(
      "La función está diseñada para regresión logística binaria."
    )
    
  }
  
  #-------------------------------------------------------
  # FÓRMULA
  #-------------------------------------------------------
  
  formula_modelo <-
    as.formula(
      paste(
        variable_dependiente,
        "~ ."
      )
    )
  
  #-------------------------------------------------------
  # TÉCNICA
  #-------------------------------------------------------
  
  tecnica <- tolower(tecnica)
  
  if(!tecnica %in% c(
    "undersampling",
    "oversampling",
    "smote"
  )){
    
    stop(
      "tecnica debe ser: undersampling, oversampling o smote"
    )
    
  }
  
  #-------------------------------------------------------
  # CARET
  #-------------------------------------------------------
  
  sampling <- switch(
    
    tecnica,
    
    undersampling = "down",
    
    oversampling = "up",
    
    smote = "smote"
    
  )
  
  set.seed(semilla)
  
  control <- trainControl(
    
    method = "cv",
    
    number = 5,
    
    sampling = sampling,
    
    classProbs = TRUE
    
  )
  
  #-------------------------------------------------------
  # MODELO
  #-------------------------------------------------------
  
  modelo <- train(
    
    formula_modelo,
    
    data = datos,
    
    method = "glm",
    
    family = binomial(),
    
    trControl = control
    
  )
  
  #-------------------------------------------------------
  # METADATOS
  #-------------------------------------------------------
  
  modelo$variable_dependiente <- variable_dependiente
  
  modelo$tipo_modelo <- "binomial"
  
  modelo$balanceo <- tecnica
  
  modelo$n_clases <-
    nlevels(
      datos[[variable_dependiente]]
    )
  
  #-------------------------------------------------------
  # RESUMEN
  #-------------------------------------------------------
  
  cat("\n")
  cat("====================================\n")
  cat(" REGRESIÓN LOGÍSTICA BALANCEADA\n")
  cat("====================================\n")
  cat("Técnica           :", tecnica, "\n")
  cat("Variable objetivo :", variable_dependiente, "\n")
  cat("Observaciones     :", nrow(datos), "\n")
  cat("====================================\n")
  
  return(modelo)
  
}