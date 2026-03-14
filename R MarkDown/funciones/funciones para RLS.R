# Funciones para implementar y evaluar modelos de 
# Regresión Lineal Simple
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


f_diagrama_dispersion_tendencia <- function(modelo, datos){
  
  library(ggplot2)
  
  #----------------------------------------------------------
  # La función recibe el modelo y los datos de entrenamiento
  # Extraer los nombres de las variables del modelo
  # Calcular correlación de Pearson
  # Construir gráfico de dispersión con línea de tendencia
  
  formula_modelo <- formula(modelo)
  
  variable_y <- all.vars(formula_modelo)[1]
  variable_x <- all.vars(formula_modelo)[2]
  
  
  r <- cor(datos[[variable_x]], datos[[variable_y]])
  
  
  #----------------------------------------------------------
  # 3. Obtener R² del modelo
  #----------------------------------------------------------
  r_square <- summary(modelo)$r.squared
  

  ggplot(datos, aes(x = .data[[variable_x]], y = .data[[variable_y]])) +
    
    # puntos de dispersión
    geom_point(color = "red", size = 3) +
    
    # línea de regresión del modelo
    geom_smooth(method = "lm", se = FALSE, color = "blue", linewidth = 1) +
    
    # título y subtítulo con estadísticos
    ggtitle(
      label = paste("Dispersión y tendencia:"),
      subtitle = paste( variable_x, "vs", variable_y,";",
        "r =", round(r,3),
        "; R² =", round(r_square,3)
      )
    ) +
    
    theme_minimal()
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