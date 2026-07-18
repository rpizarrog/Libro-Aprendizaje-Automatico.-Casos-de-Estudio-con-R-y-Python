def f_check_numeric_df(X):
    """
    f_check_numeric_df()
    Objetivo:
      Diagnosticar columnas problemáticas para statsmodels:
      - columnas tipo object/category
      - valores no numéricos en columnas que deberían ser numéricas
      - NaN/Inf

    Acepta:
      DataFrame o Series (la convierte a DataFrame)
    Retorna:
      dict con hallazgos y además imprime un resumen útil.
    """
    # Convertir Series -> DataFrame
    if isinstance(X, pd.Series):
        X = X.to_frame()

    if not isinstance(X, pd.DataFrame):
        raise ValueError("X debe ser pandas DataFrame o Series.")

    # Tipos
    dtypes = X.dtypes

    # Columnas object/category
    cols_obj = X.select_dtypes(include=["object"]).columns.tolist()
    cols_cat = X.select_dtypes(include=["category"]).columns.tolist()

    # NaN por columna
    nan_cols = X.columns[X.isna().any()].tolist()

    # Inf por columna (solo en numéricas)
    cols_num = X.select_dtypes(include=["number"]).columns.tolist()
    inf_cols = []
    if cols_num:
        inf_cols = [c for c in cols_num if np.isinf(X[c].to_numpy()).any()]

    # Columnas "mixtas": object pero parecen numéricas (tienen strings)
    sospechosas = []
    ejemplos = {}
    for c in cols_obj:
        # tomar 10 valores únicos (no nulos) para inspección
        vals = X[c].dropna().astype(str).unique()[:10]
        ejemplos[c] = vals
        # si muchos parecen números, marcar como sospechosa
        try:
            pd.to_numeric(X[c].dropna().astype(str), errors="raise")
            sospechosas.append(c)
        except Exception:
            pass

    print("=== Diagnóstico de X ===")
    print(f"Dimensiones: {X.shape}")
    print("\nColumnas object:", cols_obj)
    print("Columnas category:", cols_cat)
    print("Columnas con NaN:", nan_cols)
    print("Columnas numéricas con Inf:", inf_cols)

    if cols_obj:
        print("\nEjemplos (primeros valores únicos) de columnas object:")
        for c in cols_obj:
            print(f" - {c}: {ejemplos[c]}")

    if sospechosas:
        print("\n Columnas object que parecen numéricas (conviene convertir con pd.to_numeric):")
        print(sospechosas)

    return {
        "cols_object": cols_obj,
        "cols_category": cols_cat,
        "cols_nan": nan_cols,
        "cols_inf": inf_cols,
        "sospechosas_numeric_as_object": sospechosas
    }

def f_implementar_modelo_RL_multinomial(
    datos_entrenamiento: pd.DataFrame,
    datos_validacion: pd.DataFrame,
    variable_dependiente: str,
    variables_independientes: list | None = None,
    drop_first_dummies: bool = True,
    alpha: float = 1.0,
    maxiter: int = 500,
    method: str = "l1",
    semilla: int = 2026
) -> dict:
    #------------------------------------------------------------
    # f_implementar_modelo_RL_multinomial()
    #
    # Objetivo:
    #   Construir un modelo de Regresión Logística Multinomial en Python
    #   usando statsmodels (MNLogit) con regularización (fit_regularized)
    #   para evitar problemas como separación perfecta/overflow.
    #
    # Argumentos:
    #   datos_entrenamiento   : DataFrame con datos de entrenamiento.
    #   datos_validacion      : DataFrame con datos de validación (para alinear X).
    #   variable_dependiente  : Nombre (str) de la variable objetivo (p.ej. "shopping_preference").
    #   variables_independientes : Lista de predictores. Si None, usa todas excepto la dependiente.
    #   drop_first_dummies    : Elimina la primera categoría al crear dummies (evita colinealidad).
    #   alpha                 : Fuerza de regularización (mayor = más penalización).
    #   maxiter               : Iteraciones máximas.
    #   method                : Tipo de regularización ("l1" o "l2").
    #   semilla               : Semilla para reproducibilidad (por defecto 2026).
    #
    # Retorna:
    #   dict con:
    #     - 'modelo'     : objeto MNLogit (no entrenado)
    #     - 'resultado'  : resultado entrenado (fit_regularized)
    #     - 'clases'     : lista de clases en el orden del modelo (clase base = clases[0])
    #     - 'scaler'     : StandardScaler ajustado en entrenamiento
    #     - 'X_train_s'  : matriz X de entrenamiento (escalada + const)
    #     - 'X_valid_s'  : matriz X de validación (escalada + const, alineada)
    #     - 'y_codes'    : códigos enteros de y (0..K-1)
    #     - 'X_cols'     : nombres de columnas usadas en X (útil para predicción consistente)
    #
    # Uso:
    #   res = f_implementar_modelo_RL_multinomial(datos_entrenamiento, datos_validacion,
    #                                            "shopping_preference")
    #   res["resultado"].params
    #------------------------------------------------------------

    np.random.seed(semilla)

    #-------------------------
    # 0) Validaciones
    #-------------------------
    if not isinstance(datos_entrenamiento, pd.DataFrame):
        raise ValueError("datos_entrenamiento debe ser un pandas.DataFrame.")
    if not isinstance(datos_validacion, pd.DataFrame):
        raise ValueError("datos_validacion debe ser un pandas.DataFrame.")
    if variable_dependiente not in datos_entrenamiento.columns:
        raise ValueError(f"No existe '{variable_dependiente}' en datos_entrenamiento.")
    if variable_dependiente not in datos_validacion.columns:
        raise ValueError(f"No existe '{variable_dependiente}' en datos_validacion.")

    # Si no se especifican independientes, usar todas menos la dependiente
    if variables_independientes is None:
        variables_independientes = [
            c for c in datos_entrenamiento.columns if c != variable_dependiente
        ]

    faltantes_train = [c for c in variables_independientes if c not in datos_entrenamiento.columns]
    faltantes_valid = [c for c in variables_independientes if c not in datos_validacion.columns]
    if faltantes_train:
        raise ValueError(f"Predictoras faltantes en entrenamiento: {faltantes_train}")
    if faltantes_valid:
        raise ValueError(f"Predictoras faltantes en validación: {faltantes_valid}")

    #-------------------------
    # 1) y -> category -> codes
    #-------------------------
    y_train_cat = datos_entrenamiento[variable_dependiente].astype("category")
    y_codes = y_train_cat.cat.codes.astype(int)
    clases = list(y_train_cat.cat.categories)

    if len(clases) < 3:
        raise ValueError("La variable dependiente debe tener al menos 3 clases (multinomial).")

    #-------------------------
    # 2) X raw (desde train y valid)
    #-------------------------
    X_train_raw = datos_entrenamiento[variables_independientes].copy()
    X_valid_raw = datos_validacion[variables_independientes].copy()

    #-------------------------
    # 3) Dummies
    #-------------------------
    X_train = pd.get_dummies(X_train_raw, drop_first=drop_first_dummies)
    X_valid = pd.get_dummies(X_valid_raw, drop_first=drop_first_dummies)

    #-------------------------
    # 4) Alinear columnas (valid a train)
    #-------------------------
    X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)

    #-------------------------
    # 5) Forzar float
    #-------------------------
    X_train = X_train.astype(float)
    X_valid = X_valid.astype(float)

    #-------------------------
    # 6) Escalar (para estabilidad numérica)
    #-------------------------
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_valid_s = pd.DataFrame(scaler.transform(X_valid), columns=X_train.columns)

    #-------------------------
    # 7) Agregar constante UNA sola vez
    #-------------------------
    X_train_s = sm.add_constant(X_train_s, has_constant="add")
    X_valid_s = sm.add_constant(X_valid_s, has_constant="add")

    #-------------------------
    # 8) Quitar columnas duplicadas (seguridad)
    #-------------------------
    X_train_s = X_train_s.loc[:, ~X_train_s.columns.duplicated()].copy()
    X_valid_s = X_valid_s.loc[:, ~X_valid_s.columns.duplicated()].copy()

    #-------------------------
    # 9) Validación de finitud
    #-------------------------
    if not np.isfinite(X_train_s.to_numpy()).all():
        raise ValueError("X_train_s contiene NaN/Inf. Revisa datos y transformaciones.")
    if not np.isfinite(X_valid_s.to_numpy()).all():
        raise ValueError("X_valid_s contiene NaN/Inf. Revisa datos y transformaciones.")

    #-------------------------
    # 10) Alinear índices (CRÍTICO para statsmodels)
    #-------------------------
    X_train_s = X_train_s.reset_index(drop=True)
    y_codes = y_codes.reset_index(drop=True)

    #-------------------------
    # 11) Construir y entrenar MNLogit con regularización
    #-------------------------
    modelo = sm.MNLogit(y_codes, X_train_s)

    resultado = modelo.fit_regularized(
        method=method,   # "l1" o "l2"
        alpha=alpha,
        maxiter=maxiter,
        disp=False
    )

    return {
        "modelo": modelo,
        "resultado": resultado,
        "clases": clases,
        "scaler": scaler,
        "X_train_s": X_train_s,
        "X_valid_s": X_valid_s,
        "y_codes": y_codes,
        "X_cols": list(X_train_s.columns),
        "variable_dependiente": variable_dependiente,
        "variables_independientes": variables_independientes,
        "config": {
            "drop_first_dummies": drop_first_dummies,
            "alpha": alpha,
            "maxiter": maxiter,
            "method": method,
            "semilla": semilla
        }
    }

def f_evaluar_modelo(
    resultado_modelo: dict,
    datos_validacion: pd.DataFrame,
    umbral_accuracy: float = 0.70,
    fill_na: float = 0.0
) -> dict:
    #------------------------------------------------------------
    # f_evaluar_modelo()
    #
    # Objetivo:
    #   Evaluar un modelo de clasificación multinomial entrenado con
    #   statsmodels MNLogit (fit_regularized), usando datos de validación.
    #   Calcula matriz de confusión y métricas: Accuracy, Precision, Recall,
    #   F1 (por clase), además de Macro-F1 y decisión ACEPTAR/NO ACEPTAR.
    #
    # Argumentos:
    #   resultado_modelo : dict retornado por f_implementar_modelo_RL_multinomial()
    #                     (debe incluir: 'resultado', 'clases', 'scaler',
    #                      'variables_independientes', 'variable_dependiente',
    #                      y opcionalmente 'config').
    #   datos_validacion : DataFrame con predictores + variable dependiente.
    #   umbral_accuracy  : Umbral mínimo de exactitud para aceptar el modelo.
    #   fill_na          : Valor con el que se reemplazan NaN/Inf tras transformar.
    #
    # Retorna:
    #   dict con:
    #     - 'matriz_confusion'     : DataFrame (conteos)
    #     - 'accuracy'             : float
    #     - 'precision_por_clase'  : Series
    #     - 'recall_por_clase'     : Series
    #     - 'f1_por_clase'         : Series
    #     - 'macro_f1'             : float
    #     - 'decision'             : str ("ACEPTAR"/"NO ACEPTAR")
    #     - 'y_true'               : Series (clase real)
    #     - 'y_pred'               : Series (clase predicha)
    #     - 'probs'                : ndarray (probabilidades por clase)
    #
    # Requiere:
    #   numpy, pandas, sklearn.metrics.confusion_matrix
    #------------------------------------------------------------

    #-------------------------
    # 0) Validaciones
    #-------------------------
    for k in ["resultado", "clases", "scaler", "variables_independientes", "variable_dependiente"]:
        if k not in resultado_modelo:
            raise ValueError(f"resultado_modelo no contiene la llave requerida: '{k}'")

    res = resultado_modelo["resultado"]          # objeto fit_regularized
    clases = resultado_modelo["clases"]          # lista de clases (orden)
    scaler = resultado_modelo["scaler"]          # StandardScaler ajustado
    vars_ind = resultado_modelo["variables_independientes"]
    var_dep = resultado_modelo["variable_dependiente"]

    if var_dep not in datos_validacion.columns:
        raise ValueError(f"En datos_validacion no existe la variable dependiente '{var_dep}'")
    falt = [c for c in vars_ind if c not in datos_validacion.columns]
    if falt:
        raise ValueError(f"Faltan predictores en datos_validacion: {falt}")

    # y verdadero
    y_true = datos_validacion[var_dep].astype(str)

    #-------------------------
    # 1) Preparar X_valid: dummies + alineación + escalado + const
    #    (debe ser consistente con entrenamiento)
    #-------------------------
    X_valid_raw = datos_validacion[vars_ind].copy()

    # Crear dummies (mismo criterio)
    drop_first = True
    if "config" in resultado_modelo and "drop_first_dummies" in resultado_modelo["config"]:
        drop_first = bool(resultado_modelo["config"]["drop_first_dummies"])

    X_valid = pd.get_dummies(X_valid_raw, drop_first=drop_first)

    # Alinear columnas con entrenamiento (sin 'const' todavía)
    # X_cols incluye const; para alinear dummies, usamos columnas sin const
    X_cols = resultado_modelo.get("X_cols", None)
    if X_cols is None:
        raise ValueError("resultado_modelo debe incluir 'X_cols' (columnas usadas en entrenamiento).")

    cols_sin_const = [c for c in X_cols if c != "const"]
    X_valid = X_valid.reindex(columns=cols_sin_const, fill_value=0)

    # Forzar float
    X_valid = X_valid.astype(float)

    # Escalar con scaler entrenado
    X_valid_s = pd.DataFrame(scaler.transform(X_valid), columns=cols_sin_const)

    # Agregar constante y alinear exactamente a X_cols
    import statsmodels.api as sm
    X_valid_s = sm.add_constant(X_valid_s, has_constant="add")
    X_valid_s = X_valid_s.reindex(columns=X_cols, fill_value=0.0)

    # Limpiar NaN/Inf (por seguridad)
    X_valid_s = X_valid_s.replace([np.inf, -np.inf], np.nan).fillna(fill_na).astype(float)

    #-------------------------
    # 2) Predicción de probabilidades y clases
    #-------------------------
    probs = np.asarray(res.predict(X_valid_s))  # normalmente (n, K-1)

    K = len(clases)
    if probs.ndim == 1:
        # Caso raro: convertir a 2D
        probs = probs.reshape(-1, 1)

    # Reconstruir probabilidad de la clase base si el resultado trae K-1 columnas
    if probs.shape[1] == (K - 1):
        p_base = 1 - probs.sum(axis=1, keepdims=True)
        probs = np.hstack([p_base, probs])

    # Si por alguna razón trae K columnas, seguimos normal
    if probs.shape[1] != K:
        raise ValueError(f"Dimensión inesperada de probs: {probs.shape}. Se esperaban K={K} columnas.")

    idx = np.argmax(probs, axis=1)
    y_pred = pd.Series([clases[i] for i in idx], index=y_true.index)

    #-------------------------
    # 3) Matriz de confusión y métricas
    #-------------------------
    cm = confusion_matrix(y_true, y_pred, labels=clases)
    cm_df = pd.DataFrame(cm, index=clases, columns=clases)
    cm_df.index.name = "Real"
    cm_df.columns.name = "Predicho"

    total = cm.sum()
    accuracy = (np.trace(cm) / total) if total > 0 else 0.0

    # Recall por clase = TP / (TP+FN) = diag / row sum
    row_sums = cm.sum(axis=1)
    recall = np.divide(np.diag(cm), row_sums, out=np.zeros_like(row_sums, dtype=float), where=row_sums != 0)

    # Precision por clase = TP / (TP+FP) = diag / col sum
    col_sums = cm.sum(axis=0)
    precision = np.divide(np.diag(cm), col_sums, out=np.zeros_like(col_sums, dtype=float), where=col_sums != 0)

    # F1 por clase
    denom = (precision + recall)
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom, dtype=float), where=denom != 0)

    precision_s = pd.Series(precision, index=clases, name="precision")
    recall_s = pd.Series(recall, index=clases, name="recall")
    f1_s = pd.Series(f1, index=clases, name="f1")

    macro_f1 = float(f1_s.mean())

    # Decisión
    decision = "ACEPTAR" if accuracy >= umbral_accuracy else "NO ACEPTAR"

    return {
        "matriz_confusion": cm_df,
        "accuracy": float(accuracy),
        "precision_por_clase": precision_s,
        "recall_por_clase": recall_s,
        "f1_por_clase": f1_s,
        "macro_f1": macro_f1,
        "decision": decision,
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs
    }