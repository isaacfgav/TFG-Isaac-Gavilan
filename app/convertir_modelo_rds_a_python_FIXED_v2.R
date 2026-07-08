# ==============================================================================
# convertir_modelo_rds_a_python.R
# Conversión única del modelo RDS entrenado en R a artefactos rápidos para Python.
# Uso:
#   Rscript convertir_modelo_rds_a_python.R output/modeloXGBoost.RDS data/datos_modelling.RDS output/python_model
# ============================================================================== 

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Uso: Rscript convertir_modelo_rds_a_python.R modeloXGBoost.RDS datos_modelling.RDS output_dir")
}

model_path <- args[1]
data_path <- args[2]
out_dir <- args[3]

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(jsonlite)
  library(caret)
  library(xgboost)
})

obj <- readRDS(model_path)
model <- obj$model
dummy_model <- obj$dummy_model
nivells_k3 <- obj$nivells_k3
variables_train <- obj$variables_train

if (is.null(model) || is.null(dummy_model) || is.null(nivells_k3) || is.null(variables_train)) {
  stop("El RDS no tiene la estructura esperada: model, dummy_model, nivells_k3, variables_train.")
}

model_json_path <- file.path(out_dir, "xgboost_model.json")
xgb.save(model, model_json_path)

vars_raw <- dummy_model$vars
if (is.null(vars_raw)) vars_raw <- all.vars(dummy_model$terms)
lvls <- dummy_model$lvls

# -------------------------
# Datos de referencia
# -------------------------
dades <- NULL
if (!is.na(data_path) && file.exists(data_path)) {
  dades <- readRDS(data_path)
  if (is.list(dades) && !is.null(dades$train)) dades <- dades$train
  write.csv(dades, file.path(out_dir, "reference_train.csv"), row.names = FALSE, fileEncoding = "UTF-8")
}

# -------------------------
# Metadata formulario
# -------------------------
metadata_vars <- list()
base_values <- list()

for (v in vars_raw) {
  item <- list(name = v, type = "numeric", levels = NULL, min = NULL, max = NULL, median = NULL)

  if (!is.null(lvls) && !is.null(lvls[[v]])) {
    item$type <- "categorical"
    item$levels <- as.character(lvls[[v]])
    base_values[[v]] <- item$levels[1]
  }

  if (!is.null(dades) && v %in% names(dades)) {
    x <- dades[[v]]
    if (item$type == "categorical") {
      if (is.null(item$levels)) item$levels <- as.character(unique(x[!is.na(x)]))
      valid <- as.character(x[!is.na(x)])
      if (length(valid) > 0 && is.null(base_values[[v]])) base_values[[v]] <- valid[1]
    } else if (is.numeric(x) || is.integer(x)) {
      x_num <- as.numeric(x)
      x_num <- x_num[is.finite(x_num)]
      if (length(x_num) > 0) {
        item$min <- min(x_num, na.rm = TRUE)
        item$max <- max(x_num, na.rm = TRUE)
        item$median <- median(x_num, na.rm = TRUE)
        base_values[[v]] <- item$median
      } else {
        base_values[[v]] <- 0
      }
    } else if (is.factor(x) || is.character(x)) {
      item$type <- "categorical"
      item$levels <- as.character(unique(x[!is.na(x)]))
      if (length(item$levels) > 0) base_values[[v]] <- item$levels[1]
    }
  }

  if (is.null(base_values[[v]])) {
    if (item$type == "categorical" && length(item$levels) > 0) base_values[[v]] <- item$levels[1]
    else base_values[[v]] <- 0
  }

  metadata_vars[[length(metadata_vars) + 1]] <- item
}

make_newdata <- function(values) {
  df <- as.data.frame(as.list(values), stringsAsFactors = FALSE)
  df <- df[, vars_raw, drop = FALSE]
  for (v in vars_raw) {
    if (!is.null(lvls) && !is.null(lvls[[v]])) {
      df[[v]] <- factor(as.character(df[[v]]), levels = as.character(lvls[[v]]))
    } else {
      df[[v]] <- as.numeric(df[[v]])
    }
  }
  df
}

predict_dummy <- function(values) {
  m <- predict(dummy_model, newdata = make_newdata(values))
  m <- as.data.frame(m)
  names(m) <- make.names(names(m), unique = TRUE)
  m
}

# -------------------------
# Mapa de transformación dummyVars -> columnas del modelo
# -------------------------
feature_map <- list()
base_pred <- predict_dummy(base_values)

# IMPORTANT:
# Las columnas que genera caret::dummyVars pasan por make.names().
# Si variables_train viene guardado con nombres originales o con pequeñas diferencias
# respecto a dummyVars, Python no rellena las columnas y el modelo recibe casi todo ceros.
dummy_columns <- names(base_pred)
variables_train_original <- as.character(variables_train)
variables_train_make <- make.names(variables_train_original, unique = TRUE)

if (all(variables_train_original %in% dummy_columns)) {
  variables_train_python <- variables_train_original
} else if (all(variables_train_make %in% dummy_columns)) {
  variables_train_python <- variables_train_make
} else {
  warning(
    "variables_train no coincide exactamente con las columnas de dummyVars. ",
    "Se usarán las columnas generadas por dummyVars en su orden."
  )
  variables_train_python <- dummy_columns
}

for (v in vars_raw) {
  if (!is.null(lvls) && !is.null(lvls[[v]])) {
    levels_v <- as.character(lvls[[v]])
    rows <- list()
    for (lev in levels_v) {
      vals <- base_values
      vals[[v]] <- lev
      rows[[lev]] <- predict_dummy(vals)
    }
    mat <- do.call(rbind, rows)
    varying_cols <- names(mat)[sapply(mat, function(col) length(unique(as.numeric(col))) > 1)]

    level_map <- list()
    for (lev in levels_v) {
      row <- rows[[lev]]
      active <- varying_cols[as.numeric(row[1, varying_cols, drop = TRUE]) != 0]
      level_map[[lev]] <- as.character(active)
    }

    feature_map[[v]] <- list(type = "categorical", levels = levels_v, level_to_columns = level_map)
  } else {
    vals2 <- base_values
    base_num <- suppressWarnings(as.numeric(base_values[[v]]))
    if (!is.finite(base_num)) base_num <- 0
    vals2[[v]] <- base_num + 1
    p2 <- predict_dummy(vals2)
    differing <- names(base_pred)[as.numeric(base_pred[1, ]) != as.numeric(p2[1, ])]
    if (length(differing) == 0) differing <- make.names(v, unique = TRUE)
    feature_map[[v]] <- list(type = "numeric", columns = as.character(differing))
  }
}

# -------------------------
# Validación rápida contra R
# -------------------------
# En algunos proyectos, datos_modelling.RDS contiene columnas auxiliares o ya transformadas,
# pero no todas las variables originales usadas por dummy_model. En ese caso NO debe fallar
# la conversión: se guarda el diagnóstico y se omite solo la validación.
missing_vars_in_dades <- character(0)

if (!is.null(dades)) {
  missing_vars_in_dades <- setdiff(vars_raw, names(dades))
  if (length(missing_vars_in_dades) > 0) {
    warning(
      "datos_modelling.RDS no contiene todas las variables originales del dummy_model. ",
      "Se omite validation_matrix_python_order.csv. Variables faltantes: ",
      paste(missing_vars_in_dades, collapse = ", ")
    )
    write.csv(
      data.frame(variable_faltante = missing_vars_in_dades),
      file.path(out_dir, "diagnostico_variables_faltantes_en_datos_modelling.csv"),
      row.names = FALSE,
      fileEncoding = "UTF-8"
    )
  } else {
    n_valid <- min(20, nrow(dades))
    if (n_valid > 0) {
      valid_input <- dades[seq_len(n_valid), vars_raw, drop = FALSE]
      valid_dummy <- predict(dummy_model, newdata = valid_input)
      valid_dummy <- as.data.frame(valid_dummy)
      names(valid_dummy) <- make.names(names(valid_dummy), unique = TRUE)
      missing_valid_cols <- setdiff(variables_train_python, names(valid_dummy))
      if (length(missing_valid_cols) == 0) {
        valid_dummy <- valid_dummy[, variables_train_python, drop = FALSE]
        write.csv(valid_dummy, file.path(out_dir, "validation_matrix_python_order.csv"), row.names = FALSE, fileEncoding = "UTF-8")
        valid_pred <- predict(model, as.matrix(valid_dummy))
        write.csv(data.frame(prediction = valid_pred), file.path(out_dir, "validation_prediction_r.csv"), row.names = FALSE, fileEncoding = "UTF-8")
      } else {
        warning(
          "La matriz de validación no contiene todas las columnas de entrenamiento. Columnas faltantes: ",
          paste(missing_valid_cols, collapse = ", ")
        )
      }
    }
  }
}

artifacts <- list(
  variables = metadata_vars,
  variable_names = as.character(vars_raw),
  variables_train = as.character(variables_train_python),
  variables_train_original = as.character(variables_train_original),
  dummy_columns = as.character(dummy_columns),
  variables_faltantes_en_datos_modelling = as.character(missing_vars_in_dades),
  nivells_k3 = as.character(nivells_k3),
  feature_map = feature_map
)

writeLines(
  jsonlite::toJSON(artifacts, auto_unbox = TRUE, pretty = TRUE, na = "null"),
  file.path(out_dir, "model_artifacts.json")
)

cat("Conversión completada en:", normalizePath(out_dir), "\n")
cat("Archivos generados:\n")
cat("- xgboost_model.json\n")
cat("- model_artifacts.json\n")
if (!is.null(dades)) cat("- reference_train.csv\n")
