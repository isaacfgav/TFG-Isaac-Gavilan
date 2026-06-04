# ==============================================================================
# [TFG] predict_xgboost.R
#
# Script auxiliar per predir amb el model XGBoost guardat en RDS
# ==============================================================================

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
  
  cat("No s'han rebut arguments des de Streamlit.\n")
  cat("S'utilitzaran rutes per defecte per provar el script des de RStudio.\n")
  
  input_csv  <- "input.csv"
  output_csv <- "output.csv"
  model_path <- "modeloXGBoost.RDS"
  
} else {
  
  input_csv  <- args[1]
  output_csv <- args[2]
  model_path <- args[3]
}
# ------------------------------------------------------------------------------
# Paquets necessaris

paquetes <- c("caret", "xgboost")

for (pkg in paquetes) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(paste("Falta instal·lar el paquet:", pkg))
  }
}

library(caret)
library(xgboost)

# ------------------------------------------------------------------------------
# Carreguem el model

modelo_xgboost <- readRDS(model_path)

# El model s'havia guardat com una llista
model          <- modelo_xgboost$model
dummy_model    <- modelo_xgboost$dummy_model
nivells_k3     <- modelo_xgboost$nivells_k3
variables_train <- modelo_xgboost$variables_train

if (is.null(model) || is.null(dummy_model) || is.null(nivells_k3) || is.null(variables_train)) {
  stop("El fitxer RDS no té l'estructura esperada: model, dummy_model, nivells_k3 i variables_train.")
}

# ------------------------------------------------------------------------------
# Llegim les dades noves

newdata_original <- read.csv(
  input_csv,
  stringsAsFactors = FALSE,
  check.names = FALSE,
  fileEncoding = "UTF-8"
)

newdata <- newdata_original

# ------------------------------------------------------------------------------
# Detectem les variables originals que espera el dummy_model

vars_raw <- dummy_model$vars

if (is.null(vars_raw)) {
  vars_raw <- all.vars(dummy_model$terms)
}

vars_falten <- setdiff(vars_raw, names(newdata))

if (length(vars_falten) > 0) {
  stop(
    paste0(
      "Falten variables al CSV d'entrada:\n",
      paste(vars_falten, collapse = ", ")
    )
  )
}

# Ens quedem només amb les variables necessàries
newdata <- newdata[, vars_raw, drop = FALSE]

# ------------------------------------------------------------------------------
# Convertim les dades al mateix format dummy utilitzat en l'entrenament

x_new <- predict(dummy_model, newdata = newdata)
x_new <- as.data.frame(x_new)

# Mateix tractament de noms que al training
names(x_new) <- make.names(names(x_new), unique = TRUE)

# Si falta alguna columna dummy, la creem amb 0
vars_falten_post <- setdiff(variables_train, names(x_new))

if (length(vars_falten_post) > 0) {
  for (v in vars_falten_post) {
    x_new[[v]] <- 0
  }
}

# Eliminem columnes sobrants i ordenem igual que al training
x_new <- x_new[, variables_train, drop = FALSE]

# Convertim tot a numèric
x_new[] <- lapply(x_new, as.numeric)

# Substituïm NA, NaN, Inf i -Inf per 0
x_new[] <- lapply(x_new, function(x) {
  x[!is.finite(x)] <- NA
  x[is.na(x)] <- 0
  return(x)
})

# ------------------------------------------------------------------------------
# Predicció amb XGBoost

dnew <- xgb.DMatrix(data = as.matrix(x_new))

pred_raw <- predict(model, newdata = dnew)

num_class <- length(nivells_k3)

# Cas 1: el model retorna directament classes 0, 1, 2
if (length(pred_raw) == nrow(x_new)) {
  
  pred_num <- as.integer(round(pred_raw))
  
} else {
  
  # Cas 2: el model retorna probabilitats per classe
  pred_prob_matrix <- matrix(
    pred_raw,
    ncol = num_class,
    byrow = TRUE
  )
  
  pred_num <- max.col(pred_prob_matrix) - 1
}

# Convertim 0, 1, 2 a Cluster1, Cluster2, Cluster3
pred <- factor(
  nivells_k3[pred_num + 1],
  levels = nivells_k3
)

# ------------------------------------------------------------------------------
# Creem resultat final

resultats <- newdata_original
resultats$prediccio_cluster <- pred

write.csv(
  resultats,
  output_csv,
  row.names = FALSE,
  fileEncoding = "UTF-8"
)