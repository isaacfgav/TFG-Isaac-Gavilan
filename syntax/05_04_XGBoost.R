# ==============================================================================
# [TFG]  05_04_XGBoost.R
# 
# Autor(s)   : (c) Isaac Fernández, XX.XX.XXXX
# Revisio    :     -
# Descripcio : Model XGBoost per predir el cluster k3
# ==============================================================================
{script <- paste('[TFG]  05_04_XGBoost.R')
cat(script, '\n', rep('=', nchar(script)), '\n', sep = '')
start.time <- Sys.time()
cat('Inici:', format(start.time, '%d.%m.%Y %H:%M'), '\n')}
# ==============================================================================

# Cargamos las rutas necesarias 
source("syntax/00_inicioProyecto.R")

# Cargamos las funciones necesarias 
source(paste0(SYNTAXDIR, "99_funciones.R"))

# Cargamos los paquetes necesarios 
paquetes <- c(
  "readxl", "dplyr", "cluster", "FactoMineR",
  "factoextra", "ggplot2", "caret", "dendextend",
  "xgboost"
)

paquetes_faltantes <- c()

for (pkg in paquetes) {
  ok <- tryCatch(
    requireNamespace(pkg, quietly = TRUE),
    error = function(e) FALSE,
    warning = function(w) FALSE
  )
  
  if (!ok) {
    paquetes_faltantes <- c(paquetes_faltantes, pkg)
  }
}

if (length(paquetes_faltantes) > 0) {
  cat("Falten paquets per instal·lar:\n")
  print(paquetes_faltantes)
  stop("Instal·la aquests paquets manualment abans d'executar l'script.\n")
}

invisible(lapply(paquetes, library, character.only = TRUE))
rm(paquetes, paquetes_faltantes)

# ==============================================================================
# Cargamos la base de datos 

dataset <- readRDS(paste0(DATADIR, "datos_modelling.RDS"))

# ==============================================================================
# Preparació de les dades

dataTrain <- dataset$train
dataTest  <- dataset$test

# Eliminem files sense variable resposta
dataTrain <- dataTrain[!is.na(dataTrain$k3), ]
dataTest  <- dataTest[!is.na(dataTest$k3), ]

# Ens assegurem que k3 és factor
dataTrain$k3 <- droplevels(as.factor(dataTrain$k3))
dataTest$k3  <- factor(dataTest$k3, levels = levels(dataTrain$k3))

# Eliminem possibles files del test amb classe no reconeguda
dataTest <- dataTest[!is.na(dataTest$k3), ]

# Guardem nivells de la resposta
nivells_k3 <- levels(dataTrain$k3)

cat("Distribució de classes al train:\n")
print(table(dataTrain$k3))

cat("Distribució de classes al test:\n")
print(table(dataTest$k3))

# ==============================================================================
# Separació de predictors i resposta

x_train_raw <- dataTrain[, setdiff(names(dataTrain), "k3"), drop = FALSE]
x_test_raw  <- dataTest[, setdiff(names(dataTest), "k3"), drop = FALSE]

y_train <- dataTrain$k3
y_test  <- dataTest$k3

# ==============================================================================
# Conversió de predictors a format numèric

# XGBoost necessita matriu numèrica.
# Si totes les variables ja són numèriques, això no altera res important.
# Si hi ha factors o caràcters, els converteix a dummies.

dummy_model <- dummyVars(
  ~ .,
  data = x_train_raw,
  fullRank = TRUE
)

x_train <- predict(dummy_model, newdata = x_train_raw)
x_test  <- predict(dummy_model, newdata = x_test_raw)

x_train <- as.data.frame(x_train)
x_test  <- as.data.frame(x_test)

# Arreglem noms de columnes per evitar problemes amb XGBoost
names(x_train) <- make.names(names(x_train), unique = TRUE)
names(x_test)  <- names(x_train)

# ==============================================================================
# Neteja de NA, NaN i Inf

x_train[] <- lapply(x_train, as.numeric)
x_test[]  <- lapply(x_test, as.numeric)

x_train[] <- lapply(x_train, function(x) {
  x[!is.finite(x)] <- NA
  return(x)
})

x_test[] <- lapply(x_test, function(x) {
  x[!is.finite(x)] <- NA
  return(x)
})

# Imputem NA amb la mediana del train
for (var in names(x_train)) {
  
  mediana_train <- median(x_train[[var]], na.rm = TRUE)
  
  if (is.na(mediana_train)) {
    mediana_train <- 0
  }
  
  x_train[[var]][is.na(x_train[[var]])] <- mediana_train
  x_test[[var]][is.na(x_test[[var]])]   <- mediana_train
}

# Eliminem variables amb variància zero o quasi zero
nzv <- nearZeroVar(x_train)

if (length(nzv) > 0) {
  vars_eliminar <- names(x_train)[nzv]
  
  cat("Variables eliminades per variància zero o quasi zero:\n")
  print(vars_eliminar)
  
  x_train <- x_train[, -nzv, drop = FALSE]
  x_test  <- x_test[, names(x_train), drop = FALSE]
}

cat("NA finals a x_train:", sum(is.na(x_train)), "\n")
cat("NA finals a x_test :", sum(is.na(x_test)), "\n")

cat("Dimensions x_train:\n")
print(dim(x_train))

cat("Dimensions x_test:\n")
print(dim(x_test))

# ==============================================================================
# Preparació de la resposta per XGBoost

# XGBoost necessita classes codificades com 0, 1, 2, ...
y_train_num <- as.numeric(y_train) - 1
y_test_num  <- as.numeric(y_test) - 1

num_class <- length(nivells_k3)

cat("Nombre de classes:", num_class, "\n")
cat("Nivells de k3:\n")
print(nivells_k3)

# Creem DMatrix
dtrain <- xgb.DMatrix(
  data = as.matrix(x_train),
  label = y_train_num
)

dtest <- xgb.DMatrix(
  data = as.matrix(x_test),
  label = y_test_num
)

# ==============================================================================
# Validació creuada per seleccionar hiperparàmetres

tuneGrid <- expand.grid(
  nrounds = c(50, 100, 150),
  max_depth = c(2, 3, 4),
  eta = c(0.05, 0.1),
  gamma = c(0, 1),
  colsample_bytree = c(0.8),
  min_child_weight = c(1),
  subsample = c(0.8)
)

resultats_cv <- data.frame()

# Nombre de folds
min_classe <- min(table(y_train))

if (min_classe >= 5) {
  nfold_cv <- 5
} else if (min_classe >= 3) {
  nfold_cv <- 3
} else if (min_classe >= 2) {
  nfold_cv <- 2
} else {
  stop("Hi ha alguna classe amb només una observació. No es pot fer validació creuada.")
}

cat("Nombre de folds utilitzat:", nfold_cv, "\n")

set.seed(101)

for (i in seq_len(nrow(tuneGrid))) {
  
  cat("Entrenant combinació", i, "de", nrow(tuneGrid), "\n")
  
  params <- list(
    objective = "multi:softmax",
    eval_metric = "merror",
    num_class = num_class,
    max_depth = tuneGrid$max_depth[i],
    eta = tuneGrid$eta[i],
    gamma = tuneGrid$gamma[i],
    colsample_bytree = tuneGrid$colsample_bytree[i],
    min_child_weight = tuneGrid$min_child_weight[i],
    subsample = tuneGrid$subsample[i]
  )
  
  cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = tuneGrid$nrounds[i],
    nfold = nfold_cv,
    stratified = TRUE,
    verbose = FALSE
  )
  
  eval_log <- as.data.frame(cv$evaluation_log)
  
  col_error <- grep("test.*merror.*mean", names(eval_log), value = TRUE)
  
  if (length(col_error) == 0) {
    stop("No s'ha trobat la columna test_merror_mean dins de cv$evaluation_log.")
  }
  
  col_error <- col_error[1]
  
  millor_iteracio <- which.min(eval_log[[col_error]])
  millor_error     <- eval_log[[col_error]][millor_iteracio]
  
  fila_resultat <- data.frame(
    nrounds = tuneGrid$nrounds[i],
    best_iteration = millor_iteracio,
    max_depth = tuneGrid$max_depth[i],
    eta = tuneGrid$eta[i],
    gamma = tuneGrid$gamma[i],
    colsample_bytree = tuneGrid$colsample_bytree[i],
    min_child_weight = tuneGrid$min_child_weight[i],
    subsample = tuneGrid$subsample[i],
    test_error = millor_error,
    Accuracy_CV = 1 - millor_error
  )
  
  resultats_cv <- rbind(resultats_cv, fila_resultat)
}

# Ordenem pel menor error
resultats_cv <- resultats_cv[order(resultats_cv$test_error), ]

cat("Millors resultats de validació creuada:\n")
print(head(resultats_cv, 10))

millor <- resultats_cv[1, ]

cat("Millor combinació seleccionada:\n")
print(millor)

# ==============================================================================
# Entrenament del model final

params_final <- list(
  objective = "multi:softmax",
  eval_metric = "merror",
  num_class = num_class,
  max_depth = millor$max_depth,
  eta = millor$eta,
  gamma = millor$gamma,
  colsample_bytree = millor$colsample_bytree,
  min_child_weight = millor$min_child_weight,
  subsample = millor$subsample
)

set.seed(101)

model <- xgb.train(
  params = params_final,
  data = dtrain,
  nrounds = as.integer(millor$best_iteration),
  verbose = FALSE
)

# Guardem el model i els objectes necessaris
modelo_xgboost <- list(
  model = model,
  dummy_model = dummy_model,
  nivells_k3 = nivells_k3,
  variables_train = names(x_train),
  params_final = params_final,
  resultats_cv = resultats_cv
)

saveRDS(modelo_xgboost, file = paste0(OUTPUTDIR, "modeloXGBoost.RDS"))

# ==============================================================================
# Predicció sobre test

pred_num <- predict(model, newdata = dtest)

# XGBoost retorna 0, 1, 2...
# Ho tornem a convertir a Cluster1, Cluster2, Cluster3
pred <- factor(
  nivells_k3[pred_num + 1],
  levels = nivells_k3
)

# Matriu de confusió
matriu_confusio <- confusionMatrix(pred, y_test)

print(matriu_confusio)

# ==============================================================================
# Predicció sobre train

pred_train_num <- predict(model, newdata = dtrain)

# XGBoost retorna 0, 1, 2...
# Ho tornem a convertir a Cluster1, Cluster2, Cluster3
pred_train <- factor(
  nivells_k3[pred_train_num + 1],
  levels = nivells_k3
)

# Matriu de confusió train
matriu_confusio_train <- confusionMatrix(pred_train, y_train)

cat("\nMatriu de confusió TRAIN:\n")
print(matriu_confusio_train)

cat("\nAccuracy train:\n")
print(matriu_confusio_train$overall["Accuracy"])

cat("\nKappa train:\n")
print(matriu_confusio_train$overall["Kappa"])

# ==============================================================================
# Importància de variables

imp_df <- xgb.importance(
  feature_names = names(x_train),
  model = model
)

cat("Importància de variables:\n")
print(imp_df)

saveRDS(imp_df, file = paste0(OUTPUTDIR, "importanciaXGBoost.RDS"))

# ==============================================================================
# Resultats finals

cat("\nAccuracy test:\n")
print(matriu_confusio$overall["Accuracy"])

cat("\nKappa test:\n")
print(matriu_confusio$overall["Kappa"])

cat("\nMillors hiperparàmetres:\n")
print(millor)

# ==============================================================================
# Finalització

end.time <- Sys.time()
cat('Fi:', format(end.time, '%d.%m.%Y %H:%M'), '\n')
cat('Temps total:', round(difftime(end.time, start.time, units = "mins"), 2), 'minuts\n')
# ==============================================================================
