# ==============================================================================
# [TFG]  05_03_SVM.R
# 
# Autor(s)   : (c) Isaac Fernández, XX.XX.XXXX
# Revisio    :     -
# Descripcio : 
# ==============================================================================
{script <- paste('[TFG]  05_03_SVM.R')
cat(script, '\n', rep('=', nchar(script)), '\n', sep = '')
start.time <- Sys.time()
cat('Inici:', format(start.time, '%d.%m.%Y %H:%M'), '\n')}
# ==============================================================================
# Cargamos las rutas necesarias 
ruta <- "D:/DOCENCIA/CURS/TFG Alumnos/2025-2026/Q2/ISAAC FERNANDEZ/TFG-Isaac-Gavilan/syntax/"
setwd(ruta); source(file ="00_InicioProyecto.R")

# Cargamos las funciones necesarias 
source(file = paste0(SYNTAXDIR, "99_funciones.R"))

# ==============================================================================
# Cargamos los paquetes necesarios 
paquetes <- c("dplyr", "FactoMineR", "reshape", "ggplot2", "caret", "e1071")

new.packages <- paquetes[!(paquetes %in% installed.packages()[, "Package"])]
if (length(new.packages) > 0) {
  install.packages(new.packages)
}

invisible(lapply(paquetes, require, character.only = TRUE))
rm(paquetes, new.packages)

# ==============================================================================
# Cargamos la base de datos 
dataset <- readRDS(paste0(DATADIR, "datos_modelling.RDS"))

# ==============================================================================

trControl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = multiClassSummary
)

modelLookup("svmRadial")

# Se especifica un rango de valores para los hiperparámetros
tuneGrid <- expand.grid(sigma = seq(from=0.1, to=0.2, by=0.05),
                        C = 10**(-2:4))

# Se fija la semilla aleatoria
set.seed(101)

# Se entrena el modelo
model <- train(k3 ~ ., data = dataset$train, method = "svmRadial",
               metric = "Accuracy", trControl = trControl, tuneGrid = tuneGrid)

saveRDS(model, file = paste0(OUTPUTDIR, "modeloSVM.RDS"))

pred <- predict(model, newdata = dataset$test)

confusionMatrix(pred, dataset$test$k3) # matriz de confusión del resu

# Falta la matriz de importancia de la variables 



