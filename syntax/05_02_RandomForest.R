# ==============================================================================
# [TFG]   05_02_RandomForest.R
#
# Este script permite realizar el preprocessing de los datos
# ==============================================================================
# Cargamos las rutas necesarias 
setwd("D:/DOCENCIA/CURS/TFG Alumnos/2025-2026/Q2/ISAAC FERNANDEZ/TFG-Isaac-Gavilan/syntax/")
source(file = paste0(SYNTAXDIR, "00_InicioProyecto.R"))

# Cargamos las funciones necesarias 
source(file = paste0(SYNTAXDIR, "99_funciones.R"))

# Cargamos los paquetes necesarios 
paquetes <- c("readxl", "dplyr", "cluster", "FactoMineR", "factoextra", "ggplot2", 
              "caret", "dendextend")

new.packages <- paquetes[!(paquetes %in% installed.packages()[, "Package"])]
if (length(new.packages) > 0) {
  install.packages(new.packages)
}

invisible(lapply(paquetes, require, character.only = TRUE))
rm(paquetes, new.packages)

# ------------------------------------------------------------------------------
# Cargamos la base de datos 
dataset <- readRDS(paste0(DATADIR, "datos_modelling.RDS"))

# ------------------------------------------------------------------------------
library("randomForest")

# se fija la semilla aleatoria
set.seed(222243112)

modelLookup("rf")

# Se especifica un rango de valores posibles de mtry
tuneGrid <- expand.grid(mtry = 1:30)

# se entrena el modelo
model <- train(k3~., data=dataset$train, 
               method="rf", metric="Accuracy", ntree=500,
               tuneGrid = tuneGrid,
               trControl=trainControl(method="cv", 
                                      number=10, 
                                      classProbs = TRUE))

tablaResultados <- data.frame(model$results)

ggplot(tablaResultados, aes(x = "mtry", y = "Accuracy")) +
         geom_line() +
         geom_point()

pred <- predict(model, newdata = dataset$test)

confusionMatrix(pred, dataset$test$k3) # matriz de confusión del resu

