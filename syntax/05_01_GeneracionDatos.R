# ==============================================================================
# [TFG]   05_01_GeneracionDatos.R
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
datos <- readRDS(paste0(DATADIR, "datos_cluster.RDS"))

# ------------------------------------------------------------------------------
# Creamos las particiones correspondientes 
## Eliminamos las k que no vamso a usar 
varsNo <- c("k4", "k5", "k6")
datos <- datos[, which(!colnames(datos) %in% varsNo)]

# Creamos los datos correspondientes 
datos$k3 <- factor(
  datos$k3,
  labels = c("Cluster1", "Cluster2", "Cluster3")
)

# Creamos las particiones 
library(caret)
set.seed(2111234)

trainIndex <- createDataPartition(datos$k3, p = .8, 
                                  list = FALSE)

dataTrain <- datos[ trainIndex,]
dataTest  <- datos[-trainIndex,]


dataset <- list()
dataset$train <- dataTrain
dataset$test <- dataTest

# ------------------------------------------------------------------------------
# Guardamos el objeto en un RDS para poder usar para otros modelos 
saveRDS(dataset, file = paste0(DATADIR, "datos_modelling.RDS"))

# ==============================================================================

