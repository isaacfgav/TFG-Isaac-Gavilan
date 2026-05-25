# ==============================================================================
# [TFG]   02_Preprocessing.R
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
              "caret")

new.packages <- paquetes[!(paquetes %in% installed.packages()[, "Package"])]
if (length(new.packages) > 0) {
  install.packages(new.packages)
}

invisible(lapply(paquetes, require, character.only = TRUE))
rm(paquetes, new.packages)

# ------------------------------------------------------------------------------
# Cargamos la base de datos 
datos_raw <- readRDS(paste0(DATADIR, "data_inicial.RDS"))

dim(datos_raw)
head(datos_raw)

# ------------------------------------------------------------------------------
# Realizamos la selección de las variables que nos quedamos
datos <- datos_raw %>%
  select(
    -identificacion,
    -localizacion_de_la_lesion_es,
    -estructura_lesionada
  ) %>%
  rename(
    min_entreno_fisico = indica_los_minutos_totales_que_entrenas_fisico_a_lo_largo_de_la_semana,
    min_entreno_pista  = indica_los_minutos_totales_que_entrenas_pista_a_lo_largo_de_la_semana,
    perc_fatiga        = fatiga_num
  )

vars_bin <- names(datos)[sapply(datos, function(x) is.numeric(x) && dplyr::n_distinct(na.omit(x)) <= 2)]
vars_char <- names(datos)[sapply(datos, is.character)]
vars_factor <- unique(c(vars_bin, vars_char))


for (var in vars_factor) {
  datos[[var]] <- as.factor(datos[[var]])
}

str(datos)

# ------------------------------------------------------------------------------
cat("Nombre d'observacions:", nrow(datos), "\n")
cat("Nombre de variables:", ncol(datos), "\n\n")

names(datos)

# ------------------------------------------------------------------------------
# Guardamos los datos 
saveRDS(datos, file = paste0(DATADIR, "datos_preprocessing.RDS"))

# ==============================================================================