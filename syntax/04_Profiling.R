# ==============================================================================
# [TFG]   04_Profiling.R
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
# Creamos los profilings
## k = 3
datosk3_catdes <- datos %>%
  select(-k4, -k5, -k6) %>%
  select(-k3, everything(), k3)

res_catdes_k3 <- FactoMineR::catdes(datosk3_catdes, num.var = ncol(datosk3_catdes))
res_catdes_k3

plot(res_catdes_k3)


datos_k3 <- datos %>% select(-k4, -k5, -k6)

vars_num_k3 <- names(datos_k3)[sapply(datos_k3, is.numeric)]
vars_num_k3 <- setdiff(vars_num_k3, c("k3"))

res_num_k3 <- lapply(vars_num_k3, function(v) {
  aggregate(datos_k3[[v]], by = list(cluster = datos_k3$k3), function(x) {
    c(
      n = length(x),
      mean = mean(x, na.rm = TRUE),
      sd = sd(x, na.rm = TRUE),
      median = median(x, na.rm = TRUE),
      min = min(x, na.rm = TRUE),
      max = max(x, na.rm = TRUE)
    )
  })
})

names(res_num_k3) <- vars_num_k3
res_num_k3

vars_cat_k3 <- names(datos_k3)[sapply(datos_k3, is.factor)]
vars_cat_k3 <- setdiff(vars_cat_k3, c("k3"))

for (v in vars_cat_k3) {
  cat("\n====================", v, "====================\n")
  print(prop.table(table(datos_k3$k3, datos_k3[[v]]), margin = 1))
}

tabla_k3 <- data.frame(table(datos$k3))
colnames(tabla_k3) <- c("Cluster", "FreqAbs")
tabla_k3$FreqRel <- round(tabla_k3$FreqAbs / sum(tabla_k3$FreqAbs), 4)
