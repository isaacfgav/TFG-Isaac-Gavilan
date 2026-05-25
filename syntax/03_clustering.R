# ==============================================================================
# [TFG]   03_clustering.R
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
datos <- readRDS(paste0(DATADIR, "datos_preprocessing.RDS"))

# ------------------------------------------------------------------------------
# Calculamos la distancia de Gower
distancia <- daisy(datos, metric = "gower")
distancia <- distancia^2
cluster_hc <- hclust(distancia, method = "ward.D2")

# Visualizamos el dendograma obtenido 
plot(cluster_hc, labels = FALSE, main = "Dendrograma clustering jeràrquic", 
     xlab = "", sub = "")

# ------------------------------------------------------------------------------
# Seleccionamos el mejor valor de k a través del índice de Calinski-Harabasz 
# probar varios k
k_vals <- 2:10

sil_scores <- sapply(k_vals, function(k) {
  grupos <- cutree(cluster_hc, k = k)
  mean(silhouette(grupos, distancia)[, 3])   # media silhouette
})

# tabla resultados
tabla <- data.frame(
  k = k_vals,
  silhouette = sil_scores
)

# Top 3 mejores k
ggplot(tabla, aes(x = k, y = silhouette)) + 
  geom_line() + 
  geom_point() + 
  theme_minimal()

# Guardamos la información 
datos$k3 <- factor(cutree(cluster_hc, k = 3))
datos$k4 <- factor(cutree(cluster_hc, k = 4))
datos$k5 <- factor(cutree(cluster_hc, k = 5))
datos$k6 <- factor(cutree(cluster_hc, k = 6))

cluster_hc %>% as.dendrogram() %>% set("branches_k_color", k = 3) %>% plot(main = "")

# Guardamos la información en un dataframe
saveRDS(datos, file = paste0(DATADIR, "datos_cluster.RDS"))

