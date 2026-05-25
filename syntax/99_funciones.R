# ==============================================================================
# [TFG]   99_funciones.R
#
# Este script permite contener todas las funciones necesarias para la generación
# del pipeline de trabajo. 
# ==============================================================================

recode_count <- function(x) {
  # Recodifica comptadors respectant els NA
  case_when(
    is.na(x) ~ NA_real_,
    x == "0" ~ 0,
    x == "1" ~ 1,
    x == "2" ~ 2,
    x == "3" ~ 3,
    x == "4" ~ 4,
    x == "5" ~ 5,
    x == ">5" ~ 5,
    TRUE ~ NA_real_
  )
}

# ------------------------------------------------------------------------------
sample_empirical <- function(x, n) {
  # Mostreig empíric per reconstruir variables categòriques
  x_valid <- x[!is.na(x)]
  sample(x_valid, size = n, replace = TRUE)
}

# ------------------------------------------------------------------------------
suggested.level <- function(hc, min = 3, max = 10) {
  if (min < 2) stop("Min should be equal or higher than 2")
  intra <- rev(cumsum(hc[["height"]]))
  quot <- intra[min:max] / intra[(min - 1):(max - 1)]
  nb.clust <- which.min(quot) + min - 1
  return(nb.clust)
}

# ------------------------------------------------------------------------------
suggested.level.top3 <- function(hc, min = 3, max = 10, top_n = 3) {
  if (min < 2) stop("Min should be equal or higher than 2")
  if (max <= min) stop("Max should be higher than min")
  
  intra <- rev(cumsum(hc[["height"]]))
  ks <- min:max
  quot <- intra[min:max] / intra[(min - 1):(max - 1)]
  
  res <- data.frame(
    k = ks,
    criteri = quot
  )
  
  res <- res[order(res$criteri), ]
  head(res, top_n)
}

# ------------------------------------------------------------------------------
suggested.level.all <- function(hc, min = 3, max = 10) {
  if (min < 2) stop("Min should be equal or higher than 2")
  if (max <= min) stop("Max should be higher than min")
  
  intra <- rev(cumsum(hc[["height"]]))
  ks <- min:max
  quot <- intra[min:max] / intra[(min - 1):(max - 1)]
  
  data.frame(
    k = ks,
    criteri = quot
  )
}