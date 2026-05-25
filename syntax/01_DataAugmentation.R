# ==============================================================================
# [TFG]   01_DataAugmentation.R
#
# Este script permite generar la base de datos por la cuál vamos a realizar todo
# el proyecto. 
# ==============================================================================

# Cargamos las rutas necesarias 
setwd("D:/DOCENCIA/CURS/TFG Alumnos/2025-2026/Q2/ISAAC FERNANDEZ/TFG-Isaac-Gavilan/syntax/")
source(file = paste0(SYNTAXDIR, "00_InicioProyecto.R"))

# Cargamos las funciones necesarias 
source(file = paste0(SYNTAXDIR, "99_funciones.R"))

# Cargamos los paquetes necesarios 

paquetes <- c("readxl", "janitor", "dplyr", "stringr", "simstudy", "writexl")

new.packages <- paquetes[!(paquetes %in% installed.packages()[, "Package"])]
if (length(new.packages) > 0) {
  install.packages(new.packages)
}

invisible(lapply(paquetes, require, character.only = TRUE))
rm(paquetes, new.packages)

# ------------------------------------------------------------------------------
# Cargamos la base de datos 
## Llegim la base i netegem noms de columnes
injuries <- read_excel("BBDD_INJURIES_v1.xlsx") %>%
  clean_names()

str(injuries)
summary(injuries)

# CÒPIA DE BBDD ----------------------------------------------------------------
## Mantenim injuries intacte i treballem sobre mod_injuries
mod_injuries <- injuries

# CONVERSIÓ VARIABLES NUMÈRIQUES -----------------------------------------------
## Convertim a numèric les variables quantitatives
mod_injuries <- mod_injuries %>%
  mutate(
    edad_anos = as.numeric(gsub(",", ".", edad_anos)),
    peso_kg = as.numeric(gsub(",", ".", peso_kg)),
    altura_corporal_cm = as.numeric(gsub(",", ".", altura_corporal_cm)),
    indica_los_minutos_totales_que_entrenas_fisico_a_lo_largo_de_la_semana =
      as.numeric(gsub(",", ".", indica_los_minutos_totales_que_entrenas_fisico_a_lo_largo_de_la_semana)),
    indica_los_minutos_totales_que_entrenas_pista_a_lo_largo_de_la_semana =
      as.numeric(gsub(",", ".", indica_los_minutos_totales_que_entrenas_pista_a_lo_largo_de_la_semana))
  )

# IMPUTACIÓ MÍNIMA -------------------------------------------------------------
## Imputem només el pes perdut amb la mediana
mod_injuries <- mod_injuries %>%
  mutate(
    peso_kg = ifelse(is.na(peso_kg), median(peso_kg, na.rm = TRUE), peso_kg)
  )

# VARIABLES DERIVADES PER A LA SIMULACIÓ ---------------------------------------
mod_injuries <- mod_injuries %>%
  mutate(
    # Variables binàries bàsiques
    seleccion_bin = case_when(
      is.na(seleccion) ~ NA_real_,
      seleccion == "Masculina" ~ 1,
      seleccion == "Femenina" ~ 0,
      TRUE ~ NA_real_
    ),
    lesiones_previas_bin = case_when(
      is.na(lesiones_previas) ~ NA_real_,
      lesiones_previas == "Sí" ~ 1,
      lesiones_previas == "No" ~ 0,
      TRUE ~ NA_real_
    ),
    
    # Variables quantitatives auxiliars
    min_fisico = indica_los_minutos_totales_que_entrenas_fisico_a_lo_largo_de_la_semana,
    min_pista = indica_los_minutos_totales_que_entrenas_pista_a_lo_largo_de_la_semana,
    
    # Fatiga com a escala numèrica
    fatiga_num = case_when(
      str_detect(cual_es_tu_percepcion_de_fatiga_despues_de_la_ultima_competicion_entrenamiento_de_la_semana, "^10") ~ 10,
      str_detect(cual_es_tu_percepcion_de_fatiga_despues_de_la_ultima_competicion_entrenamiento_de_la_semana, "^9") ~ 9,
      str_detect(cual_es_tu_percepcion_de_fatiga_despues_de_la_ultima_competicion_entrenamiento_de_la_semana, "^8") ~ 8,
      str_detect(cual_es_tu_percepcion_de_fatiga_despues_de_la_ultima_competicion_entrenamiento_de_la_semana, "^7") ~ 7,
      str_detect(cual_es_tu_percepcion_de_fatiga_despues_de_la_ultima_competicion_entrenamiento_de_la_semana, "^6") ~ 6,
      str_detect(cual_es_tu_percepcion_de_fatiga_despues_de_la_ultima_competicion_entrenamiento_de_la_semana, "^5") ~ 5,
      str_detect(cual_es_tu_percepcion_de_fatiga_despues_de_la_ultima_competicion_entrenamiento_de_la_semana, "^4") ~ 4,
      str_detect(cual_es_tu_percepcion_de_fatiga_despues_de_la_ultima_competicion_entrenamiento_de_la_semana, "^3") ~ 3,
      str_detect(cual_es_tu_percepcion_de_fatiga_despues_de_la_ultima_competicion_entrenamiento_de_la_semana, "^2") ~ 2,
      str_detect(cual_es_tu_percepcion_de_fatiga_despues_de_la_ultima_competicion_entrenamiento_de_la_semana, "^1") ~ 1,
      TRUE ~ NA_real_
    ),
    
    # Comptadors de lesions
    cabeza_tronco_n = recode_count(cabeza_y_tronco),
    miembro_superior_n = recode_count(miembro_superior),
    miembro_inferior_n = recode_count(miembro_inferior),
    
    # Binàries derivades
    cabeza_tronco_bin = case_when(
      is.na(cabeza_tronco_n) ~ NA_real_,
      cabeza_tronco_n > 0 ~ 1,
      cabeza_tronco_n == 0 ~ 0
    ),
    miembro_superior_bin = case_when(
      is.na(miembro_superior_n) ~ NA_real_,
      miembro_superior_n > 0 ~ 1,
      miembro_superior_n == 0 ~ 0
    ),
    miembro_inferior_bin = case_when(
      is.na(miembro_inferior_n) ~ NA_real_,
      miembro_inferior_n > 0 ~ 1,
      miembro_inferior_n == 0 ~ 0
    ),
    
    # Localitzacions
    loc_hombro_clavicula = case_when(
      is.na(localizacion_de_la_lesion_es) ~ NA_real_,
      str_detect(localizacion_de_la_lesion_es, fixed("Hombro/clavícula")) ~ 1,
      TRUE ~ 0
    ),
    loc_brazo = case_when(
      is.na(localizacion_de_la_lesion_es) ~ NA_real_,
      str_detect(localizacion_de_la_lesion_es, fixed("Brazo")) ~ 1,
      TRUE ~ 0
    ),
    loc_rodilla = case_when(
      is.na(localizacion_de_la_lesion_es) ~ NA_real_,
      str_detect(localizacion_de_la_lesion_es, fixed("Rodilla")) ~ 1,
      TRUE ~ 0
    ),
    loc_tobillo = case_when(
      is.na(localizacion_de_la_lesion_es) ~ NA_real_,
      str_detect(localizacion_de_la_lesion_es, fixed("Tobillo")) ~ 1,
      TRUE ~ 0
    ),
    loc_dorso = case_when(
      is.na(localizacion_de_la_lesion_es) ~ NA_real_,
      str_detect(localizacion_de_la_lesion_es, fixed("Dorso")) ~ 1,
      TRUE ~ 0
    ),
    loc_columna_lumbar = case_when(
      is.na(localizacion_de_la_lesion_es) ~ NA_real_,
      str_detect(localizacion_de_la_lesion_es, fixed("Columna lumbar")) ~ 1,
      TRUE ~ 0
    ),
    loc_cara = case_when(
      is.na(localizacion_de_la_lesion_es) ~ NA_real_,
      str_detect(localizacion_de_la_lesion_es, fixed("Cara")) ~ 1,
      TRUE ~ 0
    ),
    
    # Estructures lesionades
    est_musculo = case_when(
      is.na(estructura_lesionada) ~ NA_real_,
      str_detect(estructura_lesionada, fixed("Músculo")) ~ 1,
      TRUE ~ 0
    ),
    est_ligamento = case_when(
      is.na(estructura_lesionada) ~ NA_real_,
      str_detect(estructura_lesionada, fixed("Ligamento")) ~ 1,
      TRUE ~ 0
    ),
    est_hueso = case_when(
      is.na(estructura_lesionada) ~ NA_real_,
      str_detect(estructura_lesionada, fixed("Hueso")) ~ 1,
      TRUE ~ 0
    ),
    est_menisco = case_when(
      is.na(estructura_lesionada) ~ NA_real_,
      str_detect(estructura_lesionada, fixed("Menisco")) ~ 1,
      TRUE ~ 0
    )
  )

# REVISIÓ BBDD MODIFICADA ------------------------------------------------------
ncol(mod_injuries)
names(mod_injuries)
summary(mod_injuries)

# VARIABLES BASE MODEL SIMSTUDY ------------------------------------------------
def_base <- defData(
  varname = "edad_anos",
  formula = mean(mod_injuries$edad_anos, na.rm = TRUE),
  variance = var(mod_injuries$edad_anos, na.rm = TRUE),
  dist = "normal"
)

def_base <- defData(def_base,
                    varname = "peso_kg",
                    formula = mean(mod_injuries$peso_kg, na.rm = TRUE),
                    variance = var(mod_injuries$peso_kg, na.rm = TRUE),
                    dist = "normal"
)

def_base <- defData(def_base,
                    varname = "altura_corporal_cm",
                    formula = mean(mod_injuries$altura_corporal_cm, na.rm = TRUE),
                    variance = var(mod_injuries$altura_corporal_cm, na.rm = TRUE),
                    dist = "normal"
)

def_base <- defData(def_base,
                    varname = "min_fisico",
                    formula = mean(mod_injuries$min_fisico, na.rm = TRUE),
                    variance = var(mod_injuries$min_fisico, na.rm = TRUE),
                    dist = "normal"
)

def_base <- defData(def_base,
                    varname = "min_pista",
                    formula = mean(mod_injuries$min_pista, na.rm = TRUE),
                    variance = var(mod_injuries$min_pista, na.rm = TRUE),
                    dist = "normal"
)

def_base <- defData(def_base,
                    varname = "fatiga_num",
                    formula = mean(mod_injuries$fatiga_num, na.rm = TRUE),
                    variance = var(mod_injuries$fatiga_num, na.rm = TRUE),
                    dist = "normal"
)

def_base <- defData(def_base,
                    varname = "seleccion_bin",
                    formula = mean(mod_injuries$seleccion_bin, na.rm = TRUE),
                    dist = "binary"
)

def_base <- defData(def_base,
                    varname = "lesiones_previas_bin",
                    formula = mean(mod_injuries$lesiones_previas_bin, na.rm = TRUE),
                    dist = "binary"
)

# GENERACIÓ BASE SINTÈTICA -----------------------------------------------------
# Generem una mostra sintètica
sim_injuries <- genData(5000, def_base)

# AFEGIR VARIABLES DEPENDENTS ..................................................
def_add <- defDataAdd(
  varname = "cabeza_tronco_bin",
  formula = "-1 + 0.8*lesiones_previas_bin",
  dist = "binary",
  link = "logit"
)

def_add <- defDataAdd(def_add,
                      varname = "miembro_superior_bin",
                      formula = "-1 + 1.1*lesiones_previas_bin + 0.2*seleccion_bin",
                      dist = "binary",
                      link = "logit"
)

def_add <- defDataAdd(def_add,
                      varname = "miembro_inferior_bin",
                      formula = "0.3 + 1.2*lesiones_previas_bin - 0.02*edad_anos + 0.001*min_pista",
                      dist = "binary",
                      link = "logit"
)

def_add <- defDataAdd(def_add,
                      varname = "loc_hombro_clavicula",
                      formula = "-1 + 0.9*miembro_superior_bin",
                      dist = "binary",
                      link = "logit"
)

def_add <- defDataAdd(def_add,
                      varname = "loc_brazo",
                      formula = "-1.2 + 0.8*miembro_superior_bin",
                      dist = "binary",
                      link = "logit"
)

def_add <- defDataAdd(def_add,
                      varname = "loc_rodilla",
                      formula = "-1 + 0.8*miembro_inferior_bin",
                      dist = "binary",
                      link = "logit"
)

def_add <- defDataAdd(def_add,
                      varname = "loc_tobillo",
                      formula = "-0.5 + 1.1*miembro_inferior_bin",
                      dist = "binary",
                      link = "logit"
)

def_add <- defDataAdd(def_add,
                      varname = "loc_dorso",
                      formula = "-1.4 + 0.8*cabeza_tronco_bin",
                      dist = "binary",
                      link = "logit"
)

def_add <- defDataAdd(def_add,
                      varname = "loc_columna_lumbar",
                      formula = "-1.1 + 0.9*cabeza_tronco_bin",
                      dist = "binary",
                      link = "logit"
)

def_add <- defDataAdd(def_add,
                      varname = "loc_cara",
                      formula = "-2 + 0.7*cabeza_tronco_bin",
                      dist = "binary",
                      link = "logit"
)

def_add <- defDataAdd(def_add,
                      varname = "est_musculo",
                      formula = "-0.8 + 0.4*min_fisico/100 + 0.4*min_pista/100",
                      dist = "binary",
                      link = "logit"
)

def_add <- defDataAdd(def_add,
                      varname = "est_ligamento",
                      formula = "-0.4 + 0.8*miembro_inferior_bin + 0.3*miembro_superior_bin",
                      dist = "binary",
                      link = "logit"
)

def_add <- defDataAdd(def_add,
                      varname = "est_hueso",
                      formula = "-1.8 + 0.5*cabeza_tronco_bin + 0.3*miembro_superior_bin",
                      dist = "binary",
                      link = "logit"
)

def_add <- defDataAdd(def_add,
                      varname = "est_menisco",
                      formula = "-2 + 1.2*loc_rodilla",
                      dist = "binary",
                      link = "logit"
)

sim_injuries <- addColumns(def_add, sim_injuries)

# COMPTADORS SIMPLES ...........................................................
sim_injuries <- sim_injuries %>%
  mutate(
    cabeza_tronco_n = ifelse(cabeza_tronco_bin == 1, 1, 0),
    miembro_superior_n = ifelse(miembro_superior_bin == 1, 1, 0),
    miembro_inferior_n = ifelse(miembro_inferior_bin == 1, 1, 0)
  )

# RECONSTRUCCIÓ VARIABLES ORIGINALS ............................................
n_sim <- nrow(sim_injuries)

sim_injuries <- sim_injuries %>%
  mutate(
    identificacion = 1:n(),
    seleccion = ifelse(seleccion_bin == 1, "Masculina", "Femenina"),
    edad_anos = round(edad_anos),
    peso_kg = round(peso_kg, 1),
    altura_corporal_cm = round(altura_corporal_cm),
    indica_los_minutos_totales_que_entrenas_fisico_a_lo_largo_de_la_semana = pmax(0, round(min_fisico)),
    indica_los_minutos_totales_que_entrenas_pista_a_lo_largo_de_la_semana = pmax(0, round(min_pista)),
    lesiones_previas = ifelse(lesiones_previas_bin == 1, "Sí", "No"),
    cabeza_y_tronco = ifelse(cabeza_tronco_bin == 1, "1", "0"),
    miembro_superior = ifelse(miembro_superior_bin == 1, "1", "0"),
    miembro_inferior = ifelse(miembro_inferior_bin == 1, "1", "0")
  )

# RECONSTRUCCIÓ TEXTOS .........................................................
sim_injuries <- sim_injuries %>%
  rowwise() %>%
  mutate(
    localizacion_de_la_lesion_es = ifelse(
      lesiones_previas_bin == 0,
      "",
      paste0(
        ifelse(loc_hombro_clavicula == 1, "Hombro/clavícula;", ""),
        ifelse(loc_brazo == 1, "Brazo;", ""),
        ifelse(loc_rodilla == 1, "Rodilla;", ""),
        ifelse(loc_tobillo == 1, "Tobillo;", ""),
        ifelse(loc_dorso == 1, "Dorso;", ""),
        ifelse(loc_columna_lumbar == 1, "Columna lumbar;", ""),
        ifelse(loc_cara == 1, "Cara;", "")
      )
    ),
    
    cabeza_y_tronco = ifelse(
      lesiones_previas_bin == 0,
      "0",
      ifelse(cabeza_tronco_bin == 1, "1", "0")
    ),
    
    miembro_superior = ifelse(
      lesiones_previas_bin == 0,
      "0",
      ifelse(miembro_superior_bin == 1, "1", "0")
    ),
    
    miembro_inferior = ifelse(
      lesiones_previas_bin == 0,
      "0",
      ifelse(miembro_inferior_bin == 1, "1", "0")
    ),
    
    estructura_lesionada = ifelse(
      lesiones_previas_bin == 0,
      "",
      paste0(
        ifelse(est_musculo == 1, "Músculo;", ""),
        ifelse(est_ligamento == 1, "Ligamento;", ""),
        ifelse(est_hueso == 1, "Hueso;", ""),
        ifelse(est_menisco == 1, "Menisco;", "")
      )
    )
  ) %>%
  ungroup()

# VARIABLES CATEGÒRIQUES RECONSTRUÏDES -----------------------------------------
sim_injuries <- sim_injuries %>%
  mutate(
    raza = sample_empirical(mod_injuries$raza, n_sim),
    x1_partido = sample_empirical(mod_injuries$x1_partido, n_sim),
    x2_partidos = sample_empirical(mod_injuries$x2_partidos, n_sim)
  ) %>%
  mutate(
    x2_partidos = ifelse(trimws(x1_partido) %in% c("0", "0'"), "0", x2_partidos)
  )

# FATIGA RECONSTRUÏDA ----------------------------------------------------------
sim_injuries <- sim_injuries %>%
  mutate(
    fatiga_num = pmin(10, pmax(1, round(fatiga_num))),
    cual_es_tu_percepcion_de_fatiga_despues_de_la_ultima_competicion_entrenamiento_de_la_semana =
      paste0(fatiga_num)
  )

# CORRECCIÓN -------------------------------------------------------------------
sim_injuries <- sim_injuries %>%
  mutate(
    # Si no hi ha lesions prèvies, no hi pot haver localitzacions ni estructures
    loc_hombro_clavicula = ifelse(lesiones_previas == 0, 0, loc_hombro_clavicula),
    loc_brazo = ifelse(lesiones_previas == 0, 0, loc_brazo),
    loc_rodilla = ifelse(lesiones_previas == 0, 0, loc_rodilla),
    loc_tobillo = ifelse(lesiones_previas == 0, 0, loc_tobillo),
    loc_dorso = ifelse(lesiones_previas == 0, 0, loc_dorso),
    loc_columna_lumbar = ifelse(lesiones_previas == 0, 0, loc_columna_lumbar),
    loc_cara = ifelse(lesiones_previas == 0, 0, loc_cara),
    
    est_musculo = ifelse(lesiones_previas == 0, 0, est_musculo),
    est_ligamento = ifelse(lesiones_previas == 0, 0, est_ligamento),
    est_hueso = ifelse(lesiones_previas == 0, 0, est_hueso),
    est_menisco = ifelse(lesiones_previas == 0, 0, est_menisco),
    
    # Coherència amb les zones corporals
    loc_hombro_clavicula = ifelse(miembro_superior_bin == 0, 0, loc_hombro_clavicula),
    loc_brazo = ifelse(miembro_superior == 0, 0, loc_brazo),
    
    loc_rodilla = ifelse(miembro_inferior == 0, 0, loc_rodilla),
    loc_tobillo = ifelse(miembro_inferior == 0, 0, loc_tobillo),
    
    loc_dorso = ifelse(cabeza_y_tronco == 0, 0, loc_dorso),
    loc_columna_lumbar = ifelse(cabeza_y_tronco == 0, 0, loc_columna_lumbar),
    loc_cara = ifelse(cabeza_y_tronco == 0, 0, loc_cara)
  )

# REORDENAR COLUMNES -----------------------------------------------------------
original_cols <- c(
  "identificacion",
  "seleccion",
  "edad_anos",
  "peso_kg",
  "altura_corporal_cm",
  "raza",
  "indica_los_minutos_totales_que_entrenas_fisico_a_lo_largo_de_la_semana",
  "indica_los_minutos_totales_que_entrenas_pista_a_lo_largo_de_la_semana",
  "x1_partido",
  "x2_partidos",
  "cual_es_tu_percepcion_de_fatiga_despues_de_la_ultima_competicion_entrenamiento_de_la_semana",
  "lesiones_previas",
  "localizacion_de_la_lesion_es",
  "cabeza_y_tronco",
  "miembro_superior",
  "miembro_inferior",
  "estructura_lesionada"
)

derived_cols <- setdiff(names(sim_injuries), original_cols) 

sim_injuries <- sim_injuries %>% select(all_of(original_cols), all_of(derived_cols))

sim_injuries <- sim_injuries %>% 
  select(-seleccion_bin, -id, -min_fisico, -min_pista, -cabeza_tronco_bin, 
         -lesiones_previas_bin, -miembro_superior_bin, -miembro_inferior_bin, 
         -cabeza_tronco_n, -miembro_superior_n, -miembro_inferior_n, -cabeza_y_tronco, 
         -miembro_superior, -miembro_inferior)

# EXPORTAR ---------------------------------------------------------------------
saveRDS(sim_injuries, file = paste0(DATADIR, "data_inicial.RDS"))

# ==============================================================================