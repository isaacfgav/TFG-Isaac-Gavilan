# ==============================================================================
# [TFG]  00_inicioProyecto.R
# 
# Autor(s)   : (c) Isaac Fernández, XX.XX.XXXX
# Revisio    :     -
# Descripcio : Definicio de rutes i constants globals
# ==============================================================================
{script <- paste('[TFG]  00_inicioProyecto.R')
cat(script, '\n', rep('=', nchar(script)), '\n', sep = '')
start.time <- Sys.time()
cat('Inici:', format(start.time, '%d.%m.%Y %H:%M'), '\n')}

# ==============================================================================
# 1. Sistema Operatiu
if (.Platform['OS.type'] == 'windows') {
  # servidor <- strsplit(Sys.info()['nodename'], '\\.')[[1]][1]
  if (Sys.info()['user'] == 'Usuario') {
    PATH <- 'C:/Users/Usuario/Desktop/TFG/'
  } else if (Sys.info()['user'] == 'sergi') {
    PATH <- 'D:/DOCENCIA/CURS/TFG Alumnos/2025-2026/Q2/ISAAC FERNANDEZ/TFG-Isaac-Gavilan/'
  } else {
    stop('Ordenador no soportado! PREFIX no se ha podido definir.\n')
  }
} else if (.Platform['OS.type'] == 'darwin') {
  if (Sys.info()['user'] == "ramitjans") {
    PATH <- "Volumen/user/"
  }
}

# ==============================================================================
# 2. Parametres

# ==============================================================================
# 3. Rutes
# Definim i creem els directoris si es necessari
DATADIR    <- paste(PATH, 'data',   sep = '')
INPUTDIR   <- paste(PATH, 'input',  sep = '')
OUTPUTDIR  <- paste(PATH, 'output', sep = '')
TEMPDIR    <- paste(PATH, 'temp',   sep = '')
SYNTAXDIR  <- paste(PATH, 'syntax', sep = '')
DOCDIR     <- paste(PATH, 'doc',    sep = '')

if (! file.exists(DATADIR))    { dir.create(DATADIR);    cat('Carpeta creada:', DATADIR,    '\n') }
if (! file.exists(INPUTDIR))   { dir.create(INPUTDIR);   cat('Carpeta creada:', INPUTDIR,   '\n') }
if (! file.exists(OUTPUTDIR))  { dir.create(OUTPUTDIR);  cat('Carpeta creada:', OUTPUTDIR,  '\n') }
if (! file.exists(TEMPDIR))    { dir.create(TEMPDIR);    cat('Carpeta creada:', TEMPDIR,    '\n') }
if (! file.exists(SYNTAXDIR))  { dir.create(SYNTAXDIR);  cat('Carpeta creada:', SYNTAXDIR,  '\n') }
if (! file.exists(DOCDIR))     { dir.create(DOCDIR);     cat('Carpeta creada:', DOCDIR,     '\n') }

# Adaptem rutes
INPUTDIR   <- paste(INPUTDIR,   '/', sep = '')
OUTPUTDIR  <- paste(OUTPUTDIR,  '/', sep = '')
TEMPDIR    <- paste(TEMPDIR,    '/', sep = '')
SYNTAXDIR  <- paste(SYNTAXDIR,  '/', sep = '')
DATADIR    <- paste(DATADIR,    '/', sep = '')
DOCDIR     <- paste(DOCDIR,     '/', sep = '')

# ==============================================================================
# 4. Libraries
list.of.packages <-c() 
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages) > 0) {
  install.packages(new.packages)
}
lapply(list.of.packages, require, character.only = T)
rm(list.of.packages, new.packages)

# ==============================================================================
# 5. Opcions de l'entorn
options(scipen = 999)
options(stringsAsFactors = FALSE)

# ==============================================================================
# 6. Indice de proyecto
# source(paste(SYNTAXDIR, '00_indiceProyecto.R', sep = ''))

# Calcul del timing
# END OF SCRIPT