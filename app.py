# ==============================================================================
# [TFG] app.py
#
# Aplicació Streamlit per predir el risc lesiu amb modeloXGBoost.RDS
# Inclou:
#   1) Formulari tipus enquesta individual
#   2) Predicció massiva mitjançant CSV
#   3) Semàfor de risc
#   4) Recomanacions personalitzades segons el clúster
#   5) Radial plot: observació analitzada vs mitjana del clúster predit
#   6) Top 10 casos més semblants dins del clúster predit
#   7) Gràfics comparatius observació vs clúster
# ==============================================================================

import os
import json
import textwrap
import tempfile
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go


# ==============================================================================
# CONFIGURACIÓ GENERAL
# ==============================================================================

st.set_page_config(
    page_title="Predicció de risc lesiu - TFG",
    page_icon="🚦",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent
RSCRIPT = os.environ.get("RSCRIPT_PATH", "Rscript")

MODEL_CANDIDATES = [
    PROJECT_ROOT / "output" / "modeloXGBoost.RDS",
    PROJECT_ROOT / "modeloXGBoost.RDS",
]

DATA_CANDIDATES = [
    PROJECT_ROOT / "data" / "datos_modelling.RDS",
    PROJECT_ROOT / "input" / "datos_modelling.RDS",
    PROJECT_ROOT / "output" / "datos_modelling.RDS",
    PROJECT_ROOT / "datos_modelling.RDS",
]

MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), None)
DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), None)


# ==============================================================================
# ESTIL VISUAL
# ==============================================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        color: #1F2937;
    }

    .subtitle {
        color: #4B5563;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .cluster-card {
        padding: 1.1rem;
        border-radius: 0.8rem;
        border: 1px solid #E5E7EB;
        background-color: #FAFAFA;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .recommendation-card {
        padding: 1.1rem;
        border-radius: 0.8rem;
        border: 1px solid #E5E7EB;
        background-color: #FFFFFF;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .risk-title {
        font-size: 1.25rem;
        font-weight: 750;
        margin-bottom: 0.4rem;
    }

    .small-note {
        color: #6B7280;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# FUNCIONS AUXILIARS PYTHON
# ==============================================================================

def llegir_csv_robust(uploaded_file):
    intents = [
        {"sep": ",", "encoding": "utf-8"},
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "utf-8-sig"},
        {"sep": ";", "encoding": "utf-8-sig"},
        {"sep": ",", "encoding": "latin1"},
        {"sep": ";", "encoding": "latin1"},
        {"sep": ",", "encoding": "cp1252"},
        {"sep": ";", "encoding": "cp1252"},
    ]

    ultim_error = None

    for params in intents:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, **params)

            if df.shape[1] > 1:
                return df

        except Exception as e:
            ultim_error = e

    raise ValueError(
        "No s'ha pogut llegir el CSV. Revisa que sigui un fitxer CSV real "
        "i que les columnes estiguin separades correctament."
    ) from ultim_error


def normalitzar_nom(nom):
    return (
        str(nom)
        .lower()
        .strip()
        .replace("á", "a")
        .replace("à", "a")
        .replace("é", "e")
        .replace("è", "e")
        .replace("í", "i")
        .replace("ï", "i")
        .replace("ó", "o")
        .replace("ò", "o")
        .replace("ú", "u")
        .replace("ü", "u")
        .replace("ç", "c")
        .replace("ñ", "n")
    )


def netejar_nom_variable(nom):
    nom_norm = normalitzar_nom(nom)

    diccionari = {
        "genero": "Gènere",
        "genere": "Gènere",
        "gender": "Gènere",
        "sexo": "Gènere",
        "sexe": "Gènere",
        "seleccion": "Gènere",
        "seleccio": "Gènere",

        "edad": "Edat",
        "edat": "Edat",
        "age": "Edat",
        "edad_anos": "Edat",
        "edad_años": "Edat",
        "edat_anys": "Edat",
        "edat_anos": "Edat",

        "peso_kg": "Pes corporal (kg)",
        "pes_kg": "Pes corporal (kg)",
        "peso": "Pes corporal (kg)",
        "pes": "Pes corporal (kg)",

        "altura_corporal_cm": "Alçada corporal (cm)",
        "alcada_corporal_cm": "Alçada corporal (cm)",
        "altura_cm": "Alçada corporal (cm)",
        "alcada_cm": "Alçada corporal (cm)",

        "raza": "Raça",
        "raca": "Raça",
        "raça": "Raça",

        "min_entreno_fisico": "Indica els minuts d'entreno físic setmanal",
        "minuts_entreno_fisic": "Indica els minuts d'entreno físic setmanal",
        "minutos_entreno_fisico": "Indica els minuts d'entreno físic setmanal",
        "entreno_fisico": "Indica els minuts d'entreno físic setmanal",
        "entrenament_fisic": "Indica els minuts d'entreno físic setmanal",

        "min_entreno_pista": "Indica els minuts d'entreno a pista setmanal",
        "minuts_entreno_pista": "Indica els minuts d'entreno a pista setmanal",
        "minutos_entreno_pista": "Indica els minuts d'entreno a pista setmanal",
        "entreno_pista": "Indica els minuts d'entreno a pista setmanal",
        "entrenament_pista": "Indica els minuts d'entreno a pista setmanal",

        "x1_partido": "Minuts jugats al partit",
        "x1_partit": "Minuts jugats al partit",
        "minuts_partit": "Minuts jugats al partit",
        "minutos_partido": "Minuts jugats al partit",

        "x2_partidos": "Minuts jugats al segon partit (si no hi ha indica 0)",
        "x2_partits": "Minuts jugats al segon partit (si no hi ha indica 0)",
        "minuts_segon_partit": "Minuts jugats al segon partit (si no hi ha indica 0)",
        "minutos_segundo_partido": "Minuts jugats al segon partit (si no hi ha indica 0)",

        "perc_fatiga": "Nivell de fatiga (habitualment)",
        "fatiga": "Nivell de fatiga (habitualment)",
        "percepcio_fatiga": "Percentatge de fatiga després de l'última competició o entrenament",
        "percepcion_fatiga": "Percentatge de fatiga després de l'última competició o entrenament",
        "cual_es_tu_percepcion_de_fatiga_despues_de_la_ultima_competicion_entrenamiento_de_la_semana":
            "Percentatge de fatiga després de l'última competició o entrenament",

        "lesiones_previas": "Ha tingut lesions prèvies?",
        "lesions_previes": "Ha tingut lesions prèvies?",

        "loc_tobillo": "Lesió al turmell",
        "loc_turmell": "Lesió al turmell",

        "loc_rodilla": "Lesió al genoll",
        "loc_genoll": "Lesió al genoll",

        "loc_brazo": "Lesió al braç",
        "loc_brac": "Lesió al braç",

        "loc_hombro_clavicula": "Lesió a l'espatlla o clavícula",
        "loc_espatlla_clavicula": "Lesió a l'espatlla o clavícula",

        "loc_columna_lumbar": "Lesió a la columna lumbar",
        "loc_cara": "Lesió a la cara",
        "loc_dorso": "Lesió al dors",

        "est_ligamento": "Afectació de lligament",
        "est_lligament": "Afectació de lligament",

        "est_menisco": "Afectació de menisc",
        "est_menisc": "Afectació de menisc",

        "est_hueso": "Afectació òssia",
        "est_os": "Afectació òssia",

        "est_musculo": "Afectació muscular",
        "est_muculo": "Afectació muscular",
        "est_muscle": "Afectació muscular",
        "est_muscul": "Afectació muscular",
    }

    if nom_norm in diccionari:
        return diccionari[nom_norm]

    return str(nom).replace("_", " ").capitalize()


def format_opcio(nom_variable, valor):
    valor_str = str(valor)
    valor_norm = normalitzar_nom(valor_str)
    nom_norm = normalitzar_nom(nom_variable)

    if nom_norm in ["genero", "genere", "gender", "sexo", "sexe", "seleccion", "seleccio"]:
        if valor_norm in ["masculina", "masculino", "masculi", "masculí", "home", "hombre"]:
            return "Masculí"
        if valor_norm in ["femenina", "femenino", "femeni", "femení", "dona", "mujer"]:
            return "Femení"

    if nom_norm in ["raza", "raca", "raça"]:
        mapa_raca = {
            "africana": "Africana",
            "africano": "Africana",
            "africa": "Africana",
            "africà": "Africana",

            "afrodescendent": "Afrodescendent",
            "afrodescendiente": "Afrodescendent",

            "caucasica": "Caucàsica/europea",
            "caucasico": "Caucàsica/europea",
            "caucasica/europea": "Caucàsica/europea",
            "caucasico/europeo": "Caucàsica/europea",
            "caucàsica": "Caucàsica/europea",
            "caucàsica/europea": "Caucàsica/europea",
            "blanca": "Caucàsica/europea",
            "blanco": "Caucàsica/europea",
            "europea": "Caucàsica/europea",
            "europeo": "Caucàsica/europea",

            "asiatica": "Asiàtica",
            "asiàtica": "Asiàtica",
            "asiatico": "Asiàtica",

            "llatina": "Llatinoamericana",
            "latina": "Llatinoamericana",
            "latino": "Llatinoamericana",
            "latinoamericana": "Llatinoamericana",
            "latinoamericano": "Llatinoamericana",
            "llatinoamericana": "Llatinoamericana",
            "llatinoamericà": "Llatinoamericana",

            "altres": "Altres",
            "otros": "Altres",
            "otro": "Altres",
            "altra": "Altres",
        }

        return mapa_raca.get(valor_norm, valor_str)

    if nom_norm.startswith("loc_") or nom_norm.startswith("est_"):
        if valor_str == "0":
            return "No"
        if valor_str == "1":
            return "Sí"

    return valor_str


def es_variable_localitzacio_lesio(nom):
    nom_norm = normalitzar_nom(nom)
    return nom_norm.startswith("loc_")


def es_variable_afectacio(nom):
    nom_norm = normalitzar_nom(nom)
    return nom_norm.startswith("est_")


def valor_binari_model(var_meta, seleccionat):
    levels = var_meta.get("levels", None)

    if levels is not None:
        levels_str = [str(x) for x in levels]

        if "0" in levels_str and "1" in levels_str:
            return "1" if seleccionat else "0"

        if "No" in levels_str and "Sí" in levels_str:
            return "Sí" if seleccionat else "No"

        if "No" in levels_str and "Si" in levels_str:
            return "Si" if seleccionat else "No"

    return "1" if seleccionat else "0"


def es_variable_fatiga_slider(nom):
    nom_norm = normalitzar_nom(nom)

    variables_slider = [
        "perc_fatiga",
        "fatiga",
        "percepcio_fatiga",
        "percepcion_fatiga",
        "cual_es_tu_percepcion_de_fatiga_despues_de_la_ultima_competicion_entrenamiento_de_la_semana",
    ]

    return nom_norm in variables_slider or "fatiga" in nom_norm


def descripcio_cluster(cluster):
    descripcions = {
        "Cluster1": (
            "Aquest resultat correspon a un perfil de baixa presència lesiva. "
            "S'associa amb absència o menor presència de lesions prèvies i menor "
            "afectació en localitzacions corporals."
        ),
        "Cluster2": (
            "Aquest resultat correspon al perfil de major risc. "
            "S'associa amb més presència de lesions prèvies i amb afectacions "
            "en diferents zones corporals o estructures."
        ),
        "Cluster3": (
            "Aquest resultat correspon a un perfil intermedi o mixt. "
            "No mostra un patró tan extrem com el perfil de risc alt, però tampoc "
            "es pot considerar completament equivalent al grup de baixa presència lesiva."
        ),
    }

    return descripcions.get(
        str(cluster),
        "No hi ha descripció disponible per a aquest clúster."
    )


def nivell_risc(cluster):
    cluster = str(cluster)

    if cluster == "Cluster1":
        return "Baix", "Verd"

    if cluster == "Cluster2":
        return "Alt", "Vermell"

    if cluster == "Cluster3":
        return "Mitjà", "Groc"

    return "No disponible", "Gris"


def recomanacio_breu(cluster):
    cluster = str(cluster)

    if cluster == "Cluster1":
        return (
            "Mantenir la planificació actual, continuar amb prevenció, controlar la fatiga "
            "i registrar qualsevol molèstia nova."
        )

    if cluster == "Cluster2":
        return (
            "Revisar càrrega, valorar seguiment individualitzat, reforçar prevenció específica "
            "i consultar amb personal tècnic o sanitari."
        )

    if cluster == "Cluster3":
        return (
            "Fer seguiment de la fatiga i molèsties, ajustar càrrega si cal i reforçar "
            "treball preventiu."
        )

    return "No hi ha recomanació disponible."


def recomanacions_cluster(cluster):
    cluster = str(cluster)

    if cluster == "Cluster1":
        return """
        <div class="recommendation-card">
        <div class="risk-title">Recomanacions per al perfil de risc baix</div>
        <ul>
            <li>Mantenir la planificació actual d'entrenament i competició si no apareixen molèsties.</li>
            <li>Continuar amb rutines d'escalfament, mobilitat i prevenció abans de cada sessió.</li>
            <li>Controlar periòdicament la percepció de fatiga, especialment després dels partits.</li>
            <li>Respectar els temps de descans i recuperació entre entrenaments i competicions.</li>
            <li>Registrar qualsevol molèstia nova encara que sigui lleu, per evitar que evolucioni.</li>
            <li>Mantenir hàbits de son, hidratació i alimentació adequats per afavorir la recuperació.</li>
        </ul>
        </div>
        """

    if cluster == "Cluster2":
        return """
        <div class="recommendation-card">
        <div class="risk-title">Recomanacions per al perfil de risc alt</div>
        <ul>
            <li>Revisar la càrrega total d'entrenament i els minuts acumulats de competició.</li>
            <li>Fer una valoració individualitzada amb el cos tècnic, preparador físic o personal sanitari.</li>
            <li>Prioritzar treball preventiu específic segons la zona afectada o l'antecedent lesiu.</li>
            <li>Evitar increments bruscos de càrrega i retorns precipitats després d'una lesió.</li>
            <li>Monitoritzar dolor, fatiga i sensació de sobrecàrrega abans i després de cada sessió.</li>
            <li>Valorar una reducció temporal de càrrega o adaptació de l'entrenament si hi ha símptomes persistents.</li>
        </ul>
        </div>
        """

    if cluster == "Cluster3":
        return """
        <div class="recommendation-card">
        <div class="risk-title">Recomanacions per al perfil de risc mitjà</div>
        <ul>
            <li>Fer seguiment regular de la fatiga i de possibles molèsties durant la setmana.</li>
            <li>Ajustar la càrrega si apareixen signes de sobrecàrrega o acumulació de minuts.</li>
            <li>Reforçar exercicis compensatoris, de força i de control motor.</li>
            <li>Controlar especialment els minuts acumulats entre entrenaments físics, pista i partits.</li>
            <li>Reavaluar el perfil si apareix una nova lesió o si augmenta la càrrega competitiva.</li>
            <li>Aplicar mesures preventives abans que el perfil evolucioni cap a una situació de risc alt.</li>
        </ul>
        </div>
        """

    return """
    <div class="recommendation-card">
    No hi ha recomanacions disponibles per a aquest resultat.
    </div>
    """


def semafor_risc(cluster):
    risc, color = nivell_risc(cluster)

    active_red = color == "Vermell"
    active_yellow = color == "Groc"
    active_green = color == "Verd"

    red_style = "#EF4444" if active_red else "#4A1F1F"
    yellow_style = "#FACC15" if active_yellow else "#4A3A12"
    green_style = "#22C55E" if active_green else "#12351F"

    shadow_red = "0 0 16px #EF4444" if active_red else "none"
    shadow_yellow = "0 0 16px #FACC15" if active_yellow else "none"
    shadow_green = "0 0 16px #22C55E" if active_green else "none"

    if color == "Vermell":
        etiqueta_color = "#EF4444"
    elif color == "Groc":
        etiqueta_color = "#CA8A04"
    elif color == "Verd":
        etiqueta_color = "#16A34A"
    else:
        etiqueta_color = "#6B7280"

    return f"""
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: Arial, sans-serif;
                background: transparent;
                overflow: hidden;
            }}

            .semafor-wrapper {{
                display: flex;
                align-items: center;
                gap: 1.2rem;
                padding: 1rem;
                border: 1px solid #E5E7EB;
                border-radius: 0.9rem;
                background: #FAFAFA;
                box-sizing: border-box;
                width: 100%;
                min-height: 150px;
            }}

            .semafor-box {{
                width: 58px;
                padding: 9px;
                border-radius: 20px;
                background: #111827;
                display: flex;
                flex-direction: column;
                gap: 8px;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
                margin-left: 2px;
            }}

            .llum {{
                width: 29px;
                height: 29px;
                border-radius: 50%;
            }}

            .text-risc {{
                font-size: 1.25rem;
                font-weight: 800;
                color: {etiqueta_color};
            }}

            .text-cluster {{
                color: #4B5563;
                margin-top: 0.25rem;
                font-size: 0.95rem;
            }}
        </style>
    </head>

    <body>
        <div class="semafor-wrapper">
            <div class="semafor-box">
                <div class="llum" style="background:{red_style}; box-shadow:{shadow_red};"></div>
                <div class="llum" style="background:{yellow_style}; box-shadow:{shadow_yellow};"></div>
                <div class="llum" style="background:{green_style}; box-shadow:{shadow_green};"></div>
            </div>

            <div>
                <div class="text-risc">Semàfor de risc: {risc}</div>
                <div class="text-cluster">Resultat associat a <b>{cluster}</b></div>
            </div>
        </div>
    </body>
    </html>
    """


def executar_r(script_text, args):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        script_path = tmpdir / "script_temporal.R"

        script_path.write_text(script_text, encoding="utf-8")

        cmd = [RSCRIPT, str(script_path)] + [str(a) for a in args]

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )

        return result


# ==============================================================================
# CODI R: METADADES DEL MODEL
# ==============================================================================

R_METADATA_CODE = textwrap.dedent(
    r"""
    args <- commandArgs(trailingOnly = TRUE)

    if (length(args) < 2) {
      stop("Ús: Rscript metadata.R model_path output_json [data_path]")
    }

    model_path <- args[1]
    output_json <- args[2]

    if (length(args) >= 3) {
      data_path <- args[3]
    } else {
      data_path <- ""
    }

    local_lib <- file.path(getwd(), "r_packages")

    if (!dir.exists(local_lib)) {
      dir.create(local_lib, recursive = TRUE)
    }

    .libPaths(c(local_lib, .libPaths()))

    paquets <- c("jsonlite", "caret", "xgboost")

    for (pkg in paquets) {
      if (!requireNamespace(pkg, quietly = TRUE)) {
        install.packages(
          pkg,
          lib = local_lib,
          repos = "https://cloud.r-project.org",
          dependencies = TRUE
        )
      }
    }

    library(jsonlite)
    library(caret)
    library(xgboost)

    modelo_xgboost <- readRDS(model_path)

    dummy_model <- modelo_xgboost$dummy_model

    if (is.null(dummy_model)) {
      stop("El model no conté dummy_model.")
    }

    vars_raw <- dummy_model$vars

    if (is.null(vars_raw)) {
      vars_raw <- all.vars(dummy_model$terms)
    }

    lvls <- dummy_model$lvls

    meta <- list()

    for (v in vars_raw) {

      item <- list(
        name = v,
        type = "numeric",
        levels = NULL,
        min = NULL,
        max = NULL,
        median = NULL
      )

      if (!is.null(lvls) && !is.null(lvls[[v]])) {
        item$type <- "categorical"
        item$levels <- as.character(lvls[[v]])
      }

      meta[[length(meta) + 1]] <- item
    }

    if (!is.null(data_path) && file.exists(data_path)) {

      dades <- readRDS(data_path)

      if (is.list(dades) && !is.null(dades$train)) {
        dades <- dades$train
      }

      for (i in seq_along(meta)) {

        v <- meta[[i]]$name

        if (v %in% names(dades)) {

          x <- dades[[v]]

          if (meta[[i]]$type == "categorical") {

            if (is.null(meta[[i]]$levels)) {
              meta[[i]]$levels <- as.character(unique(x[!is.na(x)]))
            }

          } else if (is.numeric(x) || is.integer(x)) {

            x_num <- as.numeric(x)
            x_num <- x_num[is.finite(x_num)]

            if (length(x_num) > 0) {
              meta[[i]]$min <- min(x_num, na.rm = TRUE)
              meta[[i]]$max <- max(x_num, na.rm = TRUE)
              meta[[i]]$median <- median(x_num, na.rm = TRUE)
            }

          } else if (is.factor(x) || is.character(x)) {

            meta[[i]]$type <- "categorical"
            meta[[i]]$levels <- as.character(unique(x[!is.na(x)]))
          }
        }
      }
    }

    out <- list(
      variables = meta,
      variable_names = vars_raw
    )

    writeLines(
      jsonlite::toJSON(out, auto_unbox = TRUE, pretty = TRUE, na = "null"),
      output_json
    )
    """
)


@st.cache_data(show_spinner=False)
def obtenir_metadata_model(model_path_str, data_path_str):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_json = tmpdir / "metadata.json"

        args = [model_path_str, output_json]

        if data_path_str is not None:
            args.append(data_path_str)

        result = executar_r(R_METADATA_CODE, args)

        if result.returncode != 0:
            missatge = result.stderr
            if result.stdout:
                missatge += "\n\nSortida R:\n" + result.stdout
            raise RuntimeError(missatge)

        with open(output_json, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return metadata


# ==============================================================================
# CODI R: PREDICCIÓ
# ==============================================================================

R_PREDICT_CODE = textwrap.dedent(
    r"""
    args <- commandArgs(trailingOnly = TRUE)

    if (length(args) < 3) {
      stop("Ús: Rscript predict.R input_csv output_csv model_path")
    }

    input_csv <- args[1]
    output_csv <- args[2]
    model_path <- args[3]

    local_lib <- file.path(getwd(), "r_packages")

    if (!dir.exists(local_lib)) {
      dir.create(local_lib, recursive = TRUE)
    }

    .libPaths(c(local_lib, .libPaths()))

    paquets <- c("caret", "xgboost")

    for (pkg in paquets) {
      if (!requireNamespace(pkg, quietly = TRUE)) {
        install.packages(
          pkg,
          lib = local_lib,
          repos = "https://cloud.r-project.org",
          dependencies = TRUE
        )
      }
    }

    library(caret)
    library(xgboost)

    modelo_xgboost <- readRDS(model_path)

    model <- modelo_xgboost$model
    dummy_model <- modelo_xgboost$dummy_model
    nivells_k3 <- modelo_xgboost$nivells_k3
    variables_train <- modelo_xgboost$variables_train

    if (is.null(model) || is.null(dummy_model) || is.null(nivells_k3) || is.null(variables_train)) {
      stop("El fitxer RDS no té l'estructura esperada.")
    }

    vars_raw <- dummy_model$vars

    if (is.null(vars_raw)) {
      vars_raw <- all.vars(dummy_model$terms)
    }

    lvls <- dummy_model$lvls

    newdata_original <- read.csv(
      input_csv,
      stringsAsFactors = FALSE,
      check.names = FALSE,
      fileEncoding = "UTF-8"
    )

    vars_falten <- setdiff(vars_raw, names(newdata_original))

    if (length(vars_falten) > 0) {
      stop(
        paste0(
          "Falten variables al CSV d'entrada: ",
          paste(vars_falten, collapse = ", ")
        )
      )
    }

    newdata <- newdata_original[, vars_raw, drop = FALSE]

    for (v in vars_raw) {

      if (!is.null(lvls) && !is.null(lvls[[v]])) {

        nivells_v <- as.character(lvls[[v]])
        valors <- as.character(newdata[[v]])

        valors[valors == ""] <- NA
        valors <- gsub("\\.0$", "", valors)

        valors_no_valids <- setdiff(unique(na.omit(valors)), nivells_v)

        if (length(valors_no_valids) > 0) {
          stop(
            paste0(
              "La variable '", v, "' té valors no vàlids: ",
              paste(valors_no_valids, collapse = ", "),
              ". Valors acceptats: ",
              paste(nivells_v, collapse = ", ")
            )
          )
        }

        newdata[[v]] <- factor(valors, levels = nivells_v)

      } else {

        valors <- as.character(newdata[[v]])
        valors <- gsub(",", ".", valors)
        valors[valors == ""] <- NA

        valors_num <- suppressWarnings(as.numeric(valors))

        mediana <- median(valors_num, na.rm = TRUE)

        if (is.na(mediana)) {
          mediana <- 0
        }

        valors_num[is.na(valors_num)] <- mediana

        newdata[[v]] <- valors_num
      }
    }

    x_new <- predict(dummy_model, newdata = newdata)
    x_new <- as.data.frame(x_new)

    names(x_new) <- make.names(names(x_new), unique = TRUE)

    vars_falten_post <- setdiff(variables_train, names(x_new))

    if (length(vars_falten_post) > 0) {
      for (v in vars_falten_post) {
        x_new[[v]] <- 0
      }
    }

    x_new <- x_new[, variables_train, drop = FALSE]

    x_new[] <- lapply(x_new, as.numeric)

    x_new[] <- lapply(x_new, function(x) {
      x[!is.finite(x)] <- NA
      x[is.na(x)] <- 0
      return(x)
    })

    dnew <- xgb.DMatrix(data = as.matrix(x_new))

    pred_raw <- predict(model, newdata = dnew)

    num_class <- length(nivells_k3)

    if (length(pred_raw) == nrow(x_new)) {

      pred_num <- as.integer(round(pred_raw))
      probs <- NULL

    } else {

      pred_prob_matrix <- matrix(
        pred_raw,
        ncol = num_class,
        byrow = TRUE
      )

      pred_num <- max.col(pred_prob_matrix) - 1

      probs <- as.data.frame(pred_prob_matrix)
      colnames(probs) <- paste0("prob_", nivells_k3)
    }

    pred <- factor(
      nivells_k3[pred_num + 1],
      levels = nivells_k3
    )

    resultats <- newdata_original
    resultats$prediccio_cluster <- pred

    if (!is.null(probs)) {
      resultats <- cbind(resultats, probs)
    }

    write.csv(
      resultats,
      output_csv,
      row.names = FALSE,
      fileEncoding = "UTF-8"
    )
    """
)


# ==============================================================================
# CODI R: DADES PER AL RADIAL PLOT
# ==============================================================================

R_RADIAL_CODE = textwrap.dedent(
    r"""
    args <- commandArgs(trailingOnly = TRUE)

    if (length(args) < 4) {
      stop("Ús: Rscript radial.R input_csv data_path cluster_pred output_csv")
    }

    input_csv <- args[1]
    data_path <- args[2]
    cluster_pred <- args[3]
    output_csv <- args[4]

    dades <- readRDS(data_path)

    if (is.list(dades) && !is.null(dades$train)) {
      dades <- dades$train
    }

    if (!"k3" %in% names(dades)) {
      stop("No s'ha trobat la variable k3 a les dades de modelització.")
    }

    obs <- read.csv(
      input_csv,
      stringsAsFactors = FALSE,
      check.names = FALSE,
      fileEncoding = "UTF-8"
    )

    dades$k3 <- as.character(dades$k3)

    dades_cluster <- dades[dades$k3 == cluster_pred, , drop = FALSE]

    if (nrow(dades_cluster) == 0) {
      stop(paste0("No hi ha observacions del clúster ", cluster_pred, " a les dades train."))
    }

    convertir_numeric <- function(x) {
      x <- as.character(x)
      x <- trimws(x)
      x <- tolower(x)

      x <- gsub("á", "a", x)
      x <- gsub("à", "a", x)
      x <- gsub("é", "e", x)
      x <- gsub("è", "e", x)
      x <- gsub("í", "i", x)
      x <- gsub("ï", "i", x)
      x <- gsub("ó", "o", x)
      x <- gsub("ò", "o", x)
      x <- gsub("ú", "u", x)
      x <- gsub("ü", "u", x)
      x <- gsub("ç", "c", x)
      x <- gsub("ñ", "n", x)

      x[x %in% c("si", "sí", "s", "yes", "true", "1")] <- "1"
      x[x %in% c("no", "n", "false", "0")] <- "0"

      x <- gsub(",", ".", x)
      suppressWarnings(as.numeric(x))
    }

    variables_excloses <- c(
      "k3",
      "genero", "genere", "gender", "sexo", "sexe",
      "seleccion", "seleccio",
      "raza", "raca", "raça"
    )

    vars <- intersect(names(obs), names(dades))
    vars <- setdiff(vars, variables_excloses)

    resultats <- data.frame(
      variable = character(),
      observacio_original = numeric(),
      mitjana_cluster_original = numeric(),
      observacio_norm = numeric(),
      mitjana_cluster_norm = numeric(),
      stringsAsFactors = FALSE
    )

    for (v in vars) {

      train_num <- convertir_numeric(dades[[v]])
      cluster_num <- convertir_numeric(dades_cluster[[v]])
      obs_num <- convertir_numeric(obs[[v]][1])

      if (all(is.na(train_num))) {
        next
      }

      if (length(obs_num) == 0 || is.na(obs_num)) {
        next
      }

      min_v <- min(train_num, na.rm = TRUE)
      max_v <- max(train_num, na.rm = TRUE)

      if (!is.finite(min_v) || !is.finite(max_v)) {
        next
      }

      if (min_v == max_v) {
        next
      }

      mitjana_cluster <- mean(cluster_num, na.rm = TRUE)

      if (!is.finite(mitjana_cluster)) {
        next
      }

      obs_norm <- (obs_num - min_v) / (max_v - min_v)
      mitjana_norm <- (mitjana_cluster - min_v) / (max_v - min_v)

      obs_norm <- max(0, min(1, obs_norm))
      mitjana_norm <- max(0, min(1, mitjana_norm))

      fila <- data.frame(
        variable = v,
        observacio_original = obs_num,
        mitjana_cluster_original = mitjana_cluster,
        observacio_norm = obs_norm,
        mitjana_cluster_norm = mitjana_norm,
        stringsAsFactors = FALSE
      )

      resultats <- rbind(resultats, fila)
    }

    write.csv(
      resultats,
      output_csv,
      row.names = FALSE,
      fileEncoding = "UTF-8"
    )
    """
)


# ==============================================================================
# CODI R: TOP 10 CASOS MÉS SEMBLANTS DINS DEL CLÚSTER PREDIT
# ==============================================================================

R_TOP10_CODE = textwrap.dedent(
    r"""
    args <- commandArgs(trailingOnly = TRUE)

    if (length(args) < 4) {
      stop("Ús: Rscript top10.R input_csv data_path cluster_pred output_csv")
    }

    input_csv <- args[1]
    data_path <- args[2]
    cluster_pred <- args[3]
    output_csv <- args[4]

    dades <- readRDS(data_path)

    if (is.list(dades) && !is.null(dades$train)) {
      dades <- dades$train
    }

    if (!"k3" %in% names(dades)) {
      stop("No s'ha trobat la variable k3 a les dades de modelització.")
    }

    obs <- read.csv(
      input_csv,
      stringsAsFactors = FALSE,
      check.names = FALSE,
      fileEncoding = "UTF-8"
    )

    dades$k3 <- as.character(dades$k3)

    dades_cluster <- dades[dades$k3 == cluster_pred, , drop = FALSE]

    if (nrow(dades_cluster) == 0) {
      stop(paste0("No hi ha observacions del clúster ", cluster_pred, " a les dades train."))
    }

    convertir_numeric <- function(x) {
      x <- as.character(x)
      x <- trimws(x)
      x <- tolower(x)

      x <- gsub("á", "a", x)
      x <- gsub("à", "a", x)
      x <- gsub("é", "e", x)
      x <- gsub("è", "e", x)
      x <- gsub("í", "i", x)
      x <- gsub("ï", "i", x)
      x <- gsub("ó", "o", x)
      x <- gsub("ò", "o", x)
      x <- gsub("ú", "u", x)
      x <- gsub("ü", "u", x)
      x <- gsub("ç", "c", x)
      x <- gsub("ñ", "n", x)

      x[x %in% c("si", "sí", "s", "yes", "true", "1")] <- "1"
      x[x %in% c("no", "n", "false", "0")] <- "0"

      x <- gsub(",", ".", x)
      suppressWarnings(as.numeric(x))
    }

    variables_excloses_distancia <- c(
      "k3",
      "genero", "genere", "gender", "sexo", "sexe",
      "seleccion", "seleccio",
      "raza", "raca", "raça"
    )

    vars_distancia <- intersect(names(obs), names(dades))
    vars_distancia <- setdiff(vars_distancia, variables_excloses_distancia)

    vars_mostrar <- intersect(names(obs), names(dades))
    vars_mostrar <- setdiff(vars_mostrar, "k3")

    vars_valides <- c()
    mat_cluster_list <- list()
    obs_norm <- c()

    for (v in vars_distancia) {

      train_num <- convertir_numeric(dades[[v]])
      cluster_num <- convertir_numeric(dades_cluster[[v]])
      obs_num <- convertir_numeric(obs[[v]][1])

      if (all(is.na(train_num))) {
        next
      }

      if (length(obs_num) == 0 || is.na(obs_num)) {
        next
      }

      min_v <- min(train_num, na.rm = TRUE)
      max_v <- max(train_num, na.rm = TRUE)

      if (!is.finite(min_v) || !is.finite(max_v)) {
        next
      }

      if (min_v == max_v) {
        next
      }

      cluster_norm <- (cluster_num - min_v) / (max_v - min_v)
      obs_v_norm <- (obs_num - min_v) / (max_v - min_v)

      cluster_norm[cluster_norm < 0] <- 0
      cluster_norm[cluster_norm > 1] <- 1

      obs_v_norm <- max(0, min(1, obs_v_norm))

      mediana_cluster <- median(cluster_norm, na.rm = TRUE)

      if (is.na(mediana_cluster)) {
        mediana_cluster <- 0
      }

      cluster_norm[is.na(cluster_norm)] <- mediana_cluster

      mat_cluster_list[[v]] <- cluster_norm
      obs_norm <- c(obs_norm, obs_v_norm)
      vars_valides <- c(vars_valides, v)
    }

    if (length(vars_valides) == 0) {
      stop("No hi ha variables numèriques vàlides per calcular similituds.")
    }

    mat_cluster <- as.data.frame(mat_cluster_list)
    names(obs_norm) <- vars_valides

    distancies <- apply(
      mat_cluster,
      1,
      function(x) {
        sqrt(mean((x - obs_norm)^2, na.rm = TRUE))
      }
    )

    similitud <- 1 - distancies

    similitud[similitud < 0] <- 0
    similitud[similitud > 1] <- 1

    ordre <- order(similitud, decreasing = TRUE)
    n_top <- min(10, length(ordre))
    idx_top <- ordre[seq_len(n_top)]

    resultat_base <- data.frame(
      cas_train = rownames(dades_cluster)[idx_top],
      cluster = cluster_pred,
      similitud = round(similitud[idx_top], 4),
      similitud_percentatge = round(similitud[idx_top] * 100, 2),
      distancia = round(distancies[idx_top], 4),
      n_variables_comparades = length(vars_valides),
      stringsAsFactors = FALSE
    )

    valors_casos <- dades_cluster[idx_top, vars_mostrar, drop = FALSE]
    valors_casos <- as.data.frame(lapply(valors_casos, as.character), stringsAsFactors = FALSE)
    names(valors_casos) <- paste0("cas_", names(valors_casos))

    valors_introduits <- data.frame(.fila_temp = seq_len(n_top))

    for (v in vars_mostrar) {
      valors_introduits[[paste0("introduit_", v)]] <- rep(as.character(obs[[v]][1]), n_top)
    }

    valors_introduits$.fila_temp <- NULL

    resultat <- cbind(
      resultat_base,
      valors_casos,
      valors_introduits
    )

    rownames(resultat) <- NULL

    write.csv(
      resultat,
      output_csv,
      row.names = FALSE,
      fileEncoding = "UTF-8"
    )
    """
)


# ==============================================================================
# FUNCIONS DE PREDICCIÓ I VISUALITZACIÓ
# ==============================================================================

def predir_amb_model(df_input):
    if MODEL_PATH is None:
        raise FileNotFoundError("No s'ha trobat modeloXGBoost.RDS.")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        input_csv = tmpdir / "input.csv"
        output_csv = tmpdir / "output.csv"

        df_input.to_csv(input_csv, index=False, encoding="utf-8")

        result = executar_r(
            R_PREDICT_CODE,
            [input_csv, output_csv, MODEL_PATH]
        )

        if result.returncode != 0:
            missatge = result.stderr
            if result.stdout:
                missatge += "\n\nSortida R:\n" + result.stdout
            raise RuntimeError(missatge)

        resultats = pd.read_csv(output_csv)

        if "prediccio_cluster" in resultats.columns:
            resultats["nivell_risc"] = resultats["prediccio_cluster"].apply(
                lambda x: nivell_risc(x)[0]
            )
            resultats["semafor"] = resultats["prediccio_cluster"].apply(
                lambda x: nivell_risc(x)[1]
            )
            resultats["recomanacio"] = resultats["prediccio_cluster"].apply(
                recomanacio_breu
            )

        return resultats


def seleccionar_columnes_resultat(df):
    columnes_resultat = [
        "prediccio_cluster",
        "nivell_risc",
        "semafor",
        "recomanacio"
    ]

    columnes_existents = [
        col for col in columnes_resultat
        if col in df.columns
    ]

    return df[columnes_existents]


def obtenir_dades_radial(df_input, cluster_pred):
    if DATA_PATH is None:
        raise FileNotFoundError(
            "No s'ha trobat datos_modelling.RDS. "
            "És necessari per calcular la mitjana del clúster."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        input_csv = tmpdir / "input_radial.csv"
        output_csv = tmpdir / "radial_output.csv"

        df_input.to_csv(input_csv, index=False, encoding="utf-8")

        result = executar_r(
            R_RADIAL_CODE,
            [input_csv, DATA_PATH, cluster_pred, output_csv]
        )

        if result.returncode != 0:
            missatge = result.stderr
            if result.stdout:
                missatge += "\n\nSortida R:\n" + result.stdout
            raise RuntimeError(missatge)

        dades_radial = pd.read_csv(output_csv)

        if dades_radial.empty:
            return dades_radial

        dades_radial["variable_mostrar"] = dades_radial["variable"].apply(
            netejar_nom_variable
        )

        dades_radial["diferencia"] = (
            dades_radial["observacio_norm"] -
            dades_radial["mitjana_cluster_norm"]
        ).abs()

        return dades_radial


VARIABLES_PRIORITARIES_RADIAL = [
    "perc_fatiga",
    "fatiga",
    "percepcio_fatiga",
    "percepcion_fatiga",
    "cual_es_tu_percepcion_de_fatiga_despues_de_la_ultima_competicion_entrenamiento_de_la_semana",

    "min_entreno_fisico",
    "minuts_entreno_fisic",
    "minutos_entreno_fisico",
    "entreno_fisico",
    "entrenament_fisic",

    "min_entreno_pista",
    "minuts_entreno_pista",
    "minutos_entreno_pista",
    "entreno_pista",
    "entrenament_pista",

    "x1_partido",
    "x1_partit",
    "minuts_partit",
    "minutos_partido",

    "x2_partidos",
    "x2_partits",
    "minuts_segon_partit",
    "minutos_segundo_partido",

    "lesiones_previas",
    "lesions_previes",

    "loc_tobillo",
    "loc_turmell",
    "loc_rodilla",
    "loc_genoll",
    "loc_brazo",
    "loc_brac",
    "loc_hombro_clavicula",
    "loc_espatlla_clavicula",
    "loc_columna_lumbar",
    "loc_cara",
    "loc_dorso",

    "est_ligamento",
    "est_lligament",
    "est_menisco",
    "est_menisc",
    "est_hueso",
    "est_os",
    "est_musculo",
    "est_muculo",
    "est_muscle",
    "est_muscul",
]


def seleccionar_variables_radial(dades_radial, max_variables=14):
    dades_plot = dades_radial.copy()

    dades_plot["variable_norm"] = dades_plot["variable"].apply(normalitzar_nom)

    variables_prioritaries_norm = [
        normalitzar_nom(v) for v in VARIABLES_PRIORITARIES_RADIAL
    ]

    dades_prioritaries = dades_plot[
        dades_plot["variable_norm"].isin(variables_prioritaries_norm)
    ].copy()

    ordre_prioritari = {
        v: i for i, v in enumerate(variables_prioritaries_norm)
    }

    dades_prioritaries["ordre"] = dades_prioritaries["variable_norm"].map(
        ordre_prioritari
    )

    dades_prioritaries = dades_prioritaries.sort_values(
        by=["ordre", "diferencia"],
        ascending=[True, False]
    )

    dades_restants = dades_plot[
        ~dades_plot["variable_norm"].isin(variables_prioritaries_norm)
    ].copy()

    dades_restants = dades_restants.sort_values(
        by="diferencia",
        ascending=False
    )

    dades_plot = pd.concat(
        [dades_prioritaries, dades_restants],
        axis=0
    ).head(max_variables)

    return dades_plot


def crear_radial_plot(dades_radial, cluster_pred, max_variables=14):
    dades_plot = seleccionar_variables_radial(dades_radial, max_variables)

    categories = dades_plot["variable_mostrar"].tolist()

    if len(categories) == 0:
        return None

    observacio = dades_plot["observacio_norm"].tolist()
    mitjana_cluster = dades_plot["mitjana_cluster_norm"].tolist()

    categories_tancat = categories + [categories[0]]
    observacio_tancat = observacio + [observacio[0]]
    mitjana_tancat = mitjana_cluster + [mitjana_cluster[0]]

    zona_alta = [1.00] * len(categories_tancat)
    zona_mitjana = [0.66] * len(categories_tancat)
    zona_baixa = [0.33] * len(categories_tancat)

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=zona_alta,
            theta=categories_tancat,
            fill="toself",
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            fillcolor="rgba(255, 99, 99, 0.18)",
            name="Zona alta",
            hoverinfo="skip"
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=zona_mitjana,
            theta=categories_tancat,
            fill="toself",
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            fillcolor="rgba(255, 220, 80, 0.28)",
            name="Zona mitjana",
            hoverinfo="skip"
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=zona_baixa,
            theta=categories_tancat,
            fill="toself",
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            fillcolor="rgba(90, 220, 140, 0.30)",
            name="Zona baixa",
            hoverinfo="skip"
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=mitjana_tancat,
            theta=categories_tancat,
            fill="toself",
            fillcolor="rgba(0, 0, 0, 0.10)",
            name=f"Mitjana {cluster_pred}",
            line=dict(
                color="black",
                width=4
            ),
            marker=dict(
                color="black",
                size=6
            )
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=observacio_tancat,
            theta=categories_tancat,
            fill="toself",
            fillcolor="rgba(255, 255, 255, 0.35)",
            name="Observació analitzada",
            line=dict(
                color="white",
                width=4
            ),
            marker=dict(
                color="white",
                size=7,
                line=dict(
                    color="black",
                    width=1
                )
            )
        )
    )

    fig.update_layout(
        title="Comparació radial amb la mitjana del clúster",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.33, 0.66, 1],
                ticktext=["Baix", "Mitjà", "Alt"]
            )
        ),
        showlegend=True,
        margin=dict(l=40, r=40, t=70, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    return fig


def crear_grafic_barres_comparacio(dades_radial, cluster_pred, max_variables=14):
    dades_plot = seleccionar_variables_radial(dades_radial, max_variables)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=dades_plot["variable_mostrar"],
            y=dades_plot["mitjana_cluster_norm"],
            name=f"Mitjana {cluster_pred}",
            marker=dict(color="black")
        )
    )

    fig.add_trace(
        go.Bar(
            x=dades_plot["variable_mostrar"],
            y=dades_plot["observacio_norm"],
            name="Observació analitzada",
            marker=dict(
                color="white",
                line=dict(
                    color="black",
                    width=1.5
                )
            )
        )
    )

    fig.update_layout(
        title="Comparació normalitzada observació vs clúster",
        xaxis_title="Variable",
        yaxis_title="Valor normalitzat",
        yaxis=dict(range=[0, 1]),
        barmode="group",

        # Llegenda apartada a la dreta
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1
        ),

        margin=dict(
            l=40,
            r=220,
            t=70,
            b=150
        )
    )

    fig.update_xaxes(tickangle=45)

    return fig


def crear_grafic_diferencies(dades_radial, max_variables=14):
    dades_plot = seleccionar_variables_radial(dades_radial, max_variables)

    dades_plot = dades_plot.copy()
    dades_plot["diferencia_signada"] = (
        dades_plot["observacio_norm"] -
        dades_plot["mitjana_cluster_norm"]
    )

    dades_plot = dades_plot.sort_values(
        by="diferencia_signada",
        ascending=True
    )

    colors = [
        "#EF4444" if x > 0 else "#3B82F6"
        for x in dades_plot["diferencia_signada"]
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=dades_plot["diferencia_signada"],
            y=dades_plot["variable_mostrar"],
            orientation="h",
            marker=dict(color=colors),
            name="Diferència"
        )
    )

    fig.add_vline(
        x=0,
        line_width=2,
        line_dash="dash",
        line_color="black"
    )

    fig.update_layout(
        title="Diferència respecte a la mitjana del clúster",
        xaxis_title="Observació - mitjana del clúster",
        yaxis_title="Variable",
        margin=dict(l=40, r=40, t=70, b=40),
        showlegend=False
    )

    return fig


def obtenir_top10_similars(df_input, cluster_pred):
    if DATA_PATH is None:
        raise FileNotFoundError(
            "No s'ha trobat datos_modelling.RDS. "
            "És necessari per calcular els casos més semblants."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        input_csv = tmpdir / "input_top10.csv"
        output_csv = tmpdir / "top10_output.csv"

        df_input.to_csv(input_csv, index=False, encoding="utf-8")

        result = executar_r(
            R_TOP10_CODE,
            [input_csv, DATA_PATH, cluster_pred, output_csv]
        )

        if result.returncode != 0:
            missatge = result.stderr
            if result.stdout:
                missatge += "\n\nSortida R:\n" + result.stdout
            raise RuntimeError(missatge)

        top10 = pd.read_csv(output_csv)

        return top10


# ==============================================================================
# FALLBACK SI NO ES PODEN EXTREURE METADADES
# ==============================================================================

FALLBACK_METADATA = {
    "variables": [
        {"name": "genero", "type": "categorical", "levels": ["Femenina", "Masculina"], "min": None, "max": None, "median": None},
        {"name": "edad", "type": "numeric", "levels": None, "min": 10, "max": 60, "median": 20},
        {"name": "peso_kg", "type": "numeric", "levels": None, "min": 40, "max": 120, "median": 70},
        {"name": "altura_corporal_cm", "type": "numeric", "levels": None, "min": 140, "max": 220, "median": 175},
        {"name": "raza", "type": "categorical", "levels": ["Africana", "Caucàsica/europea", "Asiàtica", "Llatinoamericana", "Afrodescendent", "Altres"], "min": None, "max": None, "median": None},
        {"name": "min_entreno_fisico", "type": "numeric", "levels": None, "min": 0, "max": 1000, "median": 180},
        {"name": "min_entreno_pista", "type": "numeric", "levels": None, "min": 0, "max": 1000, "median": 240},
        {"name": "x1_partido", "type": "numeric", "levels": None, "min": 0, "max": 120, "median": 30},
        {"name": "x2_partidos", "type": "numeric", "levels": None, "min": 0, "max": 120, "median": 0},
        {"name": "perc_fatiga", "type": "numeric", "levels": None, "min": 0, "max": 10, "median": 5},
        {"name": "seleccion", "type": "categorical", "levels": ["Femenina", "Masculina"], "min": None, "max": None, "median": None},
        {"name": "lesiones_previas", "type": "categorical", "levels": ["No", "Sí"], "min": None, "max": None, "median": None},
        {"name": "loc_tobillo", "type": "categorical", "levels": ["0", "1"], "min": None, "max": None, "median": None},
        {"name": "loc_rodilla", "type": "categorical", "levels": ["0", "1"], "min": None, "max": None, "median": None},
        {"name": "loc_brazo", "type": "categorical", "levels": ["0", "1"], "min": None, "max": None, "median": None},
        {"name": "loc_hombro_clavicula", "type": "categorical", "levels": ["0", "1"], "min": None, "max": None, "median": None},
        {"name": "loc_columna_lumbar", "type": "categorical", "levels": ["0", "1"], "min": None, "max": None, "median": None},
        {"name": "loc_cara", "type": "categorical", "levels": ["0", "1"], "min": None, "max": None, "median": None},
        {"name": "loc_dorso", "type": "categorical", "levels": ["0", "1"], "min": None, "max": None, "median": None},
        {"name": "est_ligamento", "type": "categorical", "levels": ["0", "1"], "min": None, "max": None, "median": None},
        {"name": "est_menisco", "type": "categorical", "levels": ["0", "1"], "min": None, "max": None, "median": None},
        {"name": "est_hueso", "type": "categorical", "levels": ["0", "1"], "min": None, "max": None, "median": None},
        {"name": "est_musculo", "type": "categorical", "levels": ["0", "1"], "min": None, "max": None, "median": None},
    ]
}


# ==============================================================================
# CAPÇALERA
# ==============================================================================

st.markdown(
    '<div class="main-title">📊 Aplicació de predicció de risc lesiu</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Aquesta eina permet predir el clúster d'un esportista a partir del model XGBoost entrenat.
    Es pot utilitzar mitjançant una enquesta individual o carregant un fitxer CSV.
    </div>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:
    st.header("Configuració")

    if MODEL_PATH is not None:
        st.success("Model trobat")
        st.caption(str(MODEL_PATH.relative_to(PROJECT_ROOT)))
    else:
        st.error("No s'ha trobat el model")
        st.caption("Busca: output/modeloXGBoost.RDS")

    if DATA_PATH is not None:
        st.success("Dades de referència trobades")
        st.caption(str(DATA_PATH.relative_to(PROJECT_ROOT)))
    else:
        st.info("Sense dades de referència")
        st.caption("L'app funcionarà igualment amb el model.")

    st.markdown("---")
    st.caption("Rscript utilitzat:")
    st.code(RSCRIPT)


if MODEL_PATH is None:
    st.error(
        "No s'ha trobat `modeloXGBoost.RDS`. "
        "Comprova que estigui a `output/modeloXGBoost.RDS` o a la mateixa carpeta que `app.py`."
    )
    st.stop()


# ==============================================================================
# CARREGUEM METADADES
# ==============================================================================

try:
    with st.spinner("Carregant informació del model..."):
        metadata = obtenir_metadata_model(
            str(MODEL_PATH),
            str(DATA_PATH) if DATA_PATH is not None else None
        )

except Exception as e:
    st.warning(
        "No s'han pogut extreure automàticament les variables del model. "
        "S'utilitzarà una configuració bàsica de suport."
    )
    st.code(str(e))
    metadata = FALLBACK_METADATA


variables = metadata.get("variables", [])

if len(variables) == 0:
    st.error("No s'han pogut obtenir les variables del model.")
    st.stop()


# ==============================================================================
# PESTANYES
# ==============================================================================

tab_enquesta, tab_csv, tab_info = st.tabs(
    ["📝 Enquesta individual", "📁 Predicció amb CSV", "ℹ️ Informació"]
)


# ==============================================================================
# PESTANYA 1: ENQUESTA INDIVIDUAL
# ==============================================================================

with tab_enquesta:

    st.subheader("📝 Enquesta individual")

    st.write(
        "Omple les variables següents i l'aplicació predirà el nivell de risc corresponent."
    )

    with st.form("formulari_enquesta"):

        respostes = {}

        variables_lesions = [
            v for v in variables
            if es_variable_localitzacio_lesio(v.get("name"))
        ]

        variables_afectacions = [
            v for v in variables
            if es_variable_afectacio(v.get("name"))
        ]

        variables_formulari = [
            v for v in variables
            if not es_variable_localitzacio_lesio(v.get("name"))
            and not es_variable_afectacio(v.get("name"))
        ]

        cols = st.columns(2)

        for idx, var_meta in enumerate(variables_formulari):

            nom = var_meta.get("name")
            tipus = var_meta.get("type", "numeric")
            levels = var_meta.get("levels", None)

            etiqueta = netejar_nom_variable(nom)
            col = cols[idx % 2]

            with col:

                if es_variable_fatiga_slider(nom):

                    minim = var_meta.get("min", None)
                    maxim = var_meta.get("max", None)
                    mediana = var_meta.get("median", None)

                    min_slider = 0.0
                    max_slider = 10.0

                    if minim is not None and maxim is not None:
                        try:
                            min_slider = float(minim)
                            max_slider = float(maxim)
                        except Exception:
                            min_slider = 0.0
                            max_slider = 10.0

                    if mediana is None:
                        valor_defecte = 5.0
                    else:
                        try:
                            valor_defecte = float(mediana)
                        except Exception:
                            valor_defecte = 5.0

                    valor_defecte = min(
                        max(valor_defecte, min_slider),
                        max_slider
                    )

                    valor = st.slider(
                        etiqueta,
                        min_value=min_slider,
                        max_value=max_slider,
                        value=valor_defecte,
                        step=1.0,
                        help=f"Variable original: {nom}"
                    )

                    if tipus == "categorical":
                        respostes[nom] = str(int(valor)) if float(valor).is_integer() else str(valor)
                    else:
                        respostes[nom] = valor

                elif tipus == "categorical" and levels is not None:

                    opcions = [str(x) for x in levels if str(x) != ""]

                    if len(opcions) == 0:
                        opcions = [""]

                    index_defecte = 0
                    nom_norm = normalitzar_nom(nom)

                    if (
                        nom_norm in ["genero", "genere", "gender", "sexo", "sexe", "seleccion", "seleccio"]
                        and "Femenina" in opcions
                    ):
                        index_defecte = opcions.index("Femenina")

                    if (
                        ("lesiones_previas" in nom_norm or "lesions_previes" in nom_norm)
                        and "No" in opcions
                    ):
                        index_defecte = opcions.index("No")

                    valor = st.selectbox(
                        etiqueta,
                        options=opcions,
                        index=index_defecte,
                        format_func=lambda x, n=nom: format_opcio(n, x),
                        help=f"Variable original: {nom}"
                    )

                    respostes[nom] = valor

                else:

                    minim = var_meta.get("min", None)
                    maxim = var_meta.get("max", None)
                    mediana = var_meta.get("median", None)

                    if mediana is None:
                        mediana = 0.0

                    try:
                        valor_defecte = float(mediana)
                    except Exception:
                        valor_defecte = 0.0

                    kwargs = {
                        "label": etiqueta,
                        "value": valor_defecte,
                        "help": f"Variable original: {nom}"
                    }

                    if minim is not None:
                        try:
                            kwargs["min_value"] = float(minim)
                        except Exception:
                            pass

                    if maxim is not None:
                        try:
                            kwargs["max_value"] = float(maxim)
                        except Exception:
                            pass

                    valor = st.number_input(**kwargs)

                    respostes[nom] = valor

        st.markdown("### Lesions i afectacions")

        col_lesions, col_afectacions = st.columns(2)

        with col_lesions:

            opcions_lesions = [
                v.get("name") for v in variables_lesions
            ]

            lesions_seleccionades = st.multiselect(
                "Selecciona les localitzacions de lesió",
                options=opcions_lesions,
                format_func=lambda x: netejar_nom_variable(x),
                help="Pots seleccionar més d'una localització. Si no n'hi ha cap, deixa-ho buit."
            )

        with col_afectacions:

            opcions_afectacions = [
                v.get("name") for v in variables_afectacions
            ]

            afectacions_seleccionades = st.multiselect(
                "Selecciona les estructures afectades",
                options=opcions_afectacions,
                format_func=lambda x: netejar_nom_variable(x),
                help="Pots seleccionar més d'una afectació. Si no n'hi ha cap, deixa-ho buit."
            )

        for var_meta in variables_lesions:
            nom = var_meta.get("name")
            respostes[nom] = valor_binari_model(
                var_meta,
                nom in lesions_seleccionades
            )

        for var_meta in variables_afectacions:
            nom = var_meta.get("name")
            respostes[nom] = valor_binari_model(
                var_meta,
                nom in afectacions_seleccionades
            )

        enviar = st.form_submit_button("Predir risc", type="primary")

    if enviar:

        df_enquesta = pd.DataFrame([respostes])

        try:
            with st.spinner("Executant el model XGBoost..."):
                resultats = predir_amb_model(df_enquesta)

            st.subheader("Resultat de la predicció")

            cluster_pred = resultats.loc[0, "prediccio_cluster"]

            st.success(f"Clúster predit: {cluster_pred}")

            components.html(
                semafor_risc(cluster_pred),
                height=170,
                scrolling=False
            )

            st.markdown(
                f"""
                <div class="cluster-card">
                <b>Interpretació del clúster:</b><br>
                {descripcio_cluster(cluster_pred)}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                recomanacions_cluster(cluster_pred),
                unsafe_allow_html=True
            )

            # ------------------------------------------------------------------
            # RADIAL PLOT I GRÀFICS COMPARATIUS

            try:
                dades_radial = obtenir_dades_radial(df_enquesta, cluster_pred)

                if not dades_radial.empty:

                    st.subheader("Comparació radial amb la mitjana del clúster")

                    fig_radial = crear_radial_plot(
                        dades_radial,
                        cluster_pred,
                        max_variables=14
                    )

                    if fig_radial is not None:
                        st.plotly_chart(
                            fig_radial,
                            use_container_width=True
                        )

                    st.subheader("Gràfics comparatius amb el clúster")

                    st.write(
                        "Els gràfics següents comparen l'observació analitzada amb la mitjana "
                        "del clúster predit utilitzant valors normalitzats entre 0 i 1."
                    )

                    col_grafic_1, col_grafic_2 = st.columns(2)

                    with col_grafic_1:
                        fig_barres = crear_grafic_barres_comparacio(
                            dades_radial,
                            cluster_pred,
                            max_variables=14
                        )

                        st.plotly_chart(
                            fig_barres,
                            use_container_width=True
                        )

                    with col_grafic_2:
                        fig_diferencies = crear_grafic_diferencies(
                            dades_radial,
                            max_variables=14
                        )

                        st.plotly_chart(
                            fig_diferencies,
                            use_container_width=True
                        )

                    with st.expander("Veure variables utilitzades en el radial plot"):
                        st.dataframe(
                            dades_radial[
                                [
                                    "variable_mostrar",
                                    "observacio_original",
                                    "mitjana_cluster_original"
                                ]
                            ],
                            use_container_width=True
                        )

                else:
                    st.info(
                        "No s'han trobat variables numèriques suficients per generar el radial plot."
                    )

            except Exception as e:
                st.warning(
                    "No s'ha pogut generar el radial plot o els gràfics comparatius."
                )
                st.code(str(e))

            # ------------------------------------------------------------------
            # TOP 10 CASOS MÉS SEMBLANTS

            try:
                top10_similars = obtenir_top10_similars(df_enquesta, cluster_pred)

                if not top10_similars.empty:

                    st.subheader("Top 10 casos més semblants dins del clúster predit")

                    st.write(
                        "Aquesta taula mostra els 10 casos del mateix clúster que més s'assemblen "
                        "a l'observació analitzada. També s'afegeixen els valors introduïts "
                        "per poder comparar-los amb els casos més semblants."
                    )

                    st.dataframe(
                        top10_similars,
                        use_container_width=True
                    )

                else:
                    st.info(
                        "No s'han trobat casos semblants dins del clúster predit."
                    )

            except Exception as e:
                st.warning(
                    "No s'ha pogut calcular el top 10 de casos més semblants."
                )
                st.code(str(e))

        except Exception as e:
            st.error("Hi ha hagut un error fent la predicció.")
            st.code(str(e))


# ==============================================================================
# PESTANYA 2: CSV
# ==============================================================================

with tab_csv:

    st.subheader("📁 Predicció mitjançant CSV")

    st.write(
        "Puja un CSV amb les variables necessàries. L'aplicació retornarà el mateix fitxer "
        "amb una nova columna anomenada `prediccio_cluster`, el nivell de risc i una recomanació."
    )

    uploaded_file = st.file_uploader(
        "Puja un fitxer CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            df_csv = llegir_csv_robust(uploaded_file)

            st.subheader("Vista prèvia de les dades carregades")
            st.dataframe(df_csv, use_container_width=True)

            st.write(f"Files carregades: **{df_csv.shape[0]}**")
            st.write(f"Columnes carregades: **{df_csv.shape[1]}**")

            vars_model = [v.get("name") for v in variables]
            vars_falten = [v for v in vars_model if v not in df_csv.columns]

            if len(vars_falten) > 0:
                st.warning(
                    "El CSV no conté totes les variables que espera el model."
                )
                st.write("Variables que falten:")
                st.code(", ".join(vars_falten))
            else:
                st.success("El CSV conté totes les variables necessàries.")

            if st.button("Predir clústers del CSV", type="primary"):

                try:
                    with st.spinner("Executant el model XGBoost..."):
                        resultats_csv = predir_amb_model(df_csv)

                    st.subheader("Resultats de la predicció")

                    resultats_csv_mostrar = seleccionar_columnes_resultat(resultats_csv)

                    st.dataframe(resultats_csv_mostrar, use_container_width=True)

                    if "prediccio_cluster" in resultats_csv.columns:

                        st.subheader("Resum de clústers predits")

                        resum = (
                            resultats_csv["prediccio_cluster"]
                            .value_counts()
                            .reset_index()
                        )

                        resum.columns = ["Cluster", "Freqüència"]

                        resum["Percentatge"] = (
                            resum["Freqüència"] / resum["Freqüència"].sum() * 100
                        ).round(2)

                        st.dataframe(resum, use_container_width=True)
                        st.bar_chart(resum.set_index("Cluster")["Freqüència"])

                        st.subheader("Resum per nivell de risc")

                        resum_risc = (
                            resultats_csv["nivell_risc"]
                            .value_counts()
                            .reset_index()
                        )

                        resum_risc.columns = ["Nivell de risc", "Freqüència"]

                        resum_risc["Percentatge"] = (
                            resum_risc["Freqüència"] / resum_risc["Freqüència"].sum() * 100
                        ).round(2)

                        st.dataframe(resum_risc, use_container_width=True)

                    csv_resultats = resultats_csv_mostrar.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Descarregar resultats en CSV",
                        data=csv_resultats,
                        file_name="resultats_prediccio_xgboost.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error("Hi ha hagut un error fent la predicció.")
                    st.code(str(e))

        except Exception as e:
            st.error("No s'ha pogut llegir el fitxer CSV.")
            st.code(str(e))

    else:
        st.info("Carrega un fitxer CSV per començar.")


# ==============================================================================
# PESTANYA 3: INFORMACIÓ
# ==============================================================================

with tab_info:

    st.subheader("ℹ️ Informació de l'aplicació")

    st.write(
        """
        Aquesta aplicació utilitza un model XGBoost entrenat en R i guardat en format `.RDS`.
        L'aplicació està desenvolupada amb Streamlit i executa codi R internament per carregar
        el model i generar les prediccions.
        """
    )

    st.markdown("### Variables utilitzades pel model")

    df_variables = pd.DataFrame(
        [
            {
                "Variable original": v.get("name"),
                "Pregunta en l'enquesta": netejar_nom_variable(v.get("name")),
                "Tipus": v.get("type"),
                "Valors possibles": ", ".join([format_opcio(v.get("name"), x) for x in v.get("levels", [])])
                if v.get("levels") is not None else ""
            }
            for v in variables
        ]
    )

    st.dataframe(df_variables, use_container_width=True)

    st.markdown("### Interpretació del semàfor de risc")

    st.write(
        """
        - **Cluster1 — Risc baix / semàfor verd**: perfil amb baixa presència lesiva.
        - **Cluster2 — Risc alt / semàfor vermell**: perfil amb major presència de lesions i major necessitat de seguiment.
        - **Cluster3 — Risc mitjà / semàfor groc**: perfil mixt o intermedi, que requereix control i prevenció.
        """
    )

    st.markdown("### Recomanacions generals per cada cas")

    st.markdown(recomanacions_cluster("Cluster1"), unsafe_allow_html=True)
    st.markdown(recomanacions_cluster("Cluster2"), unsafe_allow_html=True)
    st.markdown(recomanacions_cluster("Cluster3"), unsafe_allow_html=True)
