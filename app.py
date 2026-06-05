# ==============================================================================
# [TFG] app.py
#
# Aplicació Streamlit per predir el risc lesiu amb modeloXGBoost.RDS
# ==============================================================================

import os
import json
import textwrap
import tempfile
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st


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
    PROJECT_ROOT / "data" / "datos_cluster.RDS",
    PROJECT_ROOT / "input" / "datos_cluster.RDS",
    PROJECT_ROOT / "datos_cluster.RDS",
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
        "No s'ha pogut llegir el CSV. Revisa que sigui un CSV real "
        "i que les columnes estiguin separades correctament."
    ) from ultim_error


def normalitzar_nom(nom):
    return (
        str(nom)
        .lower()
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
        "genero": "1) Gènere",
        "genere": "1) Gènere",
        "gender": "1) Gènere",
        "sexo": "1) Gènere",
        "sexe": "1) Gènere",

        "edad": "2) Edat",
        "edat": "2) Edat",
        "age": "2) Edat",

        "peso_kg": "3) Pes corporal (kg)",
        "pes_kg": "3) Pes corporal (kg)",
        "peso": "3) Pes corporal (kg)",
        "pes": "3) Pes corporal (kg)",

        "altura_corporal_cm": "4) Alçada corporal (cm)",
        "alcada_corporal_cm": "4) Alçada corporal (cm)",
        "altura_cm": "4) Alçada corporal (cm)",
        "alcada_cm": "4) Alçada corporal (cm)",

        "raza": "5) Raça",
        "raca": "5) Raça",
        "raça": "5) Raça",

        "min_entreno_fisico": "6) Indica els minuts d'entreno físic setmanal",
        "minuts_entreno_fisic": "6) Indica els minuts d'entreno físic setmanal",
        "minutos_entreno_fisico": "6) Indica els minuts d'entreno físic setmanal",
        "entreno_fisico": "6) Indica els minuts d'entreno físic setmanal",
        "entrenament_fisic": "6) Indica els minuts d'entreno físic setmanal",

        "min_entreno_pista": "7) Indica els minuts d'entreno a pista setmanal",
        "minuts_entreno_pista": "7) Indica els minuts d'entreno a pista setmanal",
        "minutos_entreno_pista": "7) Indica els minuts d'entreno a pista setmanal",
        "entreno_pista": "7) Indica els minuts d'entreno a pista setmanal",
        "entrenament_pista": "7) Indica els minuts d'entreno a pista setmanal",

        "x1_partido": "8) Minuts jugats al partit",
        "x1_partit": "8) Minuts jugats al partit",
        "minuts_partit": "8) Minuts jugats al partit",
        "minutos_partido": "8) Minuts jugats al partit",

        "x2_partidos": "9) Minuts jugats al segon partit (si no hi ha indica 0)",
        "x2_partits": "9) Minuts jugats al segon partit (si no hi ha indica 0)",
        "minuts_segon_partit": "9) Minuts jugats al segon partit (si no hi ha indica 0)",
        "minutos_segundo_partido": "9) Minuts jugats al segon partit (si no hi ha indica 0)",

        "perc_fatiga": "10) Percentatge / nivell de fatiga",
        "fatiga": "10) Percentatge / nivell de fatiga",
        "percepcio_fatiga": "10) Percepció de fatiga després de l'última competició o entrenament",
        "cual_es_tu_percepcion_de_fatiga_despues_de_la_ultima_competicion_entrenamiento_de_la_semana":
            "10) Percepció de fatiga després de l'última competició o entrenament de la setmana",

        "seleccion": "Selecció",
        "seleccio": "Selecció",

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
    nom_norm = normalitzar_nom(nom_variable)

    if nom_norm.startswith("loc_") or nom_norm.startswith("est_"):
        if valor_str == "0":
            return "No"
        if valor_str == "1":
            return "Sí"

    return valor_str


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
    <div style="
        display:flex;
        align-items:center;
        gap:1.2rem;
        margin-top:1rem;
        margin-bottom:1rem;
        padding:1rem;
        border:1px solid #E5E7EB;
        border-radius:0.9rem;
        background:#FAFAFA;
    ">
        <div style="
            width:58px;
            padding:9px;
            border-radius:20px;
            background:#111827;
            display:flex;
            flex-direction:column;
            gap:8px;
            align-items:center;
        ">
            <div style="width:29px;height:29px;border-radius:50%;background:{red_style};box-shadow:{shadow_red};"></div>
            <div style="width:29px;height:29px;border-radius:50%;background:{yellow_style};box-shadow:{shadow_yellow};"></div>
            <div style="width:29px;height:29px;border-radius:50%;background:{green_style};box-shadow:{shadow_green};"></div>
        </div>

        <div>
            <div style="font-size:1.25rem;font-weight:800;color:{etiqueta_color};">
                Semàfor de risc: {risc}
            </div>
            <div style="color:#4B5563;margin-top:0.25rem;">
                Resultat associat a <b>{cluster}</b>
            </div>
        </div>
    </div>
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


# ==============================================================================
# FALLBACK SI NO ES PODEN EXTREURE METADADES
# ==============================================================================

FALLBACK_METADATA = {
    "variables": [
        {"name": "genero", "type": "categorical", "levels": ["Femenina", "Masculina"], "min": None, "max": None, "median": None},
        {"name": "edad", "type": "numeric", "levels": None, "min": 10, "max": 60, "median": 20},
        {"name": "peso_kg", "type": "numeric", "levels": None, "min": 40, "max": 120, "median": 70},
        {"name": "altura_corporal_cm", "type": "numeric", "levels": None, "min": 140, "max": 220, "median": 175},
        {"name": "raza", "type": "categorical", "levels": ["Caucàsica", "Afrodescendent", "Asiàtica", "Llatina", "Altres"], "min": None, "max": None, "median": None},
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
        cols = st.columns(2)

        for idx, var_meta in enumerate(variables):

            nom = var_meta.get("name")
            tipus = var_meta.get("type", "numeric")
            levels = var_meta.get("levels", None)

            etiqueta = netejar_nom_variable(nom)
            col = cols[idx % 2]

            with col:

                if tipus == "categorical" and levels is not None:

                    opcions = [str(x) for x in levels if str(x) != ""]

                    if len(opcions) == 0:
                        opcions = [""]

                    index_defecte = 0
                    nom_norm = normalitzar_nom(nom)

                    if ("lesiones_previas" in nom_norm or "lesions_previes" in nom_norm) and "No" in opcions:
                        index_defecte = opcions.index("No")

                    if (nom_norm.startswith("loc_") or nom_norm.startswith("est_")) and "0" in opcions:
                        index_defecte = opcions.index("0")

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

                    nom_norm = normalitzar_nom(nom)

                    if "fatiga" in nom_norm:
                        min_slider = 0.0
                        max_slider = 10.0

                        if minim is not None and maxim is not None:
                            try:
                                min_slider = float(minim)
                                max_slider = float(maxim)
                            except Exception:
                                pass

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

                    else:
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

        enviar = st.form_submit_button("Predir risc", type="primary")

    if enviar:

        df_enquesta = pd.DataFrame([respostes])

        st.subheader("Dades introduïdes")
        st.dataframe(df_enquesta, use_container_width=True)

        try:
            with st.spinner("Executant el model XGBoost..."):
                resultats = predir_amb_model(df_enquesta)

            st.subheader("Resultat de la predicció")

            cluster_pred = resultats.loc[0, "prediccio_cluster"]

            st.success(f"Clúster predit: {cluster_pred}")

            st.markdown(
                semafor_risc(cluster_pred),
                unsafe_allow_html=True
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

            st.subheader("Resultat complet")
            st.dataframe(resultats, use_container_width=True)

            csv_resultats = resultats.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Descarregar resultat en CSV",
                data=csv_resultats,
                file_name="resultat_enquesta_xgboost.csv",
                mime="text/csv"
            )

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
                    st.dataframe(resultats_csv, use_container_width=True)

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

                    csv_resultats = resultats_csv.to_csv(index=False).encode("utf-8")

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
                "Valors possibles": ", ".join([str(x) for x in v.get("levels", [])])
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
