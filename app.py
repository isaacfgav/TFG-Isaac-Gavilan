# ==============================================================================
# [TFG] app.py
#
# Aplicació Streamlit per predir el clúster amb modeloXGBoost.RDS
# Inclou:
#   1) Formulari tipus enquesta individual
#   2) Predicció massiva mitjançant CSV
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
# Configuració general

st.set_page_config(
    page_title="Predicció de clúster - TFG",
    page_icon="📊",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent

# Ruta de Rscript.
# En Streamlit Cloud normalment funciona amb "Rscript".
# En local, si cal, pots definir la variable d'entorn RSCRIPT_PATH.
RSCRIPT = os.environ.get("RSCRIPT_PATH", "Rscript")

# Ruta del model
MODEL_CANDIDATES = [
    PROJECT_ROOT / "output" / "modeloXGBoost.RDS",
    PROJECT_ROOT / "modeloXGBoost.RDS",
]

MODEL_PATH = None

for candidate in MODEL_CANDIDATES:
    if candidate.exists():
        MODEL_PATH = candidate
        break

# Ruta de dades originals, només per agafar rangs i valors per defecte
DATA_CANDIDATES = [
    PROJECT_ROOT / "data" / "datos_cluster.RDS",
    PROJECT_ROOT / "input" / "datos_cluster.RDS",
    PROJECT_ROOT / "datos_cluster.RDS",
]

DATA_PATH = None

for candidate in DATA_CANDIDATES:
    if candidate.exists():
        DATA_PATH = candidate
        break


# ==============================================================================
# Estil visual

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #555;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .cluster-card {
        padding: 1rem;
        border-radius: 0.8rem;
        border: 1px solid #DDD;
        background-color: #FAFAFA;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# Funcions auxiliars Python

def llegir_csv_robust(uploaded_file):
    """
    Llegeix CSV provant diferents separadors i codificacions.
    Evita errors típics amb CSV exportats des d'Excel.
    """

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
        "No s'ha pogut llegir el CSV. Revisa que sigui un fitxer CSV real, "
        "no un Excel ni un RDS, i que les columnes estiguin separades correctament."
    ) from ultim_error


def netejar_nom_variable(nom):
    """
    Converteix noms de variables en etiquetes més llegibles per a l'enquesta.
    """

    diccionari = {
        "peso_kg": "Pes corporal (kg)",
        "altura_corporal_cm": "Alçada corporal (cm)",
        "perc_fatiga": "Percentatge / nivell de fatiga",
        "seleccion": "Selecció",
        "lesiones_previas": "Ha tingut lesions prèvies?",
        "loc_tobillo": "Lesió al turmell",
        "loc_rodilla": "Lesió al genoll",
        "loc_brazo": "Lesió al braç",
        "loc_hombro_clavicula": "Lesió a l'espatlla o clavícula",
        "loc_columna_lumbar": "Lesió a la columna lumbar",
        "loc_cara": "Lesió a la cara",
        "loc_dorso": "Lesió al dors",
        "est_ligamento": "Afectació de lligament",
        "est_menisco": "Afectació de menisc",
        "est_hueso": "Afectació òssia",
        "x1_partido": "Minuts jugats en 1 partit",
        "x2_partidos": "Minuts jugats en 2 partits",
        "cual_es_tu_percepcion_de_fatiga_despues_de_la_ultima_competicion_entrenamiento_de_la_semana":
            "Percepció de fatiga després de l'última competició o entrenament de la setmana",
    }

    if nom in diccionari:
        return diccionari[nom]

    return nom.replace("_", " ").capitalize()


def format_opcio(nom_variable, valor):
    """
    Millora la visualització de valors 0/1 sense canviar el valor real enviat al model.
    """

    valor_str = str(valor)

    if nom_variable.startswith("loc_") or nom_variable.startswith("est_"):
        if valor_str == "0":
            return "No"
        if valor_str == "1":
            return "Sí"

    return valor_str


def descripcio_cluster(cluster):
    """
    Text interpretatiu dels clústers segons el profiling.
    """

    descripcions = {
        "Cluster1": (
            "Perfil de baixa presència lesiva. Es caracteritza per una menor "
            "presència de lesions prèvies i menor afectació en localitzacions corporals."
        ),
        "Cluster2": (
            "Perfil d'alta presència lesiva. Es caracteritza per major presència "
            "de lesions prèvies i lesions en diferents zones corporals."
        ),
        "Cluster3": (
            "Perfil mixt o intermedi. Presenta un comportament més heterogeni, "
            "sense un patró tan clar com els altres dos grups."
        ),
    }

    return descripcions.get(
        str(cluster),
        "No hi ha descripció disponible per a aquest clúster."
    )


def executar_r(script_text, args):
    """
    Executa un script R temporal amb Rscript.
    """

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
# Codi R: metadades del model

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

    # --------------------------------------------------------------------------
    # Llibreria local de R per evitar errors de permisos a Streamlit Cloud

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

    # --------------------------------------------------------------------------
    # Carreguem model

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

    # --------------------------------------------------------------------------
    # Intentem obtenir rangs i medianes del dataset original

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
    """
    Extreu automàticament les variables originals que necessita el model.
    """

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
# Codi R: predicció

R_PREDICT_CODE = textwrap.dedent(
    r"""
    args <- commandArgs(trailingOnly = TRUE)

    if (length(args) < 3) {
      stop("Ús: Rscript predict.R input_csv output_csv model_path")
    }

    input_csv <- args[1]
    output_csv <- args[2]
    model_path <- args[3]

    # --------------------------------------------------------------------------
    # Llibreria local de R per evitar errors de permisos a Streamlit Cloud

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

    # --------------------------------------------------------------------------
    # Carreguem model

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

    # --------------------------------------------------------------------------
    # Llegim dades d'entrada

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

    # --------------------------------------------------------------------------
    # Convertim tipus segons el dummy_model

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

    # --------------------------------------------------------------------------
    # Transformació dummy

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

    # --------------------------------------------------------------------------
    # Predicció

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
    """
    Rep un DataFrame amb les variables del model i retorna les prediccions.
    """

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

        return resultats


# ==============================================================================
# Fallback si no es poden extreure metadades

FALLBACK_METADATA = {
    "variables": [
        {"name": "peso_kg", "type": "numeric", "levels": None, "min": 40, "max": 120, "median": 70},
        {"name": "altura_corporal_cm", "type": "numeric", "levels": None, "min": 140, "max": 220, "median": 175},
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
    ]
}


# ==============================================================================
# Capçalera principal

st.markdown(
    '<div class="main-title">📊 Aplicació de predicció de clúster</div>',
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
# Sidebar

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
# Carreguem metadades

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
# Pestanyes

tab_enquesta, tab_csv, tab_info = st.tabs(
    ["📝 Enquesta individual", "📁 Predicció amb CSV", "ℹ️ Informació"]
)


# ==============================================================================
# Pestanya 1: Enquesta individual

with tab_enquesta:

    st.subheader("📝 Enquesta individual")

    st.write(
        "Omple les variables següents i l'aplicació predirà el clúster corresponent."
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

                    if nom == "lesiones_previas" and "No" in opcions:
                        index_defecte = opcions.index("No")

                    if (nom.startswith("loc_") or nom.startswith("est_")) and "0" in opcions:
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

                    if "fatiga" in nom.lower():
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

        enviar = st.form_submit_button("Predir clúster", type="primary")

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
                f"""
                <div class="cluster-card">
                <b>Interpretació:</b><br>
                {descripcio_cluster(cluster_pred)}
                </div>
                """,
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
# Pestanya 2: CSV

with tab_csv:

    st.subheader("📁 Predicció mitjançant CSV")

    st.write(
        "Puja un CSV amb les variables necessàries. L'aplicació retornarà el mateix fitxer "
        "amb una nova columna anomenada `prediccio_cluster`."
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
# Pestanya 3: Informació

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
                "Etiqueta en l'enquesta": netejar_nom_variable(v.get("name")),
                "Tipus": v.get("type"),
                "Valors possibles": ", ".join([str(x) for x in v.get("levels", [])])
                if v.get("levels") is not None else ""
            }
            for v in variables
        ]
    )

    st.dataframe(df_variables, use_container_width=True)

    st.markdown("### Interpretació dels clústers")

    st.write(
        """
        - **Cluster1**: perfil de baixa presència lesiva.
        - **Cluster2**: perfil d'alta presència lesiva.
        - **Cluster3**: perfil mixt o intermedi.
        """
    )
