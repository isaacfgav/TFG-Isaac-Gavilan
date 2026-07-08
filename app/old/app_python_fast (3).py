
# ==============================================================================
# [TFG] app_python_fast.py
# Streamlit 100% Python per a la predicció del risc lesiu.
#
# Requereix la conversió prèvia:
# Rscript convertir_modelo_rds_a_python.R \
# output/modeloXGBoost.RDS \
# data/datos_modelling.RDS \
# output/python_model
# ==============================================================================

import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import xgboost as xgb


# ==============================================================================
# CONFIGURACIÓ GENERAL
# ==============================================================================

st.set_page_config(
    page_title="Predicció de risc lesiu - TFG",
    page_icon="🚦",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent

ARTIFACT_DIR_CANDIDATES = [
    PROJECT_ROOT / "output" / "python_model",
    PROJECT_ROOT / "python_model",
]

ARTIFACT_DIR = next(
    (
        path
        for path in ARTIFACT_DIR_CANDIDATES
        if (path / "model_artifacts.json").exists()
    ),
    None
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: .2rem;
        color: #1F2937;
    }

    .subtitle {
        color: #4B5563;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .cluster-card,
    .recommendation-card {
        padding: 1.1rem;
        border-radius: .8rem;
        border: 1px solid #E5E7EB;
        background: #FAFAFA;
        margin: 1rem 0;
    }

    .risk-title {
        font-size: 1.25rem;
        font-weight: 750;
        margin-bottom: .4rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# UTILITATS
# ==============================================================================

def normalitzar_nom(nom):
    text = str(nom).lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        caracter
        for caracter in text
        if not unicodedata.combining(caracter)
    )

    return (
        text
        .replace("ç", "c")
        .replace("ñ", "n")
    )


def valor_numerico_meta(valor, defecte=0.0):
    """
    Converteix valors procedents de les metadades R/JSON a float.
    """

    if valor is None:
        return float(defecte)

    if isinstance(valor, (int, float, np.integer, np.floating)):
        if pd.isna(valor):
            return float(defecte)

        return float(valor)

    if isinstance(valor, dict):
        claus_habituals = [
            "value",
            "values",
            "x",
            "data",
            "min",
            "max",
            "median",
            "0"
        ]

        for clau in claus_habituals:
            if clau in valor:
                try:
                    return valor_numerico_meta(valor[clau], defecte)
                except Exception:
                    pass

        for element in valor.values():
            try:
                return valor_numerico_meta(element, defecte)
            except Exception:
                continue

        return float(defecte)

    if isinstance(
        valor,
        (list, tuple, np.ndarray, pd.Series)
    ):
        for element in list(valor):
            try:
                return valor_numerico_meta(element, defecte)
            except Exception:
                continue

        return float(defecte)

    try:
        text = str(valor).replace(",", ".").strip()

        if text == "" or text.lower() in [
            "nan",
            "na",
            "null",
            "none"
        ]:
            return float(defecte)

        return float(text)

    except Exception:
        return float(defecte)


def make_names_py(name):
    """
    Aproximació a make.names() de R.
    Només s'utilitza com a alternativa de seguretat.
    """

    resultat = []

    for caracter in str(name):
        if caracter.isalnum() or caracter == "_":
            resultat.append(caracter)
        else:
            resultat.append(".")

    text = "".join(resultat)

    if not text or text[0].isdigit():
        text = "X" + text

    return text


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

    for parametres in intents:
        try:
            uploaded_file.seek(0)

            dataframe = pd.read_csv(
                uploaded_file,
                **parametres
            )

            if dataframe.shape[1] > 1:
                return dataframe

        except Exception as error:
            ultim_error = error

    raise ValueError(
        "No s'ha pogut llegir el CSV. "
        "Revisa el separador i la codificació."
    ) from ultim_error


def netejar_nom_variable(nom):
    nom_normalitzat = normalitzar_nom(nom)

    diccionari = {
        "genero": "Gènere",
        "genere": "Gènere",
        "gender": "Gènere",
        "sexo": "Gènere",
        "sexe": "Gènere",
        "seleccion": "Gènere",
        "seleccio": "Gènere",

        "edad": "Edat",
        "edad_anos": "Edat",
        "edat": "Edat",
        "age": "Edat",

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

        "min_entreno_fisico": "Minuts d'entreno físic setmanal",
        "min_entreno_pista": "Minuts d'entreno a pista setmanal",

        "x1_partido": "Minuts jugats al partit",
        "x2_partidos": "Minuts jugats al segon partit",

        "perc_fatiga": "Nivell de fatiga",
        "fatiga": "Nivell de fatiga",

        "lesiones_previas": "Ha tingut lesions prèvies?",
        "lesions_previes": "Ha tingut lesions prèvies?",

        "loc_tobillo": "Lesió al turmell",
        "loc_rodilla": "Lesió al genoll",
        "loc_brazo": "Lesió al braç",

        "loc_hombro_clavicula":
            "Lesió a l'espatlla o clavícula",

        "loc_columna_lumbar":
            "Lesió a la columna lumbar",

        "loc_cara": "Lesió a la cara",
        "loc_dorso": "Lesió al dors",

        "est_ligamento": "Afectació de lligament",
        "est_menisco": "Afectació de menisc",
        "est_hueso": "Afectació òssia",
        "est_musculo": "Afectació muscular",
    }

    return diccionari.get(
        nom_normalitzat,
        str(nom).replace("_", " ").capitalize()
    )


def format_opcio(nom_variable, valor):
    valor_text = str(valor)
    valor_normalitzat = normalitzar_nom(valor_text)
    nom_normalitzat = normalitzar_nom(nom_variable)

    variables_genere = [
        "genero",
        "genere",
        "gender",
        "sexo",
        "sexe",
        "seleccion",
        "seleccio"
    ]

    if nom_normalitzat in variables_genere:
        if valor_normalitzat in [
            "masculina",
            "masculino",
            "masculi",
            "home",
            "hombre"
        ]:
            return "Masculí"

        if valor_normalitzat in [
            "femenina",
            "femenino",
            "femeni",
            "dona",
            "mujer"
        ]:
            return "Femení"

    if (
        nom_normalitzat.startswith("loc_")
        or nom_normalitzat.startswith("est_")
    ):
        if valor_text == "1":
            return "Sí"

        if valor_text == "0":
            return "No"

    return valor_text


def es_variable_localitzacio_lesio(nom):
    return normalitzar_nom(nom).startswith("loc_")


def es_variable_afectacio(nom):
    return normalitzar_nom(nom).startswith("est_")


def es_variable_fatiga_slider(nom):
    return "fatiga" in normalitzar_nom(nom)


def valor_binari_model(var_meta, seleccionat):
    nivells = var_meta.get("levels")

    if nivells:
        nivells = [str(nivell) for nivell in nivells]

        if "0" in nivells and "1" in nivells:
            return "1" if seleccionat else "0"

        if "No" in nivells and "Sí" in nivells:
            return "Sí" if seleccionat else "No"

        if "No" in nivells and "Si" in nivells:
            return "Si" if seleccionat else "No"

    return "1" if seleccionat else "0"


# ==============================================================================
# INTERPRETACIÓ DELS CLÚSTERS
# ==============================================================================

def nivell_risc(cluster):
    cluster = str(cluster)

    if cluster == "Cluster1":
        return "Baix", "Verd"

    if cluster == "Cluster2":
        return "Mitjà", "Groc"

    if cluster == "Cluster3":
        return "Alt", "Vermell"

    return "No disponible", "Gris"


def recomanacio_breu(cluster):
    cluster = str(cluster)

    if cluster == "Cluster1":
        return (
            "Mantenir la planificació, les mesures preventives "
            "i el control regular de la fatiga."
        )

    if cluster == "Cluster2":
        return (
            "Fer seguiment de la fatiga, ajustar la càrrega si cal "
            "i reforçar el treball preventiu."
        )

    if cluster == "Cluster3":
        return (
            "Revisar la càrrega, fer un seguiment individualitzat "
            "i reforçar la prevenció específica."
        )

    return "No hi ha cap recomanació disponible."


def descripcio_cluster(cluster):
    descripcions = {
        "Cluster1": (
            "Perfil de baixa presència lesiva, associat a un menor "
            "historial de lesions i a una menor afectació corporal."
        ),

        "Cluster2": (
            "Perfil intermedi o mixt, amb presència d'historial "
            "lesional i una distribució heterogènia de les afectacions."
        ),

        "Cluster3": (
            "Perfil de risc alt, associat a una major presència de "
            "lesions prèvies i afectacions en localitzacions o "
            "estructures especialment rellevants."
        ),
    }

    return descripcions.get(
        str(cluster),
        "No hi ha cap descripció disponible."
    )


def recomanacions_cluster(cluster):
    cluster = str(cluster)
    risc, _ = nivell_risc(cluster)

    if cluster == "Cluster3":
        recomanacions = [
            "Revisar la càrrega total i els minuts acumulats.",
            "Fer una valoració individualitzada amb el cos tècnic o sanitari.",
            "Prioritzar el treball preventiu específic.",
            "Evitar increments bruscos de la càrrega.",
            "Fer seguiment de les zones i estructures lesionades."
        ]

    elif cluster == "Cluster2":
        recomanacions = [
            "Monitoritzar la fatiga i les possibles molèsties.",
            "Ajustar la càrrega si apareixen signes de sobrecàrrega.",
            "Reforçar la força i el control motor.",
            "Incorporar exercicis preventius i compensatoris.",
            "Reavaluar el perfil si apareix una nova lesió."
        ]

    else:
        recomanacions = [
            "Mantenir la planificació actual.",
            "Continuar les rutines d'escalfament i prevenció.",
            "Controlar periòdicament la fatiga.",
            "Registrar les molèsties noves encara que siguin lleus."
        ]

    elements = "".join(
        f"<li>{recomanacio}</li>"
        for recomanacio in recomanacions
    )

    return f"""
    <div class="recommendation-card">
        <div class="risk-title">
            Recomanacions per al perfil de risc {risc.lower()}
        </div>
        <ul>{elements}</ul>
    </div>
    """


def semafor_risc(cluster):
    risc, color = nivell_risc(cluster)

    vermell = (
        "#EF4444"
        if color == "Vermell"
        else "#4A1F1F"
    )

    groc = (
        "#FACC15"
        if color == "Groc"
        else "#4A3A12"
    )

    verd = (
        "#22C55E"
        if color == "Verd"
        else "#12351F"
    )

    color_etiqueta = {
        "Vermell": "#EF4444",
        "Groc": "#CA8A04",
        "Verd": "#16A34A"
    }.get(
        color,
        "#6B7280"
    )

    return f"""
    <div style="
        display:flex;
        align-items:center;
        gap:1.2rem;
        padding:1rem;
        border:1px solid #E5E7EB;
        border-radius:.9rem;
        background:#FAFAFA;
        min-height:140px;
        font-family:Arial;
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
            <div style="
                width:29px;
                height:29px;
                border-radius:50%;
                background:{vermell};
            "></div>

            <div style="
                width:29px;
                height:29px;
                border-radius:50%;
                background:{groc};
            "></div>

            <div style="
                width:29px;
                height:29px;
                border-radius:50%;
                background:{verd};
            "></div>
        </div>

        <div>
            <div style="
                font-size:1.25rem;
                font-weight:800;
                color:{color_etiqueta};
            ">
                Semàfor de risc: {risc}
            </div>

            <div style="color:#4B5563">
                Resultat associat a <b>{cluster}</b>
            </div>
        </div>
    </div>
    """


# ==============================================================================
# CÀRREGA DEL MODEL I DELS ARTEFACTES
# ==============================================================================

@st.cache_data(show_spinner=False)
def carregar_artifacts(artifact_dir):
    ruta = Path(artifact_dir) / "model_artifacts.json"

    with open(
        ruta,
        "r",
        encoding="utf-8"
    ) as fitxer:
        return json.load(fitxer)


@st.cache_resource(show_spinner=False)
def carregar_model(artifact_dir):
    booster = xgb.Booster()

    booster.load_model(
        str(
            Path(artifact_dir)
            / "xgboost_model.json"
        )
    )

    return booster


@st.cache_data(show_spinner=False)
def carregar_reference_train(artifact_dir):
    ruta = (
        Path(artifact_dir)
        / "reference_train.csv"
    )

    if not ruta.exists():
        return pd.DataFrame()

    return pd.read_csv(ruta)


# ==============================================================================
# CONSTRUCCIÓ DE LA MATRIU I PREDICCIÓ
# ==============================================================================

def _to_numeric_series(serie):
    text = (
        serie
        .astype(str)
        .str.strip()
        .str.lower()
    )

    substitucions = {
        "sí": "1",
        "si": "1",
        "s": "1",
        "yes": "1",
        "true": "1",
        "no": "0",
        "n": "0",
        "false": "0"
    }

    text = (
        text
        .replace(substitucions)
        .str.replace(
            ",",
            ".",
            regex=False
        )
    )

    return pd.to_numeric(
        text,
        errors="coerce"
    )


def construir_matriz_modelo(df_input, artifacts):
    variables_originals = artifacts["variable_names"]
    variables_train = artifacts["variables_train"]
    feature_map = artifacts["feature_map"]

    variables_absents = [
        variable
        for variable in variables_originals
        if variable not in df_input.columns
    ]

    if variables_absents:
        raise ValueError(
            "Falten variables d'entrada: "
            + ", ".join(variables_absents)
        )

    x_new = pd.DataFrame(
        0.0,
        index=df_input.index,
        columns=variables_train
    )

    for variable in variables_originals:
        mapa = feature_map.get(
            variable,
            {}
        )

        if mapa.get("type") == "categorical":
            nivells = [
                str(nivell)
                for nivell in mapa.get("levels", [])
            ]

            nivell_columnes = mapa.get(
                "level_to_columns",
                {}
            )

            valors = (
                df_input[variable]
                .astype(str)
                .str.replace(
                    r"\.0$",
                    "",
                    regex=True
                )
            )

            invalids = sorted(
                set(valors.dropna().unique())
                - set(nivells)
            )

            invalids = [
                valor
                for valor in invalids
                if valor != ""
                and valor.lower() != "nan"
            ]

            if invalids:
                raise ValueError(
                    f"La variable '{variable}' conté valors no vàlids: "
                    f"{invalids}. Valors acceptats: {nivells}"
                )

            for nivell, columnes in nivell_columnes.items():
                mascara = valors == str(nivell)

                columnes_valides = [
                    columna
                    for columna in columnes
                    if columna in x_new.columns
                ]

                if columnes_valides:
                    x_new.loc[
                        mascara,
                        columnes_valides
                    ] = 1.0

        else:
            valors = _to_numeric_series(
                df_input[variable]
            )

            mediana = valors.median(
                skipna=True
            )

            if not np.isfinite(mediana):
                mediana = 0.0

            valors = (
                valors
                .fillna(mediana)
                .astype(float)
            )

            columnes = [
                columna
                for columna in mapa.get("columns", [])
                if columna in x_new.columns
            ]

            if not columnes:
                alternativa = make_names_py(variable)

                columnes = (
                    [alternativa]
                    if alternativa in x_new.columns
                    else []
                )

            for columna in columnes:
                x_new[columna] = valors.values

    x_new = (
        x_new
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0.0)
    )

    return x_new


def predir_amb_model(df_input, booster, artifacts):
    x_new = construir_matriz_modelo(
        df_input,
        artifacts
    )

    dnew = xgb.DMatrix(
        x_new.to_numpy(dtype=float),
        feature_names=list(x_new.columns)
    )

    prediccio_bruta = booster.predict(dnew)
    nivells = artifacts["nivells_k3"]

    if (
        prediccio_bruta.ndim == 1
        and len(prediccio_bruta) == len(df_input)
    ):
        prediccio_numerica = np.rint(
            prediccio_bruta
        ).astype(int)

        probabilitats = None

    else:
        probabilitats_array = prediccio_bruta.reshape(
            (
                len(df_input),
                len(nivells)
            )
        )

        prediccio_numerica = np.argmax(
            probabilitats_array,
            axis=1
        )

        probabilitats = pd.DataFrame(
            probabilitats_array,
            columns=[
                f"prob_{classe}"
                for classe in nivells
            ],
            index=df_input.index
        )

    clusters = [
        nivells[index]
        for index in prediccio_numerica
    ]

    resultats = df_input.copy()

    resultats["prediccio_cluster"] = clusters

    if probabilitats is not None:
        resultats = pd.concat(
            [
                resultats,
                probabilitats
            ],
            axis=1
        )

    resultats["nivell_risc"] = (
        resultats["prediccio_cluster"]
        .apply(
            lambda cluster:
            nivell_risc(cluster)[0]
        )
    )

    resultats["semafor"] = (
        resultats["prediccio_cluster"]
        .apply(
            lambda cluster:
            nivell_risc(cluster)[1]
        )
    )

    resultats["recomanacio"] = (
        resultats["prediccio_cluster"]
        .apply(recomanacio_breu)
    )

    return resultats


# ==============================================================================
# GRÀFICS I CASOS SIMILARS
# ==============================================================================

VARIABLES_EXCLOSES = {
    "k3",
    "genero",
    "genere",
    "gender",
    "sexo",
    "sexe",
    "seleccion",
    "seleccio",
    "raza",
    "raca"
}


def obtenir_dades_radial(
    df_input,
    cluster_pred,
    reference_train
):
    if (
        reference_train.empty
        or "k3" not in reference_train.columns
    ):
        return pd.DataFrame()

    dades_cluster = reference_train[
        reference_train["k3"].astype(str)
        == str(cluster_pred)
    ]

    if dades_cluster.empty:
        return pd.DataFrame()

    files = []

    variables_comunes = [
        variable
        for variable in df_input.columns
        if variable in reference_train.columns
        and normalitzar_nom(variable)
        not in VARIABLES_EXCLOSES
    ]

    for variable in variables_comunes:
        train_numeric = _to_numeric_series(
            reference_train[variable]
        )

        cluster_numeric = _to_numeric_series(
            dades_cluster[variable]
        )

        observacio_numeric = _to_numeric_series(
            pd.Series(
                [df_input[variable].iloc[0]]
            )
        ).iloc[0]

        if (
            train_numeric.dropna().empty
            or not np.isfinite(observacio_numeric)
        ):
            continue

        minim = train_numeric.min(
            skipna=True
        )

        maxim = train_numeric.max(
            skipna=True
        )

        if (
            not np.isfinite(minim)
            or not np.isfinite(maxim)
            or minim == maxim
        ):
            continue

        mitjana_cluster = cluster_numeric.mean(
            skipna=True
        )

        if not np.isfinite(mitjana_cluster):
            continue

        observacio_norm = float(
            np.clip(
                (
                    observacio_numeric
                    - minim
                )
                / (
                    maxim
                    - minim
                ),
                0,
                1
            )
        )

        mitjana_norm = float(
            np.clip(
                (
                    mitjana_cluster
                    - minim
                )
                / (
                    maxim
                    - minim
                ),
                0,
                1
            )
        )

        files.append(
            {
                "variable": variable,
                "variable_mostrar":
                    netejar_nom_variable(variable),

                "observacio_original":
                    observacio_numeric,

                "mitjana_cluster_original":
                    mitjana_cluster,

                "observacio_norm":
                    observacio_norm,

                "mitjana_cluster_norm":
                    mitjana_norm,

                "diferencia":
                    abs(
                        observacio_norm
                        - mitjana_norm
                    )
            }
        )

    return pd.DataFrame(files)


def seleccionar_variables_radial(
    dades_radial,
    max_variables=14
):
    if dades_radial.empty:
        return dades_radial

    return (
        dades_radial
        .sort_values(
            "diferencia",
            ascending=False
        )
        .head(max_variables)
    )


def crear_radial_plot(
    dades_radial,
    cluster_pred,
    max_variables=14
):
    dades = seleccionar_variables_radial(
        dades_radial,
        max_variables
    )

    if dades.empty:
        return None

    categories = (
        dades["variable_mostrar"]
        .tolist()
    )

    observacio = (
        dades["observacio_norm"]
        .tolist()
    )

    mitjana = (
        dades["mitjana_cluster_norm"]
        .tolist()
    )

    categories_tancades = (
        categories
        + [categories[0]]
    )

    observacio_tancada = (
        observacio
        + [observacio[0]]
    )

    mitjana_tancada = (
        mitjana
        + [mitjana[0]]
    )

    figura = go.Figure()

    figura.add_trace(
        go.Scatterpolar(
            r=mitjana_tancada,
            theta=categories_tancades,
            fill="toself",
            name=f"Mitjana {cluster_pred}",
            line=dict(
                color="black",
                width=4
            )
        )
    )

    figura.add_trace(
        go.Scatterpolar(
            r=observacio_tancada,
            theta=categories_tancades,
            fill="toself",
            name="Observació",
            line=dict(
                color="white",
                width=4
            ),
            marker=dict(
                color="white",
                line=dict(
                    color="black",
                    width=1
                )
            )
        )
    )

    figura.update_layout(
        title="Comparació radial amb la mitjana del clúster",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True
    )

    return figura


def crear_grafic_barres_comparacio(
    dades_radial,
    cluster_pred,
    max_variables=14
):
    dades = seleccionar_variables_radial(
        dades_radial,
        max_variables
    )

    figura = go.Figure()

    figura.add_trace(
        go.Bar(
            x=dades["variable_mostrar"],
            y=dades["mitjana_cluster_norm"],
            name=f"Mitjana {cluster_pred}",
            marker=dict(
                color="black"
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Mitjana: %{y:.2f}"
                "<extra></extra>"
            )
        )
    )

    figura.add_trace(
        go.Bar(
            x=dades["variable_mostrar"],
            y=dades["observacio_norm"],
            name="Observació",
            marker=dict(
                color="white",
                line=dict(
                    color="black",
                    width=1.2
                )
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Observació: %{y:.2f}"
                "<extra></extra>"
            )
        )
    )

    figura.update_layout(
        title=dict(
            text=(
                "Comparació normalitzada "
                "observació vs clúster"
            ),
            font=dict(size=16),
            x=0.01
        ),

        height=390,

        barmode="group",

        bargap=0.20,
        bargroupgap=0.05,

        yaxis=dict(
            range=[0, 1],
            title=None,
            tickfont=dict(size=10),
            gridcolor="#E5E7EB"
        ),

        xaxis=dict(
            title=None,
            tickangle=35,
            tickfont=dict(size=9),
            automargin=True
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        ),

        margin=dict(
            l=35,
            r=15,
            t=70,
            b=85
        )
    )

    return figura


def crear_grafic_diferencies(
    dades_radial,
    max_variables=14
):
    dades = seleccionar_variables_radial(
        dades_radial,
        max_variables
    ).copy()

    dades["diferencia_signada"] = (
        dades["observacio_norm"]
        - dades["mitjana_cluster_norm"]
    )

    dades = dades.sort_values(
        "diferencia_signada",
        ascending=True
    )

    colors = [
        "#2563EB"
        if valor < 0
        else "#DC2626"
        if valor > 0
        else "#9CA3AF"
        for valor
        in dades["diferencia_signada"]
    ]

    maxim_absolut = (
        dades["diferencia_signada"]
        .abs()
        .max()
    )

    if (
        not np.isfinite(maxim_absolut)
        or maxim_absolut < 0.05
    ):
        maxim_absolut = 0.05

    maxim_absolut = maxim_absolut * 1.10

    figura = go.Figure()

    figura.add_trace(
        go.Bar(
            x=dades["diferencia_signada"],
            y=dades["variable_mostrar"],
            orientation="h",
            marker=dict(
                color=colors
            ),
            showlegend=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Diferència: %{x:.2f}"
                "<extra></extra>"
            )
        )
    )

    figura.add_vline(
        x=0,
        line_width=2,
        line_dash="dash",
        line_color="black"
    )

    figura.update_layout(
        title=dict(
            text=(
                "Diferència respecte a "
                "la mitjana del clúster"
            ),
            font=dict(size=16),
            x=0.01
        ),

        height=420,

        showlegend=False,

        plot_bgcolor="white",
        paper_bgcolor="white",

        margin=dict(
            l=185,
            r=20,
            t=55,
            b=35
        ),

        xaxis=dict(
            title=None,
            range=[
                -maxim_absolut,
                maxim_absolut
            ],
            zeroline=False,
            showgrid=True,
            gridcolor="#E5E7EB"
        ),

        yaxis=dict(
            title=None,
            tickfont=dict(size=10),
            categoryorder="array",
            categoryarray=(
                dades["variable_mostrar"]
                .tolist()
            )
        )
    )

    return figura


def obtenir_top10_similars(
    df_input,
    cluster_pred,
    reference_train
):
    if (
        reference_train.empty
        or "k3" not in reference_train.columns
    ):
        return pd.DataFrame()

    cluster_df = reference_train[
        reference_train["k3"].astype(str)
        == str(cluster_pred)
    ].copy()

    if cluster_df.empty:
        return pd.DataFrame()

    variables_comunes = [
        variable
        for variable in df_input.columns
        if variable in reference_train.columns
        and normalitzar_nom(variable)
        not in VARIABLES_EXCLOSES
    ]

    columnes_matriu = []
    valors_observacio = []

    for variable in variables_comunes:
        train_numeric = _to_numeric_series(
            reference_train[variable]
        )

        cluster_numeric = _to_numeric_series(
            cluster_df[variable]
        )

        observacio_numeric = _to_numeric_series(
            pd.Series(
                [df_input[variable].iloc[0]]
            )
        ).iloc[0]

        if (
            train_numeric.dropna().empty
            or not np.isfinite(observacio_numeric)
        ):
            continue

        minim = train_numeric.min(
            skipna=True
        )

        maxim = train_numeric.max(
            skipna=True
        )

        if (
            not np.isfinite(minim)
            or not np.isfinite(maxim)
            or minim == maxim
        ):
            continue

        normalitzat = (
            (
                cluster_numeric
                - minim
            )
            / (
                maxim
                - minim
            )
        ).clip(
            0,
            1
        )

        mediana_norm = normalitzat.median(
            skipna=True
        )

        if not np.isfinite(mediana_norm):
            mediana_norm = 0

        normalitzat = normalitzat.fillna(
            mediana_norm
        )

        columnes_matriu.append(
            normalitzat.to_numpy()
        )

        valors_observacio.append(
            float(
                np.clip(
                    (
                        observacio_numeric
                        - minim
                    )
                    / (
                        maxim
                        - minim
                    ),
                    0,
                    1
                )
            )
        )

    if not columnes_matriu:
        return pd.DataFrame()

    matriu = np.vstack(
        columnes_matriu
    ).T

    observacio = np.array(
        valors_observacio
    )

    distancia = np.sqrt(
        np.mean(
            (
                matriu
                - observacio
            ) ** 2,
            axis=1
        )
    )

    similitud = np.clip(
        1 - distancia,
        0,
        1
    )

    ordre = np.argsort(
        -similitud
    )[:10]

    resultat = pd.DataFrame(
        {
            "cas_train":
                cluster_df.index
                .to_numpy()[ordre],

            "cluster":
                cluster_pred,

            "similitud":
                np.round(
                    similitud[ordre],
                    4
                ),

            "similitud_percentatge":
                np.round(
                    similitud[ordre] * 100,
                    2
                ),

            "distancia":
                np.round(
                    distancia[ordre],
                    4
                ),

            "n_variables_comparades":
                len(valors_observacio)
        }
    )

    return resultat


# ==============================================================================
# INTERFÍCIE DE L'APLICACIÓ
# ==============================================================================

st.markdown(
    """
    <div class="main-title">
        📊 Aplicació de predicció de risc lesiu
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Configuració")

    if ARTIFACT_DIR is None:
        st.error(
            "No s'han trobat els artefactes Python."
        )

        st.caption(
            "Executa primer la conversió del model RDS."
        )

    else:
        st.success(
            "Artefactes Python trobats."
        )

        try:
            ruta_relativa = ARTIFACT_DIR.relative_to(
                PROJECT_ROOT
            )

            st.caption(
                str(ruta_relativa)
            )

        except ValueError:
            st.caption(
                str(ARTIFACT_DIR)
            )


if ARTIFACT_DIR is None:
    st.error(
        "Falten els fitxers `model_artifacts.json` "
        "i `xgboost_model.json`."
    )

    st.code(
        "Rscript convertir_modelo_rds_a_python.R "
        "output/modeloXGBoost.RDS "
        "data/datos_modelling.RDS "
        "output/python_model"
    )

    st.stop()


artifacts = carregar_artifacts(
    str(ARTIFACT_DIR)
)

booster = carregar_model(
    str(ARTIFACT_DIR)
)

reference_train = carregar_reference_train(
    str(ARTIFACT_DIR)
)

variables = artifacts.get(
    "variables",
    []
)


tab_enquesta, tab_csv, tab_info = st.tabs(
    [
        "📝 Enquesta individual",
        "📁 Predicció amb CSV",
        "ℹ️ Informació"
    ]
)


# ==============================================================================
# PESTANYA: ENQUESTA INDIVIDUAL
# ==============================================================================

with tab_enquesta:
    st.subheader(
        "📝 Enquesta individual"
    )

    with st.form(
        "formulari_enquesta"
    ):
        respostes = {}

        variables_lesions = [
            variable
            for variable in variables
            if es_variable_localitzacio_lesio(
                variable.get("name")
            )
        ]

        variables_afectacions = [
            variable
            for variable in variables
            if es_variable_afectacio(
                variable.get("name")
            )
        ]

        variables_formulari = [
            variable
            for variable in variables
            if not es_variable_localitzacio_lesio(
                variable.get("name")
            )
            and not es_variable_afectacio(
                variable.get("name")
            )
        ]

        columnes = st.columns(2)

        for index, var_meta in enumerate(
            variables_formulari
        ):
            nom = var_meta.get("name")

            tipus = var_meta.get(
                "type",
                "numeric"
            )

            nivells = var_meta.get("levels")

            etiqueta = netejar_nom_variable(
                nom
            )

            with columnes[index % 2]:
                if es_variable_fatiga_slider(nom):
                    minim = valor_numerico_meta(
                        var_meta.get("min"),
                        0
                    )

                    maxim = valor_numerico_meta(
                        var_meta.get("max"),
                        10
                    )

                    mediana = valor_numerico_meta(
                        var_meta.get("median"),
                        5
                    )

                    valor = st.slider(
                        etiqueta,
                        min_value=minim,
                        max_value=maxim,
                        value=float(
                            np.clip(
                                mediana,
                                minim,
                                maxim
                            )
                        ),
                        step=1.0,
                        help=(
                            f"Variable original: {nom}"
                        )
                    )

                    if (
                        tipus == "categorical"
                        and float(valor).is_integer()
                    ):
                        respostes[nom] = str(
                            int(valor)
                        )

                    else:
                        respostes[nom] = valor

                elif (
                    tipus == "categorical"
                    and nivells
                ):
                    opcions = [
                        str(nivell)
                        for nivell in nivells
                        if str(nivell) != ""
                    ] or [""]

                    index_defecte = 0

                    if (
                        "No" in opcions
                        and (
                            "lesion"
                            in normalitzar_nom(nom)
                            or "lesio"
                            in normalitzar_nom(nom)
                        )
                    ):
                        index_defecte = opcions.index(
                            "No"
                        )

                    valor = st.selectbox(
                        etiqueta,
                        opcions,
                        index=index_defecte,
                        format_func=lambda opcio, n=nom:
                            format_opcio(n, opcio),
                        help=(
                            f"Variable original: {nom}"
                        )
                    )

                    respostes[nom] = valor

                else:
                    mediana = valor_numerico_meta(
                        var_meta.get("median"),
                        0
                    )

                    parametres = {
                        "label": etiqueta,
                        "value": mediana,
                        "help": (
                            f"Variable original: {nom}"
                        )
                    }

                    if var_meta.get("min") is not None:
                        parametres["min_value"] = (
                            valor_numerico_meta(
                                var_meta.get("min"),
                                0
                            )
                        )

                    if var_meta.get("max") is not None:
                        parametres["max_value"] = (
                            valor_numerico_meta(
                                var_meta.get("max"),
                                mediana
                            )
                        )

                    respostes[nom] = st.number_input(
                        **parametres
                    )

        st.markdown(
            "### Lesions i afectacions"
        )

        columna_lesions, columna_afectacions = (
            st.columns(2)
        )

        with columna_lesions:
            lesions_seleccionades = st.multiselect(
                "Selecciona les localitzacions de lesió",
                [
                    variable.get("name")
                    for variable in variables_lesions
                ],
                format_func=netejar_nom_variable,
                help=(
                    "Selecciona les zones on hi ha hagut "
                    "alguna lesió."
                )
            )

        with columna_afectacions:
            afectacions_seleccionades = st.multiselect(
                "Selecciona les estructures afectades",
                [
                    variable.get("name")
                    for variable
                    in variables_afectacions
                ],
                format_func=netejar_nom_variable
            )

        nombre_lesions_zona = {}

        if lesions_seleccionades:
            st.markdown(
                "#### Nombre de vegades lesionat per zona"
            )

            st.caption(
                "Aquests camps es guarden com a informació "
                "addicional, però encara no entren en el model."
            )

            columnes_lesions = st.columns(2)

            for index, nom_lesio in enumerate(
                lesions_seleccionades
            ):
                with columnes_lesions[index % 2]:
                    columna_extra = (
                        f"num_lesions_{nom_lesio}"
                    )

                    nombre_lesions_zona[columna_extra] = (
                        st.number_input(
                            (
                                "Vegades: "
                                f"{netejar_nom_variable(nom_lesio)}"
                            ),
                            min_value=1,
                            max_value=50,
                            value=1,
                            step=1,
                            help=(
                                "Variable informativa que encara "
                                "no s'utilitza en la predicció."
                            )
                        )
                    )

        for variable_meta in variables_lesions:
            nom_lesio = variable_meta.get(
                "name"
            )

            respostes[nom_lesio] = (
                valor_binari_model(
                    variable_meta,
                    nom_lesio
                    in lesions_seleccionades
                )
            )

            columna_extra = (
                f"num_lesions_{nom_lesio}"
            )

            respostes[columna_extra] = int(
                nombre_lesions_zona.get(
                    columna_extra,
                    0
                )
            )

        for variable_meta in variables_afectacions:
            nom_afectacio = variable_meta.get(
                "name"
            )

            respostes[nom_afectacio] = (
                valor_binari_model(
                    variable_meta,
                    nom_afectacio
                    in afectacions_seleccionades
                )
            )

        enviar = st.form_submit_button(
            "Predir risc",
            type="primary"
        )

    if enviar:
        df_enquesta = pd.DataFrame(
            [respostes]
        )

        try:
            resultats = predir_amb_model(
                df_enquesta,
                booster,
                artifacts
            )

            cluster_pred = resultats.loc[
                0,
                "prediccio_cluster"
            ]

            st.success(
                f"Clúster predit: {cluster_pred}"
            )

            components.html(
                semafor_risc(cluster_pred),
                height=170,
                scrolling=False
            )

            st.markdown(
                f"""
                <div class="cluster-card">
                    <b>Interpretació:</b><br>
                    {descripcio_cluster(cluster_pred)}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                recomanacions_cluster(
                    cluster_pred
                ),
                unsafe_allow_html=True
            )

            dades_radial = obtenir_dades_radial(
                df_enquesta,
                cluster_pred,
                reference_train
            )

            if not dades_radial.empty:
                st.subheader(
                    "Comparació amb la mitjana del clúster"
                )

                figura_radial = crear_radial_plot(
                    dades_radial,
                    cluster_pred
                )

                if figura_radial is not None:
                    st.plotly_chart(
                        figura_radial,
                        use_container_width=True
                    )

                columna_1, columna_2 = st.columns(2)

                with columna_1:
                    st.plotly_chart(
                        crear_grafic_barres_comparacio(
                            dades_radial,
                            cluster_pred
                        ),
                        use_container_width=True
                    )

                with columna_2:
                    st.plotly_chart(
                        crear_grafic_diferencies(
                            dades_radial
                        ),
                        use_container_width=True
                    )

            top10 = obtenir_top10_similars(
                df_enquesta,
                cluster_pred,
                reference_train
            )

            if not top10.empty:
                st.subheader(
                    "Top 10 casos més semblants"
                )

                st.dataframe(
                    top10,
                    use_container_width=True
                )

        except Exception as error:
            st.error(
                "Hi ha hagut un error fent la predicció."
            )

            st.code(
                str(error)
            )


# ==============================================================================
# PESTANYA: PREDICCIÓ AMB CSV
# ==============================================================================

with tab_csv:
    st.subheader(
        "📁 Predicció mitjançant CSV"
    )

    uploaded_file = st.file_uploader(
        "Puja un fitxer CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            df_csv = llegir_csv_robust(
                uploaded_file
            )

            st.dataframe(
                df_csv.head(20),
                use_container_width=True
            )

            variables_model = [
                variable.get("name")
                for variable in variables
            ]

            variables_absents = [
                variable
                for variable in variables_model
                if variable not in df_csv.columns
            ]

            if variables_absents:
                st.warning(
                    "Variables que falten:"
                )

                st.code(
                    ", ".join(
                        variables_absents
                    )
                )

            else:
                st.success(
                    "El CSV conté totes les variables necessàries."
                )

            if st.button(
                "Predir clústers del CSV",
                type="primary"
            ):
                resultats_csv = predir_amb_model(
                    df_csv,
                    booster,
                    artifacts
                )

                columnes_resultat = [
                    columna
                    for columna in [
                        "prediccio_cluster",
                        "nivell_risc",
                        "semafor",
                        "recomanacio"
                    ]
                    if columna
                    in resultats_csv.columns
                ]

                st.dataframe(
                    resultats_csv[
                        columnes_resultat
                    ],
                    use_container_width=True
                )

                resum = (
                    resultats_csv[
                        "prediccio_cluster"
                    ]
                    .value_counts()
                    .rename_axis("Cluster")
                    .reset_index(
                        name="Freqüència"
                    )
                )

                resum["Percentatge"] = (
                    resum["Freqüència"]
                    / resum["Freqüència"].sum()
                    * 100
                ).round(2)

                st.dataframe(
                    resum,
                    use_container_width=True
                )

                st.bar_chart(
                    resum.set_index(
                        "Cluster"
                    )["Freqüència"]
                )

                st.download_button(
                    "Descarregar resultats en CSV",
                    resultats_csv
                    .to_csv(index=False)
                    .encode("utf-8"),
                    "resultats_prediccio_xgboost.csv",
                    "text/csv"
                )

        except Exception as error:
            st.error(
                "No s'ha pogut processar el CSV."
            )

            st.code(
                str(error)
            )

    else:
        st.info(
            "Carrega un fitxer CSV per començar."
        )


# ==============================================================================
# PESTANYA: INFORMACIÓ
# ==============================================================================

with tab_info:
    st.subheader(
        "ℹ️ Informació"
    )

    st.write(
        "Aquesta versió carrega el model XGBoost "
        "directament en Python i evita iniciar R "
        "en cada predicció."
    )

    df_variables = pd.DataFrame(
        [
            {
                "Variable original":
                    variable.get("name"),

                "Pregunta en l'enquesta":
                    netejar_nom_variable(
                        variable.get("name")
                    ),

                "Tipus":
                    variable.get("type"),

                "Valors possibles":
                    ", ".join(
                        [
                            format_opcio(
                                variable.get("name"),
                                valor
                            )
                            for valor
                            in (
                                variable.get("levels")
                                or []
                            )
                        ]
                    )
            }

            for variable in variables
        ]
    )

    st.dataframe(
        df_variables,
        use_container_width=True
    )

