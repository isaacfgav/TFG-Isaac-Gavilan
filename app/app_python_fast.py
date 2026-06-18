# ==============================================================================
# [TFG] app_python_fast.py
# Streamlit 100% Python para predicción de riesgo lesivo.
# Requiere conversión previa:
#   Rscript convertir_modelo_rds_a_python.R output/modeloXGBoost.RDS data/datos_modelling.RDS output/python_model
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

st.set_page_config(page_title="Predicció de risc lesiu - TFG", page_icon="🚦", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR_CANDIDATES = [
    PROJECT_ROOT / "output" / "python_model",
    PROJECT_ROOT / "python_model",
]
ARTIFACT_DIR = next((p for p in ARTIFACT_DIR_CANDIDATES if (p / "model_artifacts.json").exists()), None)

st.markdown(
    """
    <style>
    .main-title{font-size:2.4rem;font-weight:800;margin-bottom:.2rem;color:#1F2937;}
    .subtitle{color:#4B5563;font-size:1.05rem;margin-bottom:1.5rem;}
    .cluster-card,.recommendation-card{padding:1.1rem;border-radius:.8rem;border:1px solid #E5E7EB;background:#FAFAFA;margin:1rem 0;}
    .risk-title{font-size:1.25rem;font-weight:750;margin-bottom:.4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# UTILIDADES
# ============================================================================== 

def normalitzar_nom(nom):
    text = str(nom).lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.replace("ç", "c").replace("ñ", "n")


def valor_numerico_meta(valor, defecto=0.0):
    """Convierte valores de metadata R/json a float.
    Soporta números directos, strings y diccionarios/listas generados por jsonlite.
    """
    if valor is None:
        return float(defecto)

    if isinstance(valor, (int, float, np.integer, np.floating)):
        if pd.isna(valor):
            return float(defecto)
        return float(valor)

    if isinstance(valor, dict):
        # jsonlite puede serializar escalares R como objetos. Probamos claves habituales
        # y, si no existen, el primer valor convertible.
        for key in ["value", "values", "x", "data", "min", "max", "median", "0"]:
            if key in valor:
                try:
                    return valor_numerico_meta(valor[key], defecto)
                except Exception:
                    pass
        for v in valor.values():
            try:
                return valor_numerico_meta(v, defecto)
            except Exception:
                continue
        return float(defecto)

    if isinstance(valor, (list, tuple, np.ndarray, pd.Series)):
        for v in list(valor):
            try:
                return valor_numerico_meta(v, defecto)
            except Exception:
                continue
        return float(defecto)

    try:
        texto = str(valor).replace(",", ".").strip()
        if texto == "" or texto.lower() in ["nan", "na", "null", "none"]:
            return float(defecto)
        return float(texto)
    except Exception:
        return float(defecto)


def make_names_py(name: str) -> str:
    # Aproximación suficiente para columnas ya exportadas desde R. Solo se usa como fallback.
    out = []
    for ch in str(name):
        out.append(ch if ch.isalnum() or ch == "_" else ".")
    s = "".join(out)
    if not s or s[0].isdigit():
        s = "X" + s
    return s


def llegir_csv_robust(uploaded_file):
    intents = [
        {"sep": ",", "encoding": "utf-8"}, {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "utf-8-sig"}, {"sep": ";", "encoding": "utf-8-sig"},
        {"sep": ",", "encoding": "latin1"}, {"sep": ";", "encoding": "latin1"},
        {"sep": ",", "encoding": "cp1252"}, {"sep": ";", "encoding": "cp1252"},
    ]
    ultimo_error = None
    for params in intents:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, **params)
            if df.shape[1] > 1:
                return df
        except Exception as e:
            ultimo_error = e
    raise ValueError("No s'ha pogut llegir el CSV. Revisa separador i codificació.") from ultimo_error


def netejar_nom_variable(nom):
    nom_norm = normalitzar_nom(nom)
    dic = {
        "genero": "Gènere", "genere": "Gènere", "gender": "Gènere", "sexo": "Gènere", "sexe": "Gènere", "seleccion": "Gènere", "seleccio": "Gènere",
        "edad": "Edat", "edat": "Edat", "age": "Edat",
        "peso_kg": "Pes corporal (kg)", "pes_kg": "Pes corporal (kg)", "peso": "Pes corporal (kg)", "pes": "Pes corporal (kg)",
        "altura_corporal_cm": "Alçada corporal (cm)", "alcada_corporal_cm": "Alçada corporal (cm)", "altura_cm": "Alçada corporal (cm)", "alcada_cm": "Alçada corporal (cm)",
        "raza": "Raça", "raca": "Raça",
        "min_entreno_fisico": "Minuts d'entreno físic setmanal", "min_entreno_pista": "Minuts d'entreno a pista setmanal",
        "x1_partido": "Minuts jugats al partit", "x2_partidos": "Minuts jugats al segon partit",
        "perc_fatiga": "Nivell de fatiga", "fatiga": "Nivell de fatiga",
        "lesiones_previas": "Ha tingut lesions prèvies?", "lesions_previes": "Ha tingut lesions prèvies?",
        "loc_tobillo": "Lesió al turmell", "loc_rodilla": "Lesió al genoll", "loc_brazo": "Lesió al braç",
        "loc_hombro_clavicula": "Lesió a l'espatlla o clavícula", "loc_columna_lumbar": "Lesió a la columna lumbar",
        "loc_cara": "Lesió a la cara", "loc_dorso": "Lesió al dors",
        "est_ligamento": "Afectació de lligament", "est_menisco": "Afectació de menisc", "est_hueso": "Afectació òssia", "est_musculo": "Afectació muscular",
    }
    return dic.get(nom_norm, str(nom).replace("_", " ").capitalize())


def format_opcio(nom_variable, valor):
    valor_str = str(valor)
    valor_norm = normalitzar_nom(valor_str)
    nom_norm = normalitzar_nom(nom_variable)
    if nom_norm in ["genero", "genere", "gender", "sexo", "sexe", "seleccion", "seleccio"]:
        if valor_norm in ["masculina", "masculino", "masculi", "home", "hombre"]:
            return "Masculí"
        if valor_norm in ["femenina", "femenino", "femeni", "dona", "mujer"]:
            return "Femení"
    if nom_norm.startswith("loc_") or nom_norm.startswith("est_"):
        return "Sí" if valor_str == "1" else "No" if valor_str == "0" else valor_str
    return valor_str


def es_variable_localitzacio_lesio(nom):
    return normalitzar_nom(nom).startswith("loc_")


def es_variable_afectacio(nom):
    return normalitzar_nom(nom).startswith("est_")


def es_variable_fatiga_slider(nom):
    return "fatiga" in normalitzar_nom(nom)


def valor_binari_model(var_meta, seleccionat):
    levels = var_meta.get("levels")
    if levels:
        levels = [str(x) for x in levels]
        if "0" in levels and "1" in levels:
            return "1" if seleccionat else "0"
        if "No" in levels and "Sí" in levels:
            return "Sí" if seleccionat else "No"
        if "No" in levels and "Si" in levels:
            return "Si" if seleccionat else "No"
    return "1" if seleccionat else "0"


def nivell_risc(cluster):
    cluster = str(cluster)
    if cluster == "Cluster1": return "Baix", "Verd"
    if cluster == "Cluster2": return "Alt", "Vermell"
    if cluster == "Cluster3": return "Mitjà", "Groc"
    return "No disponible", "Gris"


def recomanacio_breu(cluster):
    if str(cluster) == "Cluster1": return "Mantenir planificació, prevenció i control regular de fatiga."
    if str(cluster) == "Cluster2": return "Revisar càrrega, fer seguiment individualitzat i reforçar prevenció específica."
    if str(cluster) == "Cluster3": return "Fer seguiment de fatiga, ajustar càrrega si cal i reforçar treball preventiu."
    return "No hi ha recomanació disponible."


def descripcio_cluster(cluster):
    return {
        "Cluster1": "Perfil de baixa presència lesiva, associat a menor historial i menor afectació corporal.",
        "Cluster2": "Perfil de major risc, associat a més lesions prèvies i afectacions en diferents zones o estructures.",
        "Cluster3": "Perfil intermedi o mixt; requereix control preventiu i seguiment de càrrega.",
    }.get(str(cluster), "No hi ha descripció disponible.")


def recomanacions_cluster(cluster):
    risc, _ = nivell_risc(cluster)
    if str(cluster) == "Cluster2":
        items = ["Revisar càrrega total i minuts acumulats.", "Fer valoració individualitzada amb cos tècnic o sanitari.", "Prioritzar treball preventiu específic.", "Evitar increments bruscos de càrrega."]
    elif str(cluster) == "Cluster3":
        items = ["Monitoritzar fatiga i molèsties.", "Ajustar càrrega si apareix sobrecàrrega.", "Reforçar força, control motor i compensatoris.", "Reavaluar si apareix nova lesió."]
    else:
        items = ["Mantenir planificació actual.", "Continuar rutines d'escalfament i prevenció.", "Controlar periòdicament la fatiga.", "Registrar molèsties noves encara que siguin lleus."]
    lis = "".join(f"<li>{x}</li>" for x in items)
    return f'<div class="recommendation-card"><div class="risk-title">Recomanacions per al perfil de risc {risc.lower()}</div><ul>{lis}</ul></div>'


def semafor_risc(cluster):
    risc, color = nivell_risc(cluster)
    red = "#EF4444" if color == "Vermell" else "#4A1F1F"
    yellow = "#FACC15" if color == "Groc" else "#4A3A12"
    green = "#22C55E" if color == "Verd" else "#12351F"
    etiqueta = {"Vermell": "#EF4444", "Groc": "#CA8A04", "Verd": "#16A34A"}.get(color, "#6B7280")
    return f"""
    <div style="display:flex;align-items:center;gap:1.2rem;padding:1rem;border:1px solid #E5E7EB;border-radius:.9rem;background:#FAFAFA;min-height:140px;font-family:Arial">
      <div style="width:58px;padding:9px;border-radius:20px;background:#111827;display:flex;flex-direction:column;gap:8px;align-items:center">
        <div style="width:29px;height:29px;border-radius:50%;background:{red}"></div>
        <div style="width:29px;height:29px;border-radius:50%;background:{yellow}"></div>
        <div style="width:29px;height:29px;border-radius:50%;background:{green}"></div>
      </div>
      <div><div style="font-size:1.25rem;font-weight:800;color:{etiqueta}">Semàfor de risc: {risc}</div><div style="color:#4B5563">Resultat associat a <b>{cluster}</b></div></div>
    </div>
    """

# ==============================================================================
# CARGA DE ARTEFACTOS Y PREDICCIÓN
# ============================================================================== 

@st.cache_data(show_spinner=False)
def carregar_artifacts(artifact_dir: str):
    with open(Path(artifact_dir) / "model_artifacts.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def carregar_model(artifact_dir: str):
    booster = xgb.Booster()
    booster.load_model(str(Path(artifact_dir) / "xgboost_model.json"))
    return booster


@st.cache_data(show_spinner=False)
def carregar_reference_train(artifact_dir: str):
    path = Path(artifact_dir) / "reference_train.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _to_numeric_series(s):
    x = s.astype(str).str.strip().str.lower()
    repl = {"sí": "1", "si": "1", "s": "1", "yes": "1", "true": "1", "no": "0", "n": "0", "false": "0"}
    x = x.replace(repl).str.replace(",", ".", regex=False)
    return pd.to_numeric(x, errors="coerce")


def construir_matriz_modelo(df_input: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    vars_raw = artifacts["variable_names"]
    variables_train = artifacts["variables_train"]
    feature_map = artifacts["feature_map"]

    missing = [v for v in vars_raw if v not in df_input.columns]
    if missing:
        raise ValueError("Falten variables d'entrada: " + ", ".join(missing))

    x_new = pd.DataFrame(0.0, index=df_input.index, columns=variables_train)

    for v in vars_raw:
        fmap = feature_map.get(v, {})
        if fmap.get("type") == "categorical":
            levels = [str(x) for x in fmap.get("levels", [])]
            level_to_columns = fmap.get("level_to_columns", {})
            values = df_input[v].astype(str).str.replace(r"\.0$", "", regex=True)
            invalid = sorted(set(values.dropna().unique()) - set(levels))
            invalid = [x for x in invalid if x != "" and x.lower() != "nan"]
            if invalid:
                raise ValueError(f"La variable '{v}' té valors no vàlids: {invalid}. Valors acceptats: {levels}")
            for level, cols in level_to_columns.items():
                mask = values == str(level)
                valid_cols = [c for c in cols if c in x_new.columns]
                if valid_cols:
                    x_new.loc[mask, valid_cols] = 1.0
        else:
            vals = _to_numeric_series(df_input[v])
            med = vals.median(skipna=True)
            if not np.isfinite(med): med = 0.0
            vals = vals.fillna(med).astype(float)
            cols = [c for c in fmap.get("columns", []) if c in x_new.columns]
            if not cols:
                fallback = make_names_py(v)
                cols = [fallback] if fallback in x_new.columns else []
            for c in cols:
                x_new[c] = vals.values

    x_new = x_new.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x_new


def predir_amb_model(df_input: pd.DataFrame, booster, artifacts: dict) -> pd.DataFrame:
    x_new = construir_matriz_modelo(df_input, artifacts)
    dnew = xgb.DMatrix(x_new.to_numpy(dtype=float), feature_names=list(x_new.columns))
    pred_raw = booster.predict(dnew)
    nivells = artifacts["nivells_k3"]

    if pred_raw.ndim == 1 and len(pred_raw) == len(df_input):
        pred_num = np.rint(pred_raw).astype(int)
        probs = None
    else:
        probs_arr = pred_raw.reshape((len(df_input), len(nivells)))
        pred_num = np.argmax(probs_arr, axis=1)
        probs = pd.DataFrame(probs_arr, columns=[f"prob_{c}" for c in nivells], index=df_input.index)

    clusters = [nivells[i] for i in pred_num]
    result = df_input.copy()
    result["prediccio_cluster"] = clusters
    if probs is not None:
        result = pd.concat([result, probs], axis=1)
    result["nivell_risc"] = result["prediccio_cluster"].apply(lambda x: nivell_risc(x)[0])
    result["semafor"] = result["prediccio_cluster"].apply(lambda x: nivell_risc(x)[1])
    result["recomanacio"] = result["prediccio_cluster"].apply(recomanacio_breu)
    return result

# ==============================================================================
# RADIAL Y TOP10 EN PYTHON
# ============================================================================== 

VARIABLES_EXCLOSES = {"k3", "genero", "genere", "gender", "sexo", "sexe", "seleccion", "seleccio", "raza", "raca"}


def obtenir_dades_radial(df_input, cluster_pred, reference_train):
    if reference_train.empty or "k3" not in reference_train.columns:
        return pd.DataFrame()
    dades_cluster = reference_train[reference_train["k3"].astype(str) == str(cluster_pred)]
    if dades_cluster.empty:
        return pd.DataFrame()

    rows = []
    vars_common = [v for v in df_input.columns if v in reference_train.columns and normalitzar_nom(v) not in VARIABLES_EXCLOSES]
    for v in vars_common:
        train_num = _to_numeric_series(reference_train[v])
        cluster_num = _to_numeric_series(dades_cluster[v])
        obs_num = _to_numeric_series(pd.Series([df_input[v].iloc[0]])).iloc[0]
        if train_num.dropna().empty or not np.isfinite(obs_num):
            continue
        min_v, max_v = train_num.min(skipna=True), train_num.max(skipna=True)
        if not np.isfinite(min_v) or not np.isfinite(max_v) or min_v == max_v:
            continue
        mean_cluster = cluster_num.mean(skipna=True)
        if not np.isfinite(mean_cluster):
            continue
        obs_norm = float(np.clip((obs_num - min_v) / (max_v - min_v), 0, 1))
        mean_norm = float(np.clip((mean_cluster - min_v) / (max_v - min_v), 0, 1))
        rows.append({
            "variable": v,
            "variable_mostrar": netejar_nom_variable(v),
            "observacio_original": obs_num,
            "mitjana_cluster_original": mean_cluster,
            "observacio_norm": obs_norm,
            "mitjana_cluster_norm": mean_norm,
            "diferencia": abs(obs_norm - mean_norm),
        })
    return pd.DataFrame(rows)


def seleccionar_variables_radial(dades_radial, max_variables=14):
    if dades_radial.empty:
        return dades_radial
    return dades_radial.sort_values("diferencia", ascending=False).head(max_variables)


def crear_radial_plot(dades_radial, cluster_pred, max_variables=14):
    d = seleccionar_variables_radial(dades_radial, max_variables)
    if d.empty:
        return None
    cats = d["variable_mostrar"].tolist()
    obs = d["observacio_norm"].tolist()
    mean = d["mitjana_cluster_norm"].tolist()
    cats_c = cats + [cats[0]]
    obs_c = obs + [obs[0]]
    mean_c = mean + [mean[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=mean_c, theta=cats_c, fill="toself", name=f"Mitjana {cluster_pred}", line=dict(color="black", width=4)))
    fig.add_trace(go.Scatterpolar(r=obs_c, theta=cats_c, fill="toself", name="Observació", line=dict(color="white", width=4), marker=dict(color="white", line=dict(color="black", width=1))))
    fig.update_layout(title="Comparació radial amb la mitjana del clúster", polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
    return fig


def crear_grafic_barres_comparacio(dades_radial, cluster_pred, max_variables=14):
    d = seleccionar_variables_radial(dades_radial, max_variables)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["variable_mostrar"], y=d["mitjana_cluster_norm"], name=f"Mitjana {cluster_pred}", marker=dict(color="black")))
    fig.add_trace(go.Bar(x=d["variable_mostrar"], y=d["observacio_norm"], name="Observació", marker=dict(color="white", line=dict(color="black", width=1.5))))
    fig.update_layout(title="Comparació normalitzada observació vs clúster", yaxis=dict(range=[0, 1]), barmode="group", margin=dict(b=150))
    fig.update_xaxes(tickangle=45)
    return fig


def crear_grafic_diferencies(dades_radial, max_variables=14):
    d = seleccionar_variables_radial(dades_radial, max_variables).copy()
    d["diferencia_signada"] = d["observacio_norm"] - d["mitjana_cluster_norm"]
    d = d.sort_values("diferencia_signada")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["diferencia_signada"], y=d["variable_mostrar"], orientation="h", name="Diferència"))
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="black")
    fig.update_layout(title="Diferència respecte a la mitjana del clúster", showlegend=False)
    return fig


def obtenir_top10_similars(df_input, cluster_pred, reference_train):
    if reference_train.empty or "k3" not in reference_train.columns:
        return pd.DataFrame()
    cluster_df = reference_train[reference_train["k3"].astype(str) == str(cluster_pred)].copy()
    if cluster_df.empty:
        return pd.DataFrame()

    vars_common = [v for v in df_input.columns if v in reference_train.columns and normalitzar_nom(v) not in VARIABLES_EXCLOSES]
    mat_cols = []
    obs_vals = []
    for v in vars_common:
        train_num = _to_numeric_series(reference_train[v])
        cluster_num = _to_numeric_series(cluster_df[v])
        obs_num = _to_numeric_series(pd.Series([df_input[v].iloc[0]])).iloc[0]
        if train_num.dropna().empty or not np.isfinite(obs_num):
            continue
        min_v, max_v = train_num.min(skipna=True), train_num.max(skipna=True)
        if not np.isfinite(min_v) or not np.isfinite(max_v) or min_v == max_v:
            continue
        norm = ((cluster_num - min_v) / (max_v - min_v)).clip(0, 1)
        norm = norm.fillna(norm.median(skipna=True) if np.isfinite(norm.median(skipna=True)) else 0)
        mat_cols.append(norm.to_numpy())
        obs_vals.append(float(np.clip((obs_num - min_v) / (max_v - min_v), 0, 1)))

    if not mat_cols:
        return pd.DataFrame()
    mat = np.vstack(mat_cols).T
    obs = np.array(obs_vals)
    dist = np.sqrt(np.mean((mat - obs) ** 2, axis=1))
    sim = np.clip(1 - dist, 0, 1)
    order = np.argsort(-sim)[:10]

    out = pd.DataFrame({
        "cas_train": cluster_df.index.to_numpy()[order],
        "cluster": cluster_pred,
        "similitud": np.round(sim[order], 4),
        "similitud_percentatge": np.round(sim[order] * 100, 2),
        "distancia": np.round(dist[order], 4),
        "n_variables_comparades": len(obs_vals),
    })
    return out

# ==============================================================================
# UI
# ============================================================================== 

st.markdown('<div class="main-title">📊 Aplicació de predicció de risc lesiu</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Versió ràpida 100% Python: sense Rscript durant l’execució de Streamlit.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Configuració")
    if ARTIFACT_DIR is None:
        st.error("No s'han trobat els artefactes Python")
        st.caption("Executa primer la conversió del model RDS.")
    else:
        st.success("Artefactes Python trobats")
        st.caption(str(ARTIFACT_DIR.relative_to(PROJECT_ROOT)))

if ARTIFACT_DIR is None:
    st.error("Falten `model_artifacts.json` i `xgboost_model.json`. Executa `convertir_modelo_rds_a_python.R` una vegada.")
    st.code("Rscript convertir_modelo_rds_a_python.R output/modeloXGBoost.RDS data/datos_modelling.RDS output/python_model")
    st.stop()

artifacts = carregar_artifacts(str(ARTIFACT_DIR))
booster = carregar_model(str(ARTIFACT_DIR))
reference_train = carregar_reference_train(str(ARTIFACT_DIR))
variables = artifacts.get("variables", [])

tab_enquesta, tab_csv, tab_info = st.tabs(["📝 Enquesta individual", "📁 Predicció amb CSV", "ℹ️ Informació"])

with tab_enquesta:
    st.subheader("📝 Enquesta individual")
    with st.form("formulari_enquesta"):
        respostes = {}
        variables_lesions = [v for v in variables if es_variable_localitzacio_lesio(v.get("name"))]
        variables_afectacions = [v for v in variables if es_variable_afectacio(v.get("name"))]
        variables_formulari = [v for v in variables if not es_variable_localitzacio_lesio(v.get("name")) and not es_variable_afectacio(v.get("name"))]
        cols = st.columns(2)

        for idx, var_meta in enumerate(variables_formulari):
            nom = var_meta.get("name")
            tipus = var_meta.get("type", "numeric")
            levels = var_meta.get("levels")
            etiqueta = netejar_nom_variable(nom)
            with cols[idx % 2]:
                if es_variable_fatiga_slider(nom):
                    min_v = valor_numerico_meta(var_meta.get("min"), 0)
                    max_v = valor_numerico_meta(var_meta.get("max"), 10)
                    med = valor_numerico_meta(var_meta.get("median"), 5)
                    valor = st.slider(etiqueta, min_value=min_v, max_value=max_v, value=float(np.clip(med, min_v, max_v)), step=1.0, help=f"Variable original: {nom}")
                    respostes[nom] = str(int(valor)) if tipus == "categorical" and float(valor).is_integer() else valor
                elif tipus == "categorical" and levels:
                    opcions = [str(x) for x in levels if str(x) != ""] or [""]
                    index_defecte = 0
                    if "No" in opcions and ("lesion" in normalitzar_nom(nom) or "lesio" in normalitzar_nom(nom)):
                        index_defecte = opcions.index("No")
                    valor = st.selectbox(etiqueta, opcions, index=index_defecte, format_func=lambda x, n=nom: format_opcio(n, x), help=f"Variable original: {nom}")
                    respostes[nom] = valor
                else:
                    med = valor_numerico_meta(var_meta.get("median"), 0)
                    kwargs = {"label": etiqueta, "value": med, "help": f"Variable original: {nom}"}
                    if var_meta.get("min") is not None: kwargs["min_value"] = valor_numerico_meta(var_meta.get("min"), 0)
                    if var_meta.get("max") is not None: kwargs["max_value"] = valor_numerico_meta(var_meta.get("max"), med)
                    respostes[nom] = st.number_input(**kwargs)

        st.markdown("### Lesions i afectacions")
        c1, c2 = st.columns(2)
        with c1:
            lesions_sel = st.multiselect(
                "Selecciona les localitzacions de lesió",
                [v.get("name") for v in variables_lesions],
                format_func=netejar_nom_variable,
                help="Selecciona les zones on hi ha hagut lesió. Después podrás indicar cuántas veces se ha lesionado en cada zona."
            )
        with c2:
            afectacions_sel = st.multiselect(
                "Selecciona les estructures afectades",
                [v.get("name") for v in variables_afectacions],
                format_func=netejar_nom_variable
            )

        # ------------------------------------------------------------------
        # Número de veces lesionado por localización
        # ------------------------------------------------------------------
        # Estas variables se recogen y se añaden al dataframe final, pero NO se
        # usan todavía para construir la matriz de entrada del modelo.
        # construir_matriz_modelo() solo utiliza artifacts["variable_names"],
        # por tanto estas columnas extra no alteran la predicción actual.
        num_lesiones_por_zona = {}

        if len(lesions_sel) > 0:
            st.markdown("#### Nombre de vegades lesionat per zona")
            st.caption("Aquests camps es guarden com a informació addicional, però ara mateix no entren en el model predictiu.")

            cols_lesions_num = st.columns(2)
            for idx_lesio, nom_lesio in enumerate(lesions_sel):
                with cols_lesions_num[idx_lesio % 2]:
                    col_extra = f"num_lesions_{nom_lesio}"
                    num_lesiones_por_zona[col_extra] = st.number_input(
                        f"Vegades: {netejar_nom_variable(nom_lesio)}",
                        min_value=1,
                        max_value=50,
                        value=1,
                        step=1,
                        help=f"Variable informativa extra. No s'utilitza encara al model: {col_extra}"
                    )

        for vm in variables_lesions:
            nom_lesio = vm.get("name")
            respostes[nom_lesio] = valor_binari_model(vm, nom_lesio in lesions_sel)

            # Si una lesión no está seleccionada, igualmente dejamos registrada
            # su frecuencia como 0 en una columna extra informativa.
            col_extra = f"num_lesions_{nom_lesio}"
            respostes[col_extra] = int(num_lesiones_por_zona.get(col_extra, 0))

        for vm in variables_afectacions:
            respostes[vm.get("name")] = valor_binari_model(vm, vm.get("name") in afectacions_sel)

        enviar = st.form_submit_button("Predir risc", type="primary")

    if enviar:
        df_enquesta = pd.DataFrame([respostes])
        try:
            resultats = predir_amb_model(df_enquesta, booster, artifacts)
            cluster_pred = resultats.loc[0, "prediccio_cluster"]
            st.success(f"Clúster predit: {cluster_pred}")
            components.html(semafor_risc(cluster_pred), height=170, scrolling=False)
            st.markdown(f'<div class="cluster-card"><b>Interpretació:</b><br>{descripcio_cluster(cluster_pred)}</div>', unsafe_allow_html=True)
            st.markdown(recomanacions_cluster(cluster_pred), unsafe_allow_html=True)

            dades_radial = obtenir_dades_radial(df_enquesta, cluster_pred, reference_train)
            if not dades_radial.empty:
                st.subheader("Comparació amb la mitjana del clúster")
                fig = crear_radial_plot(dades_radial, cluster_pred)
                if fig: st.plotly_chart(fig, use_container_width=True)
                c1, c2 = st.columns(2)
                with c1: st.plotly_chart(crear_grafic_barres_comparacio(dades_radial, cluster_pred), use_container_width=True)
                with c2: st.plotly_chart(crear_grafic_diferencies(dades_radial), use_container_width=True)

            top10 = obtenir_top10_similars(df_enquesta, cluster_pred, reference_train)
            if not top10.empty:
                st.subheader("Top 10 casos més semblants")
                st.dataframe(top10, use_container_width=True)
        except Exception as e:
            st.error("Hi ha hagut un error fent la predicció.")
            st.code(str(e))

with tab_csv:
    st.subheader("📁 Predicció mitjançant CSV")
    uploaded_file = st.file_uploader("Puja un fitxer CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df_csv = llegir_csv_robust(uploaded_file)
            st.dataframe(df_csv.head(20), use_container_width=True)
            vars_model = [v.get("name") for v in variables]
            missing = [v for v in vars_model if v not in df_csv.columns]
            if missing:
                st.warning("Variables que falten:")
                st.code(", ".join(missing))
            else:
                st.success("El CSV conté totes les variables necessàries.")
            if st.button("Predir clústers del CSV", type="primary"):
                resultats_csv = predir_amb_model(df_csv, booster, artifacts)
                cols = [c for c in ["prediccio_cluster", "nivell_risc", "semafor", "recomanacio"] if c in resultats_csv.columns]
                st.dataframe(resultats_csv[cols], use_container_width=True)
                resum = resultats_csv["prediccio_cluster"].value_counts().rename_axis("Cluster").reset_index(name="Freqüència")
                resum["Percentatge"] = (resum["Freqüència"] / resum["Freqüència"].sum() * 100).round(2)
                st.dataframe(resum, use_container_width=True)
                st.bar_chart(resum.set_index("Cluster")["Freqüència"])
                st.download_button("Descarregar resultats en CSV", resultats_csv.to_csv(index=False).encode("utf-8"), "resultats_prediccio_xgboost.csv", "text/csv")
        except Exception as e:
            st.error("No s'ha pogut processar el CSV.")
            st.code(str(e))
    else:
        st.info("Carrega un fitxer CSV per començar.")

with tab_info:
    st.subheader("ℹ️ Informació")
    st.write("Aquesta versió carrega el model XGBoost directament en Python i evita iniciar R en cada predicció.")
    df_variables = pd.DataFrame([{
        "Variable original": v.get("name"),
        "Pregunta en l'enquesta": netejar_nom_variable(v.get("name")),
        "Tipus": v.get("type"),
        "Valors possibles": ", ".join([format_opcio(v.get("name"), x) for x in (v.get("levels") or [])]),
    } for v in variables])
    st.dataframe(df_variables, use_container_width=True)
