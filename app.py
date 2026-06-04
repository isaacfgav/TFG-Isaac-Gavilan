# ==============================================================================
# [TFG] app.py
#
# Aplicació Streamlit per predir el clúster amb modeloXGBoost.RDS
# ==============================================================================

import streamlit as st
import pandas as pd
import subprocess
import tempfile
from pathlib import Path

# ------------------------------------------------------------------------------
# Configuració general de l'app

st.set_page_config(
    page_title="Predicció de clúster - XGBoost",
    page_icon="📊",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent

R_SCRIPT = PROJECT_ROOT / "syntax" / "06_Predict_XGBoost.R"
MODEL_PATH = PROJECT_ROOT / "output" / "modeloXGBoost.RDS"

# ------------------------------------------------------------------------------
# Títol principal

st.title("📊 Aplicació de predicció de clúster")

st.markdown(
    """
    Aquesta aplicació utilitza el model **XGBoost** entrenat en R i guardat en format `.RDS`
    per predir el clúster d'un nou conjunt d'observacions.
    """
)

# ------------------------------------------------------------------------------
# Comprovació dels fitxers necessaris

st.subheader("Model utilitzat")

if MODEL_PATH.exists():
    st.success(f"Model trobat correctament: `{MODEL_PATH.name}`")
else:
    st.error("No s'ha trobat el fitxer `modeloXGBoost.RDS`.")
    st.write("Assegura't que el fitxer està a la mateixa carpeta que `app.py`.")
    st.stop()

if R_SCRIPT.exists():
    st.success(f"Script R trobat correctament: `{R_SCRIPT.name}`")
else:
    st.error("No s'ha trobat el fitxer `predict_xgboost.R`.")
    st.write("Assegura't que el fitxer està a la mateixa carpeta que `app.py`.")
    st.stop()

# ------------------------------------------------------------------------------
# Càrrega del fitxer CSV

st.subheader("1. Carrega les dades")

uploaded_file = st.file_uploader(
    "Puja un fitxer CSV amb les variables necessàries per al model",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("2. Vista prèvia de les dades carregades")

        st.dataframe(df, use_container_width=True)

        st.write(f"Files carregades: **{df.shape[0]}**")
        st.write(f"Columnes carregades: **{df.shape[1]}**")

        # ----------------------------------------------------------------------
        # Botó per fer la predicció

        if st.button("Predir clúster", type="primary"):

            with tempfile.TemporaryDirectory() as tmpdir:

                tmpdir = Path(tmpdir)

                input_path = tmpdir / "input.csv"
                output_path = tmpdir / "output.csv"

                df.to_csv(input_path, index=False, encoding="utf-8")

                cmd = [
                    "Rscript",
                    str(R_SCRIPT),
                    str(input_path),
                    str(output_path),
                    str(MODEL_PATH)
                ]

                with st.spinner("Executant el model XGBoost en R..."):

                    result = subprocess.run(
                        cmd,
                        cwd=PROJECT_ROOT,
                        capture_output=True,
                        text=True
                    )

                if result.returncode != 0:

                    st.error("Hi ha hagut un error executant el model.")
                    st.write("Detall de l'error:")
                    st.code(result.stderr)

                else:

                    resultats = pd.read_csv(output_path)

                    st.subheader("3. Resultats de la predicció")

                    st.dataframe(resultats, use_container_width=True)

                    # ----------------------------------------------------------
                    # Resum dels clústers predits

                    if "prediccio_cluster" in resultats.columns:

                        st.subheader("4. Resum dels clústers predits")

                        resum = (
                            resultats["prediccio_cluster"]
                            .value_counts()
                            .reset_index()
                        )

                        resum.columns = ["Cluster", "Freqüència"]

                        resum["Percentatge"] = (
                            resum["Freqüència"] / resum["Freqüència"].sum() * 100
                        ).round(2)

                        st.dataframe(resum, use_container_width=True)

                        st.bar_chart(
                            resum.set_index("Cluster")["Freqüència"]
                        )

                    # ----------------------------------------------------------
                    # Botó per descarregar els resultats

                    csv_resultats = resultats.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Descarregar resultats en CSV",
                        data=csv_resultats,
                        file_name="resultats_prediccio_xgboost.csv",
                        mime="text/csv"
                    )

    except Exception as e:

        st.error("No s'ha pogut processar el fitxer CSV.")
        st.exception(e)

else:

    st.info("Carrega un fitxer CSV per començar.")
