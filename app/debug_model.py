# ==============================================================================
# debug_model.py
# Diagnóstico independiente de la conversión RDS -> Python para XGBoost.
#
# Uso recomendado desde la raíz del proyecto:
#   python debug_model.py --artifact-dir output/python_model
#
# También puedes indicar una fila concreta:
#   python debug_model.py --artifact-dir output/python_model --row 0
#   python debug_model.py --artifact-dir output/python_model --cluster Cluster3
# ==============================================================================

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


VAR_FATIGA_LARGA = (
    "cual_es_tu_percepcion_de_fatiga_despues_de_la_ultima_competicion_"
    "entrenamiento_de_la_semana"
)


def as_list(value):
    """Convierte valores del JSON a lista.

    jsonlite::toJSON(auto_unbox=TRUE) puede guardar:
      "columns": "edad_anos"
    en lugar de:
      "columns": ["edad_anos"]

    Si Python itera sobre un string, lo recorre letra a letra. Ese era uno de
    los bugs que podía dejar la matriz prácticamente vacía.
    """
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, dict):
        if len(value) == 0:
            return []
        return list(value.values())

    return [value]


def to_numeric_series(series):
    text = series.astype(str).str.strip().str.lower()

    replacements = {
        "sí": "1",
        "si": "1",
        "s": "1",
        "yes": "1",
        "true": "1",
        "no": "0",
        "n": "0",
        "false": "0",
    }

    text = text.replace(replacements).str.replace(",", ".", regex=False)

    return pd.to_numeric(text, errors="coerce")


def repair_missing_columns(df, artifacts):
    """Reconstruye variables necesarias si el reference_train no las contiene.

    En tus artefactos, el modelo/dummy_model espera la variable larga de fatiga,
    pero reference_train.csv no la contiene. Como sí existe perc_fatiga,
    reconstruimos la variable categórica usando perc_fatiga redondeada.
    """
    df = df.copy()

    variable_names = artifacts.get("variable_names", [])

    if (
        VAR_FATIGA_LARGA in variable_names
        and VAR_FATIGA_LARGA not in df.columns
        and "perc_fatiga" in df.columns
    ):
        fatiga = to_numeric_series(df["perc_fatiga"]).round()
        fatiga = fatiga.clip(lower=1, upper=10)
        df[VAR_FATIGA_LARGA] = (
            fatiga.astype("Int64")
            .astype(str)
            .replace("<NA>", "5")
        )

    return df


def build_model_matrix(df_input, artifacts, strict=False):
    df_input = repair_missing_columns(df_input, artifacts)

    variable_names = artifacts["variable_names"]
    variables_train = artifacts["variables_train"]
    feature_map = artifacts["feature_map"]

    missing = [
        variable
        for variable in variable_names
        if variable not in df_input.columns
    ]

    if missing and strict:
        raise ValueError(
            "Faltan variables de entrada: " + ", ".join(missing)
        )

    x_new = pd.DataFrame(
        0.0,
        index=df_input.index,
        columns=variables_train,
    )

    audit_rows = []

    for variable in variable_names:
        mapping = feature_map.get(variable, {})
        variable_type = mapping.get("type", "numeric")

        if variable not in df_input.columns:
            audit_rows.append(
                {
                    "variable": variable,
                    "type": variable_type,
                    "status": "MISSING_FILLED_ZERO",
                    "mapped_columns": "",
                    "valid_columns": "",
                    "first_input_value": "",
                }
            )
            continue

        if variable_type == "categorical":
            levels = [str(level) for level in as_list(mapping.get("levels"))]
            level_to_columns = mapping.get("level_to_columns", {}) or {}

            values = (
                df_input[variable]
                .astype(str)
                .str.replace(r"\.0$", "", regex=True)
            )

            invalid_values = sorted(
                set(values.dropna().unique()) - set(levels)
            )
            invalid_values = [
                value
                for value in invalid_values
                if value != "" and value.lower() != "nan"
            ]

            if invalid_values:
                audit_rows.append(
                    {
                        "variable": variable,
                        "type": "categorical",
                        "status": "INVALID_LEVEL",
                        "mapped_columns": "",
                        "valid_columns": "",
                        "first_input_value": str(values.iloc[0]),
                        "details": str(invalid_values),
                    }
                )

            for level, mapped_columns_raw in level_to_columns.items():
                mapped_columns = [
                    str(column)
                    for column in as_list(mapped_columns_raw)
                ]

                valid_columns = [
                    column
                    for column in mapped_columns
                    if column in x_new.columns
                ]

                if valid_columns:
                    mask = values == str(level)
                    x_new.loc[mask, valid_columns] = 1.0

                audit_rows.append(
                    {
                        "variable": variable,
                        "type": "categorical",
                        "status": "OK",
                        "level": str(level),
                        "mapped_columns": "|".join(mapped_columns),
                        "valid_columns": "|".join(valid_columns),
                        "first_input_value": str(values.iloc[0]),
                        "n_rows_activated": int(
                            x_new.loc[:, valid_columns].sum().sum()
                        )
                        if valid_columns
                        else 0,
                    }
                )

        else:
            values = to_numeric_series(df_input[variable])

            median = values.median(skipna=True)
            if not np.isfinite(median):
                median = 0.0

            values = values.fillna(median).astype(float)

            mapped_columns = [
                str(column)
                for column in as_list(mapping.get("columns"))
            ]

            valid_columns = [
                column
                for column in mapped_columns
                if column in x_new.columns
            ]

            for column in valid_columns:
                x_new[column] = values.values

            audit_rows.append(
                {
                    "variable": variable,
                    "type": "numeric",
                    "status": "OK" if valid_columns else "NO_VALID_COLUMN",
                    "mapped_columns": "|".join(mapped_columns),
                    "valid_columns": "|".join(valid_columns),
                    "first_input_value": float(values.iloc[0])
                    if len(values) > 0
                    else "",
                }
            )

    x_new = (
        x_new
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    audit = pd.DataFrame(audit_rows)

    return x_new, audit, missing


def predict_clusters(x_matrix, booster, artifacts):
    dmatrix = xgb.DMatrix(
        x_matrix.to_numpy(dtype=float),
        feature_names=list(x_matrix.columns),
    )

    raw_prediction = booster.predict(dmatrix)
    levels = artifacts["nivells_k3"]

    if raw_prediction.ndim == 1:
        # multi:softmax suele devolver directamente 0, 1, 2.
        numeric_prediction = np.rint(raw_prediction).astype(int)

        probabilities = None

    else:
        probabilities_array = raw_prediction.reshape(
            (len(x_matrix), len(levels))
        )

        numeric_prediction = np.argmax(
            probabilities_array,
            axis=1,
        )

        probabilities = pd.DataFrame(
            probabilities_array,
            columns=[f"prob_{level}" for level in levels],
            index=x_matrix.index,
        )

    numeric_prediction = np.clip(
        numeric_prediction,
        0,
        len(levels) - 1,
    )

    clusters = [
        levels[index]
        for index in numeric_prediction
    ]

    return raw_prediction, numeric_prediction, clusters, probabilities


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--artifact-dir",
        default="output/python_model",
        help="Carpeta con model_artifacts.json, xgboost_model.json y reference_train.csv.",
    )

    parser.add_argument(
        "--row",
        type=int,
        default=None,
        help="Índice de fila de reference_train.csv a diagnosticar.",
    )

    parser.add_argument(
        "--cluster",
        default=None,
        help="Primer caso de reference_train.csv cuyo k3 coincida con este cluster.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Predice todo reference_train.csv y genera resumen completo.",
    )

    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)

    artifacts_path = artifact_dir / "model_artifacts.json"
    model_path = artifact_dir / "xgboost_model.json"
    reference_path = artifact_dir / "reference_train.csv"

    if not artifacts_path.exists():
        raise FileNotFoundError(f"No existe: {artifacts_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"No existe: {model_path}")

    if not reference_path.exists():
        raise FileNotFoundError(f"No existe: {reference_path}")

    with open(artifacts_path, "r", encoding="utf-8") as file:
        artifacts = json.load(file)

    booster = xgb.Booster()
    booster.load_model(str(model_path))

    reference = pd.read_csv(reference_path)

    if "k3" not in reference.columns:
        raise ValueError("reference_train.csv no contiene la columna k3.")

    feature_names_booster = booster.feature_names or []
    variables_train = artifacts["variables_train"]

    print("\n==================== CHECK INICIAL ====================")
    print(f"Filas reference_train: {len(reference)}")
    print(f"Variables esperadas por artifacts: {len(variables_train)}")
    print(f"Variables del Booster: {len(feature_names_booster)}")
    print(f"Coinciden nombres Booster/artifacts: {feature_names_booster == variables_train}")
    print("\nDistribución real k3:")
    print(reference["k3"].value_counts())

    x_all, audit_all, missing = build_model_matrix(
        reference.drop(columns=["k3"]),
        artifacts,
        strict=False,
    )

    print("\n==================== MATRIZ CONSTRUIDA ====================")
    print(f"Shape matriz: {x_all.shape}")
    print(f"Columnas todo cero: {(x_all.sum(axis=0) == 0).sum()}")
    print(f"Filas todo cero: {((x_all != 0).sum(axis=1) == 0).sum()}")
    print(f"Media de variables no cero por fila: {((x_all != 0).sum(axis=1)).mean():.2f}")

    if missing:
        print("\nVariables ausentes en reference_train reconstruidas/ignoradas:")
        for variable in missing:
            print(f"- {variable}")

    raw, numeric, clusters, probabilities = predict_clusters(
        x_all,
        booster,
        artifacts,
    )

    prediction_summary = pd.Series(clusters, name="prediccion").value_counts()

    print("\n==================== PREDICCIÓN PYTHON ====================")
    print(prediction_summary)

    crosstab = pd.crosstab(
        reference["k3"],
        pd.Series(clusters, name="prediccion"),
    )

    print("\nTabla real vs predicha:")
    print(crosstab)

    diagnostics_dir = artifact_dir / "diagnostico_python"
    diagnostics_dir.mkdir(exist_ok=True)

    x_all.to_csv(
        diagnostics_dir / "matriz_xgboost_python.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit_all.to_csv(
        diagnostics_dir / "auditoria_feature_map.csv",
        index=False,
        encoding="utf-8-sig",
    )

    result = reference.copy()
    result["prediccion_python"] = clusters
    result["prediccion_numerica_python"] = numeric

    if probabilities is not None:
        result = pd.concat([result, probabilities], axis=1)

    result.to_csv(
        diagnostics_dir / "predicciones_python_reference_train.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nArchivos generados:")
    print(f"- {diagnostics_dir / 'matriz_xgboost_python.csv'}")
    print(f"- {diagnostics_dir / 'auditoria_feature_map.csv'}")
    print(f"- {diagnostics_dir / 'predicciones_python_reference_train.csv'}")

    if args.all:
        return

    if args.cluster is not None:
        candidates = reference.index[
            reference["k3"].astype(str) == str(args.cluster)
        ]

        if len(candidates) == 0:
            raise ValueError(f"No hay filas con k3 = {args.cluster}")

        selected_index = int(candidates[0])

    elif args.row is not None:
        selected_index = int(args.row)

    else:
        selected_index = int(reference.index[0])

    print("\n==================== CASO INDIVIDUAL ====================")
    print(f"Fila seleccionada: {selected_index}")
    print(f"Cluster real k3: {reference.loc[selected_index, 'k3']}")
    print(f"Cluster predicho Python: {result.loc[selected_index, 'prediccion_python']}")
    print(f"Predicción bruta Python: {raw[selected_index]}")

    non_zero_features = x_all.loc[selected_index][
        x_all.loc[selected_index] != 0
    ].sort_index()

    print("\nFeatures no cero enviadas a XGBoost:")
    print(non_zero_features)

    print("\nDiagnóstico terminado.")


if __name__ == "__main__":
    main()
