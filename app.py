from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import io
import os
import traceback

from data_processing import DataProcessor
from ml_models import AdvancedHealthcareModels

app = FastAPI(title="Health Analytics Pro")

# =========================
# Render-safe paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# =========================
# Global State
# =========================
global_data = {
    "df_raw": None,
    "df_processed": None,
    "processor": DataProcessor(),
    "models": AdvancedHealthcareModels(),
    "last_uploaded_name": None
}


# =========================
# Home Page
# =========================
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# =========================
# Upload Dataset
# =========================
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        if file.filename.endswith(".csv"):
            df = pd.read_csv(
                io.StringIO(contents.decode("utf-8"))
            )

        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(
                io.BytesIO(contents)
            )

        else:
            return JSONResponse(
                status_code=400,
                content={
                    "message":
                    "Invalid file format. Please upload CSV or Excel."
                }
            )

        print("\n========== UPLOADED DATA ==========")
        print(df.head())
        print(df.columns.tolist())
        print(df.dtypes)

        # Process Data
        processor = global_data["processor"]
        df_processed = processor.preprocess_data(
            df,
            training=True
        )

        # Feature Selection
        feature_names = processor.get_feature_names()

        available_features = [
            col for col in feature_names
            if col in df_processed.columns
        ]

        X_train = df_processed[available_features]

        # Train Models
        models = global_data["models"]
        models.train_models(X_train, df_processed)

        # Save State
        global_data["df_raw"] = df
        global_data["df_processed"] = df_processed
        global_data["last_uploaded_name"] = file.filename

        return {
            "message":
            "Data processed and models trained successfully!",
            "filename":
            file.filename
        }

    except Exception as e:

        print("\n========== ERROR ==========")
        print(traceback.format_exc())

        return JSONResponse(
            status_code=500,
            content={
                "message": str(e),
                "traceback": traceback.format_exc()
            }
        )


# =========================
# Dashboard Data
# =========================
@app.get("/api/dashboard_data")
async def get_dashboard_data(
    dosha: str = "All Diseases"
):
    df = global_data["df_raw"]

    if df is None:
        return JSONResponse(
            status_code=400,
            content={
                "message":
                "No data uploaded yet."
            }
        )

    filtered_df = df.copy()

    if (
        dosha != "All Diseases"
        and "Dosha" in filtered_df.columns
    ):
        filtered_df = filtered_df[
            filtered_df["Dosha"] == dosha
        ]

    total_patients = len(filtered_df)

    # Safe Treatment Days Calculation
    avg_recovery = 0

    if (
        "Treatment days" in filtered_df.columns
        and total_patients > 0
    ):

        filtered_df["Treatment days"] = pd.to_numeric(
            filtered_df["Treatment days"],
            errors="coerce"
        )

        avg_recovery = float(
            filtered_df["Treatment days"]
            .fillna(0)
            .mean()
        )

    common_condition = "N/A"

    if (
        "Dosha" in filtered_df.columns
        and total_patients > 0
    ):
        mode_vals = filtered_df["Dosha"].mode()

        if not mode_vals.empty:
            common_condition = str(mode_vals.iloc[0])

    # =========================
    # Age Chart
    # =========================
    age_treatment_data = {}

    if (
        "Age group" in filtered_df.columns
        and "Treatment days" in filtered_df.columns
    ):

        plot_df = filtered_df[
            ["Age group", "Treatment days"]
        ].copy()

        plot_df["Treatment days"] = pd.to_numeric(
            plot_df["Treatment days"],
            errors="coerce"
        )

        plot_df = plot_df.dropna()

        age_agg = (
            plot_df.groupby("Age group")["Treatment days"]
            .mean()
            .to_dict()
        )

        age_treatment_data = {
            str(k): round(float(v), 2)
            for k, v in age_agg.items()
        }

    # =========================
    # Gender Chart
    # =========================
    gender_data = {}

    if "Gender" in filtered_df.columns:
        gender_data = {
            str(k): int(v)
            for k, v in
            filtered_df["Gender"]
            .value_counts()
            .to_dict()
            .items()
        }

    # =========================
    # Visit Type Chart
    # =========================
    visit_data = {}

    if "Visit type" in filtered_df.columns:
        visit_data = {
            str(k): int(v)
            for k, v in
            filtered_df["Visit type"]
            .value_counts()
            .to_dict()
            .items()
        }

    dosha_options = ["All Diseases"]

    if "Dosha" in df.columns:
        dosha_options.extend(
            [
                str(x)
                for x in df["Dosha"]
                .dropna()
                .unique()
            ]
        )

    return {
        "metrics": {
            "total_patients": total_patients,
            "avg_recovery": round(avg_recovery, 1),
            "common_condition": common_condition
        },
        "charts": {
            "age_treatment": age_treatment_data,
            "gender_dist": gender_data,
            "visit_type": visit_data
        },
        "doshas": dosha_options
    }


# =========================
# Clear Data
# =========================
@app.post("/api/clear")
async def clear_data():

    global_data["df_raw"] = None
    global_data["df_processed"] = None
    global_data["last_uploaded_name"] = None

    return {
        "message": "Data cleared"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )