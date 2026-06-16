from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import io
import os
from data_processing import DataProcessor
from ml_models import AdvancedHealthcareModels

app = FastAPI(title="Health API V2")

# Setup directories for static and templates
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global state to mimic Streamlit's session state (for demonstration)
global_data = {
    "df_raw": None,
    "df_processed": None,
    "processor": DataProcessor(),
    "models": AdvancedHealthcareModels(),
    "last_uploaded_name": None
}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return JSONResponse(status_code=400, content={"message": "Invalid file format. Use CSV or Excel."})

        # Process data
        processor = global_data["processor"]
        df_processed = processor.preprocess_data(df, training=True)
        
        # Extract features for training
        feature_names = processor.get_feature_names()
        X_train = df_processed[[c for c in feature_names if c in df_processed.columns]]
        
        # Train Models
        models = global_data["models"]
        models.train_models(X_train, df_processed)

        # Store in state
        global_data["df_raw"] = df
        global_data["df_processed"] = df_processed
        global_data["last_uploaded_name"] = file.filename

        return {"message": "Data processed and models trained successfully!", "filename": file.filename}

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error processing file: {str(e)}"})

@app.get("/api/dashboard_data")
async def get_dashboard_data(dosha: str = "All Diseases"):
    df = global_data["df_raw"]
    if df is None:
         return JSONResponse(status_code=400, content={"message": "No data uploaded yet."})

    # Filter data
    filtered_df = df.copy()
    if dosha != "All Diseases" and 'Dosha' in df.columns:
        filtered_df = df[df['Dosha'] == dosha]

    # Calculate metrics
    total_patients = int(len(filtered_df))
    
    avg_recovery = 0
    if 'Treatment days' in filtered_df.columns and total_patients > 0:
        avg_recovery = float(filtered_df['Treatment days'].mean())
        
    common_condition = "N/A"
    if 'Dosha' in filtered_df.columns and total_patients > 0:
        if not filtered_df['Dosha'].mode().empty:
            common_condition = str(filtered_df['Dosha'].mode()[0])

    # Chart Data: Treatment Duration by Age Group
    age_treatment_data = {}
    if 'Age group' in filtered_df.columns and 'Treatment days' in filtered_df.columns:
         plot_df = filtered_df[['Age group', 'Treatment days']].dropna()
         plot_df['Treatment days'] = pd.to_numeric(plot_df['Treatment days'], errors='coerce')
         age_agg = plot_df.groupby('Age group')['Treatment days'].mean().to_dict()
         age_treatment_data = {str(k): round(float(v), 2) for k, v in age_agg.items()}

    # Chart Data: Gender Distribution
    gender_data = {}
    if 'Gender' in filtered_df.columns:
         g_counts = filtered_df['Gender'].value_counts().to_dict()
         gender_data = {str(k): int(v) for k, v in g_counts.items()}

    # Chart Data: Visit Type
    visit_data = {}
    if 'Visit type' in filtered_df.columns:
         v_counts = filtered_df['Visit type'].value_counts().to_dict()
         visit_data = {str(k): int(v) for k, v in v_counts.items()}
         
    dosha_options = ["All Diseases"]
    if 'Dosha' in df.columns:
        dosha_options.extend([str(x) for x in df['Dosha'].dropna().unique()])

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

@app.post("/api/clear")
async def clear_data():
    global_data["df_raw"] = None
    global_data["df_processed"] = None
    global_data["last_uploaded_name"] = None
    return {"message": "Data cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
