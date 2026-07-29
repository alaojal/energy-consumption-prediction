"""
FASTAPI + GRADIO SERVING APPLICATION
====================================

Production-ready application for serving an energy consumption prediction
model through both a REST API and an interactive Gradio web interface.

Architecture
------------
- FastAPI: High-performance REST API with automatic OpenAPI documentation.
- Gradio: Interactive web interface for manual testing and demonstrations.
- Pydantic: Input validation and response serialization.
"""
import logging

import gradio as gr
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference_pipeline.inference import predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__) 


# Initialize FastAPI application
API_VERSION = "1.0.0"
API_TITLE = "Energy Consumption Prediction API"
GRADIO_TITLE = "Hourly Energy Consumption Predictor"
MODEL_NAME = "GradientBoostingRegressor"

app = FastAPI(
    title=API_TITLE,
    description="REST API for predicting hourly energy consumption.",
    version=API_VERSION
    )

# ====== HEALTH CHECK ENDPOINT =====
# Health endpoint used by monitoring systems and load balancers.
# / -> simple landing endpoint to confirm API is alive
class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    model: str
    version: str

@app.get(
    "/",
    response_model=HealthResponse,
    summary="Health check",
    response_description="Current health status of the API.",
    tags=["Health"],
)
async def root() -> HealthResponse:
    """
    Verify that the API service is running.

    Returns:
        HealthResponse:
            Current service status and version information.
    """
    return HealthResponse(
        status="healthy",
        service=API_TITLE,
        model=MODEL_NAME,
        version=API_VERSION,
    )

# ======= REQUEST DATA SCHEMA =====
# Pydantic model for automatic validation and API documentation
class EnergyData(BaseModel):
    """
    Energy data schema for energy consumption prediction.

    Schema describing the input features required by the
    energy consumption prediction model.

    These features correspond to the engineered date features
    used during model training.    
    """

    year: int = Field(ge=1900, 
                      le=2100, 
                      description="Calendar year.", 
                      json_schema_extra={"example": 2020}
                    )
    month: int = Field(ge=1, 
                       le=12, 
                       description="Month of the year.", 
                       json_schema_extra={"example": 2}
                    ) 
    day: int = Field(ge=1, 
                     le=31, 
                     description="Day of the month.", 
                     json_schema_extra={"example": 15}
                    )                          
    hour: int = Field(ge=0, 
                      le=23, 
                      description="Hour of the day (0-23).", 
                      json_schema_extra={"example": 14}
                    )                         # Hour of the day (0–23).

# ============================
# Response Models
# ============================

class PredictionResponse(BaseModel):
    """
    Predicted hourly energy consumption returned by the API.
    """

    predicted_AEP_MW: float = Field(
        description="Predicted hourly energy consumption in megawatts (MW).",
        json_schema_extra={"example": 15842.73},
    )

# ===== MAIN PREDICTION API ENDPOINT ====
@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=200,
    summary="Predict hourly energy consumption",
    description=(
        "Generate an hourly energy consumption prediction "
        "using the trained GradientBoostingRegressor model."
    ),
    response_description="Predicted hourly energy consumption.",
    tags=["Prediction"],
)

def get_prediction(data: EnergyData) -> PredictionResponse:
    """
    Generate an hourly energy consumption prediction.

    Workflow:
    1. Validate the request payload.
    2. Convert the input into a pandas DataFrame.
    3. Pass the data through the inference pipeline.
    4. Return the predicted energy consumption.

    Returns:
        PredictionResponse:
            Predicted hourly energy consumption in megawatts (MW).
    """
    try:
        # Convert the Pydantic model to a DataFrame.
        df = pd.DataFrame([data.model_dump()])
        result = predict(df)
        prediction = float(result["predicted_AEP_MW"].iloc[0])
        return PredictionResponse(
            predicted_AEP_MW=prediction
        )
    except Exception:
        logger.exception("API prediction failed")        # or logger.exception(...)

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again later."
        )
    

# ==== GRADIO WEB INTERFACE ====
def gradio_interface(
        year: int, 
        month: int, 
        day: int, 
        hour: int        
) -> str: 
    """
    Gradio interface function that processes user inputs and returns a prediction.

    This function:
    1. Takes individual form inputs from Gradio UI
    2. Constructs the data dictionary matching the API schema
    3. Calls the same inference pipeline used by the API
    4. Returns user-friendly prediction
    """

    # Build a single-row DataFrame matching the model input schema.
    data = {
        "year": year,    
        "month": month,  
        "day": day,      
        "hour": hour     
    }

    try:
        # Reuse the same inference pipeline as the REST API.
        if not (1900 <= year <= 2100):
            return "Invalid year."
        if not (1 <= month <= 12):
            return "Invalid month."
        if not (1 <= day <= 31):
            return "Invalid day."
        if not (0 <= hour <= 23):
            return "Invalid hour."
        
        df = pd.DataFrame([data])
        result = predict(df)
        prediction = result["predicted_AEP_MW"].iloc[0]

        return f"Predicted Energy Consumption: {prediction:.2f} MW"

    except Exception:
        logger.exception("Gradio prediction failed")
        return "Prediction failed. Please verify the input values." 

   

# ==== GRADIO UI CONFIGURATION ===
# Configure the Gradio web interface.
demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Number(label="Year", value=2020, minimum=1900, maximum=2100),
        gr.Number(label="Month", value=1, minimum=1, maximum=12),
        gr.Number(label="Day", value=1, minimum=1, maximum=31),
        gr.Number(label="Hour", value=12, minimum=0, maximum=23)
    ],
    outputs=gr.Textbox(label="Predicted Energy Consumption (MW)", lines=2),
    title=GRADIO_TITLE,
    description="""
**Predict energy consumption using machine learning**

Enter the engineered date features below to predict hourly energy consumption. 
The model is a GradientBoostingRegressor trained on historical hourly energy consumption data.
""", 
theme=gr.themes.Soft() # Professional appearance
)    

# ==== MOUNT GRADIO UI INTO FASTAPI ====
# This creates the /ui endpoint that serves the Gradio interface
# IMPORTANT: This must be the final line to properly integrate Gradio with FASTAPI
app = gr.mount_gradio_app(
    app,            # FASTAPI application
    demo,           # Gradio interface
    path="/ui"      # URL path where Gradio will be accessible
)
