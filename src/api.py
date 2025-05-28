from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import os
import boto3 # AWS SDK for Python
import logging
from contextlib import asynccontextmanager # Import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_S3_BUCKET = os.getenv("MODEL_S3_BUCKET")
MODEL_S3_KEY = "qwen2.5-1.5b-instruct-q8_0.gguf"
# Path where the model will be stored inside the Docker container / on EC2
LOCAL_MODEL_DIR = "/models/"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, os.path.basename(MODEL_S3_KEY))

# --- Global variable for the Llama model ---
# This will be managed by the lifespan context
llm_models = {} # Use a dictionary to store models, as in lifespan example

# --- Lifespan Context Manager for Model Loading/Unloading ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Loading ML model...")

    # Ensure model directory exists
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    logger.info(f"Model directory {LOCAL_MODEL_DIR} ensured.")

    # Download model from S3 if it doesn't exist locally
    if not os.path.exists(LOCAL_MODEL_PATH):
        if not MODEL_S3_BUCKET or not MODEL_S3_KEY:
            logger.error("MODEL_S3_BUCKET or MODEL_S3_KEY environment variables not set.")
            raise RuntimeError("S3 Bucket or Key not configured for model download.")
        
        logger.info(f"Model not found at {LOCAL_MODEL_PATH}. Downloading from S3: s3://{MODEL_S3_BUCKET}/{MODEL_S3_KEY}")
        s3 = boto3.client('s3')
        try:
            s3.download_file(MODEL_S3_BUCKET, MODEL_S3_KEY, LOCAL_MODEL_PATH)
            logger.info(f"Model downloaded successfully to {LOCAL_MODEL_PATH}.")
        except Exception as e:
            logger.error(f"Error downloading model from S3: {e}")
            raise RuntimeError(f"Failed to download model: {e}")
    else:
        logger.info(f"Model already exists at {LOCAL_MODEL_PATH}. Skipping download.")

    # Load Model
    try:
        logger.info(f"Loading GGUF model from {LOCAL_MODEL_PATH}...")
        # For g4dn.xlarge (NVIDIA T4), n_gpu_layers=-1 should offload all layers.
        # If you encounter issues, try a specific number or 0 for CPU only.
        current_llm = Llama(
            model_path=LOCAL_MODEL_PATH,
            n_gpu_layers=-1,  # Offload all layers to GPU. Use 0 for CPU.
            n_ctx=2048,       # Context window size your model supports
            verbose=True,     # Enable llama.cpp verbose logging
        )
        llm_models['gguf_model'] = current_llm # Store in the dictionary
        logger.info("GGUF Model loaded successfully.")
    except Exception as e:
        logger.error(f"Fatal error loading Llama model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")
    
    yield # Application runs after this yield

    # Clean up the ML models and release the resources
    logger.info("Application shutdown: Cleaning up ML model...")
    llm_models.clear()
    logger.info("ML Model cleaned up.")

# --- FastAPI App Initialization ---
app = FastAPI(title="LLM Inference API", version="1.0.0", lifespan=lifespan) # Use the lifespan manager

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9
    # Add other generation parameters as needed

@app.post("/infer")
async def infer_endpoint(request: PromptRequest):
    if 'gguf_model' not in llm_models or llm_models['gguf_model'] is None:
        logger.error("Model not loaded, cannot process inference request.")
        raise HTTPException(status_code=503, detail="Model not loaded or unavailable.")
    
    current_llm = llm_models['gguf_model']
    logger.info(f"Received inference request: prompt='{request.prompt[:50]}...', max_tokens={request.max_tokens}")
    try:
        output = current_llm(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            echo=False  # Don't echo the prompt in the output
        )
        response_text = output["choices"][0]["text"]
        logger.info(f"Inference successful. Response: '{response_text[:50]}...'")
        return {"response": response_text}
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/health")
async def health_check_endpoint():
    if 'gguf_model' in llm_models and llm_models['gguf_model'] is not None:
        return {"status": "healthy", "model_loaded": True}
    else:
        return {"status": "unhealthy", "model_loaded": False}

# If running directly with uvicorn for local testing:
# import uvicorn
# if __name__ == "__main__":
#     # Ensure MODEL_S3_BUCKET is set if you run this locally for testing model download
#     # os.environ["MODEL_S3_BUCKET"] = "your-s3-bucket-for-testing" 
#     uvicorn.run(app, host="0.0.0.0", port=8000)
