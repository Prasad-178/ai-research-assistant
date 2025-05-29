from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import boto3
import logging
from contextlib import asynccontextmanager
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_S3_BUCKET = os.getenv("MODEL_S3_BUCKET")
# IMPORTANT: MODEL_S3_KEY should now be the S3 PREFIX (folder) for your HuggingFace model files
# e.g., "qwen2.5-1.5b-instruct-hf/" if your files are under s3://<bucket>/qwen2.5-1.5b-instruct-hf/
MODEL_S3_KEY_PREFIX = os.getenv("MODEL_S3_KEY") 

# Path where the model files will be stored locally
LOCAL_MODEL_BASE_DIR = "/models/" 
# The actual path for the downloaded model will be derived from MODEL_S3_KEY_PREFIX
# e.g. /models/qwen2.5-1.5b-instruct-hf/
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_BASE_DIR, MODEL_S3_KEY_PREFIX.strip('/') if MODEL_S3_KEY_PREFIX else "downloaded_model")


# --- Global variables for the model and tokenizer ---
llm_globals = {} # Store model and tokenizer

def download_s3_folder(bucket_name, s3_folder_prefix, local_dir):
    """
    Download the contents of a folder directory in S3.
    """
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder_prefix):
        if "Contents" not in page:
            logger.warning(f"No contents found in S3 at s3://{bucket_name}/{s3_folder_prefix}")
            return False
        
        for obj in page['Contents']:
            # Path to download to (mimicking S3 structure)
            target_path = os.path.join(local_dir, os.path.relpath(obj['Key'], s3_folder_prefix))
            
            # Skip if it's a "directory" object (ends with / and size 0)
            if obj['Key'].endswith('/') and obj.get('Size', 0) == 0:
                os.makedirs(target_path, exist_ok=True)
                continue

            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            if not os.path.exists(target_path): # Avoid re-downloading
                logger.info(f"Downloading {obj['Key']} to {target_path}")
                s3.download_file(bucket_name, obj['Key'], target_path)
            else:
                logger.info(f"File {target_path} already exists. Skipping download.")
        return True # Assuming if Contents exist, we proceed
    return False # If loop completes without finding contents

# --- Lifespan Context Manager for Model Loading/Unloading ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Loading Hugging Face model...")

    if not MODEL_S3_BUCKET or not MODEL_S3_KEY_PREFIX:
        logger.error("MODEL_S3_BUCKET or MODEL_S3_KEY_PREFIX environment variables not set.")
        raise RuntimeError("S3 Bucket or Key prefix not configured for model download.")

    os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
    logger.info(f"Model directory {LOCAL_MODEL_PATH} ensured.")

    if not os.path.exists(os.path.join(LOCAL_MODEL_PATH, "config.json")):
        logger.info(f"Model files not found locally or incomplete at {LOCAL_MODEL_PATH}. Downloading from S3: s3://{MODEL_S3_BUCKET}/{MODEL_S3_KEY_PREFIX}")
        try:
            download_success = download_s3_folder(MODEL_S3_BUCKET, MODEL_S3_KEY_PREFIX, LOCAL_MODEL_PATH)
            if not download_success:
                 raise RuntimeError(f"Failed to download model files from s3://{MODEL_S3_BUCKET}/{MODEL_S3_KEY_PREFIX}. Check S3 path and permissions.")
            logger.info(f"Model files downloaded successfully to {LOCAL_MODEL_PATH}.")
        except Exception as e:
            logger.error(f"Error downloading model from S3: {e}")
            raise RuntimeError(f"Failed to download model: {e}")
    else:
        logger.info(f"Model files appear to exist at {LOCAL_MODEL_PATH}. Skipping download.")

    # Load Model and Tokenizer
    try:
        logger.info(f"Loading Hugging Face tokenizer from {LOCAL_MODEL_PATH}...")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        logger.info("Tokenizer loaded successfully.")

        model_kwargs = {
            "pretrained_model_name_or_path": LOCAL_MODEL_PATH,
            "torch_dtype": torch.float16,
        }

        if torch.cuda.is_available():
            logger.info("CUDA is available. Attempting to load model with 8-bit quantization on GPU.")
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            logger.warning("CUDA not available. Loading model on CPU without 8-bit quantization. This will be slower and consume more RAM.")
            model_kwargs["device_map"] = "cpu"
            # Explicitly remove quantization_config if CUDA is not available
            if "quantization_config" in model_kwargs:
                del model_kwargs["quantization_config"]


        logger.info(f"Loading Hugging Face model from {LOCAL_MODEL_PATH} with arguments: { {k: v for k, v in model_kwargs.items() if k != 'pretrained_model_name_or_path'} }")
        
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        logger.info(f"Hugging Face Model loaded successfully on device: {model.device}.")
        
        llm_globals['tokenizer'] = tokenizer
        llm_globals['model'] = model
        
    except Exception as e:
        logger.error(f"Fatal error loading Hugging Face model or tokenizer: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load model/tokenizer: {e}")
    
    yield # Application runs after this yield

    # Clean up
    logger.info("Application shutdown: Cleaning up Hugging Face model and tokenizer...")
    llm_globals.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Hugging Face model and tokenizer cleaned up.")

# --- FastAPI App Initialization ---
app = FastAPI(title="LLM Inference API with HuggingFace", version="1.1.0", lifespan=lifespan)

class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 150 # Changed from max_tokens
    temperature: float = 0.7
    top_p: float = 0.9
    # Add other generation parameters as needed (e.g., top_k, repetition_penalty)

@app.post("/infer")
async def infer_endpoint(request: PromptRequest):
    if 'model' not in llm_globals or 'tokenizer' not in llm_globals:
        logger.error("Model or tokenizer not loaded, cannot process inference request.")
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded or unavailable.")
    
    model = llm_globals['model']
    tokenizer = llm_globals['tokenizer']
    
    logger.info(f"Received inference request: prompt='{request.prompt[:50]}...', max_new_tokens={request.max_new_tokens}")
    
    try:
        # Prepare inputs for the model
        # The Qwen1.5 chat model expects a specific chat format.
        # For simplicity here, we'll just pass the raw prompt, but for optimal results,
        # you should use tokenizer.apply_chat_template if you have a chat interaction.
        messages = [{"role": "user", "content": request.prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Generate output
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            pad_token_id=tokenizer.eos_token_id # Important for generation
            # Add other relevant generation parameters here
        )
        
        # Decode the generated tokens, excluding the input prompt
        # generated_ids contains the full sequence (input + output)
        # We need to slice it to get only the new tokens
        response_text = tokenizer.decode(generated_ids[0][model_inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        logger.info(f"Inference successful. Response: '{response_text[:50]}...'")
        return {"response": response_text.strip()}
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/health")
async def health_check_endpoint():
    model_loaded = 'model' in llm_globals and llm_globals['model'] is not None
    tokenizer_loaded = 'tokenizer' in llm_globals and llm_globals['tokenizer'] is not None
    if model_loaded and tokenizer_loaded:
        return {"status": "healthy", "model_loaded": True, "tokenizer_loaded": True}
    else:
        return {"status": "unhealthy", "model_loaded": model_loaded, "tokenizer_loaded": tokenizer_loaded}

# For local testing with uvicorn (optional)
# import uvicorn
# if __name__ == "__main__":
#     # Set environment variables for local testing
#     os.environ["MODEL_S3_BUCKET"] = "your-s3-bucket-name"
#     os.environ["MODEL_S3_KEY"] = "your-s3-model-prefix/" # e.g., "qwen2.5-1.5b-instruct-hf/"
#     uvicorn.run(app, host="0.0.0.0", port=8000)
