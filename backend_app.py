from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from inference import load_model, run_all_tasks

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for frontend requests
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use the trained model directory when available, but fallback to donut-base for demo
MODEL_DIR = "./p2n-model" if os.path.exists("./p2n-model") else "naver-clova-ix/donut-base"
print(f"Loading model from {MODEL_DIR}...")
try:
    model, processor, device = load_model(MODEL_DIR)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # We enforce a small max_length so it returns quickly for the demo
        results = run_all_tasks(model, processor, device, temp_file, max_length=256)
        return {"filename": file.filename, "results": results}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
