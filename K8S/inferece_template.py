from fastapi import FastAPI, Request, Path
from PIL import Image
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import requests
import os
from ..gcp.credentials_utils import Credentials_broker
from ..gcp.storage_utils import GCStorage_broker
from ..wandb.wandbmanager import WandbManager
import uvicorn

app = FastAPI()
model = None
tokenizer = None
threshold = float(os.environ.get("THRESHOLD", 0.5))  # Get threshold from environment
model_name = os.environ.get("MODEL_NAME")
model_tag = os.environ.get("MODEL_TAG")

creds_obj = Credentials_broker()
creds = creds_obj.get_credentials()
gcs = GCStorage_broker(creds)

wandb_d = WandbManager("HumanDetection")

def load_model_and_tokenizer():
    global model, tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(f"{model_name}/{model_tag}")
    tokenizer = AutoTokenizer.from_pretrained(f"{model_name}/{model_tag}")
    model.eval()

@app.on_event("startup")
async def startup():
    load_model_and_tokenizer()

@app.post("/infer")
async def inference(video_uri: Path(..., title="Video URI", description="URL of the video to be analyzed"),
                   request: Request):
    if not model:
        return {"error": "Model not loaded"}
    try:
        temp_path = os.path.join("/tmp", os.path.basename(video_uri))
        # Download video from Cloud Storage
        downloaded_video = gcs.get_blob_file(video_uri, file_destination=temp_path)
        predictions = model(downloaded_video)
        human_score = predictions.logits[0][0]  # Assuming single-class prediction for human detection
        # logging inference on wandb
        wandb_d.log({"human_score": human_score})
        return {"threshold": threshold, "score": human_score}
    except Exception as e:
        return {"error": f"Inference error: {e}"}

if __name__ == "__main__":
    uvicorn.run("inference:app", host="0.0.0.0", port=8080)