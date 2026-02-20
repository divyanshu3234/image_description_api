from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import httpx
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import ipaddress
import socket

app = FastAPI()

# -------------------------
# CORS Configuration
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourfrontend.com"],  # Replace with your frontend domain
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# -------------------------
# Device Setup (GPU/CPU)
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model once at startup
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

# -------------------------
# Request Schema
# -------------------------
class ImageRequest(BaseModel):
    image_url: HttpUrl  # Built-in URL validation


# -------------------------
# SSRF Protection Helper
# -------------------------
def is_private_ip(hostname: str) -> bool:
    try:
        ip = ipaddress.ip_address(socket.gethostbyname(hostname))
        return ip.is_private or ip.is_loopback
    except Exception:
        return True  # If resolution fails, block it


# -------------------------
# API Endpoint
# -------------------------
@app.post("/describe-url")
async def describe_image(data: ImageRequest):
    try:
        # Block private/internal URLs
        if is_private_ip(data.image_url.host):
            raise HTTPException(status_code=400, detail="Private/internal URLs not allowed")

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(str(data.image_url))

        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch image")

        # Restrict image size (max 5MB)
        if len(response.content) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large (max 5MB)")

        image = Image.open(BytesIO(response.content)).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(**inputs)

        caption = processor.decode(output[0], skip_special_tokens=True)

        return {"description": caption}

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


# -------------------------
# Health Check
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}