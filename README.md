# 🖼️ BLIP Image Captioning API

A production-ready FastAPI service that generates image descriptions from a public image URL using the BLIP (Bootstrapping Language-Image Pretraining) model by Salesforce.

# Model Used  

Model: Salesforce/blip-image-captioning-base
Framework: PyTorch + Hugging Face Transformers

BLIP is a vision-language transformer trained for image captioning and multimodal understanding tasks


# Feature

Accepts an image URL

Fetches the image securely

Generates a caption using a transformer-based vision-language model

Returns a clean JSON response


**HOW TO FETCH DESCRIPTION**

```
curl -X POST "https://image-api-322039733047.us-central1.run.app/describe-url" \
  -H "Content-Type: application/json" \
  -d '{
        "image_url": "https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d"
      }'
{"description":"a man standing in a field of tall grass"}

```
