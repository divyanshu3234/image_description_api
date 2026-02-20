# image_description_api

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