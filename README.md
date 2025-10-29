# DeepSeek-OCR on Replicate

OCR model powered by DeepSeek, deployed on Replicate via Cog.

## Features

- ✅ Support for JPG, PNG, WebP images
- ✅ Support for PDF files
- ✅ 100+ language support
- ✅ Fast GPU inference
- ✅ Simple API

## Usage

### Via Replicate API

```bash
curl -X POST https://api.replicate.com/v1/predictions \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -d '{
    "version": "your-model-version-id",
    "input": {
      "image_url": "https://example.com/image.jpg",
      "prompt": "Extract text from this document"
    }
  }'
Response
json
{
  "output": "Extracted text content...",
  "status": "succeeded"
}
Model Info
Base Model: DeepSeek-OCR

Inference Engine: Replicate + Cog

GPU: NVIDIA (auto-detected)

Supported Formats: JPG, PNG, WebP, PDF

Deployment
Automatically deployed via GitHub + Replicate.
