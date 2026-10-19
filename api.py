import asyncio
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from config import MAX_FILE_SIZE_BYTES, CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_LOW_THRESHOLD
from classifier import classify_image_bytes
from PIL import UnidentifiedImageError

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Image Intelligence API",
    description="REST API for ResNet50 Image Classification",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "ResNet50",
        "service": "image-classifier"
    }

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    # 1. MIME type validation
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type '{file.content_type}'. Please upload an image."
        )

    # 2. Read bytes
    img_data = await file.read()

    # 3. Size validation
    if len(img_data) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum allowed ({MAX_FILE_SIZE_BYTES // (1024*1024)}MB)."
        )

    # 4. Offload to thread pool
    try:
        properties, decoded_preds, result = await asyncio.to_thread(
            classify_image_bytes,
            img_data
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Corrupted or unreadable image file.")
    except Exception as e:
        logger.error("API classification error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal classification error.")

    # 5. Format Top 5
    top_5 = [
        {
            "rank": i + 1,
            "label": label.replace("_", " ").title(),
            "confidence": round(float(score), 4)
        }
        for i, (imagenet_id, label, score) in enumerate(decoded_preds)
    ]

    return {
        "properties": properties,
        "best_guess": {
            "label": result["label"].replace("_", " ").title(),
            "confidence": round(result["confidence"], 4)
        },
        "top_5": top_5
    }
