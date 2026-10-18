import io
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions  # type: ignore[import]
from tensorflow.keras.utils import img_to_array  # type: ignore[import]

print("🧠 Loading AI model...")
model = ResNet50(weights='imagenet')
print("✅ AI model loaded successfully!")

def classify_image_bytes(img_data: bytes):
    """
    Takes raw image bytes, runs ResNet50 classification, 
    and returns image properties, top predictions, and the top result dict.
    """
    # 1. Open image from bytes
    img = Image.open(io.BytesIO(img_data))
    
    # 2. Extract properties
    properties = {
        "width": img.size[0],
        "height": img.size[1],
        "format": img.format or "Unknown",
        "mode": img.mode,
        "size_kb": len(img_data) // 1024
    }
    
    # 3. Preprocess for ResNet50
    img_resized = img.resize((224, 224))
    if img_resized.mode != 'RGB':
        img_resized = img_resized.convert('RGB')
        
    x = img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # 4. Predict
    preds = model.predict(x, verbose=0)
    decoded_preds = decode_predictions(preds, top=5)[0]
    
    result = {
        "label": decoded_preds[0][1],
        "confidence": float(decoded_preds[0][2])
    }
    
    return properties, decoded_preds, result
