# 🤖 AI Image Intelligence — Telegram Bot & REST API

An asynchronous, production-ready computer vision service powered by deep learning (**ResNet-50**). 

This project provides dual interfaces—an interactive **Telegram Bot** and a high-performance **FastAPI REST API**—built on a decoupled, non-blocking inference engine designed to handle concurrent requests efficiently.

---

##  Demo & Interfaces

### 1. Telegram Bot in Action
Users can interact directly with the bot on Telegram to receive instant AI classifications with top-5 confidence breakdowns:

![Bot Demo](images/image.png)
*AI analyzing an image and reporting predictions with confidence tiers*

### 2. Interactive Swagger API Docs
The REST API automatically exposes interactive OpenAPI documentation accessible via your browser at `http://localhost:8000/docs`:

- `POST /classify` — Upload raw image multipart/form-data for real-time classification.
- `GET /health` — Health check endpoint for uptime and container orchestration.

---

##  System Architecture

The project employs a modular, shared-core architecture. Both delivery channels (Telegram Long Polling & FastAPI HTTP endpoints) decouple network I/O from heavy CPU tensor operations:

```
                            ┌────────────────────────┐
                            │      Client Request    │
                            └─────┬────────────┬─────┘
                                  │            │
             Telegram Image Upload│            │ HTTP POST /classify
                                  ▼            ▼
                         ┌─────────────┐  ┌─────────────┐
                         │ Telegram Bot│  │ FastAPI App │
                         │(main_ai.py) │  │  (api.py)   │
                         └──────┬──────┘  └──────┬──────┘
                                │                │
               Network Validation│                │ MIME & Size Guards
                                ▼                ▼
                         ┌──────────────────────────────┐
                         │      asyncio.to_thread()     │  <-- Offloads CPU inference
                         └──────────────┬───────────────┘
                                        ▼
                         ┌──────────────────────────────┐
                         │    Shared Classifier Core    │
                         │       (classifier.py)        │
                         │  - In-Memory Byte Stream     │
                         │  - Pillow / NumPy Pipeline   │
                         │  - ResNet-50 Inference       │
                         │  - Confidence Scoring        │
                         └──────────────────────────────┘
```

---

##  Key Engineering Highlights

### 1. Non-Blocking Concurrency (`asyncio.to_thread`)
Deep learning inference with TensorFlow and image resizing with Pillow are CPU-bound and synchronous. If executed directly inside an asynchronous handler, they would freeze Python's single-threaded event loop, blocking all other incoming messages. 

We offload inference into a background thread pool via `await asyncio.to_thread(...)`, keeping the event loop responsive to concurrent traffic.

### 2. Zero-Disk In-Memory Processing
Downloaded images and API uploads are handled entirely in RAM using byte streams (`io.BytesIO`). Eliminating temporary disk writes:
- Minimizes disk I/O latency.
- Removes disk cleanup overhead and dangling file leaks.
- Enhances data privacy and security.

### 3. Defensive Engineering & DoS Protection
To safeguard system resources against malformed inputs and malicious payloads:
- **Pre-flight File Size Limits**: Enforces a strict 10MB limit (`MAX_FILE_SIZE_BYTES`) before downloading or processing full payloads.
- **Decompression Bomb Guard**: Validates image dimensions against a 4096px threshold (`MAX_IMAGE_DIMENSION`) to prevent memory exhaustion attacks.
- **MIME & Corrupt File Validation**: Traps unidentifiable or corrupt files via `PIL.UnidentifiedImageError` with clean user-facing error messages.

### 4. Model Uncertainty & Tiered Confidence
Neural networks apply softmax normalization, forcing predictions to sum to 100% even on nonsensical or out-of-distribution inputs. We implement threshold-based confidence tiering:
- **$\ge 60\%$ (High)**: Clear, primary identification.
- **$30\% - 60\%$ (Moderate)**: Prompts the user to review the top 5 alternative guesses.
- **$< 30\%$ (Low)**: Warns the user of low certainty and requests a clearer photo.

---

##  Project Structure

```
AI-ImageClassifierBot/
├── classifier.py         # Modular AI inference engine (ResNet-50, Pillow pipeline)
├── config.py             # Centralized settings, thresholds, and environment loader
├── main_ai.py            # Asynchronous Telegram bot delivery layer (Polling)
├── api.py                # FastAPI REST API with Swagger docs and validation
├── requirements.txt      # Production dependencies
├── LEARNING_JOURNAL.md   # Architectural decisions, changelog, and interview notes
├── images/               # Demo assets and screenshots
├── run_bot.bat           # Windows bot launcher
├── setup_python312.bat   # Automated Python 3.12 environment setup script
└── set_token.bat         # Secure token configuration script
```

---

##  Quick Start Guide

### Prerequisites
- **Python 3.12** (TensorFlow 2.18 requires Python 3.9–3.12 on Windows)
- A Telegram Bot Token from [@BotFather](https://t.me/botfather) (if running the Telegram bot)

### 1. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/your-username/AI-Classifier-Bot.git
cd AI-Classifier-Bot/AI-ImageClassifierBot

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

*(On Windows, you can alternatively double-click `setup_python312.bat` for automated setup).*

### 2. Configure Environment

Create a `.env` file in the project root:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
```

*(Or run `set_token.bat your_token_here` on Windows).*

---

##  Running the Services

### Option A: Launch the Telegram Bot
```bash
python main_ai.py
```
*(Or double-click `run_bot.bat`).*

Open Telegram, search for your bot username, send `/start`, and upload any image!

### Option B: Launch the FastAPI REST Server
```bash
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

- **Interactive Swagger UI**: Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check**:
  ```bash
  curl http://127.0.0.1:8000/health
  ```
- **Classify an Image via cURL**:
  ```bash
  curl -X POST "http://127.0.0.1:8000/classify" \
       -H "accept: application/json" \
       -H "Content-Type: multipart/form-data" \
       -F "file=@your_photo.jpg"
  ```

#### Sample API Response:
```json
{
  "properties": {
    "format": "JPEG",
    "size_kb": 128.4,
    "width": 1024,
    "height": 768,
    "mode": "RGB"
  },
  "best_guess": {
    "label": "Golden Retriever",
    "confidence_pct": "89.4%",
    "tier": "high"
  },
  "top_5": [
    {"rank": 1, "label": "Golden Retriever", "confidence": 0.8942},
    {"rank": 2, "label": "Labrador Retriever", "confidence": 0.0521},
    {"rank": 3, "label": "Cocker Spaniel", "confidence": 0.0145},
    {"rank": 4, "label": "Kuvasz", "confidence": 0.0083},
    {"rank": 5, "label": "Tennis Ball", "confidence": 0.0031}
  ]
}
```

---

##  Engineering Journal & Revision Notes

For an in-depth record of each architectural refactor, debugging story, and technical concept learned throughout this project, see [`LEARNING_JOURNAL.md`](LEARNING_JOURNAL.md).

It documents:
- Eliminating blocking event-loop calls (`asyncio.to_thread` vs synchronous handlers).
- Migrating from monolithic scripts to modular architecture.
- Defensive programming against decompression bombs and untrusted uploads.
- Secure secrets management with decoupled configuration.

---

##  License
MIT License. Built for learning and portfolio demonstration.