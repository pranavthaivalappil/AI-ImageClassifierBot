import logging
from dotenv import load_dotenv
load_dotenv()
import os


#Configure root logger format and level 
logging.basicConfig(
    format="%(asctime)s - [%(levelname)s] - [%(name)s] - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log",mode="a",encoding="utf-8") #Appending to bot.log

    ]
)


# Get bot token from environment variable
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
if not BOT_TOKEN:
    print("❌ Error: TELEGRAM_BOT_TOKEN not found in environment variables!")
    print("Please run: set_token.bat YOUR_BOT_TOKEN")
    exit(1)

 

# Type assertion for the linter
assert BOT_TOKEN is not None

# Confidence thresholds
CONFIDENCE_HIGH_THRESHOLD = 0.60
CONFIDENCE_LOW_THRESHOLD = 0.30

#Max file size: 10 MB (in bytes)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
# Max image dimensions: 4096 x 4096 pixels
MAX_IMAGE_DIMENSION = 4096
