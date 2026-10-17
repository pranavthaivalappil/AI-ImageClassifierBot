from dotenv import load_dotenv
load_dotenv()
import os

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