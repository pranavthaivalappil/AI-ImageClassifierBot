import os
import io
import numpy as np
import requests
from PIL import Image
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# TensorFlow imports (these will work in Python 3.12 environment)
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.utils import load_img, img_to_array



# Load the AI model
print("🧠 Loading AI model...")
model = ResNet50(weights='imagenet')
print("✅ AI model loaded successfully!") 

# Get bot token from environment variable
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
if not BOT_TOKEN:
    print("❌ Error: TELEGRAM_BOT_TOKEN not found in environment variables!")
    print("Please run: set_token.bat YOUR_BOT_TOKEN")
    exit(1)

# Type assertion for the linter
assert BOT_TOKEN is not None

async def classify_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle AI image classification requests"""
    if not update.message:
        return
        
    if update.message.photo:
        try:
            # Show processing message
            processing_msg = await update.message.reply_text("🔄 Processing your image with AI...")
            
            # Get the largest photo size
            file_id = update.message.photo[-1].file_id
            file = await context.bot.get_file(file_id) #what is context.bot (is it telegram module's function)
            
            if not file.file_path:
                await update.message.reply_text("❌ Sorry, couldn't access the image file.")
                return
            
            # Download and process the image
            print(f"📥 Downloading image: {file.file_path}")
            img_data = requests.get(file.file_path).content 
            img = Image.open(io.BytesIO(img_data))  #doubt
            
            # Get image properties
            width, height = img.size
            format_name = img.format or "Unknown"
            mode = img.mode
            
            # Prepare image for AI classification
            img_resized = img.resize((224, 224))
            if img_resized.mode != 'RGB':
                img_resized = img_resized.convert('RGB')
            
            # Convert to array and preprocess for ResNet50
            x = img_to_array(img_resized)
            x = np.expand_dims(x, axis=0)  #doubt #what's a batch dimension ? 
            x = preprocess_input(x)  #doubt
            
            # Make prediction
            print("🤖 Running AI classification...")
            preds = model.predict(x, verbose=0)  #doubt verbose 
            decoded_preds = decode_predictions(preds, top=5)[0]
            # Format response
            response = f"📸 **Image Analysis Results**\n\n"
            response += f"**📏 Image Properties:**\n"
            response += f"• Size: {width}×{height} pixels\n"
            response += f"• Format: {format_name}\n"
            response += f"• File size: {len(img_data) // 1024} KB\n"
            response += f"• Color mode: {mode}\n\n"
            
            response += f"🤖 **AI Classification (Top 5):**\n"
            for i, (imagenet_id, label, score) in enumerate(decoded_preds):
                confidence = f"{score * 100:.1f}%"
                emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📍"
                response += f"{emoji} {label.replace('_', ' ').title()}: {confidence}\n" #doubt
            
            response += f"\n✨ **Best guess:** {decoded_preds[0][1].replace('_', ' ').title()}"
            
            # Edit the processing message with results
            await processing_msg.edit_text(response, parse_mode='Markdown')
            
            print(f"✅ Classification complete: {decoded_preds[0][1]}")
            
        except Exception as e:
            print(f"❌ Error processing image: {str(e)}")
            await update.message.reply_text(f"❌ Sorry, there was an error processing your image: {str(e)}")
    else:
        await update.message.reply_text('📷 Please send a photo for AI analysis!')

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    if not update.message:
        return
        
    welcome_message = (
        "🤖 **Welcome to the AI Image Classifier Bot!**\n\n"
        "🧠 Powered by ResNet50 neural network\n"
        "📸 Send me any photo and I'll identify what's in it!\n\n"
        "**Features:**\n"
        "• 🎯 AI object recognition\n"
        "• 📊 Confidence scores\n"
        "• 📏 Image analysis\n"
        "• ⚡ Fast processing\n\n"
        "Just send a photo to get started! 📸✨"
    )
    await update.message.reply_text(welcome_message, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    if not update.message:
        return
        
    help_message = (
        "🆘 **How to use this bot:**\n\n"
        "1️⃣ Send any photo to this chat\n"
        "2️⃣ Wait for AI processing (usually 2-5 seconds)\n"
        "3️⃣ Get detailed analysis and classification!\n\n"
        "**Commands:**\n"
        "• /start - Welcome message\n"
        "• /help - This help message\n\n"
        "**Supported formats:** JPG, PNG, WebP, GIF\n"
        "**Recognition:** 1000+ object categories\n\n"
        "🚀 Just send a photo to begin!"
    )
    await update.message.reply_text(help_message, parse_mode='Markdown')

def main():
    """Start the bot"""
    # Ensure token is available (this will never be None due to earlier exit)
    if not BOT_TOKEN:
        return
    
    # Create application  
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, classify_image))
    
    # Start the bot
    print("🚀 AI Telegram Bot is starting...")
    print("✅ Bot ready! Send photos for AI classification!")
    print("📱 Commands: /start, /help")
    print("🛑 Press Ctrl+C to stop")
    print("-" * 50)
    
    application.run_polling()

if __name__ == '__main__':
    main() 