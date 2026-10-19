import asyncio
import logging
logger = logging.getLogger(__name__)
from telegram.request import HTTPXRequest
from classifier import classify_image_bytes
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from PIL import UnidentifiedImageError
from config import (
    BOT_TOKEN, 
    CONFIDENCE_HIGH_THRESHOLD, 
    CONFIDENCE_LOW_THRESHOLD, 
    MAX_FILE_SIZE_BYTES
)
import httpx


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
            file = await context.bot.get_file(file_id)

            if file.file_size and file.file_size > MAX_FILE_SIZE_BYTES:
                logger.warning(f"Image rejected: file size {file.file_size} exceeds the limit ")
                await update.message.reply_text("Image is too large! Please send an image under 10MB.")
                return
            
            
            if not file.file_path:
                await update.message.reply_text("❌ Sorry, couldn't access the image file.")
                return
            
            # Download and process the image
            logger.info("📥 Downloading image: %s", file.file_path)
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(file.file_path)
                response.raise_for_status()
                img_data = response.content 
            properties,decoded_preds,result = await asyncio.to_thread(
                classify_image_bytes,
                img_data
            )
            
            # Get image properties
            width = properties["width"]
            height = properties["height"]
            format_name = properties["format"]
            size = properties["size_kb"]
            mode = properties["mode"]
            
            

            # Format response
            response = f"📸 **Image Analysis Results**\n\n"
            response += f"**📏 Image Properties:**\n"
            response += f"• Size: {width}×{height} pixels\n"
            response += f"• Format: {format_name}\n"
            response += f"• File size: {size} KB\n"
            response += f"• Color mode: {mode}\n\n"
            
            response += f"🤖 **AI Classification (Top 5):**\n"
            for i, (imagenet_id, label, score) in enumerate(decoded_preds):
                confidence = f"{score * 100:.1f}%"
                emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📍"
                response += f"{emoji} {label.replace('_', ' ').title()}: {confidence}\n"

            top_label = result['label'].replace('_',' ').title()
            top_conf = result["confidence"]
            
            # response += f"\n✨ **Best guess:** {result['label'].replace('_', ' ').title()}"
            if top_conf >= CONFIDENCE_HIGH_THRESHOLD:
                response += f"\n🎯 High confidence: {top_label} ({top_conf * 100:.1f}%)"
            elif top_conf >= CONFIDENCE_LOW_THRESHOLD:
                response += f"\n🤔 Moderate confidence: It looks like {top_label} ({top_conf * 100:.1f}%), but check the other guesses above."
            else:
                response += f"\n⚠️ Low confidence ({top_conf * 100:.1f}%): I'm not very sure about this image. Try sending a clearer or more centered photo!"

            
            # Edit the processing message with results
            await processing_msg.edit_text(response, parse_mode='Markdown')
            
            logger.info(f"✅ Classification complete: {result['label']}")

        except httpx.TimeoutException:
            logger.warning("Timeout downloading image from Telegram CDN")
            await update.message.reply_text("Sorry,couldn't download your image - Telegram took too long")
        except httpx.HTTPError as e:
            logger.warning(f"Http error downloading the image {e}")
            await update.message.reply_text("❌  Sorry,there was a network error downloading your image")
        except UnidentifiedImageError:
            logger.warning(f"Couldn't read the image-it may be corrupted")
            await update.message.reply_text(f"❌ Sorry, couldn't read the image it must be corrupted")
        except ValueError as e:
            logger.warning("Validation error: %s",e)
            await update.message.reply_text(f"❌ Invalid image: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}",exc_info=True )
            await update.message.reply_text(f"❌ Sorry, there was an error processing your image:")


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
    # Configure custom HTTP request timeouts
    request = HTTPXRequest(
        connect_timeout=20.0,
        read_timeout=20.0,
        write_timeout=20.0
    )
    
    # Ensure token is available (this will never be None due to earlier exit)
    if not BOT_TOKEN:
        return
    
    # Create application  
    application = Application.builder().token(BOT_TOKEN).request(request).build()
    
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
    
    application.run_polling(bootstrap_retries=5)

if __name__ == '__main__':
    main() 