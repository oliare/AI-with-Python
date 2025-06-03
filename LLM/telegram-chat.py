from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import requests

OLLAMA_URL = 'http://localhost:11434/api/generate'
OLLAMA_MODEL = 'llama3.1:8b'
TELEGRAM_BOT_TOKEN = '7573869982:AAGvSqByybKsxbKmNmzOcLTNK00sxCzikDc'

def ask_ollama(prompt):
    payload = {
        'model': OLLAMA_MODEL,
        'prompt': prompt,
        'stream': True
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json().get('response', 'Something went wrong')
    except Exception as e:
        print("error:", str(e))
        return ''

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "<b>Hi! What can I help with?</b>\n\n💡<i>type 'bye' to end the conversation</i>",
        parse_mode='HTML'
    )

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    print(f"👤: {text}")
    if text.lower() == 'bye':
        await update.message.reply_text("Bye! See you later again 🔆")
    else:
        answer = ask_ollama(text)
        await update.message.reply_text(answer)

app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

app.run_polling()
