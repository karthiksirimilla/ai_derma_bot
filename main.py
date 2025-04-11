import os
import tempfile
import glob
from gtts import gTTS
from ultralytics import YOLO
import telebot
from dotenv import load_dotenv
# Load environment variable or use fallback token for testing
load_dotenv()
TelegramBOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
bot = telebot.TeleBot(TelegramBOT_TOKEN)

# Load YOLOv8 model
model_path = "derma_bot.pt"
yolo_model = YOLO(model_path)

# Dictionary of diseases and prescriptions
disease_prescriptions = {
    "Chickenpox": "Keep the skin cool and avoid scratching. Use calamine lotion for itching.",
    "Eczema": "Use fragrance-free moisturizers and topical corticosteroids. Avoid triggers.",
    "Eruptive-Xanthoma": "Seek treatment for high cholesterol or diabetes. Consider dietary changes.",
    "Leukocytoclastic-Vasculitis": "Use anti-inflammatory medications. Consult a healthcare provider.",
    "Monkeypox": "Isolate to prevent spread and manage symptoms. Seek medical advice.",
    "Ringworm": "Use topical antifungal creams like clotrimazole. Keep areas clean and dry.",
    "Spider-Angioma": "Usually harmless. Laser therapy can be used for cosmetic reasons.",
    "Xanthelasma": "Lifestyle changes, especially reducing cholesterol. Surgical removal may be an option.",
    "herpes-zoster": "Use antiviral medications like acyclovir. Manage pain if necessary.",
    "vitiligo": "Consider corticosteroids or phototherapy. Consult a dermatologist."
}

# Function to run YOLO model
def detect_and_save(input_path, conf=0.25):
    results = yolo_model.predict(source=input_path, conf=conf, save=True)
    latest_folder = max(glob.glob('runs/detect/*'), key=os.path.getmtime)
    output_images = glob.glob(f'{latest_folder}/*.jpg')

    detected_diseases = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = yolo_model.names[class_id]
            detected_diseases.append(class_name)

    return output_images, detected_diseases

# Function to generate TTS audio file
def generate_audio_report(diseases):
    try:
        if not diseases:
            text = "No diseases were detected in the provided image or video."
        else:
            text = "Detected Diseases and Prescriptions:\n"
            for disease in diseases:
                prescription = disease_prescriptions.get(disease, "Consult a doctor for appropriate treatment.")
                text += f"{disease}: {prescription}\n"

        tts = gTTS(text)
        temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        tts.save(temp_audio_path)
        return temp_audio_path
    except Exception as e:
        print(f"Error generating audio report: {e}")
        return None

# Welcome / start command
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(
        message,
        "🧬 Welcome to *DermaDetect*!\n"
        "Send an image or video of a skin condition to detect possible diseases and get prescriptions.\n"
        "Use /prescriptions to view supported diseases.",
        parse_mode='Markdown'
    )

# List supported diseases and prescriptions
@bot.message_handler(commands=['prescriptions'])
def send_prescriptions(message):
    diseases_list = "\n".join([f"- {disease}" for disease in disease_prescriptions.keys()])
    bot.reply_to(message, f"📋 Supported Diseases and Prescriptions:\n{diseases_list}")

# Handle image input
@bot.message_handler(content_types=['photo'])
def handle_image(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(downloaded_file)
            temp_image_path = temp_file.name

        output_images, detected_diseases = detect_and_save(temp_image_path)
        unique_diseases = set(detected_diseases)

        if unique_diseases:
            response = "✅ Detected Disease(s):\n" + "\n".join(f"- {d}" for d in unique_diseases)
            bot.reply_to(message, response)
            for d in unique_diseases:
                bot.reply_to(message, f"💊 Prescription for {d}: {disease_prescriptions.get(d)}")

            audio_path = generate_audio_report(unique_diseases)
            if audio_path:
                with open(audio_path, 'rb') as audio_file:
                    bot.send_audio(message.chat.id, audio_file)
                os.remove(audio_path)
        else:
            bot.reply_to(message, "❌ No diseases detected.")

        for img_path in output_images:
            with open(img_path, 'rb') as img_file:
                bot.send_photo(message.chat.id, img_file)
            os.remove(img_path)

        os.remove(temp_image_path)
    except Exception as e:
        bot.reply_to(message, "⚠️ Error processing the image.")
        print(f"Image Error: {e}")

# Handle video input
@bot.message_handler(content_types=['video'])
def handle_video(message):
    try:
        file_info = bot.get_file(message.video.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(downloaded_file)
            temp_video_path = temp_file.name

        output_images, detected_diseases = detect_and_save(temp_video_path)
        unique_diseases = set(detected_diseases)

        if unique_diseases:
            response = "✅ Detected Disease(s):\n" + "\n".join(f"- {d}" for d in unique_diseases)
            bot.reply_to(message, response)
            for d in unique_diseases:
                bot.reply_to(message, f"💊 Prescription for {d}: {disease_prescriptions.get(d)}")

            audio_path = generate_audio_report(unique_diseases)
            if audio_path:
                with open(audio_path, 'rb') as audio_file:
                    bot.send_audio(message.chat.id, audio_file)
                os.remove(audio_path)
        else:
            bot.reply_to(message, "❌ No diseases detected.")

        for img_path in output_images:
            with open(img_path, 'rb') as img_file:
                bot.send_photo(message.chat.id, img_file)
            os.remove(img_path)

        os.remove(temp_video_path)
    except Exception as e:
        bot.reply_to(message, "⚠️ Error processing the video.")
        print(f"Video Error: {e}")

# Start the bot
if __name__ == "__main__":
    print("Bot is running...")
    bot.infinity_polling()
