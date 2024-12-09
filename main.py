import discord
from discord.ext import commands
import tensorflow as tf
# from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True
intents.guild_messages = True


bot = commands.Bot(command_prefix='!', intents=intents)


model = tf.keras.models.load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()


# klasyfikacja obrazow
def classify_image(image_path):
    # Prepare the image
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    return class_name, confidence_score


# statystyki
energy_stats = {
    "coal": 35.5,  
    "solar": 5.5,   
    "wind": 7.3,
    "hydropower": 16.4,
}


# komenda klasyikcaji
@bot.command()
async def classify(ctx):
    if not ctx.message.attachments:
        await ctx.send("Prosze załączyć obraz.")
        return
    
    # pobrac
    attachment = ctx.message.attachments[0]
    file_path = f"./{attachment.filename}"
    await attachment.save(file_path)

    # klasyfikacja
    try:
        class_name, confidence = classify_image(file_path)
        class_name = class_name[2:]

        percentage = energy_stats.get(class_name.lower(), "Unknown")
        response = (
            f"🌎 Te źródło energi to **{class_name}**\n"
            f"👍 Jestem tego pewnien na  **{confidence:.2f}**\n"
            f"📊 Procent globalnej energii, która pochodzi z tego źródła: **{percentage}%**"
        )
        await ctx.send(response)
    except Exception as e:
        await ctx.send(f"Error in processing the image: {e}")
    finally:
        os.remove(file_path)  # Clean up the image file

# run bot
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")

bot.run("")
