import discord
from discord.ext import commands
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np

# Configurar las intenciones (puedes ajustarlas según tus necesidades)
intents = discord.Intents.all()

# Crear la instancia de Bot con intenciones
bot = commands.Bot(command_prefix='&&', intents=intents)

# Cargar el modelo y el tokenizador
model = BertForSequenceClassification.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment")
tokenizer = BertTokenizer.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment")

# Cargar datos desde el archivo CSV
datos_entrenamiento = pd.read_csv('datos_entrenamiento.csv')


@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')


@bot.command()
async def analizar(ctx, *, texto):
    # Agregar aquí el código para procesar los datos_entrenamiento según sea necesario

    # Tokenizar el texto
    tokens = tokenizer(texto, return_tensors='pt',
                       truncation=True, max_length=300)

    # Obtener la predicción del modelo
    with torch.no_grad():
        output = model(**tokens)

    # Convertir la salida a probabilidades
    probs = torch.nn.functional.softmax(output.logits, dim=-1)

    # Obtener la clase con la probabilidad más alta
    pred_class = torch.argmax(probs).item()

    # Mapear las clases a etiquetas específicas
    labels = ["Depresivo", "Negativo", "Neutral", "Positivo", "Euforia"]
    pred_label = labels[pred_class]

    # Seleccionar un GIF según la emoción
    gif_url = ""
    if pred_label == "Depresivo":
        gif_url = "https://media.tenor.com/NvmwK04kkEQAAAAC/omori-basil.gif"
    elif pred_label == "Negativo":
        gif_url = "https://media.tenor.com/-Fg9W6BXuT0AAAAC/sad-omori-sad.gif"
    elif pred_label == "Neutral":
        gif_url = "https://media.tenor.com/HZAFyy3YA5AAAAAC/omori-neutral.gif"
    elif pred_label == "Positivo":
        gif_url = "https://media.tenor.com/On-JRNGEjykAAAAC/omori-happy-happy-omori.gif"
    elif pred_label == "Euforia":
        gif_url = "https://media.tenor.com/Vfsza4xCEEIAAAAC/basil-omori.gif"

    # Enviar el resultado al canal de Discord con un mensaje estilizado
    embed = discord.Embed(
        title=f'Sentimiento: {pred_label} ({probs[0][pred_class]:.4f})')
    embed.set_image(url=gif_url)
    await ctx.send(embed=embed)

# Ejecutar el bot con el token de tu aplicación Discord
my_secret = 'MTE3NTg4MDgzMzgxOTc1ODY1Mg.GwN4ha.MaI64NNPYf-_AgKFzT6rzkGlrzT1YGz6jww5Hk'
bot.run(my_secret)
