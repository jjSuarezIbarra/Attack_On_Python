#%%
import streamlit as st
import speech_recognition as sr
import pyttsx3
import pandas as pd
import re
import time
import calendar
import random
import inflect
import threading
import glob

# Carga los archivos de predicciones de enero, febrero, junio y diciembre
archivos = [
    '../data/predicciones_20230101_20230131.csv',
    '../data/predicciones_20230201_20230228.csv',
    '../data/predicciones_20230601_20230630.csv',
    '../data/predicciones_20231201_20231231.csv',
]

# Concatenar todos los archivos en un solo DataFrame
lista_df = []
for archivo in archivos:
    try:
        df_temp = pd.read_csv(archivo)
        df_temp['fecha'] = pd.to_datetime(df_temp['fecha'])
        lista_df.append(df_temp)
    except Exception as e:
        print(f"No se pudo cargar {archivo}: {e}")

if lista_df:
    df = pd.concat(lista_df, ignore_index=True)
else:
    df = pd.DataFrame()

# Diccionario de meses en español
MESES = {
    'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04',
    'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
    'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12'
}

p = inflect.engine()

SINONIMOS = [
    "tienes", "tendrás", "recibirás", "se espera que pagues", "te llegará"
]

MESES_NOMBRES = {
    '01': 'enero', '02': 'febrero', '03': 'marzo', '04': 'abril',
    '05': 'mayo', '06': 'junio', '07': 'julio', '08': 'agosto',
    '09': 'septiembre', '10': 'octubre', '11': 'noviembre', '12': 'diciembre'
}

# Diccionario simple para números en español (1-99)
NUMS_ES = {
    0: 'cero', 1: 'uno', 2: 'dos', 3: 'tres', 4: 'cuatro', 5: 'cinco', 6: 'seis', 7: 'siete', 8: 'ocho', 9: 'nueve',
    10: 'diez', 11: 'once', 12: 'doce', 13: 'trece', 14: 'catorce', 15: 'quince', 16: 'dieciséis', 17: 'diecisiete', 18: 'dieciocho', 19: 'diecinueve',
    20: 'veinte', 21: 'veintiuno', 22: 'veintidós', 23: 'veintitrés', 24: 'veinticuatro', 25: 'veinticinco', 26: 'veintiséis', 27: 'veintisiete', 28: 'veintiocho', 29: 'veintinueve',
    30: 'treinta', 40: 'cuarenta', 50: 'cincuenta', 60: 'sesenta', 70: 'setenta', 80: 'ochenta', 90: 'noventa'
}

def num_a_texto_es(n):
    n = int(n)
    if n <= 29:
        return NUMS_ES.get(n, str(n))
    elif n < 100:
        dec = (n // 10) * 10
        uni = n % 10
        if uni == 0:
            return NUMS_ES[dec]
        else:
            return f"{NUMS_ES[dec]} y {NUMS_ES[uni]}"
    else:
        return str(n)

def monto_a_texto(monto):
    monto = round(monto, 2)
    entero = int(monto)
    decimales = int(round((monto - entero) * 100))
    if decimales == 0:
        return f"{num_a_texto_es(entero)} pesos"
    else:
        return f"{num_a_texto_es(entero)} pesos con {num_a_texto_es(decimales)} centavos"

def extraer_fechas_naturales(texto):
    # Busca patrones como "1 enero de 2023" o "31 de enero de 2023"
    patron = r'(\d{1,2})\s+de?\s*([a-záéíóúñ]+)\s+de\s+(\d{4})'
    fechas = re.findall(patron, texto, re.IGNORECASE)
    fechas_formateadas = []
    for dia, mes, anio in fechas:
        mes_num = MESES.get(mes.lower())
        if mes_num:
            fecha = f'{anio}-{mes_num.zfill(2)}-{int(dia):02d}'
            fechas_formateadas.append(fecha)
    return fechas_formateadas

def extraer_fechas_iso(texto):
    # Busca fechas en formato YYYY-MM-DD
    return re.findall(r'\d{4}-\d{2}-\d{2}', texto)

def extraer_fechas_mes_simple(texto, anio_default='2023'):
    # Busca frases como "enero", "febrero", etc.
    meses_encontrados = [mes for mes in MESES if mes in texto.lower()]
    fechas = []
    for mes in meses_encontrados:
        mes_num = int(MESES[mes])
        # Primer y último día del mes
        primer_dia = f"{anio_default}-{mes_num:02d}-01"
        ultimo_dia_num = calendar.monthrange(int(anio_default), mes_num)[1]
        ultimo_dia = f"{anio_default}-{mes_num:02d}-{ultimo_dia_num:02d}"
        fechas.append((primer_dia, ultimo_dia))
    return fechas

def gpt_wrapper(pregunta, df, para_voz=False):
    fechas = extraer_fechas_naturales(pregunta)
    if len(fechas) < 2:
        fechas = extraer_fechas_iso(pregunta)
    if len(fechas) == 2:
        fecha_inicio, fecha_fin = pd.to_datetime(fechas[0]), pd.to_datetime(fechas[1])
    else:
        fechas_mes = extraer_fechas_mes_simple(pregunta)
        if fechas_mes:
            fecha_inicio, fecha_fin = fechas_mes[0]
            fecha_inicio, fecha_fin = pd.to_datetime(fecha_inicio), pd.to_datetime(fecha_fin)
        else:
            return "Por favor, especifica el rango de fechas o al menos el mes (ejemplo: 'enero')."
    df_filtrado = df[(df['fecha'] >= fecha_inicio) & (df['fecha'] <= fecha_fin)]
    # Filtrar solo gastos > 0
    df_filtrado = df_filtrado[df_filtrado['valor_predicho'] > 0]
    if df_filtrado.empty:
        return "No hay gastos predichos en ese rango de fechas."
    df_filtrado = df_filtrado.sort_values('fecha').head(5)
    frases = []
    for _, row in df_filtrado.iterrows():
        fecha = row['fecha']
        dia = fecha.day
        mes = MESES_NOMBRES[fecha.strftime('%m')]
        sinonimo = random.choice(SINONIMOS)
        monto = row['valor_predicho']
        comercio = row['comercio'] if 'comercio' in row else 'un comercio'
        if para_voz:
            monto_txt = monto_a_texto(monto)
        else:
            monto_txt = f"{monto:.2f} pesos"
        frases.append(f"El {dia} de {mes} {sinonimo} un gasto de {monto_txt} en {comercio}.")
    return '\n'.join(frases)

def escuchar_pregunta():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source, phrase_time_limit=12)
    try:
        texto = r.recognize_google(audio, language='es-MX')
        return texto
    except Exception as e:
        return None

def hablar_respuesta(texto):
    def tts():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        for voz in engine.getProperty('voices'):
            if "spanish" in voz.name.lower() or "español" in voz.name.lower():
                engine.setProperty('voice', voz.id)
                break
        engine.say(texto)
        engine.runAndWait()
    threading.Thread(target=tts).start()

st.title("Asistente de Predicción de Gastos Hey Banco")
st.write("Haz clic en el micrófono, pregunta por los gastos de un rango de fechas, y escucha la respuesta!")

if 'respuesta' not in st.session_state:
    st.session_state['respuesta'] = ""

if st.button("🎤 Hablar"):
    mensaje_escucha = st.empty()
    mensaje_escucha.info("Escuchando... habla ahora")
    pregunta = escuchar_pregunta()
    mensaje_escucha.empty()
    if pregunta:
        st.write(f"**Tú dijiste:** {pregunta}")
        respuesta_pantalla = gpt_wrapper(pregunta, df, para_voz=False)
        respuesta_voz = gpt_wrapper(pregunta, df, para_voz=True)
        st.session_state['respuesta'] = respuesta_pantalla
        st.success("Respuesta generada. Escuchando respuesta...")
        print("Respuesta voz:", respuesta_voz)
        hablar_respuesta(respuesta_voz)
    else:
        st.warning("No se pudo reconocer el audio. Intenta de nuevo.")

if st.session_state['respuesta']:
    st.write("**Asistente:**")
    st.code(st.session_state['respuesta'])

# %%
