import spacy
import re
from unidecode import unidecode
import subprocess
import sys

# Función para cargar o instalar el modelo spaCy
def load_spacy_model():
    try:
        return spacy.load("es_core_news_sm")
    except OSError:
        print("El modelo 'es_core_news_sm' no está instalado. Instalándolo ahora...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "es_core_news_sm"])
        return spacy.load("es_core_news_sm")

# Cargar el modelo de spaCy
nlp = load_spacy_model()

def fn_clean_text(text):
    """
    Limpia el texto eliminando caracteres especiales, acentos, stop words y aplica lematización.

    params:
        - text: Texto a limpiar.
    
    return: 
        - Texto limpio.
    """
    # 1. Conversión a minúsculas
    text = text.lower()
    
    # 2. Eliminación de caracteres especiales
    text = re.sub(r"[^a-záéíóúñü\s]", "", text)
    
    # 3. Normalización (eliminación de acentos)
    text = unidecode(text)
    
    # 4. Procesamiento NLP: Eliminación de stop words y lematización
    doc = nlp(text)
    clean_tokens = [
        token.lemma_ 
        for token in doc 
        if not token.is_stop and not token.is_punct
    ]
    
    # Unir los tokens limpios en una cadena
    return " ".join(clean_tokens)
