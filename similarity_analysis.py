import fitz  # PyMuPDF
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, accuracy_score
import joblib
import os
import numpy as np
import pandas as pd
import glob

# --- Descargas NLTK (si es necesario) ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- Funciones de Utilidad ---

def extract_text_from_pdf(pdf_path):
    """Extrae el texto de un PDF usando PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_text(text):
    """Limpia el texto: minúsculas, puntuación, números, stopwords y stemming."""
    stop_words = set(stopwords.words('spanish'))
    stemmer = SnowballStemmer('spanish')  # Stemmer para español
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Eliminar puntuación
    text = re.sub(r'\d+', '', text)  # Eliminar números
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [w for w in tokens if not w in stop_words]
    stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]  # Aplicar stemming
    return " ".join(stemmed_tokens)

def split_into_sentences(text):
    """Divide el texto en oraciones."""
    sentences = nltk.sent_tokenize(text)
    return sentences

# --- Carga o Creación de la Matriz TF-IDF y el Vectorizador ---

def load_or_create_tfidf_data(pdfs_folder):
    """Carga la matriz TF-IDF y el vectorizador desde archivos o los crea si no existen."""
    tfidf_matrix_file = 'tfidf_matrix.joblib'
    vectorizer_file = 'vectorizer.joblib'
    corpus_file = 'corpus.joblib'

    if os.path.exists(tfidf_matrix_file) and os.path.exists(vectorizer_file) and os.path.exists(corpus_file):
        tfidf_matrix = joblib.load(tfidf_matrix_file)
        vectorizer = joblib.load(vectorizer_file)
        corpus = joblib.load(corpus_file)
        print("Matriz TF-IDF, vectorizador y corpus cargados desde archivos.")
        return tfidf_matrix, vectorizer, corpus
    else:
        print("Calculando la matriz TF-IDF y el vectorizador...")
        corpus = []
        train_folder = os.path.join(pdfs_folder, 'train')
        pdf_files = [f for f in os.listdir(train_folder) if f.endswith('.pdf')]
        for filename in pdf_files:
            pdf_path = os.path.join(train_folder, filename)
            text = extract_text_from_pdf(pdf_path)
            cleaned_text = clean_text(text)
            corpus.append(cleaned_text)

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Guardar la matriz TF-IDF, el vectorizador y el corpus
        joblib.dump(tfidf_matrix, tfidf_matrix_file)
        joblib.dump(vectorizer, vectorizer_file)
        joblib.dump(corpus, corpus_file)
        print("Matriz TF-IDF, vectorizador y corpus calculados y guardados en archivos.")

        return tfidf_matrix, vectorizer, corpus

# --- Generar Pares de Prueba y Calcular Similitudes ---

def generate_test_pairs(test_folder, vectorizer, threshold=0.7):
    """
    Genera pares de archivos PDF para pruebas, calcula su similitud y asigna etiquetas.

    Args:
        test_folder: Carpeta que contiene los PDFs de prueba.
        vectorizer: Vectorizador TF-IDF entrenado.
        threshold: Umbral de similitud para la clasificación binaria.

    Returns:
        Un DataFrame con las rutas de los pares de PDFs, sus similitudes y etiquetas.
    """
    a_files = glob.glob(os.path.join(test_folder, 'a_*.pdf'))
    b_files = glob.glob(os.path.join(test_folder, 'b_*.pdf'))

    pairs = []
    similarities = []
    labels = []

    for a_file in a_files:
        for b_file in b_files:
            if a_file[-5] == b_file[-5]: #Compara que el número del nombre del archivo sea igual, antes de la extensión
                pairs.append((a_file, b_file))

                text_a = extract_text_from_pdf(a_file)
                cleaned_text_a = clean_text(text_a)
                text_b = extract_text_from_pdf(b_file)
                cleaned_text_b = clean_text(text_b)

                pdf_vectors = vectorizer.transform([cleaned_text_a, cleaned_text_b])
                similarity = cosine_similarity(pdf_vectors[0], pdf_vectors[1])[0][0]
                similarities.append(similarity)
                labels.append(1 if similarity >= threshold else 0)

    df = pd.DataFrame({
        'pdf1_path': [pair[0] for pair in pairs],
        'pdf2_path': [pair[1] for pair in pairs],
        'similitud': similarities,
        'etiqueta': labels
    })

    return df

# --- Cálculo de Métricas de Evaluación ---

def calculate_metrics(predicted_similarities, true_labels, threshold):
    """
    Calcula las métricas de evaluación (RMSE, precisión, recall, F1-score, precisión global).

    Args:
        predicted_similarities: Lista de valores de similitud predichos.
        true_labels: Lista de etiquetas "verdaderas" (1 para similar, 0 para no similar).
        threshold: Umbral para la clasificación binaria.

    Returns:
        Un diccionario con las métricas calculadas.
    """
    predicted_labels = [1 if s >= threshold else 0 for s in predicted_similarities]

    rmse = np.sqrt(mean_squared_error(true_labels, predicted_labels))
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)

    return {
        'RMSE': rmse,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'Accuracy': accuracy
    }