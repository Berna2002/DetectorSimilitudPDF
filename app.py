from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from similarity_analysis import load_or_create_tfidf_data, extract_text_from_pdf, clean_text, generate_test_pairs, calculate_metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
PDFS_FOLDER = 'pdfs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PDFS_FOLDER'] = PDFS_FOLDER

# Cargar la matriz TF-IDF, el vectorizador y el corpus al iniciar la aplicación
tfidf_matrix, vectorizer, corpus = load_or_create_tfidf_data(app.config['PDFS_FOLDER'])

print("vectorizador cargado")
print(len(vectorizer.vocabulary_))

# --- Endpoint /upload ---

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf1' not in request.files or 'pdf2' not in request.files:
        return jsonify({'error': 'Se requieren dos archivos PDF'}), 400

    pdf1 = request.files['pdf1']
    pdf2 = request.files['pdf2']

    pdf1_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf1.filename)
    pdf2_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf2.filename)
    pdf1.save(pdf1_path)
    pdf2.save(pdf2_path)

    # Extraer y limpiar el texto de los PDFs subidos
    text1 = extract_text_from_pdf(pdf1_path)
    cleaned_text1 = clean_text(text1)

    text2 = extract_text_from_pdf(pdf2_path)
    cleaned_text2 = clean_text(text2)

    # Calcular la similitud entre los dos PDFs
    # Usar el vectorizador cargado para transformar los textos de los PDFs subidos
    pdf_vectors = vectorizer.transform([cleaned_text1, cleaned_text2])
    similarity = cosine_similarity(pdf_vectors[0], pdf_vectors[1])[0][0]

    return jsonify({'similarity': float(similarity * 100)}), 200

# --- Endpoint /evaluate ---

@app.route('/evaluate', methods=['GET'])
def evaluate():
    test_folder = os.path.join(app.config['PDFS_FOLDER'], 'test')
    threshold = 0.7  # Define el umbral

    # Generar pares de prueba y calcular similitudes
    test_df = generate_test_pairs(test_folder, vectorizer, threshold)

    # Calcular métricas
    metrics = calculate_metrics(test_df['similitud'].tolist(), test_df['etiqueta'].tolist(), threshold)

    # Guardar resultados en CSV
    test_df.to_csv('resultados_comparacion.csv', index=False)

    return jsonify({'message': 'Evaluación completada', 'metrics': metrics, 'results_path': 'resultados_comparacion.csv'}), 200

if __name__ == '__main__':
    app.run(debug=True)