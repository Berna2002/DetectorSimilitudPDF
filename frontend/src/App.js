import React, { useState, useCallback } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import './App.css';

function App() {
    const [similarity, setSimilarity] = useState(0);
    const [loading, setLoading] = useState(false);
    const [files, setFiles] = useState([]);

    const onDrop = useCallback((acceptedFiles) => {
        if (acceptedFiles.length !== 2) {
            alert('Por favor, selecciona exactamente dos archivos PDF.');
            return;
        }

        const arePDFs = acceptedFiles.every((file) => file.type === 'application/pdf');
        if (!arePDFs) {
            alert('Por favor, selecciona solo archivos PDF.');
            return;
        }

        setFiles(acceptedFiles);
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: 'application/pdf',
        multiple: true,
    });

    const handleFileUpload = async (event) => {
        event.preventDefault();

        if (files.length !== 2) {
            alert('Por favor, selecciona dos archivos PDF para comparar.');
            return;
        }

        setLoading(true);

        const formData = new FormData();
        formData.append('pdf1', files[0]);
        formData.append('pdf2', files[1]);

        try {
            const response = await axios.post('/upload', formData);
            setSimilarity(response.data.similarity);
        } catch (error) {
            console.error('Error al subir los archivos:', error);
            setSimilarity(0); // En caso de error, establecer la similitud a 0
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="App">
            <h1>Detector de Similitud en Documentos PDF</h1>

            <form onSubmit={handleFileUpload}>
                <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
                    <input {...getInputProps()} />
                    <p>Arrastra y suelta dos archivos PDF aquí, o haz clic para seleccionarlos</p>
                </div>

                <div>
                    {files.length > 0 && (
                        <ul>
                            {files.map((file) => (
                                <li key={file.name}>{file.name}</li>
                            ))}
                        </ul>
                    )}
                </div>

                <button type="submit" disabled={loading}>
                    Comparar Documentos
                </button>
            </form>

            {loading && <p>Cargando...</p>}

            {!loading && (
                <div className="result-container">
                    <h2>Resultado del Análisis:</h2>
                    <p>Porcentaje de Similitud: {similarity.toFixed(2)}%</p>
                </div>
            )}
        </div>
    );
}

export default App;