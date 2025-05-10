from flask import Flask, render_template, request, jsonify
import os
import base64
import io
import time
import pickle
import numpy as np
from PIL import Image
import cv2
from threading import Thread
import logging

# ========== CONFIGURACIÓN DE LOGGING ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== INICIALIZACIÓN DE LA APP FLASK ==========
app = Flask(__name__)

# Variables globales
embedder = None       # Modelo FaceNet para embeddings
detector = None       # Detector MTCNN de rostros
base_datos = None     # Base de embeddings cargada
models_loaded = False # Indicador de carga de modelos
loading_thread = None # Hilo de carga en segundo plano

# ============= FUNCIÓN PARA CARGAR MODELOS =========
def load_resources():
    """Carga los modelos de Machine Learning y la base de datos de embeddings en segundo plano."""
    global embedder, detector, base_datos, models_loaded
    
    try:
        logger.info("Iniciando carga del modelo FaceNet…")
        t0 = time.time()
        from keras_facenet import FaceNet
        embedder = FaceNet()
        logger.info(f"FaceNet cargado en {time.time() - t0:.2f} segundos.")
        
        logger.info("Iniciando carga del detector MTCNN…")
        from mtcnn.mtcnn import MTCNN
        detector = MTCNN()
        logger.info("MTCNN cargado correctamente")
        
        logger.info("Cargando embeddings…")
        with open("embeddings.pkl", "rb") as f:
            data = pickle.load(f)
            base_datos = {k: [np.array(e) for e in v] for k, v in data.items()}
        logger.info(f"Embeddings listos: {len(base_datos)} personas registradas")
        
        models_loaded = True
    except Exception as e:
        logger.error(f"Falló la carga de recursos: {e}")
        models_loaded = False

# Arranca la carga de modelos en un hilo separado
loading_thread = Thread(target=load_resources)
loading_thread.daemon = True
loading_thread.start()

# ============= RUTAS DEL SERVIDOR ==============
@app.route("/")
def index():
    """Renderiza la página principal"""
    return render_template("index.html")

@app.route("/status")
def status():
    """Devuelve el estado actual de carga de modelos"""
    global models_loaded, embedder, detector, base_datos
    
    # Comprobar si los modelos están cargados
    actual_models_loaded = (
        embedder is not None and 
        detector is not None and 
        base_datos is not None
    )
    
    if actual_models_loaded and not models_loaded:
        models_loaded = True
        logger.info("Modelos cargados exitosamente")
    
    # Información de progreso
    progress = {
        "models_loaded": models_loaded,
        "identities": len(base_datos) if models_loaded and base_datos else 0,
        "detector_loaded": detector is not None,
        "embedder_loaded": embedder is not None
    }
    
    logger.info(f"Estado de carga: {progress}")
    return jsonify(progress)

@app.route("/reconocer", methods=["POST"])
def reconocer():
    """Realiza la detección y reconocimiento facial en una imagen recibida."""
    global models_loaded, embedder, detector, base_datos
    
    # Verificar que los modelos están listos
    if not models_loaded:
        return jsonify(success=False, error="Modelos aún no disponibles. Intente más tarde."), 503
    
    try:
        logger.info("Petición de reconocimiento recibida")
        t_start = time.time()

        # Obtener imagen base64 de JSON
        data = request.get_json(force=True)
        image_data = data.get("image", "")
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        img_bytes = base64.b64decode(image_data)

        # Convertir a arreglo numpy
        img = Image.open(io.BytesIO(img_bytes))
        arr = np.array(img)
        logger.info(f"Imagen recibida: shape={arr.shape}, dtype={arr.dtype}")

        # Detección con MTCNN
        detecciones = detector.detect_faces(arr)
        logger.info(f"MTCNN detectó {len(detecciones)} rostro(s).")

        resultados = []
        for cara in detecciones:
            x, y, w, h = cara["box"]
            # Coregir posibles valores negativos
            x, y = max(0, x), max(0, y)
            x2 = min(arr.shape[1], x + w)
            y2 = min(arr.shape[0], y + h)
            face_img = arr[y:y2, x:x2]
            nombre, distancia = reconocer_persona(face_img)
            resultados.append({
                "nombre": nombre,
                "distancia": float(distancia),
                "bbox": [int(x), int(y), int(w), int(h)]
            })

        logger.info(f"Reconocimiento completo en {time.time() - t_start:.2f}s")
        return jsonify(success=True, resultados=resultados)

    except Exception as e:
        logger.error(f"Error durante el reconocimiento: {e}")
        return jsonify(success=False, error=str(e)), 500

def reconocer_persona(img_array, umbral=0.8):
    """
    Compara una imagen de rostro contra la base de datos de embeddings.
    
    Args:
        img_array (np.ndarray): Imagen de rostro.
        umbral (float): Distancia máxima para considerar una coincidencia.

    Returns:
        nombre_identificado (str): Nombre de la persona reconocida o "Desconocido".
        distancia_minima (float): Distancia al embedding más cercano.
    """
    from numpy.linalg import norm

    try:
        # Convertir imagen a RGB si es necesario
        if img_array.ndim == 2: # Escala de grises
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4: # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif img_array.shape[2] != 3:
            raise ValueError(f"Formato de imagen inesperado: {img_array.shape}")

        # Redimensionar a 160x160
        face_resized = cv2.resize(img_array, (160, 160))

        # Obtener embedding
        embedding = embedder.embeddings([face_resized])[0]

        # Buscar identidad
        nombre_identificado = "Desconocido"
        distancia_minima = float("inf")
        for nombre, emb_list in base_datos.items():
            for emb_base in emb_list:
                d = norm(embedding - emb_base)
                if d < distancia_minima and d < umbral:
                    distancia_minima = d
                    nombre_identificado = nombre
        return nombre_identificado, distancia_minima

    except Exception as e:
        logger.error(f"Error comparando embeddings: {e}")
        return "Error", float("inf")

# ============ INICIALIZAR EL SERVIDOR ===============
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Arrancando servidor en http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

