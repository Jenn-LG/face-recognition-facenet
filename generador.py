import os
import numpy as np
import pickle
import time
from PIL import Image
from tqdm import tqdm
from deepface import DeepFace
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ========== CONFIGURACIÓN GENERAL ==========
DATA_DIR = 'output_resized_512x512'  # Carpeta de imágenes originales
PROCESSED_DIR = 'faces_processed'  # Carpeta donde se guardarán las imágenes aumentadas
EMBEDDINGS_PATH = 'embeddings.pkl'  # Archivo donde se guardarán los embeddings

MODEL_NAME = 'Facenet'  # Modelo preentrenado usado para extraer embeddings


# ========== FUNCIÓN PARA OBTENER EMBEDDINGS ==========
def obtener_embedding(img_path):
    """
    Obtiene el embedding facial usando DeepFace.represent.
    
    Args:
        img_path (str): Ruta a la imagen.
    
    Returns:
        np.ndarray: Vector de embedding.
    """
    embedding_info = DeepFace.represent(
        img_path=img_path,
        model_name=MODEL_NAME,
        enforce_detection=False # No fuerza detección de rostro
    )
    embedding = embedding_info[0]['embedding']
    return np.array(embedding)


# ========== FUNCIÓN DE DATA AUGMENTATION ==========
def augment_and_process(input_dir, output_dir):
    """
    Aplica data augmentation a las imágenes y guarda las versiones aumentadas.

    Args:
        input_dir (str): Carpeta de imágenes originales.
        output_dir (str): Carpeta para guardar imágenes aumentadas.
    """
    # Configuración de tranformaciones de data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    # Crear carpeta de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Procesa cada carpeta (una por persona)
    for person in os.listdir(input_dir):
        person_dir = os.path.join(input_dir, person)
        if os.path.isdir(person_dir):
          output_person_dir = os.path.join(output_dir, person)
          os.makedirs(output_person_dir, exist_ok=True)
          # Procesa cada imagen de la persona
          for img_file in os.listdir(person_dir):
              if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                  img_path = os.path.join(person_dir, img_file)
                  img = np.array(Image.open(img_path).convert('RGB'))
                  
                  # Guardar imagen original como versión "_aug0"
                  Image.fromarray(img).save(os.path.join(output_person_dir, f"{os.path.splitext(img_file)[0]}_aug0.jpg"))
                  
                  # Generar y guardar 4 imágenes aumentadas
                  img_batch = img.reshape((1,) + img.shape)
                  i = 0
                  for batch in datagen.flow(img_batch, batch_size=1):
                      aug_img = batch[0].astype('uint8')
                      Image.fromarray(aug_img).save(os.path.join(output_person_dir, f"{os.path.splitext(img_file)[0]}_aug{i+1}.jpg"))
                      i += 1
                      if i >= 4:
                          break


# ========== FUNCIÓN PARA GENERAR EMBEDDINGS ==========
def generar_embeddings(input_dir):
    """
    Genera embeddings de todas las imágenes aumentadas.
    
    Args:
        input_dir (str): Carpeta que contiene las imágenes aumentadas.
    """
    start_time = time.time() #Guardar tiempo de inicio
    embeddings = {} #Diccionario para guardar embeddings por persona

    persons = [p for p in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, p))]
    for person in persons:
        person_dir = os.path.join(input_dir, person)
        img_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        embeddings[person] = [] #Lista de embeddings para esta persona

        print(f"\nProcesando persona: {person} ({len(img_files)} imágenes)")

        for img_file in tqdm(img_files, desc=f"{person}", unit="img"):
            img_path = os.path.join(person_dir, img_file)
            try:
                emb = obtener_embedding(img_path)
                embeddings[person].append(emb)
            except Exception as e:
                print(f"⚠️ Error procesando {img_file}: {e}")
    # guardar embeddings en un archivo pickle
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(embeddings, f)

    total_time = time.time() - start_time
    print(f"\n✅ Embeddings generados y guardados en '{EMBEDDINGS_PATH}'.")

# Aplicamos data augmentation a las imágenes originales
augment_and_process(DATA_DIR, PROCESSED_DIR)

# Generamos y guardamos los embeddings
print("\nGenerando embeddings...")
generar_embeddings(PROCESSED_DIR)