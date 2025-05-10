# Sistema de Reconocimiento Facial

![MCD](https://mcd.unison.mx/wp-content/themes/awaken/img/logo_mcd.png)

## Descripción

Este proyecto implementa un sistema de reconocimiento facial en tiempo real utilizando Python, Flask y modelos de deep learning. El sistema puede detectar y reconocer rostros en video en directo, mostrando información sobre las personas identificadas.

## Características principales

- 🎭 Detección facial en tiempo real
- 👤 Reconocimiento de personas previamente registradas
- 🌗 Interfaz web con modo oscuro/claro
- 📱 Diseño responsive que funciona en dispositivos móviles
- 🔄 Cambio entre cámaras disponibles
- 📊 Visualización de resultados de reconocimiento

## Estructura del proyecto

```
Jenn-LG-faces/
│
├── templates/            # Plantillas HTML
│   └── index.html        # Interfaz principal
│
├── app.py                # Aplicación Flask principal
├── Dockerfile            # Configuración para contenedor Docker
├── embeddings.pkl        # Embeddings faciales precalculados
├── generador.py          # Script para generar embeddings
├── requirements.txt      # Dependencias de Python
└── README.md             # Este archivo
```

## Tecnologías utilizadas

- **Backend**: Python, Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **Machine Learning**: FaceNet, MTCNN
- **Despliegue**: Docker (opcional)

## Requisitos previos

- Python 3.8+
- pip
- Navegador web Chrome, Firefox, Edge.

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/Jenn-LG/face-recognition-facenet.git
   cd face_recognition
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepara tus imágenes de referencia:
   - Coloca imágenes de rostros conocidos en el directorio `../faces/`
   - Cada subcarpeta debe nombrarse como `nombre_de_la_persona`

4. Genera los embeddings faciales:
   ```bash
   python generador.py
   ```

## Uso

1. Inicia el servidor Flask:
   ```bash
   python app.py
   ```

2. Abre tu navegador en:
   ```
   http://localhost:5000
   ```

3. Permite el acceso a la cámara cuando se solicite

4. El sistema comenzará a detectar y reconocer rostros automáticamente

## Configuración

Puedes modificar los siguientes parámetros en `app.py`:

- `UMBRAL_RECONOCIMIENTO`: Umbral de distancia para considerar una coincidencia (valores más bajos son más estrictos)
- `DETECCION_INTERVAL`: Intervalo entre detecciones (en ms)

## Docker (Opcional)

Para ejecutar con Docker:

1. Construye la imagen:
   ```bash
   docker build -t face_recognition .
   ```

2. Ejecuta el contenedor:
   ```bash
   docker run -p 5000:5000 face_recognition
   ```

## Contribución

Las contribuciones son bienvenidas. Por favor abre un issue o envía un pull request.

## Licencia

Este proyecto está bajo la [licencia MIT](https://github.com/Jenn-LG/face-recognition-facenet/blob/main/LICENSE) para más detalles.

## Autor

[Jenn-LG](https://github.com/Jenn-LG)

---

**Nota**: Este proyecto es parte del Curso RNP 2025. Para más información sobre el reconocimiento facial y su implementación, consulta la documentación técnica en el código.