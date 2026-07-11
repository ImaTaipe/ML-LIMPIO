# 👶 Baby Sleep Monitor AI

Sistema inteligente para la detección de posturas de sueño en bebés utilizando visión por computadora e inteligencia artificial.

## Características

- Detección en tiempo real mediante cámara web.
- Clasificación de postura:
  - ✅ Boca arriba (Seguro)
  - 🚨 Boca abajo (Riesgo)
- Modelo entrenado con YOLOv8.
- Interfaz desarrollada con Streamlit.
- Captura de video mediante WebRTC.

## Tecnologías

- Python
- Streamlit
- YOLOv8 (Ultralytics)
- OpenCV
- streamlit-webrtc

## Estructura del proyecto

```
app.py
requirements.txt
models/
└── best.pt
```

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```