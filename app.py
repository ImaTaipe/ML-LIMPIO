from pathlib import Path
import streamlit as st
import cv2
import av
import random

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(
    page_title="Monitor Inteligente del Bebé",
    layout="centered"
)

st.title("Monitor Inteligente de Posición del Bebé")
st.subheader("Curso: Machine Learning")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/2/22/UPN_-_Universidad_Privada_del_Norte.png",
        use_container_width=True
    )
st.markdown("---")
# ===========================================
# DASHBOARD AMBIENTAL (SIMULADO)
# ===========================================

temperatura = round(random.uniform(20.0, 22.0), 1)
humedad = random.randint(75, 90)

if 20 <= temperatura <= 22 and 75 <= humedad <= 90:
    estado_ambiente = "🟢 Ambiente Ideal"
    color = "green"
else:
    estado_ambiente = "🟡 Revisar Ambiente"
    color = "orange"

st.subheader("🌡 Monitoreo Ambiental")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Temperatura",
    f"{temperatura} °C"
)

col2.metric(
    "Humedad",
    f"{humedad} %"
)

col3.metric(
    "Estado",
    estado_ambiente
)

st.progress((humedad - 75) / 15)

st.markdown(
    f"""
<div style="
background:#f8f9fa;
padding:15px;
border-radius:10px;
border-left:8px solid {color};
">
<b>Condición del ambiente:</b><br>

Temperatura recomendada:
<b>20°C - 22°C</b><br>

Humedad recomendada:
<b>75% - 90%</b>

</div>
""",
unsafe_allow_html=True
)

st.markdown("---")

st.markdown("""
### Sistema de monitoreo basado en IA para detección de posturas de sueño.

* **Boca arriba** = Seguro
* **Boca abajo** = Riesgo 
""")

st.markdown("""
<style>
video {
    max-width: 700px !important;
    margin: auto !important;
}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = Path("models/best.pt")

if not MODEL_PATH.exists():
    st.error("❌ No se encontró el modelo en models/best.pt")
    st.stop()

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        resultados = model(img, verbose=False)

        img_resultado = img.copy()

        estado = "Esperando detección..."
        color = (0, 255, 255)

        for r in resultados:

            img_resultado = r.plot()

            if len(r.boxes) == 0:
                continue

            box = r.boxes[0]

            clase = int(box.cls[0])
            confianza = float(box.conf[0]) * 100

            nombre = model.names[clase]

            if nombre == "baby_on_back":
                estado = f"TODO CORRECTO ({confianza:.1f}%)"
                color = (0, 255, 0)

            elif nombre == "baby_on_stomach":
                estado = f"ALERTA - BEBÉ EN RIESGO ({confianza:.1f}%)"
                color = (0, 0, 255)

        cv2.rectangle(
            img_resultado,
            (10, 10),
            (620, 70),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            img_resultado,
            estado,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )
        return av.VideoFrame.from_ndarray(
            img_resultado,
            format="bgr24"
        )

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
)

webrtc_streamer(
    key="bebe_monitor",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)
