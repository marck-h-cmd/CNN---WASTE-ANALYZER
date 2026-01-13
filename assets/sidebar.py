import streamlit as st
#import tensorflow as tf
#go3

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x150/4CAF50/FFFFFF?text=♻️", use_container_width=True)
    
    st.markdown("### 🧭 Navegación")
    page = st.radio("", [
        "🏠 Inicio",
        "🔍 Clasificador",
        "📊 Lote de Imágenes",
        "🎓 Sobre el Modelo",
        "🌱 Educación Ambiental",
        "📈 Estadísticas"
    ])
    
    st.markdown("---")
    st.markdown("### ⚙️ Configuración")
    confidence_threshold = st.slider("Umbral de confianza", 0.0, 1.0, 0.7, 0.05)
    
    st.markdown("---")
    st.markdown("### 📊 Resumen")
    st.metric("Clasificaciones totales", len(st.session_state.classifications))
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <small>Desarrollado con ❤️ y 🤖<br>
        CNN + Transfer Learning</small>
    </div>
    """, unsafe_allow_html=True)