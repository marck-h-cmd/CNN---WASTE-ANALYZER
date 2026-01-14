import streamlit as st
import os
import sys
import yaml
import json
import shutil
from pathlib import Path

# Añadir directorio src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Cargar configuración
@st.cache_resource
def load_config():
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo config.yaml")
        st.stop()

config = load_config()

# Configuración de página
st.set_page_config(
    page_title="⚙️ Configuración",
    page_icon="⚙️",
    layout="wide"
)

# Cargar CSS personalizado
def load_custom_css():
    css_file = Path(__file__).parent.parent / "assets" / "css" / "styles.css"
    if css_file.exists():
        with open(css_file, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_custom_css()

def save_configuration():
    """Guardar configuración en archivo"""
    try:
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        st.success("✅ Configuración guardada exitosamente!")
        st.session_state.config_changed = False
        return True
    
    except Exception as e:
        st.error(f"❌ Error guardando configuración: {str(e)}")
        return False

def export_configuration():
    """Exportar configuración como archivo"""
    config_json = json.dumps(config, indent=2, default=str)
    
    st.download_button(
        label="📥 Descargar Configuración (JSON)",
        data=config_json,
        file_name="configuracion_sistema.json",
        mime="application/json"
    )

def main():
    st.markdown('<h1 class="main-header">⚙️ Configuración del Sistema</h1>', unsafe_allow_html=True)
    
    # Tabs de configuración
    tab1, tab2, tab3, tab4 = st.tabs(["🔧 Sistema", "🧠 Modelo", "📊 Dashboard", "🚀 Entrenamiento"])
    
    with tab1:
        st.markdown("### 🔧 Configuración del Sistema")
        st.info("📌 Ajusta las rutas y configuraciones del sistema")
        
        with st.form("system_config_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📁 Rutas del Sistema")
                
                data_raw = st.text_input(
                    "Ruta datos originales",
                    value=config['paths']['data_raw'],
                    help="Carpeta donde están los datos sin procesar"
                )
                
                data_processed = st.text_input(
                    "Ruta datos procesados",
                    value=config['paths']['data_processed'],
                    help="Carpeta donde irán los datos procesados para YOLO"
                )
                
                models_dir = st.text_input(
                    "Directorio de modelos",
                    value=config['paths']['models_dir'],
                    help="Carpeta principal para almacenar modelos"
                )
                
                results_dir = st.text_input(
                    "Directorio de resultados",
                    value=config['paths']['results_dir'],
                    help="Carpeta para guardar resultados y logs"
                )
            
            with col2:
                st.subheader("⚡ Rendimiento")
                
                use_gpu = st.checkbox(
                    "Usar GPU si está disponible",
                    value=config['performance']['use_gpu'],
                    help="Habilitar aceleración GPU (CUDA) si está disponible"
                )
                
                max_workers = st.slider(
                    "Máximo de workers (procesadores)",
                    min_value=1,
                    max_value=8,
                    value=config['performance']['max_workers'],
                    help="Número de procesos paralelos para carga de datos"
                )
                
                cache_predictions = st.checkbox(
                    "Cachear predicciones",
                    value=config['performance']['cache_predictions'],
                    help="Almacenar en caché resultados de predicciones"
                )
                
                optimize_model = st.checkbox(
                    "Optimizar modelo",
                    value=config['performance']['optimize_model'],
                    help="Aplicar optimizaciones de rendimiento al modelo"
                )
            
            # Guardar configuración
            submitted = st.form_submit_button("💾 Guardar Configuración del Sistema", type="primary")
            
            if submitted:
                # Actualizar configuración
                config['paths']['data_raw'] = data_raw
                config['paths']['data_processed'] = data_processed
                config['paths']['models_dir'] = models_dir
                config['paths']['results_dir'] = results_dir
                config['performance']['use_gpu'] = use_gpu
                config['performance']['max_workers'] = max_workers
                config['performance']['cache_predictions'] = cache_predictions
                config['performance']['optimize_model'] = optimize_model
                
                if save_configuration():
                    st.balloons()
    
    with tab2:
        st.markdown("### 🧠 Configuración del Modelo")
        st.info("📌 Ajusta los parámetros del modelo YOLO")
        
        with st.form("model_config_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Modelo Base")
                
                # Obtener el nombre del modelo
                model_name_config = config['model']['name']
                # Si está solo "nano", convertir a "yolov8n"
                if model_name_config == "nano":
                    model_name_config = "yolov8n"
                
                model_options = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
                try:
                    current_idx = model_options.index(model_name_config)
                except ValueError:
                    current_idx = 0  # Por defecto, usar yolov8n
                
                model_name = st.selectbox(
                    "Nombre del modelo",
                    model_options,
                    index=current_idx,
                    help="Tamaño del modelo YOLO (n=nano, s=small, m=medium, l=large, x=xlarge)"
                )
                
                input_size_options = [224, 256, 320, 416, 512]
                input_size_idx = input_size_options.index(config['model']['input_size'])
                
                input_size = st.selectbox(
                    "Tamaño de entrada (pixels)",
                    input_size_options,
                    index=input_size_idx,
                    help="Resolución de imagen para el modelo"
                )
                
                pretrained = st.checkbox(
                    "Usar modelo preentrenado",
                    value=config['model']['pretrained'],
                    help="Inicializar con pesos preentrenados en ImageNet"
                )
            
            with col2:
                st.subheader("Predicción")
                
                confidence_threshold = st.slider(
                    "Umbral de confianza",
                    min_value=0.1,
                    max_value=1.0,
                    value=config['prediction']['confidence_threshold'],
                    step=0.05,
                    help="Confianza mínima para aceptar una predicción"
                )
                
                top_k_predictions = st.slider(
                    "Top-K predicciones",
                    min_value=1,
                    max_value=10,
                    value=config['prediction']['top_k_predictions'],
                    help="Número de predicciones principales a mostrar"
                )
                
                save_predictions = st.checkbox(
                    "Guardar predicciones",
                    value=config['prediction']['save_predictions'],
                    help="Almacenar resultados de predicciones"
                )
                
                save_visualizations = st.checkbox(
                    "Guardar visualizaciones",
                    value=config['prediction']['save_visualizations'],
                    help="Guardar imágenes con anotaciones de predicción"
                )
            
            # Guardar configuración
            submitted = st.form_submit_button("💾 Guardar Configuración del Modelo", type="primary")
            
            if submitted:
                config['model']['name'] = model_name
                config['model']['input_size'] = input_size
                config['model']['pretrained'] = pretrained
                config['prediction']['confidence_threshold'] = confidence_threshold
                config['prediction']['top_k_predictions'] = top_k_predictions
                config['prediction']['save_predictions'] = save_predictions
                config['prediction']['save_visualizations'] = save_visualizations
                
                if save_configuration():
                    st.balloons()
    
    with tab3:
        st.markdown("### 📊 Configuración del Dashboard")
        st.info("📌 Ajusta la apariencia y características de la interfaz")
        
        with st.form("dashboard_config_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎨 Apariencia")
                
                theme_options = ["light", "dark"]
                theme_idx = 0 if config['dashboard']['theme'] == "light" else 1
                
                theme = st.selectbox(
                    "Tema",
                    theme_options,
                    index=theme_idx,
                    help="Tema visual de la interfaz"
                )
                
                max_file_size = st.number_input(
                    "Tamaño máximo de archivo (MB)",
                    min_value=1,
                    max_value=500,
                    value=config['dashboard']['max_file_size_mb'],
                    help="Tamaño máximo para cargar imágenes"
                )
                
                title = st.text_input(
                    "Título del Dashboard",
                    value=config['dashboard']['title'],
                    help="Nombre que aparece en la página principal"
                )
            
            with col2:
                st.subheader("🚀 Características")
                
                enable_camera = st.checkbox(
                    "Habilitar cámara web",
                    value=config['dashboard']['enable_camera'],
                    help="Permite usar la cámara para capturar imágenes"
                )
                
                enable_batch = st.checkbox(
                    "Habilitar procesamiento por lotes",
                    value=config['dashboard']['enable_batch_processing'],
                    help="Procesar múltiples imágenes a la vez"
                )
                
                enable_comparison = st.checkbox(
                    "Habilitar comparación de modelos",
                    value=config['dashboard']['enable_model_comparison'],
                    help="Comparar resultados entre diferentes modelos"
                )
                
                enable_reports = st.checkbox(
                    "Habilitar generación de reportes",
                    value=config['dashboard']['enable_report_generation'],
                    help="Generar reportes en PDF de los resultados"
                )
            
            # Guardar configuración
            submitted = st.form_submit_button("💾 Guardar Configuración del Dashboard", type="primary")
            
            if submitted:
                config['dashboard']['theme'] = theme
                config['dashboard']['max_file_size_mb'] = max_file_size
                config['dashboard']['title'] = title
                config['dashboard']['enable_camera'] = enable_camera
                config['dashboard']['enable_batch_processing'] = enable_batch
                config['dashboard']['enable_model_comparison'] = enable_comparison
                config['dashboard']['enable_report_generation'] = enable_reports
                
                if save_configuration():
                    st.balloons()
    
    with tab4:
        st.markdown("### 🚀 Configuración de Entrenamiento")
        st.info("📌 Parámetros para entrenar nuevos modelos")
        
        col_form1, col_form2 = st.columns([2, 1])
        
        with col_form1:
            with st.form("training_config_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔧 Parámetros Básicos")
                    
                    epochs = st.number_input(
                        "Épocas",
                        min_value=1,
                        max_value=500,
                        value=config['training']['epochs'],
                        help="Número de pasadas sobre el dataset completo"
                    )
                    
                    batch_size = st.selectbox(
                        "Tamaño del batch",
                        [8, 16, 32, 64, 128],
                        index=[8, 16, 32, 64, 128].index(config['training']['batch_size']),
                        help="Número de imágenes por iteración"
                    )
                    
                    learning_rate = st.number_input(
                        "Tasa de aprendizaje",
                        min_value=0.00001,
                        max_value=0.1,
                        value=float(config['training']['learning_rate']),
                        format="%.6f",
                        help="Velocidad de actualización de pesos"
                    )
                    
                    device = st.selectbox(
                        "Dispositivo",
                        ["cpu", "cuda"],
                        index=0 if config['training']['device'] == 'cpu' else 1,
                        help="CPU o GPU para entrenar"
                    )
                
                with col2:
                    st.subheader("📊 Data Augmentation")
                    
                    augment = st.checkbox(
                        "Usar augmentación",
                        value=config['training']['augment'],
                        help="Aplicar transformaciones a las imágenes"
                    )
                    
                    scale = st.slider(
                        "Escala (scale)",
                        min_value=0.0,
                        max_value=1.0,
                        value=config['training']['scale'],
                        step=0.1,
                        help="Rango de escalado de imágenes"
                    )
                    
                    translate = st.slider(
                        "Traducción (translate)",
                        min_value=0.0,
                        max_value=0.5,
                        value=config['training']['translate'],
                        step=0.05,
                        help="Rango de desplazamiento de imágenes"
                    )
                    
                    degrees = st.slider(
                        "Rotación (degrees)",
                        min_value=0,
                        max_value=90,
                        value=config['training']['degrees'],
                        help="Grados de rotación"
                    )
                
                st.markdown("---")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    st.subheader("🎯 Regularización")
                    
                    weight_decay = st.number_input(
                        "Decaimiento de pesos",
                        min_value=0.0,
                        max_value=0.01,
                        value=float(config['training']['weight_decay']),
                        format="%.6f",
                        help="Regularización L2"
                    )
                    
                    dropout = st.number_input(
                        "Dropout",
                        min_value=0.0,
                        max_value=0.5,
                        value=float(config['training']['dropout']),
                        format="%.2f",
                        help="Probabilidad de dropout"
                    )
                    
                    patience = st.number_input(
                        "Paciencia (early stopping)",
                        min_value=1,
                        max_value=50,
                        value=config['training']['patience'],
                        help="Épocas sin mejora antes de parar"
                    )
                
                with col4:
                    st.subheader("📈 Optimizador")
                    
                    momentum = st.number_input(
                        "Momentum",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(config['training']['momentum']),
                        format="%.3f",
                        help="Momentum para optimizador SGD"
                    )
                    
                    warmup_epochs = st.number_input(
                        "Épocas de calentamiento",
                        min_value=0,
                        max_value=10,
                        value=config['training']['warmup_epochs'],
                        help="Épocas iniciales con LR más baja"
                    )
                    
                    validation_split = st.slider(
                        "Split validación",
                        min_value=0.1,
                        max_value=0.5,
                        value=config['training']['validation_split'],
                        step=0.05,
                        help="Proporción de datos para validación"
                    )
                
                # Guardar configuración
                submitted = st.form_submit_button("💾 Guardar Configuración de Entrenamiento", type="primary")
                
                if submitted:
                    config['training']['epochs'] = epochs
                    config['training']['batch_size'] = batch_size
                    config['training']['learning_rate'] = learning_rate
                    config['training']['device'] = device
                    config['training']['augment'] = augment
                    config['training']['scale'] = scale
                    config['training']['translate'] = translate
                    config['training']['degrees'] = degrees
                    config['training']['weight_decay'] = weight_decay
                    config['training']['dropout'] = dropout
                    config['training']['patience'] = patience
                    config['training']['momentum'] = momentum
                    config['training']['warmup_epochs'] = warmup_epochs
                    config['training']['validation_split'] = validation_split
                    
                    if save_configuration():
                        st.balloons()
        
        with col_form2:
            st.subheader("📌 Presets")
            if st.button("⚡ Rápido", use_container_width=True, help="Configuración rápida: 10 épocas"):
                config['training']['epochs'] = 10
                config['training']['batch_size'] = 32
                config['training']['learning_rate'] = 0.001
                if save_configuration():
                    st.success("✅ Preset aplicado")
                    st.rerun()
            
            if st.button("⚖️ Balanceado", use_container_width=True, help="Configuración balanceada: 50 épocas"):
                config['training']['epochs'] = 50
                config['training']['batch_size'] = 32
                config['training']['learning_rate'] = 0.001
                if save_configuration():
                    st.success("✅ Preset aplicado")
                    st.rerun()
            
            if st.button("🔬 Profundo", use_container_width=True, help="Configuración profunda: 100 épocas"):
                config['training']['epochs'] = 100
                config['training']['batch_size'] = 16
                config['training']['learning_rate'] = 0.0001
                if save_configuration():
                    st.success("✅ Preset aplicado")
                    st.rerun()
    
    # Acciones de sistema
    st.markdown("---")
    st.markdown("### ⚡ Acciones del Sistema")
    
    col_act1, col_act2, col_act3, col_act4 = st.columns(4)
    
    with col_act1:
        if st.button("🔄 Reiniciar Sistema", use_container_width=True, help="Limpiar caché y recargar"):
            st.cache_resource.clear()
            st.success("✅ Sistema reiniciado")
            st.rerun()
    
    with col_act2:
        if st.button("🧹 Limpiar Caché", use_container_width=True, help="Eliminar archivos de caché"):
            try:
                cache_dirs = ["./__pycache__", "./.streamlit"]
                for cache_dir in cache_dirs:
                    if os.path.exists(cache_dir):
                        shutil.rmtree(cache_dir)
                st.success("✅ Caché limpiado")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    with col_act3:
        if st.button("📤 Exportar Configuración", use_container_width=True, help="Descargar config como JSON"):
            export_configuration()
    
    with col_act4:
        if st.button("🔍 Ver Configuración Actual", use_container_width=True, help="Mostrar todas las configuraciones"):
            with st.expander("📋 Configuración actual (YAML)"):
                st.code(yaml.dump(config, default_flow_style=False), language="yaml")
    
    # Información del sistema
    st.markdown("---")
    st.markdown("### 📊 Información del Sistema")
    
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        import torch
        st.metric("GPU Disponible", "✅ Sí" if torch.cuda.is_available() else "❌ No")
    
    with col_info2:
        st.metric("PyTorch Version", torch.__version__)
    
    with col_info3:
        st.metric("Clases disponibles", len(config['classes']))

if __name__ == "__main__":
    main()
