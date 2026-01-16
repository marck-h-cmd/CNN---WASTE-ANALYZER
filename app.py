
import streamlit as st
import os
import sys
from pathlib import Path
import yaml
import pandas as pd
import json
#go2
# Añadir directorio src al path
sys.path.append(str(Path(__file__).parent / "src"))



# Mapeo UI (Español) -> Dataset
CLASS_LABELS = {
    "batería": "battery",
    "biológico": "biological",
    "vidrio marrón": "brown-glass",
    "cartón": "cardboard",
    "ropa": "clothes",
    "vidrio verde": "green-glass",
    "metal": "metal",
    "papel": "paper",
    "plástico": "plastic",
    "zapatos": "shoes",
    "basura": "trash",
    "vidrio blanco": "white-glass",
}

# Configuración de página
st.set_page_config(
    page_title="Clasificador de Residuos Inteligente",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Cargar CSS personalizado
def load_custom_css():
    css_file = Path(__file__).parent / "assets" / "css" / "styles.css"
    if css_file.exists():
        with open(css_file, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        # CSS por defecto
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #2E8B57;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: bold;
        }
        .sub-header {
            font-size: 1.8rem;
            color: #3CB371;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        .metric-card {
            background: linear-gradient(135deg, #f0f8ff 0%, #e6f7ff 100%);
            padding: 1.5rem;
            border-radius: 15px;
            border-left: 5px solid #2E8B57;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        .class-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 25px;
            font-weight: 600;
            margin: 0.3rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stButton>button {
            background: linear-gradient(135deg, #2E8B57 0%, #3CB371 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(46, 139, 87, 0.3);
        }
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .warning-box {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

def main():
    """Función principal de la aplicación"""
    
    # Cargar CSS
    load_custom_css()
    
    # Sidebar
    with st.sidebar:
        # Logo
        logo_path = Path(__file__).parent / "assets" / "images" / "logo.png"
        if logo_path.exists():
            st.image(str(logo_path), width=200)
        else:
            st.title("🗑️ Clasificador")
        
        st.markdown("---")
        
        # Menú de navegación
        st.subheader("📊 Navegación")
        
        menu_options = [
            "🏠 Página Principal",
            "📁 Gestionar Datos",
            "🚀 Entrenar Modelo", 
            "🔍 Clasificar Residuos",
            "📈 Análisis y Métricas",
            "⚙️ Configuración"
        ]
        
        # Usar session_state para navegación
        if 'selected_page' not in st.session_state:
            st.session_state.selected_page = "🏠 Página Principal"
        
        selected_page = st.radio(
            "Selecciona una página:",
            menu_options,
            index=menu_options.index(st.session_state.selected_page),
            label_visibility="collapsed"
        )
        
        st.session_state.selected_page = selected_page
        
        # st.markdown("---")
        
        # # Estado del sistema
        # st.subheader("📊 Estado del Sistema")
        
        # # Verificar modelo
        # model_path = Path(config['paths']['trained_models']) / "best.pt"
        # if model_path.exists():
        #     st.success("✅ Modelo disponible")
        #     model_status = "Entrenado"
        # else:
        #     st.warning("⚠️ Sin modelo entrenado")
        #     model_status = "No entrenado"
        
        # # Verificar datos
        # data_path = Path(config['paths']['data_processed'])
        # if data_path.exists() and any(data_path.iterdir()):
        #     st.success("✅ Datos disponibles")
        #     data_status = "Procesados"
        # else:
        #     st.warning("⚠️ Datos no procesados")
        #     data_status = "Sin procesar"
        
        # col1, col2 = st.columns(2)
        # with col1:
        #     st.metric("Modelo", model_status)
        # with col2:
        #     st.metric("Datos", data_status)
        
        st.markdown("---")
        
        # Acciones rápidas
        st.subheader("⚡ Acciones Rápidas")
        
        if st.button("🔄 Verificar Sistema", width='stretch'):
            st.rerun()
        
        if st.button("🧹 Limpiar Caché", width='stretch'):
            st.cache_resource.clear()
            st.success("Caché limpiado!")
        
        if st.button("📥 Exportar Config", width='stretch'):
            export_configuration()
    
    # Contenido principal según página seleccionada
    if selected_page == "🏠 Página Principal":
        show_home_page()
    elif selected_page == "📁 Gestionar Datos":
        show_data_management_page()
    elif selected_page == "🚀 Entrenar Modelo":
        show_training_page()
    elif selected_page == "🔍 Clasificar Residuos":
        show_classification_page()
    elif selected_page == "📈 Análisis y Métricas":
        show_analysis_page()
    elif selected_page == "⚙️ Configuración":
        show_configuration_page()

def show_home_page():
    """Mostrar página de inicio"""
    st.markdown('<h1 class="main-header">🏠 Bienvenido al Clasificador de Residuos</h1>', unsafe_allow_html=True)
    
    # Introducción amigable
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🌱 ¡Ayuda al planeta clasificando residuos!
        
        Nuestra aplicación inteligente te ayuda a identificar y clasificar diferentes tipos de residuos 
        de manera rápida y precisa. Solo sube una foto y obtén resultados instantáneos.
        
        ### ✨ ¿Qué puedes hacer aquí?
        
        🎯 **Clasificar residuos** - Sube fotos o usa tu cámara  
        📊 **Analizar resultados** - Revisa métricas y estadísticas  
        🚀 **Entrenar modelos** - Mejora la precisión con tus datos  
     
        
        ### 🚀 Empieza en 3 pasos simples
        
        1. **Prepara tus datos** - Organiza las imágenes de residuos
        2. **Entrena el modelo** - Ajusta la IA con tus ejemplos  
        3. **¡Clasifica!** - Comienza a identificar residuos automáticamente
        """)
        
        # Llamado a la acción
        st.markdown("---")
        st.markdown("### 🎯 ¿Listo para comenzar?")
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("📁 Preparar Datos", width='stretch', type="secondary"):
                st.session_state.selected_page = "📁 Gestionar Datos"
                st.rerun()
        
        with col_btn2:
            if st.button("🚀 Entrenar Modelo", width='stretch', type="secondary"):
                st.session_state.selected_page = "🚀 Entrenar Modelo"
                st.rerun()
        
        with col_btn3:
            if st.button("🎯 Clasificar Ahora", width='stretch', type="primary"):
                st.session_state.selected_page = "🔍 Clasificar Residuos"
                st.rerun()
    
    with col2:
        # Tarjeta de información general
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("📱 Sobre la App")
        
        st.markdown("""
        **Tecnología**: IA Avanzada  
        **Precisión**: Hasta 95%  
        **Velocidad**: < 100ms por imagen  
        **Plataformas**: Web, Móvil, Desktop
        """)
        
        # Estado del sistema
        st.markdown("---")
        st.subheader("⚡ Estado Actual")
        
        # Verificar modelo
        model_path = Path(config['paths']['trained_models']) / "best.pt"
        if model_path.exists():
            st.success("✅ Modelo listo")
        else:
            st.warning("⚠️ Sin modelo entrenado")
        
        # Verificar datos
        data_path = Path(config['paths']['data_processed'])
        if data_path.exists() and any(data_path.iterdir()):
            st.success("✅ Datos preparados")
        else:
            st.warning("⚠️ Datos no procesados")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Mostrar clases de manera atractiva
    st.markdown("---")
    st.markdown('<h3 class="sub-header">🗂️ Tipos de Residuos que podemos identificar</h3>', unsafe_allow_html=True)
    
    # Mostrar badges de clases en un diseño más atractivo
    classes = config['classes']
    cols = st.columns(4)
    
    # Colores para badges
    colors = [
        "#FF6B6B", "#4ECDC4", "#FFD166", "#06D6A0",
        "#118AB2", "#073B4C", "#EF476F", "#7209B7",
        "#3A86FF", "#FB5607", "#8338EC", "#FF006E"
    ]
    
    for idx, class_name in enumerate(classes):
        with cols[idx % 4]:
            color = colors[idx % len(colors)]
            display_name = class_name.replace("-", " ").title()
            st.markdown(
                f'<div class="class-badge" style="background-color: {color}; color: white; '
                f'font-size: 14px; padding: 8px 12px; margin: 4px; border-radius: 20px; '
                f'text-align: center;">{display_name}</div>',
                unsafe_allow_html=True
            )
    
    # Información técnica (mantener pero hacer más accesible)
    st.markdown("---")
    st.markdown('<h3 class="sub-header">🔧 Tecnología Detrás</h3>', unsafe_allow_html=True)
    
    tech_cols = st.columns(3)
    
    with tech_cols[0]:
        st.markdown("""
        ### 🤖 Inteligencia Artificial
        - **Modelo**: YOLOv8 (última generación)
        - **Aprendizaje**: Deep Learning automático
        - **Entrenamiento**: Optimizado para velocidad
        """)
    
    with tech_cols[1]:
        st.markdown("""
        ### 📊 Dataset
        - **Fuente**: Garbage Classification Dataset
        - **Imágenes**: Miles de ejemplos reales
        - **Categorías**: 12 tipos de residuos
        """)
    
    with tech_cols[2]:
        st.markdown("""
        ### ⚡ Rendimiento
        - **Precisión**: 85-95% de acierto
        - **Velocidad**: Procesamiento instantáneo
        - **Compatibilidad**: Funciona en cualquier dispositivo
        """)
    
    # Características técnicas
    st.markdown("---")
    st.markdown('<h3 class="sub-header">🔧 Especificaciones Técnicas</h3>', unsafe_allow_html=True)
    
    tech_cols = st.columns(3)
    
    with tech_cols[0]:
        st.markdown("""
        ### 🧠 Arquitectura del Modelo
        - **Framework**: YOLOv8 (Ultralytics)
        - **Tipo**: Solo Clasificación
        - **Backbone**: CSPDarknet
        - **Pre-entrenado**: ImageNet
        - **Parámetros**: 3.2M (nano)
        """)
    
    with tech_cols[1]:
        st.markdown("""
        ### 📊 Dataset Original
        - **Nombre**: Garbage Classification
        - **Fuente**: Kaggle
        - **Clases**: 12 categorías
        - **Imágenes**: ~15,000
        - **Licencia**: CC BY-SA 4.0
        """)
    
    with tech_cols[2]:
        st.markdown("""
        ### ⚡ Rendimiento
        - **Precisión Top-1**: >85%
        - **Precisión Top-5**: >95%
        - **Tiempo Inferencia**: 45ms (GPU)
        - **Compatibilidad**: ONNX, TensorRT
        - **Plataforma**: Web, Móvil, Edge
        """)

def show_data_management_page():
    """Mostrar página de gestión de datos"""
    st.markdown('<h1 class="main-header">📁 Gestión de Datos del Dataset</h1>', unsafe_allow_html=True)
    
    # Importar funciones de preparación de datos
    from src.data_preparation import DataPreparer
    
    preparer = DataPreparer(config)
    
    # Tabs para diferentes operaciones
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Explorar Dataset", 
        "🔄 Preparar Datos", 
        "📈 Estadísticas", 
        "🔍 Ver Imágenes"
    ])
    
    with tab1:
        st.markdown("### 📊 Explorar Dataset Original")
        
        # Verificar dataset original
        raw_path = Path(config['paths']['data_raw'])
        
        if raw_path.exists():
            st.success(f"✅ Dataset encontrado en: {raw_path}")
            
            # Mostrar estructura
            st.markdown("#### Estructura de Carpetas:")
            
            import os
            folders = [f for f in os.listdir(raw_path) if os.path.isdir(raw_path / f)]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Carpetas de Clases:**")
                for folder in sorted(folders):
                    st.write(f"📁 {folder}")
            
            with col2:
                # Contar imágenes por clase
                st.write("**Conteo de Imágenes:**")
                for folder in sorted(folders)[:12]:  # Mostrar  las 12 clases
                    folder_path = raw_path / folder
                    images = [f for f in os.listdir(folder_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    st.write(f"📸 {folder}: {len(images)} imágenes")
            
            # Resumen total
            total_images = preparer.count_total_images()
            st.info(f"📦 **Total de imágenes en dataset:** {total_images:,}")
            
        else:
            st.warning(f"⚠️ No se encontró el dataset en: {raw_path}")
            st.markdown("""
            ### 📥 Descargar Dataset
            
            El dataset **Garbage Classification** está disponible en Kaggle:
            
            1. Visita: [https://www.kaggle.com/datasets/mostafaabla/garbage-classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
            2. Descarga el dataset
            3. Extrae las carpetas en: `data/raw/`
            
            Estructura esperada:
            ```
            data/raw/
            ├── battery/
            ├── biological/
            ├── brown-glass/
            ├── cardboard/
            ├── clothes/
            ├── green-glass/
            ├── metal/
            ├── paper/
            ├── plastic/
            ├── shoes/
            ├── trash/
            └── white-glass/
            ```
            """)
    
    with tab2:
        st.markdown("### 🔄 Preparar Datos para YOLO")
        
        if st.button("🔄 Procesar Dataset", type="primary", width='stretch'):
            with st.spinner("Procesando dataset para YOLO..."):
                try:
                    stats = preparer.prepare_yolo_dataset()
                    
                    st.success("✅ Dataset procesado exitosamente!")
                    
                    # Mostrar estadísticas
                    st.markdown("#### 📊 Estadísticas del Procesamiento")
                    
                    df_stats = preparer.get_statistics_dataframe()
                    st.dataframe(df_stats, width='stretch')
                    
                    # Gráfico de distribución
                    fig = preparer.plot_class_distribution()
                    st.plotly_chart(fig, width='stretch')
                    
                except Exception as e:
                    st.error(f"❌ Error procesando dataset: {str(e)}")
        
        # Opciones de procesamiento
        with st.expander("⚙️ Opciones Avanzadas"):
            col1, col2 = st.columns(2)
            
            with col1:
                validation_split = st.slider(
                    "Proporción de Validación",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.2,
                    step=0.05,
                    help="Porcentaje de datos para validación"
                )
            
            with col2:
                image_size = st.selectbox(
                    "Tamaño de Imagen",
                    [224, 256, 320, 416, 512],
                    index=0,
                    help="Tamaño al que se redimensionarán las imágenes"
                )
            
            augment_data = st.checkbox(
                "Aplicar aumento de datos",
                value=True,
                help="Aplicar transformaciones para aumentar el dataset"
            )
    
    with tab3:
        st.markdown("### 📈 Estadísticas Detalladas")
        
        try:
            # Generar reporte estadístico
            report = preparer.generate_statistics_report()
            
            # Métricas principales
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Imágenes", f"{report['total_images']:,}")
            
            with col2:
                st.metric("Clases", report['num_classes'])
            
            with col3:
                st.metric("Proporción Train/Val", f"{report['train_val_ratio']:.1%}")
            
            # Distribución por clase
            st.markdown("#### Distribución por Clase")
            st.dataframe(report['class_distribution'], width='stretch')
            
            # Balance de clases
            st.markdown("#### 📊 Balance de Clases")
            
            balance_ratio = report['balance_ratio']
            if balance_ratio > 0.7:
                st.success(f"✅ Dataset balanceado (ratio: {balance_ratio:.2f})")
            elif balance_ratio > 0.4:
                st.warning(f"⚠️ Dataset moderadamente balanceado (ratio: {balance_ratio:.2f})")
            else:
                st.error(f"❌ Dataset desbalanceado (ratio: {balance_ratio:.2f})")
                st.markdown("""
                **Recomendación:** Considera aplicar técnicas de balanceo como:
                - Sobremuestreo (oversampling)
                - Submuestreo (undersampling)
                - Aumento de datos específico por clase
                """)
            
        except Exception as e:
            st.warning("Primero procesa el dataset para ver estadísticas.")
    
    with tab4:
        st.markdown("### 🔍 Visualizar Imágenes del Dataset")
        
        # Seleccionar clase
        selected_label = st.selectbox(
            "Selecciona una clase para ver imágenes:",
             list(CLASS_LABELS.keys())
        )
        
        if selected_label:
            selected_class = CLASS_LABELS[selected_label]
            
            if selected_label :
                selected_class = CLASS_LABELS[selected_label]

                sample_images = preparer.get_sample_images(
                selected_class, 
                 num_samples=6
                )
                if sample_images:
                    st.markdown(f"#### Imágenes de: {selected_label}")
                
                # Mostrar en grid
                cols = st.columns(3)
                for idx, img_path in enumerate(sample_images):
                    with cols[idx % 3]:
                        st.image(str(img_path), width='stretch')
                        st.caption(f"{img_path.name}")
            else:
                st.info(f"No hay imágenes para la clase {selected_class}")

def show_training_page():
    """Mostrar página de entrenamiento del modelo"""
    st.markdown('<h1 class="main-header">🚀 Entrenamiento del Modelo YOLO</h1>', unsafe_allow_html=True)
    
    from src.model_trainer import ModelTrainer
    
    # Verificar datos procesados
    processed_path = Path(config['paths']['data_processed'])
    if not processed_path.exists() or not any(processed_path.iterdir()):
        st.warning("⚠️ Primero debes preparar los datos en la página 'Gestionar Datos'")
        if st.button("📁 Ir a Gestionar Datos"):
            st.switch_page("pages/02_📁_Preparar_Datos.py")
        return
    
    # Inicializar entrenador
    trainer = ModelTrainer(config)
    
    # Tabs para entrenamiento
    tab1, tab2, tab3 = st.tabs(["🎯 Configurar Entrenamiento", "🚀 Entrenar Modelo", "📊 Resultados"])
    
    with tab1:
        st.markdown("### 🎯 Configuración del Entrenamiento")
        
        # Configuración básica
        col1, col2 = st.columns(2)
        
        with col1:
            model_size = st.selectbox(
                "Tamaño del Modelo",
                ["nano (yolov8n)", "small (yolov8s)", "medium (yolov8m)", "large (yolov8l)", "xlarge (yolov8x)"],
                index=0,
                help="Modelos más grandes son más precisos pero más lentos"
            )
            
            epochs = st.number_input(
                "Número de Épocas",
                min_value=10,
                max_value=500,
                value=config['training']['epochs'],
                step=10
            )
            
            batch_size = st.selectbox(
                "Tamaño del Batch",
                [8, 16, 32, 64],
                index=2
            )
        
        with col2:
            learning_rate = st.number_input(
                "Tasa de Aprendizaje",
                min_value=0.00001,
                max_value=0.1,
                value=config['training']['learning_rate'],
                step=0.0001,
                format="%.5f"
            )
            
            device = st.selectbox(
                "Dispositivo",
                ["auto (detectar)", "cpu", "cuda (GPU)"],
                index=0
            )
            
            patience = st.number_input(
                "Paciencia (Early Stopping)",
                min_value=3,
                max_value=50,
                value=config['training']['patience'],
                step=1
            )
        
        # Configuración avanzada
        with st.expander("⚙️ Configuración Avanzada"):
            col_adv1, col_adv2 = st.columns(2)
            
            with col_adv1:
                augment = st.checkbox(
                    "Aumentación de Datos",
                    value=config['training']['augment'],
                    help="Aplica transformaciones aleatorias a las imágenes"
                )
                
                dropout = st.slider(
                    "Dropout Rate",
                    min_value=0.0,
                    max_value=0.5,
                    value=config['training']['dropout'],
                    step=0.05
                )
            
            with col_adv2:
                weight_decay = st.number_input(
                    "Weight Decay",
                    min_value=0.0,
                    max_value=0.01,
                    value=config['training']['weight_decay'],
                    step=0.0001,
                    format="%.4f"
                )
                
                warmup_epochs = st.number_input(
                    "Épocas de Warmup",
                    min_value=0,
                    max_value=10,
                    value=config['training']['warmup_epochs'],
                    step=1
                )
        
        # Guardar configuración
        if st.button("💾 Guardar Configuración", width='stretch'):
            # Actualizar configuración
            config['model']['name'] = model_size.split()[0]
            config['training']['epochs'] = epochs
            config['training']['batch_size'] = batch_size
            config['training']['learning_rate'] = learning_rate
            config['training']['patience'] = patience
            config['training']['augment'] = augment
            config['training']['dropout'] = dropout
            config['training']['weight_decay'] = weight_decay
            config['training']['warmup_epochs'] = warmup_epochs
            
            # Guardar
            with open('config.yaml', 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            st.success("✅ Configuración guardada!")
    
    with tab2:
        st.markdown("### 🚀 Entrenar Modelo YOLO")
        
        # Información previa al entrenamiento
        st.info("""
        **📋 Información del Entrenamiento:**
        
        - **Dataset**: Garbage Classification (12 clases)
        - **Tipo**: Clasificación de imágenes
        - **Modelo**: YOLOv8 (modo clasificación)
        - **Hardware recomendado**: GPU con al menos 4GB VRAM
        - **Tiempo estimado**: 30-60 minutos (depende de épocas y hardware)
        """)
        
        # Verificar recursos CON MÁS DETALLE
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            import torch
            has_gpu = torch.cuda.is_available()
            # ✅ LÍNEAS CORREGIDAS:
            if has_gpu:
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                except Exception as e:
                    gpu_name = "Error detectando GPU"
                    gpu_memory = 0
                    has_gpu = False
        
        with col_res2:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)
            if ram_gb >= 16:
                st.success(f"✅ RAM: {ram_gb:.1f} GB")
            elif ram_gb >= 8:
                st.warning(f"⚠️ RAM: {ram_gb:.1f} GB")
            else:
                st.error(f"❌ RAM: {ram_gb:.1f} GB")
        
        with col_res3:
            # Mostrar dispositivo seleccionado
            device_display = device
            if device == "auto (detectar)":
                if has_gpu:
                    device_display = "GPU (detectada)"
                else:
                    device_display = "CPU (no hay GPU)"
            
            st.info(f"🎯 Dispositivo: {device_display}")
        
        # Botón para probar GPU
        if st.button("🧪 Probar GPU", type="secondary"):
            import torch
            if torch.cuda.is_available():
                # Operación de prueba
                x = torch.randn(1000, 1000).cuda()
                y = torch.randn(1000, 1000).cuda()
                z = torch.matmul(x, y)
                st.success(f"✅ GPU funciona correctamente")
                st.write(f"Operación completada: {z.shape} en GPU")
            else:
                st.error("❌ GPU no disponible")
        
        # Botón para iniciar entrenamiento
        if st.button("🎬 Iniciar Entrenamiento", type="primary", width='stretch'):
            
            # Área para logs de entrenamiento
            training_logs = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_container = st.empty()
            
            # Callback para actualizar UI durante entrenamiento
            def training_callback(epoch, total_epochs, metrics):
                # Actualizar progreso
                progress = (epoch + 1) / total_epochs
                progress_bar.progress(progress)
                
                # Actualizar texto
                status_text.text(f"Época {epoch + 1}/{total_epochs} - Loss: {metrics.get('loss', 0):.4f}")
                
                # Mostrar métricas en logs
                with training_logs.container():
                    st.write(f"✅ Época {epoch + 1} completada")
                    st.write(f"   📉 Loss: {metrics.get('train/loss', metrics.get('loss', 0)):.4f}")
                    st.write(f"   📈 Accuracy: {metrics.get('metrics/accuracy', 0):.4f}")
                    if 'lr/pg0' in metrics:
                        st.write(f"   📚 LR: {metrics['lr/pg0']:.6f}")
            
            # Iniciar entrenamiento
            with st.spinner("🚀 Iniciando entrenamiento..."):
                try:
                    # Convertir dispositivo correctamente
                    device_param = device
                    if device == "auto (detectar)":
                        device_param = "auto"
                    elif device == "cuda (GPU)":
                        device_param = "cuda"
                    
                    # Mostrar configuración final
                    st.info(f"**Configuración final:** Épocas={epochs}, Batch={batch_size}, Device={device_param}")
                    
                    results = trainer.train_model(
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        device=device_param,  # Ya convertido
                        callback=training_callback
                    )
                    
                    st.success("✅ ¡Entrenamiento completado exitosamente!")
                    st.balloons()
                    
                    # Mostrar resumen
                    st.markdown("#### 📊 Resumen del Entrenamiento")
                    
                    col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                    
                    with col_res1:
                        st.metric("Épocas", results.get('epochs', epochs))
                    
                    with col_res2:
                        final_acc = results.get('metrics', {}).get('accuracy', 0)
                        st.metric("Precisión", f"{final_acc:.2%}")
                    
                    with col_res3:
                        training_time = results.get('training_time', 0)
                        st.metric("Tiempo", f"{training_time:.1f} min")
                    
                    with col_res4:
                        device_used = results.get('device', 'cpu')
                        st.metric("Dispositivo", "GPU" if device_used == 'cuda' else "CPU")
                    
                    # Mostrar métricas detalladas
                    with st.expander("📈 Ver métricas detalladas"):
                        if 'metrics' in results:
                            metrics = results['metrics']
                            st.write("**Métricas por clase:**")
                            if 'class_report' in metrics:
                                report_df = pd.DataFrame(metrics['class_report']).transpose()
                                st.dataframe(report_df)
                            
                            st.write(f"**Exactitud:** {metrics.get('accuracy', 0):.4f}")
                            st.write(f"**Precisión:** {metrics.get('precision', 0):.4f}")
                            st.write(f"**Recall:** {metrics.get('recall', 0):.4f}")
                            st.write(f"**F1-Score:** {metrics.get('f1_score', 0):.4f}")
                    
                    # Enlace al modelo entrenado
                    model_path = results.get('model_path', '')
                    if model_path and Path(model_path).exists():
                        st.markdown(f"**📁 Modelo guardado en:** `{model_path}`")
                        
                except Exception as e:
                    st.error(f"❌ Error durante el entrenamiento: {str(e)}")
                    st.error("""
                    **Posibles soluciones:**
                    1. Reduce el batch_size (16 o 8)
                    2. Verifica que el dataset esté correctamente organizado
                    3. Revisa los logs de error arriba
                    """)
    
    with tab3:
        st.markdown("### 📊 Resultados del Entrenamiento")

        # Verificar si hay experimentos entrenados
        results_dir = Path(config['paths']['results_dir']) / 'training_logs'
        experiments = []

        if results_dir.exists():
            experiments = [d for d in results_dir.iterdir() if d.is_dir()]

        if experiments:
            # Encontrar el experimento más reciente
            latest_experiment = max(experiments, key=lambda x: x.stat().st_mtime)
            experiment_name = latest_experiment.name

            st.success(f"✅ Mostrando resultados del último entrenamiento: **{experiment_name}**")

            # Mostrar información básica del experimento
            results_file = latest_experiment / f"results_{latest_experiment.stat().st_mtime:.0f}.json"

            # Buscar archivo de resultados más reciente
            results_files = list(latest_experiment.glob("results_*.json"))
            if results_files:
                results_file = max(results_files, key=lambda x: x.stat().st_mtime)

                try:
                    with open(results_file, 'r', encoding='utf-8') as f:
                        experiment_data = json.load(f)

                    # Mostrar métricas principales
                    metrics = experiment_data.get('metrics', {})

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("🎯 Exactitud", f"{metrics.get('accuracy', 0):.2%}")

                    with col2:
                        st.metric("📊 Precisión", f"{metrics.get('precision', 0):.2%}")

                    with col3:
                        st.metric("🔍 Recall", f"{metrics.get('recall', 0):.2%}")

                    with col4:
                        st.metric("📈 F1-Score", f"{metrics.get('f1_score', 0):.2%}")

                    # Información adicional del entrenamiento
                    st.markdown("#### 📋 Información del Entrenamiento")

                    info_cols = st.columns(4)
                    with info_cols[0]:
                        st.info(f"⏱️ Tiempo: {experiment_data.get('training_time', 0):.1f} min")
                    with info_cols[1]:
                        st.info(f"🎯 Épocas: {experiment_data.get('epochs', 0)}")
                    with info_cols[2]:
                        st.info(f"📦 Batch: {experiment_data.get('batch_size', 0)}")
                    with info_cols[3]:
                        device_used = experiment_data.get('device', 'cpu')
                        st.info(f"💻 Dispositivo: {'GPU' if 'cuda' in str(device_used) else 'CPU'}")

                except Exception as e:
                    st.warning(f"No se pudo cargar la información detallada: {e}")

            # Mostrar gráficas generadas
            st.markdown("---")
            st.markdown("#### 📈 Gráficas Generadas")

            plots_dir = latest_experiment / 'plots'
            if plots_dir.exists():
                # Buscar archivos de gráficas
                training_history_files = list(plots_dir.glob("training_history_*.png"))
                confusion_matrix_files = list(plots_dir.glob("confusion_matrix_*.png"))

                # Mostrar training history
                if training_history_files:
                    training_history_file = max(training_history_files, key=lambda x: x.stat().st_mtime)
                    try:
                        st.markdown("**📊 Historial de Entrenamiento:**")
                        st.image(str(training_history_file), caption="Evolución del entrenamiento", use_column_width=True)
                    except Exception as e:
                        st.warning(f"No se pudo cargar la gráfica de historial: {e}")

                # Mostrar confusion matrix
                if confusion_matrix_files:
                    confusion_matrix_file = max(confusion_matrix_files, key=lambda x: x.stat().st_mtime)
                    try:
                        st.markdown("**🎯 Matriz de Confusión:**")
                        st.image(str(confusion_matrix_file), caption="Matriz de Confusión", use_column_width=True)
                    except Exception as e:
                        st.warning(f"No se pudo cargar la matriz de confusión: {e}")

                if not training_history_files and not confusion_matrix_files:
                    st.info("ℹ️ No se encontraron gráficas generadas durante el entrenamiento.")
            else:
                st.info("ℹ️ No se encontraron gráficas para este experimento.")

            # Mostrar archivos generados por YOLO
            st.markdown("---")
            st.markdown("#### 📁 Archivos Generados por YOLO")

            model_yolo_dir = Path(config['paths']['trained_models']) / experiment_name
            if model_yolo_dir.exists():
                # Mostrar estructura de archivos
                with st.expander("📂 Ver estructura de archivos generados"):
                    file_structure = []

                    def get_file_structure(path, prefix=""):
                        if path.is_dir():
                            file_structure.append(f"{prefix}📁 {path.name}/")
                            for item in sorted(path.iterdir()):
                                get_file_structure(item, prefix + "  ")
                        else:
                            size = path.stat().st_size
                            if size < 1024:
                                size_str = f"{size} B"
                            elif size < 1024*1024:
                                size_str = f"{size/1024:.1f} KB"
                            else:
                                size_str = f"{size/(1024*1024):.1f} MB"
                            file_structure.append(f"{prefix}📄 {path.name} ({size_str})")

                    get_file_structure(model_yolo_dir)
                    st.code("\n".join(file_structure))

                # Mostrar contenido del results.csv si existe
                results_csv = model_yolo_dir / 'results.csv'
                if results_csv.exists():
                    try:
                        results_df = pd.read_csv(results_csv)
                        st.markdown("**📊 Resultados Detallados del Entrenamiento:**")
                        st.dataframe(results_df, use_container_width=True)
                    except Exception as e:
                        st.warning(f"No se pudo cargar results.csv: {e}")

                # Mostrar args.yaml si existe
                args_yaml = model_yolo_dir / 'args.yaml'
                if args_yaml.exists():
                    try:
                        with open(args_yaml, 'r') as f:
                            args_content = f.read()
                        with st.expander("⚙️ Ver configuración del entrenamiento (args.yaml)"):
                            st.code(args_content, language='yaml')
                    except Exception as e:
                        st.warning(f"No se pudo cargar args.yaml: {e}")

                # Botón para descargar el modelo
                model_file = model_yolo_dir / 'weights' / 'best.pt'
                if model_file.exists():
                    with open(model_file, 'rb') as f:
                        model_bytes = f.read()
                    st.download_button(
                        label="📥 Descargar Modelo Entrenado (best.pt)",
                        data=model_bytes,
                        file_name=f"{experiment_name}_best.pt",
                        mime="application/octet-stream"
                    )
            else:
                st.info("ℹ️ No se encontraron archivos del modelo YOLO para este experimento.")

            # Mostrar imágenes generadas por YOLO en la carpeta del modelo
            st.markdown("---")
            st.markdown("#### 🖼️ Imágenes Generadas por YOLO")

            # Buscar imágenes en la carpeta del modelo entrenado
            model_images_dir = Path('runs/classify/models/trained') / experiment_name
            if model_images_dir.exists():
                # Buscar todas las imágenes
                all_images = []
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    all_images.extend(list(model_images_dir.glob(ext)))

                if all_images:
                    st.success(f"✅ Se encontraron {len(all_images)} imágenes generadas por YOLO")

                    # Categorizar imágenes
                    confusion_matrices = [img for img in all_images if 'confusion_matrix' in img.name]
                    results_images = [img for img in all_images if img.name == 'results.png']
                    train_batches = [img for img in all_images if 'train_batch' in img.name]
                    val_batches = [img for img in all_images if 'val_batch' in img.name]

                    # Mostrar matrices de confusión
                    if confusion_matrices:
                        st.markdown("**🎯 Matrices de Confusión:**")
                        cols = st.columns(min(len(confusion_matrices), 2))
                        for idx, img_file in enumerate(confusion_matrices[:2]):
                            try:
                                with cols[idx]:
                                    caption = "Normalizada" if "normalized" in img_file.name else "Estándar"
                                    st.image(str(img_file), caption=f"Matriz de Confusión - {caption}", use_column_width=True)
                            except Exception as e:
                                st.warning(f"Error cargando {img_file.name}: {e}")

                    # Mostrar gráfica de resultados
                    if results_images:
                        st.markdown("**📊 Gráfica de Resultados:**")
                        try:
                            st.image(str(results_images[0]), caption="Resultados del Entrenamiento", use_column_width=True)
                        except Exception as e:
                            st.warning(f"Error cargando results.png: {e}")

                    # Mostrar batches de entrenamiento
                    if train_batches:
                        st.markdown("**🎓 Batches de Entrenamiento:**")
                        cols = st.columns(min(len(train_batches), 3))
                        for idx, img_file in enumerate(train_batches[:3]):
                            try:
                                with cols[idx]:
                                    batch_num = img_file.name.replace('train_batch', '').replace('.jpg', '')
                                    st.image(str(img_file), caption=f"Batch de Entrenamiento {batch_num}", use_column_width=True)
                            except Exception as e:
                                st.warning(f"Error cargando {img_file.name}: {e}")

                    # Mostrar batches de validación
                    if val_batches:
                        st.markdown("**✅ Batches de Validación:**")
                        # Separar labels y predicciones
                        labels_images = [img for img in val_batches if 'labels' in img.name]
                        pred_images = [img for img in val_batches if 'pred' in img.name]

                        # Mostrar máximo 2 pares de validación
                        for i in range(min(2, len(labels_images))):
                            cols = st.columns(2)
                            batch_num = str(i)

                            # Labels
                            if i < len(labels_images):
                                with cols[0]:
                                    try:
                                        st.image(str(labels_images[i]), caption=f"Validación {batch_num} - Labels Verdaderas", use_column_width=True)
                                    except Exception as e:
                                        st.warning(f"Error cargando labels {i}: {e}")

                            # Predicciones
                            if i < len(pred_images):
                                with cols[1]:
                                    try:
                                        st.image(str(pred_images[i]), caption=f"Validación {batch_num} - Predicciones", use_column_width=True)
                                    except Exception as e:
                                        st.warning(f"Error cargando pred {i}: {e}")

                    # Información sobre todas las imágenes
                    with st.expander("📋 Ver todas las imágenes disponibles"):
                        image_list = []
                        for img in sorted(all_images, key=lambda x: x.name):
                            size = img.stat().st_size
                            if size < 1024:
                                size_str = f"{size} B"
                            elif size < 1024*1024:
                                size_str = f"{size/1024:.1f} KB"
                            else:
                                size_str = f"{size/(1024*1024):.1f} MB"
                            image_list.append(f"🖼️ {img.name} ({size_str})")

                        st.code("\n".join(image_list))

                else:
                    st.info("ℹ️ No se encontraron imágenes generadas por YOLO en la carpeta del modelo.")
            else:
                st.info("ℹ️ No existe la carpeta del modelo entrenado.")

            # Opción para ver otros experimentos
            st.markdown("---")
            if len(experiments) > 1:
                with st.expander("🔄 Ver otros experimentos"):
                    selected_experiment = st.selectbox(
                        "Seleccionar experimento:",
                        [exp.name for exp in sorted(experiments, key=lambda x: x.stat().st_mtime, reverse=True)],
                        index=0
                    )

                    if selected_experiment != experiment_name:
                        st.info(f"Para ver los resultados de '{selected_experiment}', refresca la página y selecciona ese experimento.")

        else:
            st.info("ℹ️ No hay experimentos de entrenamiento completados aún. Entrena un modelo primero en la pestaña '🚀 Entrenar Modelo'.")

def show_classification_page():
    """Mostrar página de clasificación"""
    st.markdown('<h1 class="main-header">🔍 Clasificación de Residuos</h1>', unsafe_allow_html=True)
    
    from src.model_predictor import ModelPredictor
    
    # Inicializar predictor
    predictor = ModelPredictor(config)
    
    # Verificar si hay modelo entrenado
    if not predictor.model_exists():
        st.warning("⚠️ No hay modelo entrenado. Primero entrena un modelo.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Ir a Entrenar Modelo"):
                st.switch_page("pages/03_🚀_Entrenar_Modelo.py")
        
        with col2:
            st.info("Después de entrenar, vuelve aquí para clasificar")
        
        return
    
    # Obtener modelos disponibles
    available_models = predictor.get_available_models()
    
    if not available_models:
        st.error("❌ No se encontraron modelos entrenados")
        return
    
    # Selector de modelo en la parte superior
    st.markdown("### 🎯 Selección de Modelo")
    
    col_model, col_info, col_stats = st.columns([2, 1, 1])
    
    with col_model:
        # Crear opciones para el selectbox
        model_options = {model['name']: model for model in available_models}
        
        selected_model_name = st.selectbox(
            "Selecciona el modelo a usar:",
            options=list(model_options.keys()),
            help="Selecciona uno de los modelos entrenados disponibles"
        )
        
        selected_model = model_options[selected_model_name]
    
    with col_info:
        # Mostrar información del modelo seleccionado
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"**📁 Ubicación:** {selected_model['location']}")
        st.markdown(f"**🔬 Experimento:** {selected_model['experiment']}")
        st.markdown(f"**⚖️ Tipo:** {selected_model['weight_type']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_stats:
        # Mostrar estadísticas del archivo
        model_path = Path(selected_model['path'])
        if model_path.exists():
            file_size = model_path.stat().st_size / (1024 * 1024)
            st.metric("📦 Tamaño", f"{file_size:.2f} MB")
            st.metric("🗂️ Modelos", len(available_models))
    
    # Cargar el modelo seleccionado
    if predictor.current_model_path != selected_model['path']:
        with st.spinner("Cargando modelo seleccionado..."):
            try:
                predictor.load_model(selected_model['path'])
            except Exception as e:
                st.error(f"Error al cargar modelo: {e}")
                return
    
    st.markdown("---")
    
    # Seleccionar método de entrada
    st.markdown("### 📥 Seleccionar Método de Entrada")
    
    input_method = st.radio(
        "¿Cómo quieres proporcionar la imagen?",
        ["📤 Subir Imagen", "📷 Usar Cámara Web", "📂 Carpeta de Imágenes", "📁 Seleccionar del Dataset"],
        horizontal=True
    )
    
    if input_method == "📤 Subir Imagen":
        uploaded_file = st.file_uploader(
            "Sube una imagen de residuo",
            type=config['dashboard']['supported_formats'],
            help="Sube una imagen para clasificar"
        )
        
        if uploaded_file is not None:
            process_single_image(uploaded_file, predictor)
    
    elif input_method == "📷 Usar Cámara Web":
        if config['dashboard']['enable_camera']:
            camera_image = st.camera_input("Toma una foto del residuo")
            
            if camera_image is not None:
                process_single_image(camera_image, predictor)
        else:
            st.warning("La cámara web no está habilitada en la configuración.")
    
    elif input_method == "📂 Carpeta de Imágenes":
        uploaded_files = st.file_uploader(
            "Sube múltiples imágenes",
            type=config['dashboard']['supported_formats'],
            accept_multiple_files=True,
            help="Selecciona múltiples imágenes para procesar en batch"
        )
        
        if uploaded_files and len(uploaded_files) > 0:
            process_batch_images(uploaded_files, predictor)
    
    else:  # 📁 Seleccionar del Dataset
        # Seleccionar clase y luego imagen
        from src.data_preparation import DataPreparer
        preparer = DataPreparer(config)
        
        classes = config['classes']
        selected_class = st.selectbox("Selecciona una clase:", classes)
        
        if selected_class:
            # Obtener imágenes de ejemplo
            sample_images = preparer.get_sample_images(selected_class, num_samples=10)
            
            if sample_images:
                # Mostrar imágenes en un selector
                selected_image = st.selectbox(
                    "Selecciona una imagen:",
                    [img.name for img in sample_images]
                )
                
                if selected_image:
                    img_path = next(img for img in sample_images if img.name == selected_image)
                    
                    # Mostrar imagen seleccionada
                    st.image(str(img_path), caption=f"Imagen seleccionada: {selected_image}")
                    
                    if st.button("🎯 Clasificar Esta Imagen", width='stretch'):
                        process_single_image(str(img_path), predictor)
            else:
                st.info(f"No hay imágenes para la clase {selected_class}")

def process_single_image(image_source, predictor):
    """Procesar una sola imagen"""
    with st.spinner("🔍 Analizando imagen..."):
        try:
            # Realizar predicción
            predictions, processing_time, original_image = predictor.predict(image_source)
            
            if not predictions:
                st.error("No se pudo realizar la predicción")
                return
            
            # Mostrar resultados en dos columnas
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Mostrar imagen con predicción
                from src.visualizations import VisualizationManager
                viz = VisualizationManager()
                
                fig = viz.plot_prediction_result(original_image, predictions)
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                # Mostrar métricas y resultados
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                
                # Predicción principal
                top_pred = predictions[0]
                st.metric(
                    label="🏆 Predicción Principal",
                    value=top_pred['class'].replace('-', ' ').title(),
                    delta=f"{top_pred['confidence']:.1%} confianza"
                )
                
                # Tiempo de procesamiento
                st.metric("⏱️ Tiempo", f"{processing_time:.0f} ms")
                
                # Top 3 predicciones
                st.markdown("#### 🥇 Top 3 Predicciones")
                
                for i, pred in enumerate(predictions[:3], 1):
                    progress = pred['confidence']
                    st.progress(
                        progress,
                        text=f"{i}. {pred['class'].replace('-', ' ').title()}: {pred['confidence']:.1%}"
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Acciones
                st.markdown("---")
                
                col_act1, col_act2 = st.columns(2)
                
                with col_act1:
                    if st.button("📥 Guardar Resultado", width='stretch'):
                        predictor.save_prediction_result(
                            image_source, 
                            predictions, 
                            processing_time
                        )
                        st.success("✅ Resultado guardado!")
                
                with col_act2:
                    # Exportar resultados
                    import pandas as pd
                    df = pd.DataFrame(predictions)
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label="📋 Exportar CSV",
                        data=csv,
                        file_name="prediccion.csv",
                        mime="text/csv",
                        width='stretch'
                    )
            
            # Análisis detallado
            with st.expander("📈 Análisis Detallado", expanded=False):
                tab1, tab2, tab3 = st.tabs(["📊 Distribución", "🎯 Probabilidades", "📋 Metadata"])
                
                with tab1:
                    from src.visualizations import VisualizationManager
                    viz = VisualizationManager()
                    fig = viz.plot_probability_distribution(predictions)
                    st.plotly_chart(fig, width='stretch')
                
                with tab2:
                    import pandas as pd
                    df = pd.DataFrame(predictions)
                    st.dataframe(
                        df[['class', 'confidence', 'percentage']]
                        .style.background_gradient(subset=['confidence'], cmap='Greens')
                        .format({'confidence': '{:.2%}', 'percentage': '{:.1f}%'})
                    )
                
                with tab3:
                    # Información de la imagen
                    img_info = predictor.get_image_info(image_source)
                    st.json(img_info)
        
        except Exception as e:
            st.error(f"❌ Error procesando imagen: {str(e)}")

def process_batch_images(image_files, predictor):
    """Procesar múltiples imágenes"""
    st.info(f"📦 Procesando {len(image_files)} imágenes...")
    
    # Barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    for idx, img_file in enumerate(image_files):
        # Actualizar progreso
        progress = (idx + 1) / len(image_files)
        progress_bar.progress(progress)
        status_text.text(f"Procesando imagen {idx + 1} de {len(image_files)}")
        
        try:
            # Predecir
            predictions, processing_time, _ = predictor.predict(img_file)
            
            if predictions:
                results.append({
                    'filename': img_file.name,
                    'predictions': predictions,
                    'top_prediction': predictions[0]['class'],
                    'confidence': predictions[0]['confidence'],
                    'processing_time': processing_time
                })
        
        except Exception as e:
            st.warning(f"Error procesando {img_file.name}: {str(e)}")
    
    progress_bar.empty()
    status_text.empty()
    
    # Mostrar resumen
    st.success(f"✅ Procesadas {len(results)} imágenes")
    
    if results:
        # Convertir a DataFrame
        import pandas as pd
        df = pd.DataFrame(results)
        
        # Métricas principales
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Imágenes Procesadas", len(df))
        
        with col2:
            avg_conf = df['confidence'].mean()
            st.metric("Confianza Promedio", f"{avg_conf:.1%}")
        
        with col3:
            most_common = df['top_prediction'].mode()[0] if not df.empty else "N/A"
            st.metric("Clase Más Común", most_common.replace('-', ' ').title())
        
        # Tabla de resultados
        st.markdown("#### 📋 Resultados Detallados")
        st.dataframe(
            df[['filename', 'top_prediction', 'confidence', 'processing_time']]
            .rename(columns={
                'filename': 'Archivo',
                'top_prediction': 'Predicción',
                'confidence': 'Confianza',
                'processing_time': 'Tiempo (ms)'
            })
            .style.background_gradient(subset=['Confianza'], cmap='Greens')
            .format({'Confianza': '{:.1%}', 'Tiempo (ms)': '{:.0f}'})
        )
        
        # Gráfico de distribución
        st.markdown("#### 📊 Distribución de Clases")
        from src.visualizations import VisualizationManager
        viz = VisualizationManager()
        fig = viz.plot_class_distribution_batch(df)
        st.plotly_chart(fig, width='stretch')
        
        # Exportar resultados
        st.markdown("---")
        st.markdown("### 📤 Exportar Resultados")
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            # Exportar a CSV
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Descargar CSV",
                data=csv_data,
                file_name="batch_results.csv",
                mime="text/csv",
                width='stretch'
            )
        
        with col_exp2:
            # Generar reporte
            if st.button("📊 Generar Reporte Completo", width='stretch'):
                report_path = predictor.generate_batch_report(results)
                with open(report_path, 'rb') as f:
                    st.download_button(
                        label="📄 Descargar Reporte PDF",
                        data=f,
                        file_name="reporte_clasificacion.pdf",
                        mime="application/pdf",
                        width='stretch'
                    )

def show_analysis_page():
    """Mostrar página de análisis y métricas"""
    st.markdown('<h1 class="main-header">📈 Análisis y Métricas</h1>', unsafe_allow_html=True)
    
    from src.metrics_analyzer import MetricsAnalyzer
    
    analyzer = MetricsAnalyzer(config)
    
    # Tabs para diferentes tipos de análisis
    tab1, tab2, tab3, tab4, tab5= st.tabs([
        "📊 Métricas del Modelo", 
        "📈 Historial de Entrenamiento",
        "🔄 Análisis Comparativo", 
        "📊 Resultados", 
        "🔍 Diagnóstico de Errores"
    ])
    
    with tab1:
        st.markdown("### 📊 Métricas del Modelo Actual")
        
        # Cargar métricas del modelo actual
        metrics = analyzer.load_current_model_metrics()
        
        if metrics:
            # Métricas principales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
            
            with col2:
                st.metric("Precision", f"{metrics.get('precision', 0):.2%}")
            
            with col3:
                st.metric("Recall", f"{metrics.get('recall', 0):.2%}")
            
            with col4:
                st.metric("F1-Score", f"{metrics.get('f1_score', 0):.2%}")
            
            # Métricas adicionales
            col5, col6 = st.columns(2)
            with col5:
                st.metric("Top-1 Accuracy", f"{metrics.get('top1_accuracy', 0):.2%}")
            with col6:
                st.metric("Top-5 Accuracy", f"{metrics.get('top5_accuracy', 0):.2%}")
            
            st.markdown("---")
            
            # Gráficos
            st.markdown("#### 📈 Curvas de Rendimiento")
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Curva ROC
                fig = analyzer.plot_roc_curve(metrics)
                st.plotly_chart(fig, width='stretch')
            
            with col_chart2:
                # Curva Precision-Recall
                fig = analyzer.plot_precision_recall_curve(metrics)
                st.plotly_chart(fig, width='stretch')
            
            # Matriz de confusión
            st.markdown("#### 🎯 Matriz de Confusión")
            fig = analyzer.plot_confusion_matrix(metrics)
            st.plotly_chart(fig, width='stretch')
            
            # Métricas por clase
            st.markdown("#### 📋 Métricas por Clase")
            class_metrics_df = analyzer.get_class_metrics_dataframe(metrics)
            st.dataframe(
                class_metrics_df.style.highlight_max(
                    subset=['Precisión', 'Recall', 'F1-Score'], 
                    color='lightgreen'
                ),
                use_container_width=True
            )
            
        else:
            st.info("ℹ️ No hay métricas disponibles. Primero entrena un modelo.")
            st.markdown("""
            ### 🚀 Para comenzar:
            1. Ve a la página **Entrenar Modelo**
            2. Configura los parámetros de entrenamiento
            3. Inicia el entrenamiento
            4. Regresa aquí para ver las métricas
            """)
    
    with tab2:
        st.markdown("### 📈 Historial de Entrenamiento")
        
        # Selector de experimento
        experiment_dirs = []
        base_dirs = [
            Path('runs/classify/models/trained'),
            Path('models/trained')
        ]
        
        for base_dir in base_dirs:
            if base_dir.exists():
                experiment_dirs.extend([d.name for d in base_dir.iterdir() if d.is_dir()])
        
        if experiment_dirs:
            selected_experiment = st.selectbox(
                "Seleccionar Experimento:",
                ['Más reciente'] + list(set(experiment_dirs))
            )
            
            exp_name = None if selected_experiment == 'Más reciente' else selected_experiment
            
            # Cargar y mostrar historial
            history_df = analyzer.load_training_history(exp_name)
            
            if history_df is not None:
                # Resumen del entrenamiento
                summary = analyzer.get_training_summary(exp_name)
                
                if summary:
                    st.markdown("#### 📊 Resumen del Entrenamiento")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total de Épocas", summary.get('total_epochs', 0))
                    with col2:
                        st.metric("Mejor Época", summary.get('best_epoch', 0))
                    with col3:
                        st.metric("Mejor Val Acc", f"{summary.get('best_val_acc', 0):.2%}")
                    with col4:
                        hours = summary.get('training_time', 0) / 3600
                        st.metric("Tiempo Total", f"{hours:.1f}h")
                
                st.markdown("---")
                
                # Gráfico de historial
                st.markdown("#### 📉 Curvas de Entrenamiento")
                fig = analyzer.plot_training_history(exp_name)
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabla de datos
                with st.expander("📋 Ver Datos Completos"):
                    st.dataframe(history_df, use_container_width=True)
            else:
                st.info("No se encontró historial de entrenamiento para este experimento.")
        else:
            st.info("No hay experimentos de entrenamiento disponibles.")
    
    with tab3:
        st.markdown("### 📈 Análisis Comparativo de Modelos")
        
        # Comparar diferentes modelos
        available_models = analyzer.get_available_models()
        
        if len(available_models) >= 2:
            # Crear lista de nombres para mostrar
            model_names = [model['name'] for model in available_models]
            
            # Seleccionar modelos para comparar
            col1, col2 = st.columns(2)
            
            with col1:
                model_a_name = st.selectbox(
                    "Modelo A:",
                    model_names,
                    index=0
                )
            
            with col2:
                model_b_name = st.selectbox(
                    "Modelo B:",
                    model_names,
                    index=min(1, len(model_names)-1)
                )
            
            if st.button("🔄 Comparar Modelos", width='stretch'):
                # Encontrar los diccionarios completos de los modelos seleccionados
                model_a = next(model for model in available_models if model['name'] == model_a_name)
                model_b = next(model for model in available_models if model['name'] == model_b_name)
                
                comparison = analyzer.compare_models(model_a, model_b)
                
                if comparison:
                    # Mostrar comparación
                    st.markdown("#### 📊 Resultados de la Comparación")
                    st.dataframe(comparison.style.highlight_max(axis=0, color='lightgreen'))
                    
                    # Gráfico de comparación
                    fig = analyzer.plot_model_comparison(comparison)
                    st.plotly_chart(fig, width='stretch')
        else:
            st.info("Necesitas al menos 2 modelos entrenados para comparar.")
    
    with tab4:
        st.markdown("### 📊 Resultados del Entrenamiento")

        # Verificar si hay experimentos entrenados
        results_dir = Path(config['paths']['results_dir']) / 'training_logs'
        experiments = []

        if results_dir.exists():
            experiments = [d for d in results_dir.iterdir() if d.is_dir()]

        if experiments:
            # Encontrar el experimento más reciente
            latest_experiment = max(experiments, key=lambda x: x.stat().st_mtime)
            experiment_name = latest_experiment.name

            st.success(f"✅ Mostrando resultados del último entrenamiento: **{experiment_name}**")

            # Mostrar información básica del experimento
            results_file = latest_experiment / f"results_{latest_experiment.stat().st_mtime:.0f}.json"

            # Buscar archivo de resultados más reciente
            results_files = list(latest_experiment.glob("results_*.json"))
            if results_files:
                results_file = max(results_files, key=lambda x: x.stat().st_mtime)

                try:
                    with open(results_file, 'r', encoding='utf-8') as f:
                        experiment_data = json.load(f)

                    # Mostrar métricas principales
                    metrics = experiment_data.get('metrics', {})

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("🎯 Exactitud", f"{metrics.get('accuracy', 0):.2%}")

                    with col2:
                        st.metric("📊 Precisión", f"{metrics.get('precision', 0):.2%}")

                    with col3:
                        st.metric("🔍 Recall", f"{metrics.get('recall', 0):.2%}")

                    with col4:
                        st.metric("📈 F1-Score", f"{metrics.get('f1_score', 0):.2%}")

                    # Información adicional del entrenamiento
                    st.markdown("#### 📋 Información del Entrenamiento")

                    info_cols = st.columns(4)
                    with info_cols[0]:
                        st.info(f"⏱️ Tiempo: {experiment_data.get('training_time', 0):.1f} min")
                    with info_cols[1]:
                        st.info(f"🎯 Épocas: {experiment_data.get('epochs', 0)}")
                    with info_cols[2]:
                        st.info(f"📦 Batch: {experiment_data.get('batch_size', 0)}")
                    with info_cols[3]:
                        device_used = experiment_data.get('device', 'cpu')
                        st.info(f"💻 Dispositivo: {'GPU' if 'cuda' in str(device_used) else 'CPU'}")

                except Exception as e:
                    st.warning(f"No se pudo cargar la información detallada: {e}")

            # Mostrar gráficas generadas
            st.markdown("---")
            st.markdown("#### 📈 Gráficas Generadas")

            plots_dir = latest_experiment / 'plots'
            if plots_dir.exists():
                # Buscar archivos de gráficas
                training_history_files = list(plots_dir.glob("training_history_*.png"))
                confusion_matrix_files = list(plots_dir.glob("confusion_matrix_*.png"))

                # Mostrar training history
                if training_history_files:
                    training_history_file = max(training_history_files, key=lambda x: x.stat().st_mtime)
                    try:
                        st.markdown("**📊 Historial de Entrenamiento:**")
                        st.image(str(training_history_file), caption="Evolución del entrenamiento", use_column_width=True)
                    except Exception as e:
                        st.warning(f"No se pudo cargar la gráfica de historial: {e}")

                # Mostrar confusion matrix
                if confusion_matrix_files:
                    confusion_matrix_file = max(confusion_matrix_files, key=lambda x: x.stat().st_mtime)
                    try:
                        st.markdown("**🎯 Matriz de Confusión:**")
                        st.image(str(confusion_matrix_file), caption="Matriz de Confusión", use_column_width=True)
                    except Exception as e:
                        st.warning(f"No se pudo cargar la matriz de confusión: {e}")

                if not training_history_files and not confusion_matrix_files:
                    st.info("ℹ️ No se encontraron gráficas generadas durante el entrenamiento.")
            else:
                st.info("ℹ️ No se encontraron gráficas para este experimento.")

            # Mostrar archivos generados por YOLO
            st.markdown("---")
            st.markdown("#### 📁 Archivos Generados por YOLO")

            model_yolo_dir = Path(config['paths']['trained_models']) / experiment_name
            if model_yolo_dir.exists():
                # Mostrar estructura de archivos
                with st.expander("📂 Ver estructura de archivos generados"):
                    file_structure = []

                    def get_file_structure(path, prefix=""):
                        if path.is_dir():
                            file_structure.append(f"{prefix}📁 {path.name}/")
                            for item in sorted(path.iterdir()):
                                get_file_structure(item, prefix + "  ")
                        else:
                            size = path.stat().st_size
                            if size < 1024:
                                size_str = f"{size} B"
                            elif size < 1024*1024:
                                size_str = f"{size/1024:.1f} KB"
                            else:
                                size_str = f"{size/(1024*1024):.1f} MB"
                            file_structure.append(f"{prefix}📄 {path.name} ({size_str})")

                    get_file_structure(model_yolo_dir)
                    st.code("\n".join(file_structure))

                # Mostrar contenido del results.csv si existe
                results_csv = model_yolo_dir / 'results.csv'
                if results_csv.exists():
                    try:
                        results_df = pd.read_csv(results_csv)
                        st.markdown("**📊 Resultados Detallados del Entrenamiento:**")
                        st.dataframe(results_df, use_container_width=True)
                    except Exception as e:
                        st.warning(f"No se pudo cargar results.csv: {e}")

                # Mostrar args.yaml si existe
                args_yaml = model_yolo_dir / 'args.yaml'
                if args_yaml.exists():
                    try:
                        with open(args_yaml, 'r') as f:
                            args_content = f.read()
                        with st.expander("⚙️ Ver configuración del entrenamiento (args.yaml)"):
                            st.code(args_content, language='yaml')
                    except Exception as e:
                        st.warning(f"No se pudo cargar args.yaml: {e}")

                # Botón para descargar el modelo
                model_file = model_yolo_dir / 'weights' / 'best.pt'
                if model_file.exists():
                    with open(model_file, 'rb') as f:
                        model_bytes = f.read()
                    st.download_button(
                        label="📥 Descargar Modelo Entrenado (best.pt)",
                        data=model_bytes,
                        file_name=f"{experiment_name}_best.pt",
                        mime="application/octet-stream"
                    )
            else:
                st.info("ℹ️ No se encontraron archivos del modelo YOLO para este experimento.")

            # Mostrar imágenes generadas por YOLO en la carpeta del modelo
            st.markdown("---")
            st.markdown("#### 🖼️ Imágenes Generadas por YOLO")

            # Buscar imágenes en la carpeta del modelo entrenado
            model_images_dir = Path('runs/classify/models/trained') / experiment_name
            if model_images_dir.exists():
                # Buscar todas las imágenes
                all_images = []
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    all_images.extend(list(model_images_dir.glob(ext)))

                if all_images:
                    st.success(f"✅ Se encontraron {len(all_images)} imágenes generadas por YOLO")

                    # Categorizar imágenes
                    confusion_matrices = [img for img in all_images if 'confusion_matrix' in img.name]
                    results_images = [img for img in all_images if img.name == 'results.png']
                    train_batches = [img for img in all_images if 'train_batch' in img.name]
                    val_batches = [img for img in all_images if 'val_batch' in img.name]

                    # Mostrar matrices de confusión
                    if confusion_matrices:
                        st.markdown("**🎯 Matrices de Confusión:**")
                        cols = st.columns(min(len(confusion_matrices), 2))
                        for idx, img_file in enumerate(confusion_matrices[:2]):
                            try:
                                with cols[idx]:
                                    caption = "Normalizada" if "normalized" in img_file.name else "Estándar"
                                    st.image(str(img_file), caption=f"Matriz de Confusión - {caption}", use_column_width=True)
                            except Exception as e:
                                st.warning(f"Error cargando {img_file.name}: {e}")

                    # Mostrar gráfica de resultados
                    if results_images:
                        st.markdown("**📊 Gráfica de Resultados:**")
                        try:
                            st.image(str(results_images[0]), caption="Resultados del Entrenamiento", use_column_width=True)
                        except Exception as e:
                            st.warning(f"Error cargando results.png: {e}")

                    # Mostrar batches de entrenamiento
                    if train_batches:
                        st.markdown("**🎓 Batches de Entrenamiento:**")
                        cols = st.columns(min(len(train_batches), 3))
                        for idx, img_file in enumerate(train_batches[:3]):
                            try:
                                with cols[idx]:
                                    batch_num = img_file.name.replace('train_batch', '').replace('.jpg', '')
                                    st.image(str(img_file), caption=f"Batch de Entrenamiento {batch_num}", use_column_width=True)
                            except Exception as e:
                                st.warning(f"Error cargando {img_file.name}: {e}")

                    # Mostrar batches de validación
                    if val_batches:
                        st.markdown("**✅ Batches de Validación:**")
                        # Separar labels y predicciones
                        labels_images = [img for img in val_batches if 'labels' in img.name]
                        pred_images = [img for img in val_batches if 'pred' in img.name]

                        # Mostrar máximo 2 pares de validación
                        for i in range(min(2, len(labels_images))):
                            cols = st.columns(2)
                            batch_num = str(i)

                            # Labels
                            if i < len(labels_images):
                                with cols[0]:
                                    try:
                                        st.image(str(labels_images[i]), caption=f"Validación {batch_num} - Labels Verdaderas", use_column_width=True)
                                    except Exception as e:
                                        st.warning(f"Error cargando labels {i}: {e}")

                            # Predicciones
                            if i < len(pred_images):
                                with cols[1]:
                                    try:
                                        st.image(str(pred_images[i]), caption=f"Validación {batch_num} - Predicciones", use_column_width=True)
                                    except Exception as e:
                                        st.warning(f"Error cargando pred {i}: {e}")

                    # Información sobre todas las imágenes
                    with st.expander("📋 Ver todas las imágenes disponibles"):
                        image_list = []
                        for img in sorted(all_images, key=lambda x: x.name):
                            size = img.stat().st_size
                            if size < 1024:
                                size_str = f"{size} B"
                            elif size < 1024*1024:
                                size_str = f"{size/1024:.1f} KB"
                            else:
                                size_str = f"{size/(1024*1024):.1f} MB"
                            image_list.append(f"🖼️ {img.name} ({size_str})")

                        st.code("\n".join(image_list))

                else:
                    st.info("ℹ️ No se encontraron imágenes generadas por YOLO en la carpeta del modelo.")
            else:
                st.info("ℹ️ No existe la carpeta del modelo entrenado.")

            # Opción para ver otros experimentos
            st.markdown("---")
            if len(experiments) > 1:
                with st.expander("🔄 Ver otros experimentos"):
                    selected_experiment = st.selectbox(
                        "Seleccionar experimento:",
                        [exp.name for exp in sorted(experiments, key=lambda x: x.stat().st_mtime, reverse=True)],
                        index=0
                    )

                    if selected_experiment != experiment_name:
                        st.info(f"Para ver los resultados de '{selected_experiment}', refresca la página y selecciona ese experimento.")

        else:
            st.info("ℹ️ No hay experimentos de entrenamiento completados aún. Entrena un modelo primero en la pestaña '🚀 Entrenar Modelo'.")
    
    with tab5:
        
        # Cargar métricas actuales
        metrics = analyzer.load_current_model_metrics()
        
        if metrics:
            # Cargar errores comunes
            common_errors = analyzer.get_common_errors(metrics)
            
            if common_errors:
                st.markdown("#### ⚠️ Errores Más Comunes")
                
                # Mostrar top 5 errores
                for idx, error in enumerate(common_errors[:5], 1):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{idx}. {error['actual']} → {error['predicted']}**")
                            st.progress(min(error['percentage'] / 20, 1.0))  # Normalizar a escala 0-1
                        with col2:
                            st.metric("Errores", error['count'])
                            st.caption(f"{error['percentage']:.1f}%")
                        st.markdown("---")
                
                # Tabla completa
                with st.expander("📋 Ver Todos los Errores"):
                    errors_df = pd.DataFrame(common_errors)
                    errors_df.columns = ['Clase Real', 'Clase Predicha', 'Cantidad', 'Porcentaje (%)']
                    st.dataframe(errors_df, use_container_width=True)
                
                # Análisis de confianza
                st.markdown("#### 📊 Distribución de Confianza en Errores")
                st.info("Esta métrica muestra cómo de confiado está el modelo en sus predicciones incorrectas. Idealmente, los errores deberían tener baja confianza.")
                
                confidence_data = analyzer.get_error_confidence_distribution()
                
                if not confidence_data.empty:
                    from src.visualizations import VisualizationManager
                    viz = VisualizationManager()
                    fig = viz.plot_confidence_histogram(confidence_data)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No se detectaron errores significativos en la clasificación.")
        else:
            st.info("ℹ️ No hay datos de errores disponibles. Entrena un modelo primero.")
    


def show_configuration_page():
    """Mostrar página de configuración"""
    st.markdown('<h1 class="main-header">⚙️ Configuración del Sistema</h1>', unsafe_allow_html=True)
    
    # Tabs de configuración
    tab1, tab2, tab3 = st.tabs(["🔧 Sistema", "🧠 Modelo", "📊 Dashboard"])
    
    with tab1:
        st.markdown("### 🔧 Configuración del Sistema")
        
        with st.form("system_config_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Rutas
                st.subheader("📁 Rutas del Sistema")
                
                data_raw = st.text_input(
                    "Ruta datos originales",
                    value=config['paths']['data_raw']
                )
                
                data_processed = st.text_input(
                    "Ruta datos procesados",
                    value=config['paths']['data_processed']
                )
            
            with col2:
                # Rendimiento
                st.subheader("⚡ Rendimiento")
                
                use_gpu = st.checkbox(
                    "Usar GPU si está disponible",
                    value=config['performance']['use_gpu']
                )
                
                max_workers = st.slider(
                    "Máximo de workers",
                    min_value=1,
                    max_value=8,
                    value=config['performance']['max_workers']
                )
            
            # Guardar configuración
            submitted = st.form_submit_button("💾 Guardar Configuración del Sistema", type="primary")
            
            if submitted:
                # Actualizar configuración
                config['paths']['data_raw'] = data_raw
                config['paths']['data_processed'] = data_processed
                config['performance']['use_gpu'] = use_gpu
                config['performance']['max_workers'] = max_workers
                
                save_configuration()
    
    with tab2:
        st.markdown("### 🧠 Configuración del Modelo")
        
        with st.form("model_config_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Modelo base
                st.subheader("Modelo Base")
                
                model_name = st.selectbox(
                    "Nombre del modelo",
                    ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                    index=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"].index(config['model']['name'])
                )
                
                input_size = st.selectbox(
                    "Tamaño de entrada",
                    [224, 256, 320, 416, 512],
                    index=[224, 256, 320, 416, 512].index(config['model']['input_size'])
                )
            
            with col2:
                # Predicción
                st.subheader("Predicción")
                
                confidence_threshold = st.slider(
                    "Umbral de confianza",
                    min_value=0.1,
                    max_value=1.0,
                    value=config['prediction']['confidence_threshold'],
                    step=0.05
                )
                
                top_k_predictions = st.slider(
                    "Top-K predicciones",
                    min_value=1,
                    max_value=10,
                    value=config['prediction']['top_k_predictions']
                )
            
            # Guardar configuración
            submitted = st.form_submit_button("💾 Guardar Configuración del Modelo", type="primary")
            
            if submitted:
                config['model']['name'] = model_name
                config['model']['input_size'] = input_size
                config['prediction']['confidence_threshold'] = confidence_threshold
                config['prediction']['top_k_predictions'] = top_k_predictions
                
                save_configuration()
    
    with tab3:
        st.markdown("### 📊 Configuración del Dashboard")
        
        with st.form("dashboard_config_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Apariencia
                st.subheader("🎨 Apariencia")
                
                theme = st.selectbox(
                    "Tema",
                    ["light", "dark"],
                    index=0 if config['dashboard']['theme'] == "light" else 1
                )
                
                max_file_size = st.number_input(
                    "Tamaño máximo de archivo (MB)",
                    min_value=1,
                    max_value=100,
                    value=config['dashboard']['max_file_size_mb']
                )
            
            with col2:
                # Características
                st.subheader("🚀 Características")
                
                enable_camera = st.checkbox(
                    "Habilitar cámara web",
                    value=config['dashboard']['enable_camera']
                )
                
                enable_batch = st.checkbox(
                    "Habilitar procesamiento por lotes",
                    value=config['dashboard']['enable_batch_processing']
                )
            
            # Guardar configuración
            submitted = st.form_submit_button("💾 Guardar Configuración del Dashboard", type="primary")
            
            if submitted:
                config['dashboard']['theme'] = theme
                config['dashboard']['max_file_size_mb'] = max_file_size
                config['dashboard']['enable_camera'] = enable_camera
                config['dashboard']['enable_batch_processing'] = enable_batch
                
                save_configuration()
    
    # Acciones de sistema
    st.markdown("---")
    st.markdown("### ⚡ Acciones del Sistema")
    
    col_act1, col_act2, col_act3 = st.columns(3)
    
    with col_act1:
        if st.button("🔄 Reiniciar Sistema", width='stretch'):
            st.cache_resource.clear()
            st.success("✅ Sistema reiniciado")
            st.rerun()
    
    with col_act2:
        if st.button("🧹 Limpiar Caché", width='stretch'):
            import shutil
            cache_dirs = ["./__pycache__", "./streamlit_cache"]
            for cache_dir in cache_dirs:
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
            st.success("✅ Caché limpiado")
    
    with col_act3:
        if st.button("📤 Exportar Configuración", width='stretch'):
            export_configuration()

def show_configuration_page():
    """Mostrar página de configuración del sistema"""
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
                
               
            
            # Guardar configuración
            submitted = st.form_submit_button("💾 Guardar Configuración del Dashboard", type="primary")
            
            if submitted:
                config['dashboard']['theme'] = theme
                config['dashboard']['max_file_size_mb'] = max_file_size
                config['dashboard']['title'] = title
                config['dashboard']['enable_camera'] = enable_camera
                config['dashboard']['enable_batch_processing'] = enable_batch
                config['dashboard']['enable_model_comparison'] = enable_comparison
                
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
            if st.button("⚡ Rápido", width='stretch', help="Configuración rápida: 10 épocas"):
                config['training']['epochs'] = 10
                config['training']['batch_size'] = 32
                config['training']['learning_rate'] = 0.001
                if save_configuration():
                    st.success("✅ Preset aplicado")
                    st.rerun()
            
            if st.button("⚖️ Balanceado", width='stretch', help="Configuración balanceada: 50 épocas"):
                config['training']['epochs'] = 50
                config['training']['batch_size'] = 32
                config['training']['learning_rate'] = 0.001
                if save_configuration():
                    st.success("✅ Preset aplicado")
                    st.rerun()
            
            if st.button("🔬 Profundo", width='stretch', help="Configuración profunda: 100 épocas"):
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
        if st.button("🔄 Reiniciar Sistema", width='stretch', help="Limpiar caché y recargar"):
            st.cache_resource.clear()
            st.success("✅ Sistema reiniciado")
            st.rerun()
    
    with col_act2:
        if st.button("🧹 Limpiar Caché", width='stretch', help="Eliminar archivos de caché"):
            try:
                import shutil
                cache_dirs = ["./__pycache__", "./.streamlit"]
                for cache_dir in cache_dirs:
                    if os.path.exists(cache_dir):
                        shutil.rmtree(cache_dir)
                st.success("✅ Caché limpiado")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    with col_act3:
        if st.button("📤 Exportar Configuración", width='stretch', help="Descargar config como JSON"):
            export_configuration()
    
    with col_act4:
        if st.button("📋 Ver Configuración Actual", width='stretch', help="Mostrar todas las configuraciones"):
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

def save_configuration():
    """Guardar configuración en archivo"""
    try:
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        st.success("✅ Configuración guardada exitosamente!")
        st.rerun()
    
    except Exception as e:
        st.error(f"❌ Error guardando configuración: {str(e)}")

def export_configuration():
    """Exportar configuración como archivo"""
    import json
    config_json = json.dumps(config, indent=2, default=str)
    
    st.download_button(
        label="📥 Descargar Configuración",
        data=config_json,
        file_name="configuracion_sistema.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()
