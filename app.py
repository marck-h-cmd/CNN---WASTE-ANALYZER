
import streamlit as st
import os
import sys
from pathlib import Path
import yaml
import pandas as pd
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
        
        selected_page = st.radio(
            "Selecciona una página:",
            menu_options,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Estado del sistema
        st.subheader("📊 Estado del Sistema")
        
        # Verificar modelo
        model_path = Path(config['paths']['trained_models']) / "best.pt"
        if model_path.exists():
            st.success("✅ Modelo disponible")
            model_status = "Entrenado"
        else:
            st.warning("⚠️ Sin modelo entrenado")
            model_status = "No entrenado"
        
        # Verificar datos
        data_path = Path(config['paths']['data_processed'])
        if data_path.exists() and any(data_path.iterdir()):
            st.success("✅ Datos disponibles")
            data_status = "Procesados"
        else:
            st.warning("⚠️ Datos no procesados")
            data_status = "Sin procesar"
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Modelo", model_status)
        with col2:
            st.metric("Datos", data_status)
        
        st.markdown("---")
        
        # Acciones rápidas
        st.subheader("⚡ Acciones Rápidas")
        
        if st.button("🔄 Verificar Sistema", use_container_width=True):
            st.rerun()
        
        if st.button("🧹 Limpiar Caché", use_container_width=True):
            st.cache_resource.clear()
            st.success("Caché limpiado!")
        
        if st.button("📥 Exportar Config", use_container_width=True):
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
    st.markdown('<h1 class="main-header">🏠 Sistema Inteligente de Clasificación de Residuos</h1>', unsafe_allow_html=True)
    
    # Introducción
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🌍 ¡Bienvenido al Sistema de Clasificación Automática de Residuos!
        
        Este sistema utiliza **YOLOv8** y **redes neuronales convolucionales** para clasificar 
        automáticamente 12 tipos diferentes de residuos del dataset **Garbage Classification**.
        
        ### 🎯 Objetivos del Sistema
        
        ✅ **Clasificación precisa** de materiales reciclables  
        ✅ **Entrenamiento personalizado** con tu propio dataset  
        ✅ **Dashboard interactivo** con métricas en tiempo real  
        ✅ **Predicciones en tiempo real** desde imágenes o cámara  
        ✅ **Reportes detallados** para análisis de resultados  
        
        ### 📋 Flujo de Trabajo
        
        1. **📁 Preparar Datos** - Organiza el dataset de Kaggle
        2. **🚀 Entrenar Modelo** - Entrena YOLO con tus datos
        3. **🔍 Clasificar** - Prueba con nuevas imágenes
        4. **📈 Analizar** - Revisa métricas y mejora el modelo
        """)
    
    with col2:
        # Tarjeta de estadísticas
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("📊 Estadísticas del Sistema")
        
        # Contar clases
        classes = config['classes']
        st.metric("Clases de Residuos", len(classes))
        
        # Verificar imágenes
        try:
            from src.data_preparation import count_images
            total_images = count_images(Path(config['paths']['data_raw']))
            st.metric("Imágenes Totales", f"{total_images:,}")
        except:
            st.metric("Imágenes Totales", "Cargando...")
        
        st.metric("Precisión Esperada", "85-95%")
        st.metric("Tiempo Inferencia", "< 100ms")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Progreso rápido
        st.markdown("### 🚀 Comenzar Rápidamente")
        
        if st.button("📥 Preparar Datos", use_container_width=True):
            st.switch_page("pages/02_📁_Preparar_Datos.py")
        
        if st.button("🎯 Clasificar Ahora", use_container_width=True):
            st.switch_page("pages/04_🔍_Clasificar.py")
    
    # Mostrar clases
    st.markdown("---")
    st.markdown('<h3 class="sub-header">🗂️ Clases de Residuos Soportadas</h3>', unsafe_allow_html=True)
    
    # Mostrar badges de clases
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
            st.markdown(
                f'<div class="class-badge" style="background-color: {color}; color: white;">'
                f'{class_name.replace("-", " ").title()}'
                '</div>',
                unsafe_allow_html=True
            )
    
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
        
        if st.button("🔄 Procesar Dataset", type="primary", use_container_width=True):
            with st.spinner("Procesando dataset para YOLO..."):
                try:
                    stats = preparer.prepare_yolo_dataset()
                    
                    st.success("✅ Dataset procesado exitosamente!")
                    
                    # Mostrar estadísticas
                    st.markdown("#### 📊 Estadísticas del Procesamiento")
                    
                    df_stats = preparer.get_statistics_dataframe()
                    st.dataframe(df_stats, use_container_width=True)
                    
                    # Gráfico de distribución
                    fig = preparer.plot_class_distribution()
                    st.plotly_chart(fig, use_container_width=True)
                    
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
            st.dataframe(report['class_distribution'], use_container_width=True)
            
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
                        st.image(str(img_path), use_container_width=True)
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
        if st.button("💾 Guardar Configuración", use_container_width=True):
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
        if st.button("🎬 Iniciar Entrenamiento", type="primary", use_container_width=True):
            
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
        
        # Verificar si hay modelos entrenados
        trained_models = list(Path(config['paths']['trained_models']).glob("*.pt"))
        
        if trained_models:
            # Mostrar modelos disponibles
            st.success(f"✅ {len(trained_models)} modelo(s) entrenado(s) disponibles")
            
            # Seleccionar modelo
            selected_model = st.selectbox(
                "Seleccionar modelo:",
                [m.name for m in trained_models]
            )
            
            if selected_model:
                model_path = Path(config['paths']['trained_models']) / selected_model
                
                # Cargar métricas del modelo
                metrics = trainer.load_model_metrics(model_path)
                
                if metrics:
                    # Mostrar métricas principales
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
                    
                    with col2:
                        st.metric("Precision", f"{metrics.get('precision', 0):.2%}")
                    
                    with col3:
                        st.metric("Recall", f"{metrics.get('recall', 0):.2%}")
                    
                    with col4:
                        st.metric("F1-Score", f"{metrics.get('f1_score', 0):.2%}")
                    
                    # Gráficos
                    st.markdown("#### 📈 Curvas de Aprendizaje")
                    
                    if 'history' in metrics:
                        fig = trainer.plot_training_history(metrics['history'])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Matriz de confusión
                    st.markdown("#### 🎯 Matriz de Confusión")
                    
                    if 'confusion_matrix' in metrics:
                        fig = trainer.plot_confusion_matrix(metrics['confusion_matrix'])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Reporte por clase
                    st.markdown("#### 📋 Reporte por Clase")
                    
                    if 'class_report' in metrics:
                        st.dataframe(metrics['class_report'], use_container_width=True)
                else:
                    st.info("No hay métricas disponibles para este modelo.")
        else:
            st.info("No hay modelos entrenados aún. Entrena un modelo primero.")

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
                    
                    if st.button("🎯 Clasificar Esta Imagen", use_container_width=True):
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
                st.plotly_chart(fig, use_container_width=True)
            
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
                    if st.button("📥 Guardar Resultado", use_container_width=True):
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
                        use_container_width=True
                    )
            
            # Análisis detallado
            with st.expander("📈 Análisis Detallado", expanded=False):
                tab1, tab2, tab3 = st.tabs(["📊 Distribución", "🎯 Probabilidades", "📋 Metadata"])
                
                with tab1:
                    from src.visualizations import VisualizationManager
                    viz = VisualizationManager()
                    fig = viz.plot_probability_distribution(predictions)
                    st.plotly_chart(fig, use_container_width=True)
                
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
        st.plotly_chart(fig, use_container_width=True)
        
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
                use_container_width=True
            )
        
        with col_exp2:
            # Generar reporte
            if st.button("📊 Generar Reporte Completo", use_container_width=True):
                report_path = predictor.generate_batch_report(results)
                with open(report_path, 'rb') as f:
                    st.download_button(
                        label="📄 Descargar Reporte PDF",
                        data=f,
                        file_name="reporte_clasificacion.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

def show_analysis_page():
    """Mostrar página de análisis y métricas"""
    st.markdown('<h1 class="main-header">📈 Análisis y Métricas</h1>', unsafe_allow_html=True)
    
    from src.metrics_analyzer import MetricsAnalyzer
    
    analyzer = MetricsAnalyzer(config)
    
    # Tabs para diferentes tipos de análisis
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Métricas del Modelo", 
        "📈 Historial de Entrenamiento",
        "🔄 Análisis Comparativo", 
        "🔍 Diagnóstico de Errores", 
        "📋 Reportes"
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
                st.plotly_chart(fig, use_container_width=True)
            
            with col_chart2:
                # Curva Precision-Recall
                fig = analyzer.plot_precision_recall_curve(metrics)
                st.plotly_chart(fig, use_container_width=True)
            
            # Matriz de confusión
            st.markdown("#### 🎯 Matriz de Confusión")
            fig = analyzer.plot_confusion_matrix(metrics)
            st.plotly_chart(fig, use_container_width=True)
            
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
            # Seleccionar modelos para comparar
            col1, col2 = st.columns(2)
            
            with col1:
                model_a = st.selectbox(
                    "Modelo A:",
                    available_models,
                    index=0
                )
            
            with col2:
                model_b = st.selectbox(
                    "Modelo B:",
                    available_models,
                    index=min(1, len(available_models)-1)
                )
            
            if st.button("🔄 Comparar Modelos", use_container_width=True):
                comparison = analyzer.compare_models(model_a, model_b)
                
                if comparison:
                    # Mostrar comparación
                    st.markdown("#### 📊 Resultados de la Comparación")
                    st.dataframe(comparison.style.highlight_max(axis=0, color='lightgreen'))
                    
                    # Gráfico de comparación
                    fig = analyzer.plot_model_comparison(comparison)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Necesitas al menos 2 modelos entrenados para comparar.")
    
    with tab4:
        st.markdown("### 🔍 Diagnóstico de Errores")
        
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
    
    with tab4:
        st.markdown("### 📋 Generar Reportes")
        
        # Tipos de reportes
        report_type = st.selectbox(
            "Tipo de Reporte:",
            ["📊 Reporte de Métricas", "📈 Reporte de Entrenamiento", "🔍 Reporte de Errores", "📋 Reporte Completo"]
        )
        
        # Opciones del reporte
        with st.expander("⚙️ Opciones del Reporte"):
            include_charts = st.checkbox("Incluir gráficos", value=True)
            include_tables = st.checkbox("Incluir tablas", value=True)
            include_recommendations = st.checkbox("Incluir recomendaciones", value=True)
        
        # Generar reporte
        if st.button("📄 Generar Reporte", type="primary", use_container_width=True):
            with st.spinner("Generando reporte..."):
                try:
                    report_path = analyzer.generate_report(
                        report_type=report_type,
                        include_charts=include_charts,
                        include_tables=include_tables,
                        include_recommendations=include_recommendations
                    )
                    
                    # Descargar reporte
                    with open(report_path, 'rb') as f:
                        st.download_button(
                            label="📥 Descargar Reporte PDF",
                            data=f,
                            file_name=f"reporte_{report_type.lower().replace(' ', '_')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    
                    st.success("✅ Reporte generado exitosamente!")
                
                except Exception as e:
                    st.error(f"❌ Error generando reporte: {str(e)}")

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
        if st.button("🔄 Reiniciar Sistema", use_container_width=True):
            st.cache_resource.clear()
            st.success("✅ Sistema reiniciado")
            st.rerun()
    
    with col_act2:
        if st.button("🧹 Limpiar Caché", use_container_width=True):
            import shutil
            cache_dirs = ["./__pycache__", "./streamlit_cache"]
            for cache_dir in cache_dirs:
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
            st.success("✅ Caché limpiado")
    
    with col_act3:
        if st.button("📤 Exportar Configuración", use_container_width=True):
            export_configuration()

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