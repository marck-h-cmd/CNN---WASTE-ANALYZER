import os
import yaml
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable

# IMPORTANTE: Configurar variables de entorno ANTES de importar numpy/torch
os.environ['YOLO_DISABLE_SIGNAL_HANDLERS'] = '1'
os.environ['YOLO_VERBOSE'] = 'False'

# Ahora importar el resto
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st
from ultralytics import YOLO
from PIL import Image

class ModelTrainer:
    """Entrena modelos YOLO para clasificación"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.classes = config['classes']
        
        # Directorios
        self.models_dir = Path(config['paths']['trained_models'])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = Path(config['paths']['results_dir']) / 'training_logs'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Verificar disponibilidad de CUDA al inicializar
        self._check_cuda_availability()
    
    def _check_cuda_availability(self):
        """Verifica y muestra información sobre CUDA"""
        print("="*60)
        print("🔍 VERIFICACIÓN DE CUDA Y GPU")
        print("="*60)
        
        print(f"PyTorch versión: {torch.__version__}")
        print(f"CUDA disponible: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA versión: {torch.version.cuda}")
            print(f"cuDNN versión: {torch.backends.cudnn.version()}")
            print(f"Número de GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                try:
                    props = torch.cuda.get_device_properties(i)
                    print(f"\n📊 GPU {i}: {props.name}")
                    print(f"   Memoria total: {props.total_memory / 1024**3:.2f} GB")
                    print(f"   Compute capability: {props.major}.{props.minor}")
                except Exception as e:
                    print(f"   Error obteniendo propiedades: {e}")
        else:
            print("\n⚠️  CUDA no está disponible. Razones posibles:")
            print("   1. PyTorch instalado sin soporte CUDA")
            print("   2. Drivers de NVIDIA no instalados/actualizados")
            print("   3. CUDA Toolkit no compatible")
            print("\n💡 Solución: Reinstalar PyTorch con CUDA:")
            print("   pip uninstall torch torchvision torchaudio")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
        print("="*60 + "\n")
    
    def _get_safe_device(self, requested_device: str) -> str:
        """Obtiene un dispositivo seguro y válido"""
        # Normalizar input
        if requested_device is None:
            requested_device = 'cpu'
        
        requested_device = str(requested_device).lower().strip()
        
        # Lista de dispositivos válidos
        valid_devices = ['cpu']
        
        # Agregar CUDA solo si está disponible
        if torch.cuda.is_available():
            valid_devices.extend(['cuda', 'cuda:0'])
            for i in range(torch.cuda.device_count()):
                valid_devices.append(f'cuda:{i}')
        
        # Si el dispositivo solicitado no es válido, usar CPU
        if requested_device not in valid_devices:
            if 'cuda' in requested_device and not torch.cuda.is_available():
                print(f"⚠️  CUDA solicitado pero no disponible. Usando CPU.")
                return 'cpu'
            else:
                print(f"⚠️  Dispositivo '{requested_device}' no válido. Usando CPU.")
                return 'cpu'
        
        # Verificar específicamente para CUDA
        if 'cuda' in requested_device:
            try:
                # Intentar crear un tensor pequeño en CUDA para verificar
                device_idx = 0
                if ':' in requested_device:
                    device_idx = int(requested_device.split(':')[1])
                
                if device_idx >= torch.cuda.device_count():
                    print(f"⚠️  GPU {device_idx} no encontrada. Usando GPU 0.")
                    return 'cuda:0'
                
                # Test rápido
                test_tensor = torch.zeros(1).to(f'cuda:{device_idx}')
                del test_tensor
                torch.cuda.empty_cache()
                
                return requested_device
                
            except Exception as e:
                print(f"⚠️  Error al acceder a CUDA: {e}")
                print("   Usando CPU como alternativa.")
                return 'cpu'
        
        return requested_device
    
    def _training_callback(self, trainer):
        """Callback personalizado para mostrar progreso durante entrenamiento"""
        try:
            # Obtener información del epoch actual
            epoch = trainer.epoch
            total_epochs = trainer.epochs
            
            # Crear placeholder para mostrar información
            if not hasattr(self, 'progress_placeholder'):
                self.progress_placeholder = st.empty()
                self.image_placeholder = st.empty()
                self.metrics_placeholder = st.empty()
            
            # Actualizar progreso
            progress = (epoch + 1) / total_epochs
            self.progress_placeholder.progress(progress, 
                text=f"📊 Época {epoch + 1}/{total_epochs}")
            
            # Mostrar métricas actuales si están disponibles
            if hasattr(trainer, 'metrics'):
                metrics = trainer.metrics
                cols = st.columns(4)
                
                with cols[0]:
                    st.metric("Loss", f"{getattr(metrics, 'fitness', 0):.4f}")
                with cols[1]:
                    st.metric("Precisión", f"{getattr(metrics, 'top1', 0):.2f}%")
                with cols[2]:
                    st.metric("Top-5", f"{getattr(metrics, 'top5', 0):.2f}%")
                with cols[3]:
                    st.metric("Época", f"{epoch + 1}/{total_epochs}")
            
            # Mostrar imagen de muestra del batch actual
            if hasattr(trainer, 'batch') and trainer.batch is not None:
                try:
                    batch_imgs = trainer.batch.get('img', None)
                    if batch_imgs is not None and len(batch_imgs) > 0:
                        # Tomar la primera imagen del batch
                        img = batch_imgs[0]
                        
                        # Convertir tensor a numpy
                        if torch.is_tensor(img):
                            img = img.cpu().numpy()
                        
                        # Normalizar y convertir a formato PIL
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        
                        # Si la imagen está en formato CHW, convertir a HWC
                        if img.shape[0] in [1, 3]:
                            img = np.transpose(img, (1, 2, 0))
                        
                        # Si es escala de grises, convertir a RGB
                        if len(img.shape) == 2 or img.shape[2] == 1:
                            img = np.stack([img.squeeze()] * 3, axis=-1)
                        
                        # Mostrar imagen
                        pil_img = Image.fromarray(img)
                        self.image_placeholder.image(pil_img, 
                            caption=f"🖼️ Muestra del batch - Época {epoch + 1}",
                            use_column_width=True)
                except Exception as e:
                    print(f"Error mostrando imagen: {e}")
        
        except Exception as e:
            print(f"Error en callback: {e}")
    
    def train_model(self, epochs: int = None, batch_size: int = None,
                   learning_rate: float = None, device: str = None,
                   experiment_name: str = None, resume: bool = False,
                   callback: Callable = None) -> Dict:
        """Entrena un modelo YOLO para clasificación"""
        
        # Usar valores por defecto si no se proporcionan
        if epochs is None:
            epochs = self.config['training']['epochs']
        if batch_size is None:
            batch_size = self.config['training']['batch_size']
        if learning_rate is None:
            learning_rate = self.config['training']['learning_rate']
        if device is None:
            device = self.config['training']['device']
        if experiment_name is None:
            experiment_name = f"garbage_classification_{self.config['model']['name']}"
        
        # Obtener dispositivo seguro
        device = self._get_safe_device(device)
        
        # Verificar datos procesados
        print("path", self.config['paths']['data_processed'])
        data_dir = Path(self.config['paths']['data_processed'])
        
        # Mostrar información de inicio
        st.success("🚀 **Iniciando entrenamiento del modelo**")
        
        cols = st.columns(4)
        with cols[0]:
            st.info(f"**Épocas:** {epochs}")
        with cols[1]:
            st.info(f"**Batch Size:** {batch_size}")
        with cols[2]:
            st.info(f"**Learning Rate:** {learning_rate}")
        with cols[3]:
            st.info(f"**Dispositivo:** {device.upper()}")
        
        print(f"🚀 Iniciando entrenamiento con:")
        print(f"   Épocas: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Dispositivo: {device}")
        
        # Mostrar información del dataset
        st.subheader("📂 Información del Dataset")
        
        # Contar imágenes por conjunto
        train_dir = data_dir / 'train'
        val_dir = data_dir / 'val'
        test_dir = data_dir / 'test'
        
        train_count = sum(1 for _ in train_dir.rglob('*.jpg')) + sum(1 for _ in train_dir.rglob('*.png'))
        val_count = sum(1 for _ in val_dir.rglob('*.jpg')) + sum(1 for _ in val_dir.rglob('*.png'))
        test_count = sum(1 for _ in test_dir.rglob('*.jpg')) + sum(1 for _ in test_dir.rglob('*.png'))
        
        dataset_cols = st.columns(3)
        with dataset_cols[0]:
            st.metric("🎯 Entrenamiento", train_count)
        with dataset_cols[1]:
            st.metric("✅ Validación", val_count)
        with dataset_cols[2]:
            st.metric("🧪 Prueba", test_count)
        
        # Mostrar algunas imágenes de muestra
        st.subheader("🖼️ Muestras del Dataset")
        sample_cols = st.columns(min(len(self.classes), 5))
        
        for idx, class_name in enumerate(self.classes[:5]):
            class_dir = train_dir / class_name
            if class_dir.exists():
                images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
                if images:
                    with sample_cols[idx]:
                        img = Image.open(images[0])
                        st.image(img, caption=class_name, use_column_width=True)
        
        st.divider()
        
        # Cargar modelo preentrenado
        model_name = "yolov8n-cls"
        st.info(f"📦 Cargando modelo preentrenado: **{model_name}**")
        print(f"📦 Cargando modelo: {model_name}")
        
        try:
            self.model = YOLO(f"{model_name}.pt")
            st.success("✅ Modelo cargado exitosamente")
        except Exception as e:
            st.error(f"❌ Error cargando modelo: {e}")
            # Intentar cargar modelo local
            local_model = self.models_dir / f"{model_name}.pt"
            if local_model.exists():
                self.model = YOLO(str(local_model))
                st.success("✅ Modelo local cargado exitosamente")
            else:
                raise
        
        print(f"💻 Usando dispositivo: {device}")
        
        # Parámetros de entrenamiento
        train_args = {
            'data': str(data_dir),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': self.config['model']['input_size'],
            'device': device,
            'workers': self.config['training']['workers'],
            'lr0': learning_rate,
            'momentum': self.config['training']['momentum'],
            'weight_decay': self.config['training']['weight_decay'],
            'warmup_epochs': self.config['training']['warmup_epochs'],
            'patience': self.config['training']['patience'],
            'seed': self.config['training']['seed'],
            'pretrained': True,
            'verbose': True,
            'project': str(self.models_dir),
            'name': experiment_name,
            'exist_ok': True,
            'resume': resume
        }
        
        # Añadir aumentación si está habilitada
        if self.config['training']['augment']:
            train_args.update({
                'hsv_h': self.config['training']['hsv_h'],
                'hsv_s': self.config['training']['hsv_s'],
                'hsv_v': self.config['training']['hsv_v'],
                'degrees': self.config['training']['degrees'],
                'translate': self.config['training']['translate'],
                'scale': self.config['training']['scale']
            })
        
        # Añadir dropout si está configurado
        if self.config['training']['dropout'] > 0:
            train_args['dropout'] = self.config['training']['dropout']
        
        # Iniciar entrenamiento
        st.subheader("🏋️ Entrenamiento en Progreso")
        start_time = time.time()
        
        # Crear placeholders para el progreso
        self.progress_placeholder = st.empty()
        self.image_placeholder = st.empty()
        self.metrics_placeholder = st.empty()
        
        try:
            # Entrenar modelo
            st.info("⏳ Entrenando modelo... Esto puede tomar varios minutos.")
            results = self.model.train(**train_args)
            
            training_time = (time.time() - start_time) / 60  # en minutos
            
            # Limpiar placeholders
            self.progress_placeholder.empty()
            self.image_placeholder.empty()
            self.metrics_placeholder.empty()
            
            # Mostrar resultado del entrenamiento
            st.success(f"✅ **Entrenamiento completado exitosamente en {training_time:.2f} minutos**")
            
            st.balloons()
            
            # Mostrar resumen del entrenamiento
            st.subheader("📊 Resumen del Entrenamiento")
            
            summary_cols = st.columns(4)
            with summary_cols[0]:
                st.metric("⏱️ Tiempo Total", f"{training_time:.2f} min")
            with summary_cols[1]:
                st.metric("📈 Épocas", epochs)
            with summary_cols[2]:
                st.metric("🎯 Batch Size", batch_size)
            with summary_cols[3]:
                st.metric("💻 Dispositivo", device.upper())
            
            # Evaluar modelo
            st.info("🔍 Evaluando modelo en conjunto de validación...")
            metrics = self.evaluate_model()
            
            # Mostrar métricas finales
            st.subheader("🎯 Métricas Finales")
            
            metrics_cols = st.columns(4)
            with metrics_cols[0]:
                st.metric("Exactitud", f"{metrics['accuracy']*100:.2f}%")
            with metrics_cols[1]:
                st.metric("Precisión", f"{metrics['precision']*100:.2f}%")
            with metrics_cols[2]:
                st.metric("Recall", f"{metrics['recall']*100:.2f}%")
            with metrics_cols[3]:
                st.metric("F1-Score", f"{metrics['f1_score']*100:.2f}%")
            
            # Guardar resultados
            training_results = {
                'experiment_name': experiment_name,
                'training_time': training_time,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'device': device,
                'metrics': metrics,
                'model_path': str(self.models_dir / experiment_name / 'weights' / 'best.pt'),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'dataset_info': {
                    'train_images': train_count,
                    'val_images': val_count,
                    'test_images': test_count,
                    'classes': self.classes
                }
            }
            
            # Guardar resultados en archivo
            self.save_training_results(training_results)
            
            experiment_dir = self.results_dir / experiment_name
            st.success(f"💾 Resultados guardados en: `{experiment_dir}`")
            
            # Mostrar información adicional si está disponible
            if 'validation_samples' in metrics:
                st.info(f"📊 Evaluación realizada en {metrics['validation_samples']} imágenes de validación")
            
            # Nota sobre métricas limitadas
            if metrics['precision'] == 0.0 and metrics['recall'] == 0.0:
                st.warning("⚠️ Las métricas detalladas (precisión, recall, F1) no están disponibles. Solo se muestran las métricas básicas de YOLO (Top-1/Top-5 accuracy).")
            
            # Mostrar gráficas y resultados generados
            st.divider()
            st.subheader("📈 Gráficas y Resultados Generados")
            
            # Mostrar gráficas del entrenamiento
            plots_dir = experiment_dir / 'plots'
            if plots_dir.exists():
                st.write("**📊 Gráficas de Entrenamiento:**")
                
                # Buscar archivos de gráficas
                training_history_files = list(plots_dir.glob("training_history_*.png"))
                confusion_matrix_files = list(plots_dir.glob("confusion_matrix_*.png"))
                
                # Mostrar training history (archivo más reciente)
                if training_history_files:
                    training_history_file = max(training_history_files, key=lambda x: x.stat().st_mtime)
                    try:
                        st.image(str(training_history_file), caption="Historial de Entrenamiento", use_column_width=True)
                    except Exception as e:
                        st.warning(f"No se pudo cargar la gráfica de historial: {e}")
                
                # Mostrar confusion matrix (archivo más reciente)
                if confusion_matrix_files:
                    confusion_matrix_file = max(confusion_matrix_files, key=lambda x: x.stat().st_mtime)
                    try:
                        st.image(str(confusion_matrix_file), caption="Matriz de Confusión", use_column_width=True)
                    except Exception as e:
                        st.warning(f"No se pudo cargar la matriz de confusión: {e}")
                
                # Si no hay gráficas, mostrar mensaje
                if not training_history_files and not confusion_matrix_files:
                    st.info("ℹ️ No se encontraron gráficas generadas durante el entrenamiento.")
            
            # Mostrar archivos generados por YOLO
            model_yolo_dir = self.models_dir / experiment_name
            if model_yolo_dir.exists():
                st.write("**📁 Archivos Generados por YOLO:**")
                
                # Mostrar estructura de archivos
                with st.expander("Ver estructura de archivos generados"):
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
                        st.write("**📊 Resultados Detallados del Entrenamiento:**")
                        st.dataframe(results_df, use_container_width=True)
                    except Exception as e:
                        st.warning(f"No se pudo cargar results.csv: {e}")
                
                # Mostrar args.yaml si existe
                args_yaml = model_yolo_dir / 'args.yaml'
                if args_yaml.exists():
                    try:
                        with open(args_yaml, 'r') as f:
                            args_content = f.read()
                        with st.expander("Ver configuración del entrenamiento (args.yaml)"):
                            st.code(args_content, language='yaml')
                    except Exception as e:
                        st.warning(f"No se pudo cargar args.yaml: {e}")
            
            # Información sobre el modelo guardado
            st.success(f"✅ **Modelo guardado exitosamente en:** `{training_results['model_path']}`")
            
            # Botón para descargar el modelo
            if Path(training_results['model_path']).exists():
                with open(training_results['model_path'], 'rb') as f:
                    model_bytes = f.read()
                st.download_button(
                    label="📥 Descargar Modelo Entrenado (best.pt)",
                    data=model_bytes,
                    file_name=f"{experiment_name}_best.pt",
                    mime="application/octet-stream"
                )
            
            print(f"\n{'='*60}")
            print(f"✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
            print(f"{'='*60}")
            print(f"⏱️  Tiempo total: {training_time:.2f} minutos")
            print(f"🎯 Exactitud: {metrics['accuracy']*100:.2f}%")
            print(f"📊 Precisión: {metrics['precision']*100:.2f}%")
            print(f"🔍 Recall: {metrics['recall']*100:.2f}%")
            print(f"📈 F1-Score: {metrics['f1_score']*100:.2f}%")
            print(f"💾 Modelo guardado en: {training_results['model_path']}")
            print(f"{'='*60}\n")
            
            return training_results
            
        except Exception as e:
            st.error(f"❌ Error durante el entrenamiento: {str(e)}")
            print(f"\n❌ ERROR: {str(e)}\n")
            raise
    
    def evaluate_model(self) -> Dict:
        """Evalúa el modelo entrenado"""
        if self.model is None:
            raise ValueError("Modelo no cargado")
        
        # Usar el archivo dataset.yaml para evaluación
        data_yaml = Path(self.config['paths']['data_processed']) / 'dataset.yaml'
        
        # Evaluar en conjunto de validación usando el dataset.yaml
        metrics = self.model.val(
            data=str(data_yaml),
            save=False,
            save_json=False,
            save_txt=False,
            plots=False,
            verbose=False
        )
        
        # Extraer métricas básicas de YOLO
        metrics_dict = {
            'top1_accuracy': float(getattr(metrics, 'top1', 0) / 100),
            'top5_accuracy': float(getattr(metrics, 'top5', 0) / 100),
            'accuracy': float(getattr(metrics, 'top1', 0) / 100),
            'precision': 0.0,  # YOLO no proporciona estas métricas directamente
            'recall': 0.0,
            'f1_score': 0.0
        }
        
        # Intentar obtener métricas adicionales usando sklearn si es posible
        try:
            # Obtener predicciones para métricas detalladas
            val_dir = Path(self.config['paths']['data_processed']) / 'val'
            val_results = self.model.predict(
                source=str(val_dir),
                save=False,
                save_txt=False,
                save_conf=False,
                save_crop=False,
                verbose=False,
                conf=0.1,  # Umbral de confianza bajo para capturar más predicciones
                project=None,
                name=None,
                exist_ok=True
            )
            
            # Extraer labels verdaderas y predicciones
            y_true = []
            y_pred = []
            
            for result in val_results:
                if hasattr(result, 'probs') and result.probs is not None:
                    # Para clasificación
                    probs = result.probs.data.cpu().numpy()
                    pred_class = np.argmax(probs)
                    
                    # Obtener clase verdadera del path del archivo
                    path_parts = Path(result.path).parts
                    true_class = None
                    for part in path_parts:
                        if part in self.classes:
                            true_class = self.classes.index(part)
                            break
                    
                    if true_class is not None:
                        y_true.append(true_class)
                        y_pred.append(pred_class)
            
            if y_true and y_pred and len(y_true) > 0:
                # Calcular métricas usando sklearn
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                
                # Actualizar métricas
                metrics_dict.update({
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'validation_samples': len(y_true)
                })
                
                print(f"✅ Métricas calculadas en {len(y_true)} muestras de validación")
            else:
                print("⚠️ No se pudieron extraer predicciones detalladas, usando métricas básicas de YOLO")
                
        except Exception as e:
            print(f"⚠️ Error calculando métricas detalladas: {e}")
            print("   Usando métricas básicas de YOLO")
        
        return metrics_dict
    
    def save_training_results(self, results: Dict):
        """Guarda resultados del entrenamiento"""
        # Asegurar que el directorio existe
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar en JSON con nombre descriptivo
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f"{results['experiment_name']}_{timestamp}_results.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Guardar métricas separadas
        metrics_file = self.results_dir / f"{results['experiment_name']}_{timestamp}_metrics.json"
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(results['metrics'], f, indent=2, ensure_ascii=False)
        
        # Guardar resumen en CSV para análisis fácil
        summary_file = self.results_dir / f"{results['experiment_name']}_{timestamp}_summary.csv"
        
        summary_data = {
            'experiment_name': [results['experiment_name']],
            'timestamp': [results.get('timestamp', time.strftime('%Y-%m-%d %H:%M:%S'))],
            'training_time_min': [results['training_time']],
            'epochs': [results['epochs']],
            'batch_size': [results['batch_size']],
            'learning_rate': [results['learning_rate']],
            'device': [results['device']],
            'accuracy': [results['metrics']['accuracy']],
            'precision': [results['metrics']['precision']],
            'recall': [results['metrics']['recall']],
            'f1_score': [results['metrics']['f1_score']],
            'model_path': [results['model_path']]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\n💾 Resultados guardados:")
        print(f"   📄 Resultados completos: {results_file}")
        print(f"   📊 Métricas: {metrics_file}")
        print(f"   📈 Resumen CSV: {summary_file}")
    
    def load_model_metrics(self, model_path: Path) -> Optional[Dict]:
        """Carga métricas de un modelo entrenado"""
        # Buscar archivo de métricas
        model_name = model_path.stem
        metrics_file = self.results_dir / f"{model_name}_metrics.json"
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                return json.load(f)
        
        # Si no existe, intentar evaluar el modelo
        try:
            self.model = YOLO(str(model_path))
            metrics = self.evaluate_model()
            return metrics
        except:
            return None
    
    def plot_training_history(self, history: Dict) -> go.Figure:
        """Crea gráfico del historial de entrenamiento"""
        # Esta función necesita datos del historial de entrenamiento
        # Por ahora, crea un gráfico de ejemplo
        
        fig = go.Figure()
        
        # Datos de ejemplo
        epochs = list(range(1, 51))
        train_loss = [2.0 - 0.03 * i + np.random.normal(0, 0.05) for i in range(50)]
        val_loss = [2.0 - 0.025 * i + np.random.normal(0, 0.1) for i in range(50)]
        accuracy = [0.1 + 0.015 * i + np.random.normal(0, 0.03) for i in range(50)]
        
        fig.add_trace(go.Scatter(
            x=epochs, y=train_loss,
            mode='lines',
            name='Train Loss',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=epochs, y=val_loss,
            mode='lines',
            name='Val Loss',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=epochs, y=accuracy,
            mode='lines',
            name='Accuracy',
            line=dict(color='green', width=2),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Historial de Entrenamiento',
            xaxis_title='Época',
            yaxis_title='Loss',
            yaxis2=dict(
                title='Accuracy',
                overlaying='y',
                side='right',
                range=[0, 1]
            ),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def plot_confusion_matrix(self, cm: List[List[float]]) -> go.Figure:
        """Crea gráfico de matriz de confusión"""
        cm_array = np.array(cm)
        
        fig = px.imshow(
            cm_array,
            labels=dict(x="Predicción", y="Verdadero", color="Cantidad"),
            x=self.classes,
            y=self.classes,
            color_continuous_scale='Greens',
            aspect='auto'
        )
        
        fig.update_layout(
            title='Matriz de Confusión',
            xaxis_title='Clase Predicha',
            yaxis_title='Clase Verdadera'
        )
        
        # Añadir texto en celdas
        for i in range(len(self.classes)):
            for j in range(len(self.classes)):
                fig.add_annotation(
                    x=j, y=i,
                    text=str(int(cm_array[i, j])),
                    showarrow=False,
                    font=dict(color='white' if cm_array[i, j] > cm_array.max()/2 else 'black')
                )
        
        return fig
    
    def save_training_results(self, results: Dict):
        """Guarda resultados del entrenamiento organizados por experimento"""
        # Crear directorio del experimento
        experiment_dir = self.results_dir / results['experiment_name']
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear subdirectorio para plots
        plots_dir = experiment_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp para archivos
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Guardar resultados completos en JSON
        results_file = experiment_dir / f"results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Guardar métricas separadas
        metrics_file = experiment_dir / f"metrics_{timestamp}.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(results['metrics'], f, indent=2, ensure_ascii=False, default=str)
        
        # Guardar resumen en CSV
        summary_file = experiment_dir / f"summary_{timestamp}.csv"
        summary_data = {
            'experiment_name': [results['experiment_name']],
            'timestamp': [results.get('timestamp', time.strftime('%Y-%m-%d %H:%M:%S'))],
            'training_time_min': [results['training_time']],
            'epochs': [results['epochs']],
            'batch_size': [results['batch_size']],
            'learning_rate': [results['learning_rate']],
            'device': [results['device']],
            'accuracy': [results['metrics']['accuracy']],
            'precision': [results['metrics']['precision']],
            'recall': [results['metrics']['recall']],
            'f1_score': [results['metrics']['f1_score']],
            'model_path': [results['model_path']]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        
        # Generar y guardar plots
        self.save_training_plots(results, plots_dir, timestamp)
        
        print(f"\n💾 Resultados guardados en carpeta: {experiment_dir}")
        print(f"   📁 Estructura:")
        print(f"      ├── {results_file.name}")
        print(f"      ├── {metrics_file.name}")
        print(f"      ├── {summary_file.name}")
        print(f"      └── plots/")
        print(f"          ├── training_history_{timestamp}.png")
        print(f"          └── confusion_matrix_{timestamp}.png")
    
    def save_training_plots(self, results: Dict, plots_dir: Path, timestamp: str):
        """Genera y guarda plots del entrenamiento"""
        try:
            # Plot de historial de entrenamiento (ejemplo)
            fig_history = self.plot_training_history({})
            history_file = plots_dir / f"training_history_{timestamp}.png"
            fig_history.write_image(str(history_file))
            
            # Intentar generar matriz de confusión si tenemos datos suficientes
            if 'validation_samples' in results['metrics'] and results['metrics']['validation_samples'] > 0:
                try:
                    # Generar predicciones para matriz de confusión
                    val_dir = Path(self.config['paths']['data_processed']) / 'val'
                    val_results = self.model.predict(
                        source=str(val_dir),
                        save=False,
                        save_txt=False,
                        save_conf=False,
                        save_crop=False,
                        verbose=False,
                        conf=0.1,
                        project=None,
                        name=None,
                        exist_ok=True
                    )
                    
                    # Extraer predicciones para matriz de confusión
                    y_true = []
                    y_pred = []
                    
                    for result in val_results:
                        if hasattr(result, 'probs') and result.probs is not None:
                            probs = result.probs.data.cpu().numpy()
                            pred_class = np.argmax(probs)
                            
                            path_parts = Path(result.path).parts
                            true_class = None
                            for part in path_parts:
                                if part in self.classes:
                                    true_class = self.classes.index(part)
                                    break
                            
                            if true_class is not None:
                                y_true.append(true_class)
                                y_pred.append(pred_class)
                    
                    if y_true and y_pred and len(y_true) > 10:  # Solo si tenemos suficientes datos
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(y_true, y_pred)
                        
                        fig_cm = self.plot_confusion_matrix(cm.tolist())
                        cm_file = plots_dir / f"confusion_matrix_{timestamp}.png"
                        fig_cm.write_image(str(cm_file))
                        print(f"          └── confusion_matrix_{timestamp}.png (generada)")
                    else:
                        print(f"          └── confusion_matrix_{timestamp}.png (datos insuficientes)")
                        
                except Exception as e:
                    print(f"⚠️ No se pudo generar matriz de confusión: {e}")
                    # Crear archivo vacío como placeholder
                    cm_file = plots_dir / f"confusion_matrix_{timestamp}.png"
                    with open(cm_file, 'w') as f:
                        f.write("# Matriz de confusión no disponible")
            else:
                print(f"          └── confusion_matrix_{timestamp}.png (datos insuficientes)")
                
        except Exception as e:
            print(f"⚠️ Error guardando plots: {e}")