
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import cv2
from PIL import Image
import torch
from ultralytics import YOLO
import streamlit as st

class ModelPredictor:
    """Realiza predicciones con modelos YOLO entrenados"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.classes = config['classes']
        
        # Directorios
        self.models_dir = Path(config['paths']['trained_models'])
        self.uploads_dir = Path(config['paths']['uploads_dir'])
        self.predictions_dir = Path(config['paths']['results_dir']) / 'predictions'
        
        # Crear directorios si no existen
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
    
    def model_exists(self) -> bool:
        """Verifica si existe un modelo entrenado"""
        # Buscar el mejor modelo en models/trained/
        best_model = self.models_dir / 'best.pt'
        if best_model.exists():
            return True
        
        # Buscar cualquier modelo .pt en models/trained/
        model_files = list(self.models_dir.glob('*.pt'))
        if model_files:
            return True
        
        # Buscar en runs/
        runs_dir = Path('runs')
        runs_models = [
            runs_dir / 'train' / 'weights' / 'best.pt',
            runs_dir / 'classify' / 'train' / 'weights' / 'best.pt',
            runs_dir / 'classify' / 'models' / 'trained' / 'garbage_classification_nano' / 'weights' / 'best.pt'
        ]
        
        for model_path in runs_models:
            if model_path.exists():
                return True
        
        return False
    
    def load_model(self, model_path: str = None):
        """Carga el modelo entrenado"""
        if model_path is None:
            # Buscar el mejor modelo primero en runs/
            runs_dir = Path('runs')
            possible_paths = [
                runs_dir / 'train' / 'weights' / 'best.pt',
                runs_dir / 'classify' / 'train' / 'weights' / 'best.pt',
                runs_dir / 'classify' / 'models' / 'trained' / 'garbage_classification_nano' / 'weights' / 'best.pt',
                self.models_dir / 'best.pt'
            ]
            
            # Buscar el primer modelo que exista
            for path in possible_paths:
                if path.exists():
                    model_path = str(path)
                    break
            
            # Si no encontró en runs ni en models, buscar cualquier .pt
            if model_path is None:
                model_files = list(self.models_dir.glob('*.pt'))
                if model_files:
                    model_path = str(model_files[0])
                else:
                    raise FileNotFoundError("No se encontró ningún modelo entrenado")
        
        try:
            st.info(f"📦 Cargando modelo: {Path(model_path).name} desde {Path(model_path).parent}")
            self.model = YOLO(model_path)
            
            # Verificar que sea modelo de clasificación
            if not hasattr(self.model.model, 'names'):
                # Asignar nombres de clases si no los tiene
                self.model.model.names = {i: name for i, name in enumerate(self.classes)}
            
            st.success("✅ Modelo cargado exitosamente")
            
        except Exception as e:
            st.error(f"❌ Error cargando modelo: {str(e)}")
            raise
    
    def load_pretrained_model(self):
        """Carga el modelo entrenado desde runs/ o models/"""
        try:
            # Buscar el modelo best.pt en la carpeta runs/ primero
            runs_dir = Path('runs')
            possible_paths = [
                runs_dir / 'train' / 'weights' / 'best.pt',
                runs_dir / 'classify' / 'train' / 'weights' / 'best.pt',
                runs_dir / 'classify' / 'models' / 'trained' / 'garbage_classification_nano' / 'weights' / 'best.pt',
                self.models_dir / 'best.pt'
            ]
            
            model_path = None
            for path in possible_paths:
                if path.exists():
                    model_path = path
                    break
            
            if model_path is None:
                st.error("❌ No se encontró modelo entrenado en runs/ ni en models/")
                st.info("💡 Primero entrena un modelo en la página '🚀 Entrenar Modelo'")
                return
            
            st.info(f"📦 Cargando modelo desde: {model_path}")
            self.model = YOLO(str(model_path))
            
            # Verificar que sea modelo de clasificación
            if not hasattr(self.model.model, 'names'):
                self.model.model.names = {i: name for i, name in enumerate(self.classes)}
            
            st.success(f"✅ Modelo cargado exitosamente desde: {model_path.parent.parent.parent.name if 'runs' in str(model_path) else model_path.parent.name}")
            
        except Exception as e:
            st.error(f"❌ Error cargando modelo: {str(e)}")
            raise
    
    def predict(self, image_source: Union[str, bytes, Image.Image]) -> Tuple[List[Dict], float, np.ndarray]:
        """Realiza predicción en una imagen"""
        if self.model is None:
            raise ValueError("Modelo no cargado. Llama a load_model() primero.")
        
        start_time = time.time()
        
        try:
            # Cargar imagen
            if isinstance(image_source, str):
                # Ruta de archivo
                if not os.path.exists(image_source):
                    raise FileNotFoundError(f"No se encuentra la imagen: {image_source}")
                
                image = cv2.imread(image_source)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            elif isinstance(image_source, bytes):
                # Bytes de imagen
                image_array = np.frombuffer(image_source, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            elif hasattr(image_source, 'read'):
                # Archivo subido (Streamlit)
                image_bytes = image_source.read()
                image_array = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            elif isinstance(image_source, Image.Image):
                # Objeto PIL Image
                image = np.array(image_source)
                if len(image.shape) == 2:  # Convertir escala de grises a RGB
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
            else:
                raise ValueError("Formato de imagen no soportado")
            
            # Guardar copia original para visualización
            original_image = image.copy()
            
            # Realizar predicción
            results = self.model(
                image, 
                verbose=False,
                imgsz=self.config['model']['input_size']
            )
            
            processing_time = (time.time() - start_time) * 1000  # en milisegundos
            
            # Procesar resultados
            predictions = self._process_results(results[0])
            
            return predictions, processing_time, original_image
            
        except Exception as e:
            st.error(f"❌ Error en predicción: {str(e)}")
            return [], 0, None
    
    def _process_results(self, result) -> List[Dict]:
        """Procesa resultados de YOLO para clasificación"""
        predictions = []
        
        if hasattr(result, 'probs'):
            # Para clasificación
            probs = result.probs.data.cpu().numpy()
            
            for idx, prob in enumerate(probs):
                if prob >= self.config['prediction']['confidence_threshold']:
                    predictions.append({
                        'class': self.classes[idx] if idx < len(self.classes) else f"class_{idx}",
                        'class_id': idx,
                        'confidence': float(prob),
                        'percentage': float(prob * 100)
                    })
            
            # Ordenar por confianza descendente
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Limitar a top-k predicciones
            top_k = self.config['prediction']['top_k_predictions']
            predictions = predictions[:top_k]
        
        return predictions
    
    def save_prediction_result(self, image_source, predictions: List[Dict], 
                             processing_time: float, metadata: Dict = None):
        """Guarda resultado de predicción"""
        if metadata is None:
            metadata = {}
        
        # Generar nombre único
        import uuid
        from datetime import datetime
        
        prediction_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        
        # Obtener información del archivo
        filename = getattr(image_source, 'name', 'unknown')
        
        # Crear resultado
        result = {
            'id': prediction_id,
            'timestamp': timestamp,
            'filename': filename,
            'predictions': predictions,
            'processing_time_ms': processing_time,
            'top_prediction': predictions[0] if predictions else None,
            'metadata': metadata
        }
        
        # Guardar en archivo JSON
        result_file = self.predictions_dir / f"pred_{prediction_id}.json"
        
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    
    def get_image_info(self, image_source) -> Dict:
        """Obtiene información de la imagen"""
        try:
            if hasattr(image_source, 'name'):
                # Para archivos subidos en Streamlit
                return {
                    'filename': image_source.name,
                    'size_bytes': image_source.size,
                    'type': image_source.type
                }
            elif isinstance(image_source, str):
                # Para rutas de archivo
                import os
                stats = os.stat(image_source)
                return {
                    'filename': os.path.basename(image_source),
                    'size_bytes': stats.st_size,
                    'modified': stats.st_mtime
                }
        except:
            pass
        
        return {'info': 'No disponible'}
    
    def generate_batch_report(self, results: List[Dict]) -> str:
        """Genera reporte de batch en formato CSV"""
        import pandas as pd
        from datetime import datetime
        
        # Preparar datos
        data = []
        for result in results:
            if isinstance(result, dict):
                top_pred = result.get('top_prediction', {})
                if isinstance(top_pred, dict):
                    data.append({
                        'filename': result.get('filename', 'unknown'),
                        'predicted_class': top_pred.get('class', ''),
                        'confidence': top_pred.get('confidence', 0),
                        'percentage': top_pred.get('percentage', 0),
                        'processing_time_ms': result.get('processing_time', 0)
                    })
        
        # Crear DataFrame
        df = pd.DataFrame(data)
        
        # Guardar CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.predictions_dir / f"batch_report_{timestamp}.csv"
        
        df.to_csv(report_file, index=False)
        
        return str(report_file)