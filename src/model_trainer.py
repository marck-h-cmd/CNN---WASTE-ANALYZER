import os
import yaml
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable
import numpy as np
import pandas as pd
import torch
#go
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st

# SOLUCIÓN: Deshabilitar handlers de señal para threading
os.environ['YOLO_DISABLE_SIGNAL_HANDLERS'] = '1'
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO

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
        
        # Verificar datos procesados
        print("path", self.config['paths']['data_processed'])
        data_dir = Path(self.config['paths']['data_processed'])

        
        print(f"🚀 Iniciando entrenamiento con:")
        print(f"   Épocas: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Dispositivo: {device}")
        
        # Cargar modelo preentrenado
        model_name = "yolov8n-cls"
        self.model = YOLO(f"{model_name}.pt")
        print(f"📦 Cargando modelo: {model_name}")
        
        try:
            self.model = YOLO(f"{model_name}.pt")
        except Exception as e:
            st.error(f"Error cargando modelo: {e}")
            # Intentar cargar modelo local
            local_model = self.models_dir / f"{model_name}.pt"
            if local_model.exists():
                self.model = YOLO(str(local_model))
            else:
                raise
        
        # Configurar dispositivo
        # Blindaje total de device
        if device not in ['cpu', 'cuda']:
            device = 'cpu'
        
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'

        
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
        start_time = time.time()
        
        try:
            results = self.model.train(**train_args)
            
            training_time = (time.time() - start_time) / 60  # en minutos
            
            # Evaluar modelo
            metrics = self.evaluate_model()
            
            # Guardar resultados
            training_results = {
                'experiment_name': experiment_name,
                'training_time': training_time,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'device': device,
                'metrics': metrics,
                'model_path': str(self.models_dir / experiment_name / 'weights' / 'best.pt')
            }
            
            # Guardar resultados en archivo
            self.save_training_results(training_results)
            
            return training_results
            
        except Exception as e:
            st.error(f"❌ Error durante el entrenamiento: {str(e)}")
            raise
    
    def evaluate_model(self) -> Dict:
        """Evalúa el modelo entrenado"""
        if self.model is None:
            raise ValueError("Modelo no cargado")
        
        # Evaluar en conjunto de validación
        metrics = self.model.val()
        
        # Obtener predicciones para métricas adicionales
        val_results = self.model.predict(
            source=str(Path(self.config['paths']['data_processed']) / 'val'),
            save=False,
            verbose=False
        )
        
        # Extraer labels verdaderas y predicciones
        y_true = []
        y_pred = []
        y_prob = []
        
        for result in val_results:
            if hasattr(result, 'probs'):
                # Para clasificación
                probs = result.probs.data.cpu().numpy()
                pred_class = np.argmax(probs)
                true_class = getattr(result, 'names', None)
                
                # Intentar obtener clase verdadera del nombre del archivo
                if true_class is None:
                    # Inferir de la ruta del archivo
                    path_parts = Path(result.path).parts
                    for part in path_parts:
                        if part in self.classes:
                            true_class = self.classes.index(part)
                            break
                
                if true_class is not None:
                    y_true.append(true_class)
                    y_pred.append(pred_class)
                    y_prob.append(probs)
        
        if y_true and y_pred:
            # Calcular métricas adicionales
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Matriz de confusión
            cm = confusion_matrix(y_true, y_pred)
            
            # Reporte por clase
            class_report = classification_report(
                y_true, y_pred,
                target_names=self.classes,
                output_dict=True,
                zero_division=0
            )
            
            metrics_dict = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'top1_accuracy': float(getattr(metrics, 'top1', 0) / 100),
                'top5_accuracy': float(getattr(metrics, 'top5', 0) / 100),
                'confusion_matrix': cm.tolist(),
                'class_report': class_report,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_prob': [p.tolist() for p in y_prob] if y_prob else []
            }
        else:
            # Métricas básicas de YOLO
            metrics_dict = {
                'top1_accuracy': float(getattr(metrics, 'top1', 0) / 100),
                'top5_accuracy': float(getattr(metrics, 'top5', 0) / 100),
                'accuracy': float(getattr(metrics, 'top1', 0) / 100),
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        
        return metrics_dict
    
    def save_training_results(self, results: Dict):
        """Guarda resultados del entrenamiento"""
        # Guardar en JSON
        results_file = self.results_dir / f"{results['experiment_name']}_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Guardar métricas separadas
        metrics_file = self.results_dir / f"{results['experiment_name']}_metrics.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        
        print(f"💾 Resultados guardados en: {results_file}")
    
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