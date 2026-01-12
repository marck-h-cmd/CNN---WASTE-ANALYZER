
import os
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import streamlit as st

class DataPreparer:
    """Prepara el dataset Garbage Classification para YOLO"""
    
    def __init__(self, config: dict):
        self.config = config
        self.raw_path = Path(config['paths']['data_raw'])
        self.processed_path = Path(config['paths']['data_processed'])
        self.classes = config['classes']
        
    def validate_dataset_structure(self) -> bool:
        """Valida que el dataset tenga la estructura esperada"""
        if not self.raw_path.exists():
            st.error(f"❌ No se encuentra el dataset en: {self.raw_path}")
            return False
        
        # Verificar que todas las carpetas de clases existan
        missing_classes = []
        for class_name in self.classes:
            class_path = self.raw_path / class_name
            if not class_path.exists():
                missing_classes.append(class_name)
        
        if missing_classes:
            st.warning(f"⚠️ Faltan carpetas para clases: {missing_classes}")
            return False
        
        return True
    
    def count_total_images(self) -> int:
        """Cuenta el total de imágenes en el dataset"""
        total = 0
        for class_name in self.classes:
            class_path = self.raw_path / class_name
            if class_path.exists():
                images = [f for f in class_path.iterdir() 
                         if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                total += len(images)
        return total
    
    def prepare_yolo_dataset(self, validation_split: float = 0.2, 
                           random_seed: int = 42) -> Dict:
        """Prepara el dataset en formato YOLO para clasificación"""
        
        # Validar dataset
        if not self.validate_dataset_structure():
            raise ValueError("Estructura del dataset inválida")
        
        # Crear directorios
        train_dir = self.processed_path / 'train'
        val_dir = self.processed_path / 'val'
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {}
        
        for class_name in self.classes:
            st.info(f"Procesando clase: {class_name}")
            
            # Ruta a las imágenes originales
            src_class_dir = self.raw_path / class_name
            
            # Listar imágenes
            image_files = list(src_class_dir.glob('*.*'))
            image_files = [f for f in image_files 
                          if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            
            if not image_files:
                st.warning(f"No hay imágenes para la clase {class_name}")
                continue
            
            # Dividir en train/val
            train_files, val_files = train_test_split(
                image_files, 
                test_size=validation_split, 
                random_state=random_seed,
                shuffle=True
            )
            
            # Crear directorios para la clase
            train_class_dir = train_dir / class_name
            val_class_dir = val_dir / class_name
            
            train_class_dir.mkdir(exist_ok=True)
            val_class_dir.mkdir(exist_ok=True)
            
            # Copiar imágenes a train
            for img_file in train_files:
                dst_path = train_class_dir / img_file.name
                shutil.copy2(img_file, dst_path)
            
            # Copiar imágenes a val
            for img_file in val_files:
                dst_path = val_class_dir / img_file.name
                shutil.copy2(img_file, dst_path)
            
            # Guardar estadísticas
            stats[class_name] = {
                'total': len(image_files),
                'train': len(train_files),
                'val': len(val_files),
                'train_ratio': len(train_files) / len(image_files),
                'val_ratio': len(val_files) / len(image_files)
            }
        
        # Crear archivo YAML para YOLO
        self._create_yolo_yaml()
        
        # Generar reporte de estadísticas
        report = self._generate_statistics_report(stats)
        
        return report
    
    def _create_yolo_yaml(self):
        """Crea archivo YAML de configuración para YOLO"""
        yaml_content = {
            'path': str(self.processed_path.absolute()),
            'train': 'train',
            'val': 'val',
            'nc': len(self.classes),
            'names': self.classes
        }
        
        yaml_file = self.processed_path / 'dataset.yaml'
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
    
    def _generate_statistics_report(self, stats: Dict) -> Dict:
        """Genera reporte estadístico del dataset"""
        
        total_images = sum(s['total'] for s in stats.values())
        total_train = sum(s['train'] for s in stats.values())
        total_val = sum(s['val'] for s in stats.values())
        
        # Calcular balance
        class_counts = [s['total'] for s in stats.values()]
        min_count = min(class_counts)
        max_count = max(class_counts)
        balance_ratio = min_count / max_count if max_count > 0 else 0
        
        report = {
            'total_images': total_images,
            'train_images': total_train,
            'val_images': total_val,
            'train_val_ratio': total_val / total_images if total_images > 0 else 0,
            'num_classes': len(self.classes),
            'class_distribution': stats,
            'balance_ratio': balance_ratio,
            'is_balanced': balance_ratio > 0.7,
            'min_images_per_class': min_count,
            'max_images_per_class': max_count,
            'avg_images_per_class': total_images / len(self.classes) if self.classes else 0
        }
        
        # Guardar reporte
        report_file = self.processed_path / 'dataset_report.json'
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def get_statistics_dataframe(self) -> pd.DataFrame:
        """Obtiene estadísticas como DataFrame"""
        report_file = self.processed_path / 'dataset_report.json'
        
        if not report_file.exists():
            return pd.DataFrame()
        
        import json
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        # Crear DataFrame de distribución por clase
        class_stats = []
        for class_name, stats in report['class_distribution'].items():
            class_stats.append({
                'Clase': class_name,
                'Total': stats['total'],
                'Train': stats['train'],
                'Val': stats['val'],
                '% Train': f"{stats['train_ratio']:.1%}",
                '% Val': f"{stats['val_ratio']:.1%}"
            })
        
        return pd.DataFrame(class_stats)
    
    def plot_class_distribution(self) -> go.Figure:
        """Crea gráfico de distribución de clases"""
        df = self.get_statistics_dataframe()
        
        if df.empty:
            # Crear gráfico vacío
            fig = go.Figure()
            fig.update_layout(title="No hay datos disponibles")
            return fig
        
        # Gráfico de barras
        fig = px.bar(
            df,
            x='Clase',
            y=['Train', 'Val'],
            title='Distribución de Imágenes por Clase',
            labels={'value': 'Número de Imágenes', 'variable': 'Conjunto'},
            color_discrete_sequence=['#2E8B57', '#3CB371'],
            barmode='group'
        )
        
        fig.update_layout(
            xaxis_title="Clase",
            yaxis_title="Número de Imágenes",
            legend_title="Conjunto",
            hovermode='x unified'
        )
        
        return fig
    
    def get_sample_images(self, class_name: str, num_samples: int = 6) -> List[Path]:
        """Obtiene imágenes de muestra de una clase específica"""
        class_path = self.raw_path / class_name
        
        if not class_path.exists():
            return []
        
        # Obtener todas las imágenes
        all_images = list(class_path.glob('*.*'))
        all_images = [img for img in all_images 
                     if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        # Seleccionar muestra aleatoria
        if len(all_images) <= num_samples:
            return all_images
        else:
            return random.sample(all_images, num_samples)
    
    def generate_statistics_report(self) -> Dict:
        """Genera reporte estadístico completo"""
        report_file = self.processed_path / 'dataset_report.json'
        
        if not report_file.exists():
            # Procesar dataset si no existe
            return self.prepare_yolo_dataset()
        
        import json
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        return report