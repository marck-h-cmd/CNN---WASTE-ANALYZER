
import os
import sys
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

def load_config(config_path: str = 'config.yaml') -> Dict:
    """Carga configuración desde archivo YAML"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        st.error(f"❌ No se encontró el archivo de configuración: {config_path}")
        raise

def save_config(config: Dict, config_path: str = 'config.yaml'):
    """Guarda configuración en archivo YAML"""
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        st.success("✅ Configuración guardada exitosamente!")
    except Exception as e:
        st.error(f"❌ Error guardando configuración: {str(e)}")

def get_project_root() -> Path:
    """Obtiene la ruta raíz del proyecto"""
    return Path(__file__).parent.parent

def create_directory_structure(config: Dict):
    """Crea la estructura de directorios del proyecto"""
    directories = [
        config['paths']['data_raw'],
        config['paths']['data_processed'],
        config['paths']['models_dir'],
        config['paths']['trained_models'],
        config['paths']['pretrained_models'],
        config['paths']['results_dir'],
        config['paths']['uploads_dir'],
        config['paths']['assets_dir'],
        
        # Subdirectorios
        Path(config['paths']['results_dir']) / 'training_logs',
        Path(config['paths']['results_dir']) / 'predictions',
        Path(config['paths']['results_dir']) / 'reports',
        Path(config['paths']['assets_dir']) / 'css',
        Path(config['paths']['assets_dir']) / 'images',
        Path(config['paths']['assets_dir']) / 'icons'
    ]
    
    for directory in directories:
        if isinstance(directory, str):
            directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

def check_system_requirements() -> Dict:
    """Verifica requisitos del sistema"""
    requirements = {
        'python_version': sys.version_info >= (3, 8),
        'torch_available': False,
        'cuda_available': False,
        'gpu_memory': 0,
        'ram_gb': 0
    }
    
    try:
        import torch
        requirements['torch_available'] = True
        requirements['cuda_available'] = torch.cuda.is_available()
        
        if requirements['cuda_available']:
            requirements['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    except ImportError:
        pass
    
    try:
        import psutil
        requirements['ram_gb'] = psutil.virtual_memory().total / (1024**3)
    
    except ImportError:
        pass
    
    return requirements

def format_file_size(bytes: int) -> str:
    """Formatea tamaño de archivo en unidades legibles"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"

def validate_image_file(file_path: Path) -> bool:
    """Valida que un archivo sea una imagen válida"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    if file_path.suffix.lower() not in valid_extensions:
        return False
    
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except:
        return False

def get_class_colors(num_classes: int) -> List[str]:
    """Genera colores únicos para cada clase"""
    import colorsys
    
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        saturation = 0.7
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})')
    
    return colors

def progress_bar(current: int, total: int, description: str = ""):
    """Muestra barra de progreso en Streamlit"""
    progress = current / total if total > 0 else 0
    st.progress(progress, text=description)

def display_class_badges(classes: List[str], columns: int = 4):
    """Muestra badges para las clases"""
    cols = st.columns(columns)
    colors = get_class_colors(len(classes))
    
    for idx, class_name in enumerate(classes):
        with cols[idx % columns]:
            color = colors[idx]
            display_name = class_name.replace('-', ' ').title()
            
            st.markdown(
                f'<div style="background-color: {color}; color: white; '
                f'padding: 0.5rem 1rem; border-radius: 20px; '
                f'text-align: center; margin: 0.2rem; font-weight: 600;">'
                f'{display_name}'
                '</div>',
                unsafe_allow_html=True
            )

def export_to_excel(data: pd.DataFrame, filename: str) -> str:
    """Exporta datos a Excel"""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{filename}_{timestamp}.xlsx"
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        data.to_excel(writer, index=False, sheet_name='Resultados')
    
    return output_file

def load_training_history(model_name: str) -> Optional[Dict]:
    """Carga historial de entrenamiento de un modelo"""
    results_dir = Path('results') / 'training_logs'
    history_files = list(results_dir.glob(f"*{model_name}*results*.json"))
    
    if history_files:
        latest_file = max(history_files, key=lambda x: x.stat().st_mtime)
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    return None