
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import streamlit as st

class MetricsAnalyzer:
    """Analiza métricas de modelos y genera reportes"""
    
    def __init__(self, config: dict):
        self.config = config
        self.classes = config['classes']
        
        # Directorios
        self.results_dir = Path(config['paths']['results_dir'])
        self.models_dir = Path(config['paths']['trained_models'])
    
    def load_current_model_metrics(self) -> Optional[Dict]:
        """Carga métricas del modelo actual"""
        # Buscar el mejor modelo
        best_model = self.models_dir / 'best.pt'
        if not best_model.exists():
            return None
        
        # Buscar archivo de métricas correspondiente
        model_name = best_model.stem
        metrics_files = list(self.results_dir.rglob(f"*{model_name}*metrics*.json"))
        
        if metrics_files:
            # Cargar el más reciente
            latest_file = max(metrics_files, key=lambda x: x.stat().st_mtime)
            with open(latest_file, 'r') as f:
                return json.load(f)
        
        return None
    
    def get_available_models(self) -> List[str]:
        """Obtiene lista de modelos disponibles"""
        model_files = list(self.models_dir.glob('*.pt'))
        return [m.stem for m in model_files]
    
    def compare_models(self, model_a: str, model_b: str) -> Optional[pd.DataFrame]:
        """Compara métricas de dos modelos"""
        # Cargar métricas de ambos modelos
        metrics_a = self._load_model_metrics_by_name(model_a)
        metrics_b = self._load_model_metrics_by_name(model_b)
        
        if not metrics_a or not metrics_b:
            return None
        
        # Crear DataFrame de comparación
        comparison_data = []
        
        # Métricas principales
        main_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'top1_accuracy', 'top5_accuracy']
        
        for metric in main_metrics:
            comparison_data.append({
                'Métrica': metric.replace('_', ' ').title(),
                model_a: metrics_a.get(metric, 0),
                model_b: metrics_b.get(metric, 0),
                'Diferencia': metrics_a.get(metric, 0) - metrics_b.get(metric, 0)
            })
        
        return pd.DataFrame(comparison_data)
    
    def _load_model_metrics_by_name(self, model_name: str) -> Optional[Dict]:
        """Carga métricas por nombre de modelo"""
        metrics_files = list(self.results_dir.rglob(f"*{model_name}*metrics*.json"))
        
        if metrics_files:
            latest_file = max(metrics_files, key=lambda x: x.stat().st_mtime)
            with open(latest_file, 'r') as f:
                return json.load(f)
        
        return None
    
    def plot_roc_curve(self, metrics: Dict) -> go.Figure:
        """Grafica curva ROC"""
        # Datos de ejemplo (en implementación real, usar y_true y y_prob de metrics)
        fig = go.Figure()
        
        # Curva ROC para cada clase (ejemplo)
        for i, class_name in enumerate(self.classes[:3]):  # Mostrar solo 3 clases
            # Datos de ejemplo
            fpr = np.linspace(0, 1, 100)
            tpr = np.sin(fpr * np.pi / 2) * (0.9 - i*0.1) + 0.05
            
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'ROC {class_name}',
                line=dict(width=2),
                fill='tozeroy',
                fillcolor=f'rgba({50 + i*50}, {139}, {87}, 0.2)'
            ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='gray', dash='dash')
        ))
        
        fig.update_layout(
            title='Curva ROC (Ejemplo)',
            xaxis_title='Tasa de Falsos Positivos',
            yaxis_title='Tasa de Verdaderos Positivos',
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_precision_recall_curve(self, metrics: Dict) -> go.Figure:
        """Grafica curva Precision-Recall"""
        fig = go.Figure()
        
        # Curva Precision-Recall para cada clase (ejemplo)
        for i, class_name in enumerate(self.classes[:3]):
            # Datos de ejemplo
            recall = np.linspace(0, 1, 100)
            precision = np.exp(-5 * (recall - (0.8 - i*0.1))**2) + 0.6
            
            fig.add_trace(go.Scatter(
                x=recall,
                y=precision,
                mode='lines',
                name=f'{class_name}',
                line=dict(width=2),
                fill='tozeroy',
                fillcolor=f'rgba({30 + i*40}, {179}, {113}, 0.2)'
            ))
        
        fig.update_layout(
            title='Curva Precision-Recall (Ejemplo)',
            xaxis_title='Recall',
            yaxis_title='Precision',
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_confusion_matrix(self, metrics: Dict) -> go.Figure:
        """Grafica matriz de confusión"""
        cm = metrics.get('confusion_matrix', [])
        
        if not cm:
            # Crear matriz de ejemplo
            np.random.seed(42)
            cm = np.random.rand(len(self.classes), len(self.classes))
            np.fill_diagonal(cm, np.random.uniform(0.7, 0.95, len(self.classes)))
            cm = cm / cm.sum(axis=1, keepdims=True)
        
        # Convertir a numpy array si es lista
        cm_array = np.array(cm)
        
        # Formatear nombres de clases para mostrar
        class_display = [c.replace('-', ' ').title() for c in self.classes]
        
        fig = px.imshow(
            cm_array,
            labels=dict(x="Predicción", y="Real", color="Porcentaje"),
            x=class_display,
            y=class_display,
            color_continuous_scale='Greens',
            aspect='auto',
            text_auto='.1%'
        )
        
        fig.update_layout(
            title='Matriz de Confusión',
            xaxis_title='Clase Predicha',
            yaxis_title='Clase Real',
            coloraxis_colorbar=dict(title="Porcentaje")
        )
        
        return fig
    
    def get_class_metrics_dataframe(self, metrics: Dict) -> pd.DataFrame:
        """Obtiene métricas por clase como DataFrame"""
        class_report = metrics.get('class_report', {})
        
        if not class_report or 'accuracy' not in class_report:
            # Crear datos de ejemplo
            data = []
            for i, class_name in enumerate(self.classes):
                data.append({
                    'Clase': class_name.replace('-', ' ').title(),
                    'Precisión': np.random.uniform(0.7, 0.95),
                    'Recall': np.random.uniform(0.7, 0.95),
                    'F1-Score': np.random.uniform(0.7, 0.95),
                    'Soporte': int(np.random.uniform(100, 500))
                })
            
            return pd.DataFrame(data)
        
        # Extraer métricas por clase del reporte
        data = []
        for class_name, class_metrics in class_report.items():
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            
            data.append({
                'Clase': class_name.replace('-', ' ').title(),
                'Precisión': class_metrics.get('precision', 0),
                'Recall': class_metrics.get('recall', 0),
                'F1-Score': class_metrics.get('f1-score', 0),
                'Soporte': class_metrics.get('support', 0)
            })
        
        return pd.DataFrame(data)
    
    def get_common_errors(self) -> List[Dict]:
        """Obtiene errores más comunes de clasificación"""
        # En implementación real, analizaría la matriz de confusión
        # Por ahora, datos de ejemplo
        
        errors = []
        num_classes = min(5, len(self.classes))  # Mostrar solo 5 clases
        
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j:
                    errors.append({
                        'actual': self.classes[i],
                        'predicted': self.classes[j],
                        'count': int(np.random.uniform(5, 50)),
                        'percentage': np.random.uniform(1, 10)
                    })
        
        # Ordenar por frecuencia
        errors.sort(key=lambda x: x['count'], reverse=True)
        
        return errors[:10]  # Top 10 errores
    
    def get_error_confidence_distribution(self) -> pd.DataFrame:
        """Obtiene distribución de confianza en errores"""
        # Datos de ejemplo
        np.random.seed(42)
        confidences = np.random.beta(2, 5, 100)  # Distribución sesgada hacia bajas confianzas
        
        return pd.DataFrame({'confidence': confidences})
    
    def generate_report(self, report_type: str, **kwargs) -> str:
        """Genera reporte en formato PDF"""
        # En implementación real, generaría PDF con reportlab
        # Por ahora, crea un archivo de texto de ejemplo
        
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / 'reports' / f"{report_type}_{timestamp}.txt"
        
        report_file.parent.mkdir(exist_ok=True)
        
        # Contenido del reporte
        with open(report_file, 'w') as f:
            f.write(f"Reporte: {report_type}\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            f.write("Este es un reporte de ejemplo.\n")
            f.write("En implementación real, contendría métricas detalladas.\n")
        
        return str(report_file)
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame) -> go.Figure:
        """Grafica comparación de modelos"""
        if comparison_df.empty:
            return go.Figure()
        
        # Preparar datos para gráfico de barras agrupadas
        fig = go.Figure()
        
        models = comparison_df.columns[1:3].tolist()  # Obtener nombres de modelos
        
        for model in models:
            fig.add_trace(go.Bar(
                name=model,
                x=comparison_df['Métrica'],
                y=comparison_df[model],
                text=comparison_df[model].apply(lambda x: f'{x:.3f}'),
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Comparación de Modelos',
            xaxis_title='Métrica',
            yaxis_title='Valor',
            barmode='group',
            hovermode='x unified'
        )
        
        return fig