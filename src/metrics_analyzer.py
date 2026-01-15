
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import streamlit as st

class MetricsAnalyzer:
    """Analiza métricas de modelos y genera reportes"""
    
    def __init__(self, config: dict):
        self.config = config
        self.classes = config['classes']
        
        # Directorios
        base_results = Path(config['paths']['results_dir'])
        # El trainer guarda en results/training_logs/, así que ajustamos
        self.results_dir = base_results / 'training_logs'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = Path(config['paths']['trained_models'])
    
    def load_current_model_metrics(self) -> Optional[Dict]:
        """Carga métricas del modelo actual"""
        # Buscar en results/training_logs primero (donde guarda el trainer mejorado)
        if self.results_dir.exists():
            experiment_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
            
            # Buscar archivos de métricas en subdirectorios de experimentos
            metrics_files = []
            for exp_dir in experiment_dirs:
                metrics_files.extend(list(exp_dir.glob('metrics_*.json')))
                metrics_files.extend(list(exp_dir.glob('*metrics.json')))
            
            if metrics_files:
                # Obtener el archivo más reciente
                latest_file = max(metrics_files, key=lambda x: x.stat().st_mtime)
                with open(latest_file, 'r') as f:
                    return json.load(f)
        
        # Buscar en runs/ como fallback
        runs_models = [
            Path('runs/classify/models/trained'),
            Path('runs/train'),
            Path('runs/classify/train')
        ]
        
        # Buscar el directorio del modelo más reciente
        experiment_dirs = []
        for base_dir in runs_models:
            if base_dir.exists():
                experiment_dirs.extend([d for d in base_dir.iterdir() if d.is_dir()])
        
        if experiment_dirs:
            # Buscar archivos de métricas
            metrics_files = []
            for exp_dir in experiment_dirs:
                metrics_files.extend(list(exp_dir.rglob('metrics_*.json')))
                metrics_files.extend(list(exp_dir.rglob('*metrics.json')))
            
            if metrics_files:
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
        # Buscar en results/training_logs
        metrics_files = list(self.results_dir.rglob(f"*{model_name}*metrics*.json"))
        
        # Buscar también en el directorio del experimento específico
        experiment_dir = self.results_dir / model_name
        if experiment_dir.exists():
            metrics_files.extend(list(experiment_dir.glob('metrics_*.json')))
        
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
        
        # Normalizar por filas (por clase real) para obtener porcentajes
        if cm_array.sum() > len(self.classes):  # Si son valores absolutos, no porcentajes
            cm_normalized = cm_array.astype('float') / (cm_array.sum(axis=1)[:, np.newaxis] + 1e-10)
        else:
            cm_normalized = cm_array
        
        cm_normalized = np.nan_to_num(cm_normalized)
        
        # Formatear nombres de clases para mostrar
        class_display = [c.replace('-', ' ').title() for c in self.classes]
        
        # Crear heatmap con anotaciones personalizadas
        fig = go.Figure(data=go.Heatmap(
            z=cm_normalized,
            x=class_display,
            y=class_display,
            colorscale='Greens',
            hovertemplate='Real: %{y}<br>Predicha: %{x}<br>Porcentaje: %{z:.1%}<extra></extra>',
            colorbar=dict(title="Porcentaje")
        ))
        
        # Añadir anotaciones de texto con porcentajes
        for i in range(len(cm_normalized)):
            for j in range(len(cm_normalized[0])):
                fig.add_annotation(
                    text=f"{cm_normalized[i, j]:.1%}",
                    x=class_display[j],
                    y=class_display[i],
                    xref='x',
                    yref='y',
                    showarrow=False,
                    font=dict(
                        color='white' if cm_normalized[i, j] > 0.5 else 'black',
                        size=9
                    )
                )
        
        fig.update_layout(
            title='Matriz de Confusión Normalizada',
            xaxis_title='Clase Predicha',
            yaxis_title='Clase Real',
            xaxis=dict(side='bottom', tickangle=-45),
            yaxis=dict(autorange='reversed'),
            width=900,
            height=800
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
    
    def get_common_errors(self, metrics: Dict = None) -> List[Dict]:
        """Obtiene errores más comunes de clasificación desde la matriz de confusión"""
        if metrics is None:
            metrics = self.load_current_model_metrics()
        
        if not metrics or 'confusion_matrix' not in metrics:
            return []
        
        cm = np.array(metrics['confusion_matrix'])
        errors = []
        
        # Analizar matriz de confusión para encontrar confusiones frecuentes
        for i in range(len(cm)):
            for j in range(len(cm[0])):
                if i != j and cm[i, j] > 0:  # Solo errores (fuera de diagonal)
                    # Calcular porcentaje respecto al total de la clase real
                    total_class = cm[i].sum()
                    percentage = (cm[i, j] / total_class * 100) if total_class > 0 else 0
                    
                    errors.append({
                        'actual': self.classes[i].replace('-', ' ').title(),
                        'predicted': self.classes[j].replace('-', ' ').title(),
                        'count': int(cm[i, j]),
                        'percentage': float(percentage)
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
        
        colors = ['#2E8B57', '#3CB371']
        
        for idx, model in enumerate(models):
            fig.add_trace(go.Bar(
                name=model,
                x=comparison_df['Métrica'],
                y=comparison_df[model],
                text=comparison_df[model].apply(lambda x: f'{x:.3f}'),
                textposition='auto',
                marker_color=colors[idx]
            ))
        
        fig.update_layout(
            title='Comparación de Modelos',
            xaxis_title='Métrica',
            yaxis_title='Valor',
            barmode='group',
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def load_training_history(self, experiment_name: str = None) -> Optional[pd.DataFrame]:
        """Carga historial de entrenamiento desde results.csv de YOLO"""
        # Buscar archivo results.csv en múltiples ubicaciones
        search_paths = [
            Path('runs/classify/models/trained'),
            Path('runs/classify/train'),
            Path('runs/train'),
            Path('models/trained')
        ]
        
        results_files = []
        
        if experiment_name:
            # Buscar en directorio específico del experimento
            for base_path in search_paths:
                exp_path = base_path / experiment_name
                if exp_path.exists():
                    csv_file = exp_path / 'results.csv'
                    if csv_file.exists():
                        results_files.append(csv_file)
        else:
            # Buscar en todas las ubicaciones
            for base_path in search_paths:
                if base_path.exists():
                    results_files.extend(list(base_path.rglob('results.csv')))
        
        if not results_files:
            return None
        
        # Usar el archivo más reciente
        latest_file = max(results_files, key=lambda x: x.stat().st_mtime)
        
        try:
            df = pd.read_csv(latest_file)
            # Limpiar nombres de columnas
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            print(f"Error cargando historial: {e}")
            return None
    
    def plot_training_history(self, experiment_name: str = None) -> go.Figure:
        """Crea gráfico del historial de entrenamiento"""
        df = self.load_training_history(experiment_name)
        
        if df is None or df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No hay datos de entrenamiento disponibles",
                             showarrow=False,
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, font=dict(size=14))
            return fig
        
        # Crear subplots para diferentes métricas
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss de Entrenamiento y Validación', 
                          'Accuracy Top-1',
                          'Accuracy Top-5',
                          'Learning Rate'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Plot 1: Training and Validation Loss
        if 'train/loss' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['train/loss'],
                          name='Train Loss', mode='lines+markers',
                          line=dict(color='#2E8B57', width=2)),
                row=1, col=1
            )
        
        if 'val/loss' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['val/loss'],
                          name='Val Loss', mode='lines+markers',
                          line=dict(color='#FF6B6B', width=2)),
                row=1, col=1
            )
        
        # Plot 2: Top-1 Accuracy
        if 'metrics/accuracy_top1' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['metrics/accuracy_top1'],
                          name='Top-1 Acc', mode='lines+markers',
                          line=dict(color='#3CB371', width=2),
                          fill='tozeroy', fillcolor='rgba(60, 179, 113, 0.1)'),
                row=1, col=2
            )
        
        # Plot 3: Top-5 Accuracy
        if 'metrics/accuracy_top5' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['metrics/accuracy_top5'],
                          name='Top-5 Acc', mode='lines+markers',
                          line=dict(color='#4ECDC4', width=2),
                          fill='tozeroy', fillcolor='rgba(78, 205, 196, 0.1)'),
                row=2, col=1
            )
        
        # Plot 4: Learning Rate
        lr_cols = [col for col in df.columns if 'lr/' in col]
        if lr_cols:
            # Usar solo el primer learning rate group
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df[lr_cols[0]],
                          name='Learning Rate', mode='lines+markers',
                          line=dict(color='#FF6B6B', width=2)),
                row=2, col=2
            )
        
        # Actualizar layout
        fig.update_xaxes(title_text="Época", row=2, col=1)
        fig.update_xaxes(title_text="Época", row=2, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", row=2, col=1)
        fig.update_yaxes(title_text="LR", row=2, col=2, type="log")
        
        fig.update_layout(
            height=700,
            showlegend=True,
            title_text="Historial de Entrenamiento",
            title_font_size=18,
            hovermode='x unified'
        )
        
        return fig
    
    def get_training_summary(self, experiment_name: str = None) -> Dict:
        """Obtiene resumen del entrenamiento"""
        df = self.load_training_history(experiment_name)
        
        if df is None or df.empty:
            return {}
        
        summary = {
            'total_epochs': len(df),
            'best_epoch': int(df['metrics/accuracy_top1'].idxmax() + 1) if 'metrics/accuracy_top1' in df.columns else 0,
            'final_train_loss': float(df['train/loss'].iloc[-1]) if 'train/loss' in df.columns else 0,
            'final_val_loss': float(df['val/loss'].iloc[-1]) if 'val/loss' in df.columns else 0,
            'best_val_acc': float(df['metrics/accuracy_top1'].max()) if 'metrics/accuracy_top1' in df.columns else 0,
            'final_val_acc': float(df['metrics/accuracy_top1'].iloc[-1]) if 'metrics/accuracy_top1' in df.columns else 0,
            'training_time': float(df['time'].iloc[-1]) if 'time' in df.columns else 0
        }
        
        return summary