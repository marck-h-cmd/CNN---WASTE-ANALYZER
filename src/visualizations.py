
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from PIL import Image
import io
import streamlit as st

class VisualizationManager:
    """Gestiona visualizaciones para el dashboard"""
    
    def plot_prediction_result(self, image: np.ndarray, predictions: list) -> go.Figure:
        """Crea visualización de resultado de predicción"""
        
        # Crear figura con dos subplots
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.6, 0.4],
            specs=[[{"type": "image"}, {"type": "bar"}]],
            subplot_titles=("Imagen Clasificada", "Probabilidades por Clase")
        )
        
        # Mostrar imagen
        fig.add_trace(
            go.Image(z=image),
            row=1, col=1
        )
        
        # Preparar datos para gráfico de barras
        if predictions:
            df = pd.DataFrame(predictions)
            
            # Limitar a top 8 predicciones
            df = df.head(8).copy()
            
            # Formatear nombres de clases
            df['class_display'] = df['class'].apply(
                lambda x: x.replace('-', ' ').title()
            )
            
            # Gráfico de barras horizontal
            colors = px.colors.sequential.Greens[:len(df)]
            
            fig.add_trace(
                go.Bar(
                    y=df['class_display'],
                    x=df['percentage'],
                    orientation='h',
                    marker_color=colors,
                    text=df['percentage'].apply(lambda x: f'{x:.1f}%'),
                    textposition='auto',
                    hoverinfo='x+y',
                    hovertemplate='<b>%{y}</b><br>Confianza: %{x:.1f}%<extra></extra>'
                ),
                row=1, col=2
            )
        
        # Actualizar layout
        top_pred = predictions[0]['class'].replace('-', ' ').title() if predictions else 'N/A'
        
        fig.update_layout(
            height=400,
            showlegend=False,
            title_text=f"Resultados de Clasificación - Predicción: {top_pred}",
            title_font_size=16,
            margin=dict(t=100, l=20, r=20, b=20)
        )
        
        fig.update_xaxes(
            title_text="Confianza (%)", 
            row=1, col=2,
            range=[0, 105]  # Para mostrar hasta 100% + margen
        )
        
        fig.update_yaxes(
            title_text="Clases", 
            row=1, col=2,
            autorange="reversed"  # Para mostrar la más alta arriba
        )
        
        return fig
    
    def plot_probability_distribution(self, predictions: list) -> go.Figure:
        """Crea gráfico de distribución de probabilidades"""
        if not predictions:
            return go.Figure()
        
        df = pd.DataFrame(predictions)
        
        # Formatear nombres para mostrar
        df['class_display'] = df['class'].apply(
            lambda x: x.replace('-', ' ').title()
        )
        
        # Gráfico de pastel
        fig = px.pie(
            df, 
            values='confidence', 
            names='class_display',
            title='Distribución de Probabilidades',
            color_discrete_sequence=px.colors.sequential.Greens,
            hole=0.4,
            hover_data=['percentage']
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Probabilidad: %{value:.3f}<br>Porcentaje: %{percent}',
            pull=[0.1 if i == 0 else 0 for i in range(len(df))]  # Resaltar predicción principal
        )
        
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(size=10)
            ),
            margin=dict(t=50, b=100)
        )
        
        return fig
    
    def plot_class_distribution_batch(self, df: pd.DataFrame) -> go.Figure:
        """Crea gráfico de distribución de clases para batch"""
        if df.empty:
            return go.Figure()
        
        # Contar frecuencia de cada clase
        class_counts = df['top_prediction'].value_counts().reset_index()
        class_counts.columns = ['class', 'count']
        
        # Formatear nombres
        class_counts['class_display'] = class_counts['class'].apply(
            lambda x: x.replace('-', ' ').title()
        )
        
        # Ordenar por frecuencia
        class_counts = class_counts.sort_values('count', ascending=True)
        
        # Gráfico de barras horizontal
        fig = px.bar(
            class_counts,
            y='class_display',
            x='count',
            orientation='h',
            title='Distribución de Clases en el Batch',
            color='count',
            color_continuous_scale='Greens',
            text='count'
        )
        
        fig.update_layout(
            xaxis_title="Número de Imágenes",
            yaxis_title="Clase",
            coloraxis_showscale=False,
            margin=dict(t=50)
        )
        
        fig.update_traces(
            texttemplate='%{text}',
            textposition='outside'
        )
        
        return fig
    
    def plot_confidence_histogram(self, confidence_data: pd.DataFrame) -> go.Figure:
        """Crea histograma de distribuciones de confianza"""
        if confidence_data.empty:
            return go.Figure()
        
        fig = px.histogram(
            confidence_data,
            x='confidence',
            nbins=20,
            title='Distribución de Niveles de Confianza',
            color_discrete_sequence=['#2E8B57'],
            opacity=0.8,
            marginal="box"  # Añadir box plot en el margen
        )
        
        fig.update_layout(
            xaxis_title="Confianza",
            yaxis_title="Frecuencia",
            bargap=0.1,
            showlegend=False
        )
        
        # Añadir línea vertical para la media
        mean_conf = confidence_data['confidence'].mean()
        fig.add_vline(
            x=mean_conf,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Media: {mean_conf:.2f}",
            annotation_position="top right",
            annotation_font_size=10
        )
        
        # Añadir línea vertical para la mediana
        median_conf = confidence_data['confidence'].median()
        fig.add_vline(
            x=median_conf,
            line_dash="dot",
            line_color="blue",
            annotation_text=f"Mediana: {median_conf:.2f}",
            annotation_position="top left",
            annotation_font_size=10
        )
        
        return fig