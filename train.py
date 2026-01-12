#!/usr/bin/env python3
"""
Script independiente para entrenamiento del modelo YOLO
Uso: python train.py --epochs 50 --batch 32 --model yolov8n
"""

import argparse
import yaml
import sys
from pathlib import Path

# Añadir src al path
sys.path.append(str(Path(__file__).parent / "src"))

from src.model_trainer import ModelTrainer

def main():
    parser = argparse.ArgumentParser(description='Entrenar modelo YOLO para clasificación de residuos')
    
    parser.add_argument('--epochs', type=int, default=50, help='Número de épocas')
    parser.add_argument('--batch', type=int, default=32, help='Tamaño del batch')
    parser.add_argument('--model', type=str, default='yolov8n', help='Tamaño del modelo (n, s, m, l, x)')
    parser.add_argument('--lr', type=float, default=0.001, help='Tasa de aprendizaje')
    parser.add_argument('--device', type=str, default='auto', help='Dispositivo (cpu, cuda, auto)')
    parser.add_argument('--name', type=str, default='garbage_classification', help='Nombre del experimento')
    parser.add_argument('--resume', action='store_true', help='Reanudar entrenamiento previo')
    
    args = parser.parse_args()
    
    # Cargar configuración
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("🚀 Iniciando entrenamiento del modelo YOLO...")
    print(f"📋 Configuración: {args.epochs} épocas, batch {args.batch}, modelo {args.model}")
    
    # Actualizar configuración con argumentos
    config['model']['name'] = args.model
    config['training']['epochs'] = args.epochs
    config['training']['batch_size'] = args.batch
    config['training']['learning_rate'] = args.lr
    
    # Inicializar entrenador
    trainer = ModelTrainer(config)
    
    # Entrenar modelo
    results = trainer.train_model(
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        device=args.device,
        experiment_name=args.name,
        resume=args.resume
    )
    
    print(f"\n✅ Entrenamiento completado!")
    print(f"📊 Métricas finales:")
    print(f"   Accuracy: {results.get('metrics', {}).get('accuracy', 0):.2%}")
    print(f"   Tiempo total: {results.get('training_time', 0):.1f} minutos")
    
    # Guardar configuración actualizada
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

if __name__ == "__main__":
    main()