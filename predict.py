#!/usr/bin/env python3
"""
Script independiente para predicción con el modelo YOLO
Uso: python predict.py --image imagen.jpg --model best.pt
"""

import argparse
import yaml
import sys
from pathlib import Path

# Añadir src al path
sys.path.append(str(Path(__file__).parent / "src"))

from src.model_predictor import ModelPredictor

def main():
    parser = argparse.ArgumentParser(description='Predecir residuos con modelo YOLO entrenado')
    
    parser.add_argument('--image', type=str, required=True, help='Ruta a la imagen')
    parser.add_argument('--model', type=str, default='best.pt', help='Ruta al modelo entrenado')
    parser.add_argument('--conf', type=float, default=0.25, help='Umbral de confianza')
    parser.add_argument('--save', action='store_true', help='Guardar resultados')
    parser.add_argument('--batch', type=str, help='Carpeta con múltiples imágenes')
    
    args = parser.parse_args()
    
    # Cargar configuración
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Actualizar umbral de confianza
    config['prediction']['confidence_threshold'] = args.conf
    
    # Inicializar predictor
    predictor = ModelPredictor(config)
    
    # Cargar modelo
    model_path = Path(config['paths']['trained_models']) / args.model
    if not model_path.exists():
        print(f"❌ Modelo no encontrado en: {model_path}")
        return
    
    predictor.load_model(str(model_path))
    
    if args.batch:
        # Procesar batch de imágenes
        print(f"🔍 Procesando imágenes en: {args.batch}")
        
        import os
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(list(Path(args.batch).glob(f'*{ext}')))
            image_files.extend(list(Path(args.batch).glob(f'*{ext.upper()}')))
        
        print(f"📦 Encontradas {len(image_files)} imágenes")
        
        results = []
        for img_path in image_files:
            try:
                predictions, processing_time, _ = predictor.predict(str(img_path))
                
                if predictions:
                    top_pred = predictions[0]
                    results.append({
                        'image': img_path.name,
                        'prediction': top_pred['class'],
                        'confidence': top_pred['confidence'],
                        'time_ms': processing_time
                    })
                    
                    print(f"✅ {img_path.name}: {top_pred['class']} ({top_pred['confidence']:.1%})")
            
            except Exception as e:
                print(f"❌ Error procesando {img_path.name}: {str(e)}")
        
        # Guardar resultados si se solicita
        if args.save and results:
            import pandas as pd
            df = pd.DataFrame(results)
            output_file = f"predictions_{Path(args.batch).name}.csv"
            df.to_csv(output_file, index=False)
            print(f"📁 Resultados guardados en: {output_file}")
    
    else:
        # Procesar imagen individual
        print(f"🔍 Procesando imagen: {args.image}")
        
        predictions, processing_time, _ = predictor.predict(args.image)
        
        if predictions:
            print("\n🎯 Resultados de la predicción:")
            print("=" * 50)
            
            for i, pred in enumerate(predictions[:5], 1):
                print(f"{i}. {pred['class']:15} - {pred['confidence']:.2%}")
            
            print(f"\n⏱️  Tiempo de procesamiento: {processing_time:.0f} ms")
            
            # Guardar si se solicita
            if args.save:
                predictor.save_prediction_result(args.image, predictions, processing_time)
                print("📁 Resultado guardado")

if __name__ == "__main__":
    main()