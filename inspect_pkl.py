import joblib
import os
from pathlib import Path

models_dir = 'C:/Users/Anya/master_thesis/output/models_upd_30/models'
model_files = list(Path(models_dir).glob('sarima_model_*.pkl'))

print("\n" + "="*80)
print("SARIMA MODEL INSPECTION")
print("="*80)

for model_file in model_files[:10]:  # Check first 10
    try:
        obj = joblib.load(model_file)
        
        print(f"\nFile: {model_file.name}")
        print(f"  Type: {type(obj)}")
        print(f"  Has forecast method: {hasattr(obj, 'forecast')}")
        print(f"  Attributes: {[a for a in dir(obj) if not a.startswith('_')][:10]}")
        
        # Try to forecast
        try:
            result = obj.forecast(steps=5)
            print(f"  Forecast result type: {type(result)}")
            print(f"  Forecast result: {result}")
        except Exception as e:
            print(f"  Forecast ERROR: {e}")
    
    except Exception as e:
        print(f"\nFile: {model_file.name}")
        print(f"  LOAD ERROR: {e}")

print("\n" + "="*80)