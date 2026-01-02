import pandas as pd
import numpy as np
import os
import json
import pickle
import logging
from datetime import datetime

# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

CHECKPOINT_DIR = 'C:/Users/Anya/master_thesis/output/models_upd_30/checkpoints'
LOG_DIR = 'C:/Users/Anya/master_thesis/output/models_upd_30/logs'

log_file = os.path.join(LOG_DIR, f'integrated_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manages resumable execution"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.state_file = os.path.join(checkpoint_dir, 'pipeline_state.json')
        self.load_state()
    
    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
            logger.info(f"Checkpoint loaded: {list(self.state.keys())}")
        else:
            self.state = {task: False for task in [
                'data_loaded', 'data_prepared', 'features_engineered',
                'sarima_fitted', 'xgboost_fitted', 'lstm_fitted',
                'models_evaluated', 'feature_importance_computed',
                'visualizations_created', 'report_generated'
            ]}
    
    def save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def mark_complete(self, task: str):
        self.state[task] = True
        self.save_state()
        logger.info(f"Task complete: {task}")
    
    def is_complete(self, task: str) -> bool:
        return self.state.get(task, False)
    
    def save_checkpoint(self, name: str, data):
        path = os.path.join(self.checkpoint_dir, f'{name}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Checkpoint saved: {name}")
    
    def load_checkpoint(self, name: str):
        path = os.path.join(self.checkpoint_dir, f'{name}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None


checkpoint_mgr = CheckpointManager(CHECKPOINT_DIR)

def inspect_checkpoint(data, name="train_data"):
    """
    Comprehensive inspection of a loaded checkpoint object.
    Adapts to DataFrame, Dictionary, List, or Numpy Array.
    """
    print(f"n=== INSPECTING CHECKPOINT: {name} ===")
    print(f"Type: {type(data)}")

    # CASE 1: PANDAS DATAFRAME
    if isinstance(data, pd.DataFrame):
        print(f"\n[Shape]: {data.shape} (Rows, Columns)")
        print(f"[Memory Usage]: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\n--- Column Structure & Data Types ---")
        print(data.dtypes)
        
        print("\n--- Missing Values Check ---")
        null_counts = data.isnull().sum()
        if null_counts.sum() > 0:
            print(null_counts[null_counts > 0])
        else:
            print("No missing values found.")

        print("\n--- First 3 Rows ---")
        print(data.head(3))
        

    # CASE 2: DICTIONARY (Common for checkpoints with metadata)
    elif isinstance(data, dict):
        print(f"\n[Keys Found]: {list(data.keys())}")
        for key, value in data.items():
            print(f"\n> Key: '{key}'")
            print(f"  Type: {type(value)}")
            if hasattr(value, 'shape'):
                print(f"  Shape: {value.shape}")
            if isinstance(value, pd.DataFrame):
                print(f"  Columns: {list(value.columns)[:5]} ... (Total {len(value.columns)})")
            elif isinstance(value, (int, float, str)):
                print(f"  Value: {value}")

    # CASE 3: NUMPY ARRAY
    elif isinstance(data, np.ndarray):
        print(f"\n[Shape]: {data.shape}")
        print(f"[Dtype]: {data.dtype}")
        print("\n--- Statistics ---")
        print(f"Mean: {data.mean():.4f}, Std: {data.std():.4f}")

    # CASE 4: LIST/TUPLE
    elif isinstance(data, (list, tuple)):
        print(f"\n[Length]: {len(data)}")
        if len(data) > 0:
            print(f"[First Element Type]: {type(data[0])}")

    print("\n==========================================\n")

# --- MAIN EXECUTION ---
# Ensure your CheckpointManager is available here. 
# If it is defined in another file, import it: from your_module import checkpoint_mgr

try:
    # 1. Load your checkpoint
    # Note: Ensure 'checkpoint_mgr' is initialized in your environment
    train_data = checkpoint_mgr.load_checkpoint('train_data')
    
    # 2. Run inspection
    inspect_checkpoint(train_data)

except NameError:
    print("Error: 'checkpoint_mgr' is not defined. Please initialize your CheckpointManager class first.")
except Exception as e:
    print(f"An error occurred while loading: {e}")