# master_thesis

The Instructions to run the final code are as follows:

1. To make sure all dependencies are in place  - the list is provided in requirements.txt file
pip install -r requirements.txt

2. Main research pipeline to run would be as this:
- C_00_cleaning_pipeline_with_visualizations.py - Data cleaning & outlier detection
- C_00_demand_analysis.py - Demand distribution analysis & tier classification
- C_01_1_hierarchical_clustering.py - Spatiotemporal clustering with K-Means and HDBSCAN
- C_01_2_cleanup_checkpoints.py - Checkpoint management utility (if needed to delete checkpoint and start from scratch)
- C_02_1_forecasting_pipeline.py - SARIMA + XGBoost + LSTM forecasting models