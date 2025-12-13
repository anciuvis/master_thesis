
# ============================================================================
# DEMAND ANALYSIS: Average Hourly Demand per Cluster
# Master Thesis - Vilnius University
# ============================================================================
# This script calculates average demand per hour in each cluster
# to understand data sparsity and inform model selection

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
input_path = 'C:/Users/Anya/master_thesis/output'
output_path = 'C:/Users/Anya/master_thesis/output/analysis'
os.makedirs(output_path, exist_ok=True)

# Load data
print("="*80)
print("DEMAND ANALYSIS: Cluster Statistics")
print("="*80)

data = pd.read_parquet(os.path.join(input_path, 'taxi_data_with_clusters_full.parquet'))
print(f"\nRaw data shape: {data.shape}")
print(f"Date range: {data['tpep_pickup_datetime'].min()} to {data['tpep_pickup_datetime'].max()}")

# Prepare time series data
data['time_period'] = data['tpep_pickup_datetime'].dt.floor('h')
demand = data.groupby(['time_period', 'kmeans_cluster']).size().reset_index(name='demand')

# Pivot to get demand matrix
demand_matrix = demand.pivot(index='time_period', columns='kmeans_cluster', values='demand').fillna(0)

print(f"\nDemand matrix shape: {demand_matrix.shape}")
print(f"  Timesteps (hours): {len(demand_matrix)}")
print(f"  Number of clusters: {demand_matrix.shape[1]}")

# ============================================================================
# STATISTICS BY CLUSTER
# ============================================================================

print("\n" + "="*80)
print("CLUSTER DEMAND STATISTICS")
print("="*80)

# Calculate statistics for each cluster
cluster_stats = pd.DataFrame({
    'cluster_id': demand_matrix.columns,
    'avg_hourly_demand': demand_matrix.mean(),
    'median_hourly_demand': demand_matrix.median(),
    'max_hourly_demand': demand_matrix.max(),
    'min_hourly_demand': demand_matrix.min(),
    'std_hourly_demand': demand_matrix.std(),
    'total_demand': demand_matrix.sum(),
    'sparsity': (demand_matrix == 0).sum() / len(demand_matrix) * 100,  # % of zero hours
    'non_zero_hours': (demand_matrix != 0).sum()
})

cluster_stats = cluster_stats.sort_values('total_demand', ascending=False).reset_index(drop=True)
cluster_stats['rank'] = range(1, len(cluster_stats) + 1)
cluster_stats['cumulative_demand_pct'] = cluster_stats['total_demand'].cumsum() / cluster_stats['total_demand'].sum() * 100

print("\nTop 20 Clusters by Demand:")
print(cluster_stats.head(20)[['rank', 'cluster_id', 'avg_hourly_demand', 'total_demand', 'cumulative_demand_pct', 'sparsity']].to_string(index=False))

print("\n" + "-"*80)
print("Bottom 20 Clusters by Demand:")
print(cluster_stats.tail(20)[['rank', 'cluster_id', 'avg_hourly_demand', 'total_demand', 'cumulative_demand_pct', 'sparsity']].to_string(index=False))

# ============================================================================
# DEMAND TIERS FOR HIERARCHICAL MODELING
# ============================================================================

print("\n" + "="*80)
print("HIERARCHICAL TIER CLASSIFICATION")
print("="*80)

# Find natural breakpoints for tiers
cumsum_pct = cluster_stats['cumulative_demand_pct'].values
# Tier 1: Up to 80% of demand
tier1_idx = np.argmax(cumsum_pct >= 80)
tier1_clusters = cluster_stats.iloc[:tier1_idx+1]['cluster_id'].tolist()

# Tier 2: 80% to 95% of demand
tier2_end_idx = np.argmax(cumsum_pct >= 95)
tier2_clusters = cluster_stats.iloc[tier1_idx+1:tier2_end_idx+1]['cluster_id'].tolist()

# Tier 3: 95% to 100% of demand
tier3_clusters = cluster_stats.iloc[tier2_end_idx+1:]['cluster_id'].tolist()

print(f"\nTIER 1 (High-Demand, Sophisticated Models):")
print(f"  Count: {len(tier1_clusters)} clusters")
print(f"  Demand: {cluster_stats.iloc[tier1_idx]['cumulative_demand_pct']:.1f}% of total")
print(f"  Avg hourly demand per cluster: {cluster_stats.iloc[:tier1_idx+1]['avg_hourly_demand'].mean():.2f} trips/hour")
print(f"  Avg sparsity: {cluster_stats.iloc[:tier1_idx+1]['sparsity'].mean():.1f}% zero hours")
print(f"  Recommendation: ConvLSTM or XGBoost with extensive features")

print(f"\nTIER 2 (Medium-Demand, Medium Complexity):")
print(f"  Count: {len(tier2_clusters)} clusters")
print(f"  Demand: {cluster_stats.iloc[tier2_end_idx]['cumulative_demand_pct'] - cluster_stats.iloc[tier1_idx]['cumulative_demand_pct']:.1f}% of total")
print(f"  Avg hourly demand per cluster: {cluster_stats.iloc[tier1_idx+1:tier2_end_idx+1]['avg_hourly_demand'].mean():.2f} trips/hour")
print(f"  Avg sparsity: {cluster_stats.iloc[tier1_idx+1:tier2_end_idx+1]['sparsity'].mean():.1f}% zero hours")
print(f"  Recommendation: XGBoost with lag features")

print(f"\nTIER 3 (Low-Demand, Simple Models):")
print(f"  Count: {len(tier3_clusters)} clusters")
print(f"  Demand: {100 - cluster_stats.iloc[tier2_end_idx]['cumulative_demand_pct']:.1f}% of total")
print(f"  Avg hourly demand per cluster: {cluster_stats.iloc[tier2_end_idx+1:]['avg_hourly_demand'].mean():.2f} trips/hour")
print(f"  Avg sparsity: {cluster_stats.iloc[tier2_end_idx+1:]['sparsity'].mean():.1f}% zero hours")
print(f"  Recommendation: AutoRegressive AR(7) or Exponential Smoothing")

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Save full statistics
cluster_stats.to_csv(os.path.join(output_path, 'cluster_demand_statistics.csv'), index=False)

# Save tier assignments
tier_assignments = pd.DataFrame({
    'cluster_id': tier1_clusters + tier2_clusters + tier3_clusters,
    'tier': ['Tier 1']*len(tier1_clusters) + ['Tier 2']*len(tier2_clusters) + ['Tier 3']*len(tier3_clusters)
})
tier_assignments.to_csv(os.path.join(output_path, 'cluster_tier_assignments.csv'), index=False)

# Save tier cluster lists for Python
tier_summary = {
    'tier_1_clusters': tier1_clusters,
    'tier_2_clusters': tier2_clusters,
    'tier_3_clusters': tier3_clusters
}

# Save as Python pickle for easy loading
import pickle
with open(os.path.join(output_path, 'tier_clusters.pkl'), 'wb') as f:
    pickle.dump(tier_summary, f)

print(f"\n✓ Saved to {output_path}/cluster_demand_statistics.csv")
print(f"✓ Saved to {output_path}/cluster_tier_assignments.csv")
print(f"✓ Saved to {output_path}/tier_clusters.pkl")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Figure 1: Demand Distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1.1 Top 30 clusters
ax = axes[0, 0]
top_30 = cluster_stats.head(30)
ax.barh(range(len(top_30)), top_30['total_demand'], color='steelblue')
ax.set_yticks(range(len(top_30)))
ax.set_yticklabels(top_30['cluster_id'])
ax.set_xlabel('Total Demand (trips)')
ax.set_title('Top 30 Clusters by Total Demand')
ax.invert_yaxis()

# 1.2 Cumulative demand distribution
ax = axes[0, 1]
ax.plot(range(len(cluster_stats)), cluster_stats['cumulative_demand_pct'], linewidth=2, marker='o', markersize=3)
ax.axhline(80, color='red', linestyle='--', label='Tier 1 boundary (80%)')
ax.axhline(95, color='orange', linestyle='--', label='Tier 2 boundary (95%)')
ax.fill_between(range(len(tier1_clusters)), 0, 100, alpha=0.2, color='green', label='Tier 1')
ax.fill_between(range(len(tier1_clusters), len(tier1_clusters)+len(tier2_clusters)), 0, 100, 
                alpha=0.2, color='yellow', label='Tier 2')
ax.fill_between(range(len(tier1_clusters)+len(tier2_clusters), len(cluster_stats)), 0, 100, 
                alpha=0.2, color='red', label='Tier 3')
ax.set_xlabel('Cluster Rank (by demand)')
ax.set_ylabel('Cumulative Demand (%)')
ax.set_title('Cumulative Demand Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 1.3 Average hourly demand by tier
ax = axes[1, 0]
tier_avg_demand = [
    cluster_stats.iloc[:tier1_idx+1]['avg_hourly_demand'].mean(),
    cluster_stats.iloc[tier1_idx+1:tier2_end_idx+1]['avg_hourly_demand'].mean(),
    cluster_stats.iloc[tier2_end_idx+1:]['avg_hourly_demand'].mean()
]
tier_names = [f'Tier 1\n({len(tier1_clusters)} clusters)', 
              f'Tier 2\n({len(tier2_clusters)} clusters)',
              f'Tier 3\n({len(tier3_clusters)} clusters)']
bars = ax.bar(tier_names, tier_avg_demand, color=['green', 'orange', 'red'])
ax.set_ylabel('Average Hourly Demand (trips/hour)')
ax.set_title('Average Hourly Demand by Tier')
ax.grid(axis='y', alpha=0.3)
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom')

# 1.4 Data sparsity by tier
ax = axes[1, 1]
tier_sparsity = [
    cluster_stats.iloc[:tier1_idx+1]['sparsity'].mean(),
    cluster_stats.iloc[tier1_idx+1:tier2_end_idx+1]['sparsity'].mean(),
    cluster_stats.iloc[tier2_end_idx+1:]['sparsity'].mean()
]
bars = ax.bar(tier_names, tier_sparsity, color=['green', 'orange', 'red'])
ax.set_ylabel('Sparsity (% zero-demand hours)')
ax.set_title('Data Sparsity by Tier')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 100])
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(output_path, 'demand_distribution_analysis.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved demand distribution plot")

# Figure 2: Scatter plot - demand vs sparsity
fig, ax = plt.subplots(figsize=(12, 8))
colors = ['green' if c in tier1_clusters else 'orange' if c in tier2_clusters else 'red' 
          for c in cluster_stats['cluster_id']]
scatter = ax.scatter(cluster_stats['avg_hourly_demand'], cluster_stats['sparsity'], 
                     c=colors, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Average Hourly Demand (trips/hour)')
ax.set_ylabel('Data Sparsity (% zero-demand hours)')
ax.set_title('Cluster Characteristics: Demand vs Sparsity')
ax.grid(True, alpha=0.3)
# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', edgecolor='black', label=f'Tier 1 (n={len(tier1_clusters)})'),
                  Patch(facecolor='orange', edgecolor='black', label=f'Tier 2 (n={len(tier2_clusters)})'),
                  Patch(facecolor='red', edgecolor='black', label=f'Tier 3 (n={len(tier3_clusters)})')]
ax.legend(handles=legend_elements, loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'demand_sparsity_scatter.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved demand-sparsity scatter plot")

# Figure 3: Time series for sample clusters from each tier
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Sample clusters from each tier
sample_tier1 = tier1_clusters[0]
sample_tier2 = tier2_clusters[0] if tier2_clusters else tier1_clusters[1]
sample_tier3 = tier3_clusters[0] if tier3_clusters else tier2_clusters[0]

for ax, cluster, tier_name in zip(axes, [sample_tier1, sample_tier2, sample_tier3],
                                   ['Tier 1 (High-Demand)', 'Tier 2 (Medium-Demand)', 'Tier 3 (Low-Demand)']):
    ts = demand_matrix[cluster]
    ax.plot(ts.index, ts.values, color='steelblue', linewidth=1)
    ax.fill_between(ts.index, 0, ts.values, alpha=0.3, color='steelblue')
    ax.set_ylabel('Hourly Demand')
    ax.set_title(f'{tier_name} - Cluster {cluster}')
    ax.grid(True, alpha=0.3)
    stats = cluster_stats[cluster_stats['cluster_id'] == cluster].iloc[0]
    ax.text(0.02, 0.95, f"Avg: {stats['avg_hourly_demand']:.2f} trips/h | Sparsity: {stats['sparsity']:.1f}%",
            transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[-1].set_xlabel('Date-Time')
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'sample_timeseries_by_tier.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved sample time series plot")

print("\n" + "="*80)
print("DEMAND ANALYSIS COMPLETE")
print("="*80)

print(f"""
SUMMARY FOR YOUR THESIS:

Dataset contains {len(demand_matrix.columns)} clusters with {len(demand_matrix)} hourly observations.

Hierarchical Classification for Modeling:
├─ TIER 1 (High-Demand): {len(tier1_clusters)} clusters, {cluster_stats.iloc[tier1_idx]['cumulative_demand_pct']:.1f}% of trips
│  └─ Suitable for: ConvLSTM, Deep Learning
│
├─ TIER 2 (Medium-Demand): {len(tier2_clusters)} clusters, {cluster_stats.iloc[tier2_end_idx]['cumulative_demand_pct'] - cluster_stats.iloc[tier1_idx]['cumulative_demand_pct']:.1f}% of trips
│  └─ Suitable for: XGBoost, Gradient Boosting
│
└─ TIER 3 (Low-Demand): {len(tier3_clusters)} clusters, {100 - cluster_stats.iloc[tier2_end_idx]['cumulative_demand_pct']:.1f}% of trips
   └─ Suitable for: AutoRegressive (AR), Simple models

Memory Recommendation (32GB RAM):
- Full dataset (all 100 clusters): ~800 MB in memory
- Batch processing: Process 50-100 cluster groups at a time
- Checkpoint saves: Every tier completion or every 24 hours of processing
""")