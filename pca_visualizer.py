# ============================================================================
# PCA VISUALIZATION MODULE
# ============================================================================
# Generates PCA analysis visualizations from pipeline results
# Input: PCA model and data from OptimizedLSTMForecaster
# Output:PNG figures
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class PCAVisualizer:
    """Generate comprehensive PCA analysis visualizations"""
    
    def __init__(self, pca_model, X_original_shape, X_reduced_shape, 
                 explained_variance_ratio, output_dir='./thesis_figures/'):
        """
        Parameters:
        -----------
        pca_model : PCA object
            Fitted PCA model from OptimizedLSTMForecaster
        X_original_shape : tuple
            Shape of original feature matrix (n_samples, n_features)
        X_reduced_shape : tuple
            Shape of reduced feature matrix (n_samples, n_components)
        explained_variance_ratio : array
            Explained variance ratio array from PCA
        output_dir : str
            Directory to save figures
        """
        self.pca = pca_model
        self.n_samples_orig, self.n_features_orig = X_original_shape
        self.n_samples_red, self.n_components = X_reduced_shape
        self.explained_variance_ratio = explained_variance_ratio
        self.cumsum_var = np.cumsum(explained_variance_ratio)
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)
        plt.rcParams['font.size'] = 11
        
    def plot_comprehensive_analysis(self, save=True):
        """Create 6-panel comprehensive PCA analysis figure"""
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Scree plot (first 100 components or all if fewer)
        ax1 = plt.subplot(2, 3, 1)
        n_plot = min(100, len(self.explained_variance_ratio))
        ax1.bar(range(1, n_plot + 1), 
                self.explained_variance_ratio[:n_plot],
                alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        ax1.axvline(x=self.n_components, color='red', linestyle='--', 
                   linewidth=2.5, label=f'Selected: {self.n_components} components')
        ax1.set_xlabel('Principal Component Index', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Explained Variance Ratio', fontsize=11, fontweight='bold')
        ax1.set_title('Scree Plot: Variance per Component', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, n_plot + 1])
        
        # 2. Cumulative explained variance
        ax2 = plt.subplot(2, 3, 2)
        n_range = min(500, len(self.cumsum_var))
        ax2.plot(range(1, n_range + 1), self.cumsum_var[:n_range], 
                linewidth=2.5, color='darkgreen', marker='o', 
                markersize=3, markevery=max(1, n_range//20))
        ax2.axhline(y=0.95, color='orange', linestyle='--', linewidth=2, 
                   label='95% Variance')
        ax2.axhline(y=self.cumsum_var[self.n_components - 1], color='red', 
                   linestyle='--', linewidth=2.5, 
                   label=f'{self.n_components} Comps: {self.cumsum_var[self.n_components - 1]:.2%}')
        ax2.axvline(x=self.n_components, color='red', linestyle=':', linewidth=2.5, alpha=0.6)
        ax2.set_xlabel('Number of Components', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Cumulative Explained Variance', fontsize=11, fontweight='bold')
        ax2.set_title('Cumulative Explained Variance', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10, loc='lower right')
        ax2.set_ylim([self.cumsum_var[0] - 0.05, 1.005])
        ax2.grid(True, alpha=0.3)
        
        # 3. Compression ratios
        ax3 = plt.subplot(2, 3, 3)
        comp_values = [24, 48, 96, 168, self.n_components, 512]
        comp_values = [c for c in comp_values if c <= len(self.explained_variance_ratio)]
        comp_ratios = [self.n_features_orig / c for c in comp_values]
        colors = ['lightcoral' if c != self.n_components else 'darkred' for c in comp_values]
        
        bars = ax3.bar(range(len(comp_values)), comp_ratios, 
                       alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
        ax3.set_xticks(range(len(comp_values)))
        ax3.set_xticklabels(comp_values, fontsize=10)
        ax3.set_xlabel('Number of Components', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Compression Ratio', fontsize=11, fontweight='bold')
        ax3.set_title('Compression Ratio by Component Count', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, comp) in enumerate(zip(bars, comp_values)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}x', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
        
        # 4. PCA loadings heatmap (top components, sample of features)
        ax4 = plt.subplot(2, 3, 4)
        n_top_comp = min(20, self.n_components)
        n_sample_feat = min(100, self.n_features_orig)
        
        loadings = self.pca.components_[:n_top_comp, :n_sample_feat]
        im = ax4.imshow(loadings, aspect='auto', cmap='RdBu_r', 
                       vmin=-0.15, vmax=0.15, interpolation='nearest')
        ax4.set_xlabel(f'Original Features (sample: 1-{n_sample_feat})', 
                      fontsize=11, fontweight='bold')
        ax4.set_ylabel('Principal Component', fontsize=11, fontweight='bold')
        ax4.set_title(f'PCA Loadings: Top {n_top_comp} Components', 
                     fontsize=12, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax4, label='Loading Value')
        
        # 5. 2D data projection
        ax5 = plt.subplot(2, 3, 5)
        if self.n_components >= 2:
            pca_2d = PCA(n_components=2)
            # Note: You'll need to fit this on scaled data
            ax5.text(0.5, 0.5, '2D Projection\n(requires X_scaled data)', 
                    ha='center', va='center', transform=ax5.transAxes, fontsize=11)
            ax5.set_xlabel(f'PC1', fontsize=11, fontweight='bold')
            ax5.set_ylabel(f'PC2', fontsize=11, fontweight='bold')
            ax5.set_title('2D PCA Projection', fontsize=12, fontweight='bold')
        
        # 6. Summary statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""
DIMENSIONALITY REDUCTION SUMMARY

Original Space:
  - Features: {self.n_features_orig:,}
  - Samples: {self.n_samples_orig:,}
  
Reduced Space (PCA-{self.n_components}):
  - Components: {self.n_components}
  - Samples: {self.n_samples_red:,}
  
Compression Statistics:
  - Ratio: {self.n_features_orig / self.n_components:.0f}x
  - Variance: {self.cumsum_var[self.n_components - 1]:.2%}
  - Loss: {(1 - self.cumsum_var[self.n_components - 1]):.2%}
  
Information Preserved:
  + Spatio-temporal patterns
  + Seasonal structures
  + Temporal correlations
  + Zone relationships
"""
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', 
                         alpha=0.8, pad=1))
        
        plt.tight_layout()
        
        if save:
            filepath = f"{self.output_dir}01_pca_comprehensive_analysis.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        return fig
    
    def plot_memory_performance(self, save=True):
        """Create memory and performance impact figure"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 2.1 Memory usage comparison
        ax = axes[0, 0]
        # Estimate memory usage (assuming float32)
        mem_original = (self.n_samples_orig * self.n_features_orig * 4) / (1024**3)
        mem_reduced = (self.n_samples_red * self.n_components * 4) / (1024**3)
        
        models = [f'Original\nAll {self.n_features_orig:,} Features',
                 f'PCA-{self.n_components}\n(Selected)']
        memory_gb = [mem_original, mem_reduced]
        colors = ['#d62728', '#2ca02c']
        
        bars = ax.bar(models, memory_gb, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=2)
        ax.set_ylabel('Memory Required (GB)', fontsize=12, fontweight='bold')
        ax.set_title('Memory Usage Comparison', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, mem in zip(bars, memory_gb):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mem:.2f} GB', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')
            if mem < memory_gb[0]:
                reduction = (1 - mem/memory_gb[0]) * 100
                ax.text(bar.get_x() + bar.get_width()/2., height/2,
                       f'↓ {reduction:.1f}%', ha='center', va='center',
                       fontsize=11, color='white', fontweight='bold')
        
        # 2.2 Variance-Compression trade-off
        ax = axes[0, 1]
        
        # Create trade-off curve
        component_range = range(1, min(len(self.cumsum_var) + 1, 1001))
        variance = self.cumsum_var[min(len(self.cumsum_var) - 1, 1000)]
        compression = [self.n_features_orig / c for c in component_range]
        variance_values = self.cumsum_var[:len(component_range)]
        
        ax2 = ax.twinx()
        
        line1 = ax.plot(list(component_range), variance_values, 'o-', 
                       linewidth=3, markersize=4, color='steelblue', 
                       label='Variance Retained', markevery=max(1, len(component_range)//20))
        line2 = ax2.plot(list(component_range), compression, 's--', 
                        linewidth=3, markersize=4, color='darkorange', 
                        label='Compression Ratio', markevery=max(1, len(component_range)//20))
        
        # Highlight selected point
        ax.plot(self.n_components, self.cumsum_var[self.n_components - 1], 
               'D', markersize=12, color='red', markeredgecolor='darkred', 
               markeredgewidth=2.5, label=f'Selected ({self.n_components})', zorder=5)
        
        ax.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
        ax.set_ylabel('Variance Retained', fontsize=12, fontweight='bold', color='steelblue')
        ax2.set_ylabel('Compression Ratio', fontsize=12, fontweight='bold', color='darkorange')
        ax.set_title('Variance-Compression Trade-off', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax2.tick_params(axis='y', labelcolor='darkorange')
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)
        
        # 2.3 Information retention by component ranges
        ax = axes[1, 0]
        
        ranges = [
            (1, 50, 'First 50'),
            (51, 100, '51-100'),
            (101, 150, '101-150'),
            (151, 256, '151-256') if self.n_components >= 256 else (151, self.n_components, f'151-{self.n_components}')
        ]
        
        retention_pcts = []
        labels_range = []
        
        for start, end, label in ranges:
            if start <= len(self.cumsum_var):
                end_idx = min(end, len(self.cumsum_var)) - 1
                start_idx = start - 2
                start_val = self.cumsum_var[start_idx] if start_idx >= 0 else 0
                end_val = self.cumsum_var[end_idx]
                retention = (end_val - start_val) * 100
                retention_pcts.append(retention)
                labels_range.append(label)
        
        colors_range = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(retention_pcts)]
        bars = ax.bar(labels_range, retention_pcts, color=colors_range, alpha=0.8, 
                     edgecolor='black', linewidth=2)
        ax.set_ylabel('Cumulative Variance Added (%)', fontsize=12, fontweight='bold')
        ax.set_title('Variance Contribution by Component Range', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, pct in zip(bars, retention_pcts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2.4 Key metrics summary
        ax = axes[1, 1]
        ax.axis('off')
        
        mem_saved = (1 - mem_reduced / mem_original) * 100
        comp_ratio = self.n_features_orig / self.n_components
        
        metrics_text = f"""
KEY METRICS & JUSTIFICATION

Dimensionality Reduction:
  Original - Reduced: {self.n_features_orig:,} → {self.n_components}
  Compression Ratio: {comp_ratio:.0f}x
  
Information Preservation:
  Variance Retained: {self.cumsum_var[self.n_components - 1]:.2%}
  Signal Preserved: {self.cumsum_var[self.n_components - 1]*100:.1f}%
  Information Loss: {(1-self.cumsum_var[self.n_components - 1])*100:.2f}%
  
Memory Impact:
  Original Size: {mem_original:.2f} GB
  Reduced Size: {mem_reduced:.4f} GB
  Memory Saved: {mem_saved:.1f}%
"""
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=1))
        
        plt.tight_layout()
        
        if save:
            filepath = f"{self.output_dir}02_pca_memory_performance.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        return fig
    
    def plot_variance_distribution(self, save=True):
        """Create variance distribution analysis figure"""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Top 50 components variance
        ax = axes[0]
        n_top = min(50, len(self.explained_variance_ratio))
        top_var = self.explained_variance_ratio[:n_top]
        
        colors_grad = plt.cm.RdYlGn(np.linspace(0.3, 0.9, n_top))
        bars = ax.bar(range(1, n_top + 1), top_var, color=colors_grad, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Component Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Explained Variance', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {n_top} Principal Components Variance Distribution', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Cumulative with annotations
        ax = axes[1]
        n_annot = min(10, self.n_components)
        ax.plot(range(1, n_top + 1), self.cumsum_var[:n_top], 
               linewidth=3, color='darkgreen', marker='o', markersize=5, markevery=5)
        ax.fill_between(range(1, n_top + 1), self.cumsum_var[:n_top], alpha=0.3, color='green')
        
        # Annotate key points
        key_points = [1, 10, 25, 50]
        for kp in key_points:
            if kp <= n_top:
                ax.plot(kp, self.cumsum_var[kp - 1], 'ro', markersize=8)
                ax.annotate(f'{self.cumsum_var[kp - 1]:.1%}', 
                           xy=(kp, self.cumsum_var[kp - 1]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Variance', fontsize=12, fontweight='bold')
        ax.set_title('Cumulative Variance Accumulation (Top 50 Components)', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if save:
            filepath = f"{self.output_dir}03_pca_variance_distribution.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        
        return fig
    
    def generate_all(self):
        """Generate all PCA visualizations"""
        print("\n" + "="*80)
        print("GENERATING PCA ANALYSIS VISUALIZATIONS")
        print("="*80)
        
        self.plot_comprehensive_analysis(save=True)
        self.plot_memory_performance(save=True)
        self.plot_variance_distribution(save=True)
        
        print("\n" + "="*80)
        print("VISUALIZATION SUMMARY")
        print("="*80)
        print(f"\nGenerated 3 comprehensive figures:")
        print(f"  1. 01_pca_comprehensive_analysis.png - 6-panel overview")
        print(f"  2. 02_pca_memory_performance.png - Memory and trade-offs")
        print(f"  3. 03_pca_variance_distribution.png - Variance analysis")
        print(f"\nAll figures saved to: {self.output_dir}")
        print(f"\nKey Statistics:")
        print(f"  Original dimensions: {self.n_features_orig:,}")
        print(f"  Reduced dimensions: {self.n_components}")
        print(f"  Variance retained: {self.cumsum_var[self.n_components - 1]:.2%}")
        print(f"  Compression ratio: {self.n_features_orig / self.n_components:.0f}x")
        print("="*80 + "\n")
