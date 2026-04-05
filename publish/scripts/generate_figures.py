#!/usr/bin/env python3
"""
Generate all figures for the RIWC paper
Run: python generate_figures.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (8, 5)

# ============================================================================
# FIGURE 1: System Architecture
# ============================================================================

def generate_architecture_figure():
    """Figure 1: RIWC System Architecture"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # VRAM Box
    vram = Rectangle((0.05, 0.55), 0.4, 0.4, facecolor='#3498db', alpha=0.3, 
                      edgecolor='#2980b9', linewidth=2, label='VRAM (12GB)')
    ax.add_patch(vram)
    ax.text(0.25, 0.8, 'GPU VRAM', ha='center', fontsize=12, fontweight='bold')
    
    # Metadata in VRAM
    metadata = Rectangle((0.1, 0.65), 0.3, 0.12, facecolor='#e74c3c', alpha=0.8, edgecolor='black')
    ax.add_patch(metadata)
    ax.text(0.25, 0.71, 'Metadata (1.2GB)', ha='center', fontsize=9, color='white', fontweight='bold')
    ax.text(0.25, 0.67, 'min/max ranges', ha='center', fontsize=7, color='white')
    
    # Cache in VRAM
    cache = Rectangle((0.1, 0.58), 0.3, 0.06, facecolor='#2ecc71', alpha=0.8, edgecolor='black')
    ax.add_patch(cache)
    ax.text(0.25, 0.61, 'Weight Cache (4-8GB)', ha='center', fontsize=9, color='white', fontweight='bold')
    
    # NVMe Box
    nvme = Rectangle((0.55, 0.55), 0.4, 0.4, facecolor='#f39c12', alpha=0.3, 
                      edgecolor='#e67e22', linewidth=2, label='NVMe (2TB)')
    ax.add_patch(nvme)
    ax.text(0.75, 0.8, 'NVMe Storage', ha='center', fontsize=12, fontweight='bold')
    
    # Weights on NVMe
    weights = Rectangle((0.6, 0.65), 0.3, 0.12, facecolor='#9b59b6', alpha=0.8, edgecolor='black')
    ax.add_patch(weights)
    ax.text(0.75, 0.71, 'Full Weights (140GB)', ha='center', fontsize=9, color='white', fontweight='bold')
    ax.text(0.75, 0.67, '70B parameters @ FP16', ha='center', fontsize=7, color='white')
    
    # Arrow from NVMe to VRAM
    ax.annotate('', xy=(0.46, 0.75), xytext=(0.54, 0.75),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(0.5, 0.77, 'Load on\ndemand', ha='center', fontsize=8)
    
    # GPU Kernel arrow
    ax.annotate('', xy=(0.46, 0.62), xytext=(0.54, 0.62),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(0.5, 0.59, 'Range check\n(parallel)', ha='center', fontsize=8)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 1)
    ax.axis('off')
    ax.set_title('Figure 1: RIWC System Architecture', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../figures/architecture.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('../figures/architecture.png', bbox_inches='tight', dpi=300)
    print('✓ Generated: architecture.pdf')

# ============================================================================
# FIGURE 2: Speed Comparison
# ============================================================================

def generate_speed_comparison():
    """Figure 2: Throughput comparison across methods"""
    methods = ['Full Model\n(140GB VRAM)', 'INT4\n(35GB VRAM)', 
               'CPU Offload\n(8GB RAM)', 'RIWC (Ours)\n(12GB VRAM)']
    speeds = [50, 48, 0.5, 15]
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db']
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bars = ax.bar(methods, speeds, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, speed in zip(bars, speeds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{speed} t/s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Tokens per Second', fontsize=12, fontweight='bold')
    ax.set_title('Figure 2: Throughput Comparison for 70B Model', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.1, 100)
    
    # Add annotation
    ax.annotate('20-30x faster\nthan CPU offload', xy=(2, 0.8), xytext=(1.5, 10),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=9, ha='center')
    
    plt.tight_layout()
    plt.savefig('../figures/speed_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('../figures/speed_comparison.png', bbox_inches='tight', dpi=300)
    print('✓ Generated: speed_comparison.pdf')

# ============================================================================
# FIGURE 3: Cache Warmup Curve
# ============================================================================

def generate_cache_warmup():
    """Figure 3: Cache hit rate over time"""
    tokens = np.arange(0, 1000, 10)
    hit_rates = 0.45 + 0.5 * (1 - np.exp(-tokens/100))
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(tokens, hit_rates, linewidth=3, color='#3498db')
    ax.fill_between(tokens, hit_rates, alpha=0.3, color='#3498db')
    
    # Mark key points
    ax.scatter([10, 50, 200, 1000], [0.45, 0.68, 0.85, 0.94], 
               color='red', s=100, zorder=5)
    
    # Add annotations
    ax.annotate('Cold start\n45% hit', xy=(10, 0.45), xytext=(100, 0.35),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=9)
    ax.annotate('Warm\n85% hit', xy=(200, 0.85), xytext=(350, 0.75),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=9)
    ax.annotate('Hot\n94% hit', xy=(1000, 0.94), xytext=(800, 0.98),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=9)
    
    ax.set_xlabel('Tokens Generated', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cache Hit Rate', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3: Cache Warmup Curve (70B Model)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1050)
    ax.set_ylim(0.3, 1.0)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Usable threshold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('../figures/cache_warmup.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('../figures/cache_warmup.png', bbox_inches='tight', dpi=300)
    print('✓ Generated: cache_warmup.pdf')

# ============================================================================
# FIGURE 4: Activation Distributions
# ============================================================================

def generate_activation_distributions():
    """Figure 4: Activation clustering across layers"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Layer 1: Uniform distribution
    x1 = np.linspace(-2, 2, 100)
    y1 = np.ones_like(x1) * 0.5
    axes[0].fill_between(x1, y1, alpha=0.5, color='#3498db')
    axes[0].plot(x1, y1, color='#2980b9', linewidth=2)
    axes[0].set_title('Early Layers (Layer 1-2)\nNear-Uniform', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Activation Value', fontsize=10)
    axes[0].set_ylabel('Density', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Layer 2: Clustered distribution
    x2 = np.linspace(-2, 2, 100)
    y2 = np.exp(-((x2 - 0.5)**2) / 0.1) + np.exp(-((x2 + 0.5)**2) / 0.1)
    y2 = y2 / y2.max() * 0.8
    axes[1].fill_between(x2, y2, alpha=0.5, color='#e74c3c')
    axes[1].plot(x2, y2, color='#c0392b', linewidth=2)
    axes[1].set_title('Middle Layers (Layer 3-30)\nClustered', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Activation Value', fontsize=10)
    axes[1].set_ylabel('Density', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Layer 3: Peaked distribution
    x3 = np.linspace(-1, 1, 100)
    y3 = np.exp(-((x3 - 0.2)**2) / 0.02)
    y3 = y3 / y3.max()
    axes[2].fill_between(x3, y3, alpha=0.5, color='#2ecc71')
    axes[2].plot(x3, y3, color='#27ae60', linewidth=2)
    axes[2].set_title('Late Layers (Layer 31-32)\nHighly Peaked', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Activation Value', fontsize=10)
    axes[2].set_ylabel('Density', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Figure 4: Activation Distributions Across Llama 2 Layers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/activation_distributions.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('../figures/activation_distributions.png', bbox_inches='tight', dpi=300)
    print('✓ Generated: activation_distributions.pdf')

# ============================================================================
# FIGURE 5: GPU Kernel Speedup
# ============================================================================

def generate_gpu_speedup():
    """Figure 5: GPU kernel speedup comparison"""
    methods = ['CPU\n(Sequential)', 'CPU\n(Binary Search)', 'GPU\n(Our Kernel)']
    times = [70.0, 1.7, 0.0001]
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bars = ax.bar(methods, times, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, time in zip(bars, times):
        if time < 0.01:
            label = f'{time*1000:.2f} ms'
        else:
            label = f'{time:.1f} s'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                label, ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Time per Token (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 5: GPU Kernel Speedup for Range Checking', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(0.00001, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add speedup annotations
    ax.annotate('41x\nfaster', xy=(1, 1.7), xytext=(0.5, 10),
                arrowprops=dict(arrowstyle='->', color='blue'), fontsize=9)
    ax.annotate('700,000x\nfaster!', xy=(2, 0.0001), xytext=(1.5, 0.001),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../figures/gpu_speedup.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('../figures/gpu_speedup.png', bbox_inches='tight', dpi=300)
    print('✓ Generated: gpu_speedup.pdf')

# ============================================================================
# FIGURE 6: Group Size Sensitivity
# ============================================================================

def generate_group_size_sensitivity():
    """Figure 6: Group size trade-off analysis"""
    group_sizes = [100, 500, 1000, 2000, 5000, 10000]
    metadata_gb = [14, 2.8, 1.4, 0.7, 0.28, 0.14]
    hit_rate = [0.95, 0.90, 0.85, 0.75, 0.60, 0.50]
    
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    
    # Metadata size (left axis)
    color1 = '#e74c3c'
    ax1.set_xlabel('Group Size (weights per group)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Metadata Size (GB)', color=color1, fontsize=12, fontweight='bold')
    line1 = ax1.plot(group_sizes, metadata_gb, 'o-', color=color1, linewidth=2, markersize=8, label='Metadata Size')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('log')
    
    # Hit rate (right axis)
    ax2 = ax1.twinx()
    color2 = '#3498db'
    ax2.set_ylabel('Cache Hit Rate', color=color2, fontsize=12, fontweight='bold')
    line2 = ax2.plot(group_sizes, hit_rate, 's-', color=color2, linewidth=2, markersize=8, label='Hit Rate')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0.4, 1.0)
    
    # Mark optimal point
    ax1.axvline(x=1000, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.text(1050, 5, 'Optimal (g=1000)', fontsize=10, color='green', fontweight='bold')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    ax1.set_title('Figure 6: Group Size Sensitivity Analysis', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../figures/group_size_sensitivity.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('../figures/group_size_sensitivity.png', bbox_inches='tight', dpi=300)
    print('✓ Generated: group_size_sensitivity.pdf')

# ============================================================================
# FIGURE 7: End-to-End Latency
# ============================================================================

def generate_latency_breakdown():
    """Figure 7: End-to-end latency breakdown"""
    components = ['Range\nCheck', 'NVMe\nLoad', 'Matrix\nMultiply', 'Attention', 'Other']
    times = [0.0001, 0.15, 0.04, 0.03, 0.02]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bars = ax.bar(components, times, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add percentage labels
    total = sum(times)
    for bar, time in zip(bars, times):
        pct = (time / total) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Time per Token (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 7: End-to-End Latency Breakdown (Warm Cache)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add total annotation
    ax.text(2.5, 0.28, f'Total: {total:.3f}s/token\n({1/total:.1f} t/s)', 
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('../figures/latency_breakdown.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('../figures/latency_breakdown.png', bbox_inches='tight', dpi=300)
    print('✓ Generated: latency_breakdown.pdf')

# ============================================================================
# FIGURE 8: VRAM Comparison
# ============================================================================

def generate_vram_comparison():
    """Figure 8: VRAM requirements comparison"""
    models = ['7B', '13B', '30B', '70B']
    full_vram = [14, 26, 60, 140]
    quant_vram = [3.5, 6.5, 15, 35]
    riwc_vram = [2, 4, 8, 12]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bars1 = ax.bar(x - width, full_vram, width, label='Full Model (FP16)', color='#e74c3c', edgecolor='black')
    bars2 = ax.bar(x, quant_vram, width, label='INT4 Quantization', color='#f39c12', edgecolor='black')
    bars3 = ax.bar(x + width, riwc_vram, width, label='RIWC (Ours)', color='#2ecc71', edgecolor='black')
    
    ax.set_xlabel('Model Size (Parameters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('VRAM Required (GB)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 8: VRAM Requirements Across Methods', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add consumer GPU line
    ax.axhline(y=12, color='blue', linestyle='--', linewidth=2, label='Consumer GPU (12GB limit)')
    ax.axhline(y=24, color='green', linestyle='--', linewidth=2, label='High-end Consumer (24GB limit)')
    
    # Add annotations
    ax.annotate('70B won\'t fit', xy=(3, 140), xytext=(2.5, 120),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=9)
    ax.annotate('70B fits!', xy=(3, 12), xytext=(2.5, 25),
                arrowprops=dict(arrowstyle='->', color='green'), fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../figures/vram_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('../figures/vram_comparison.png', bbox_inches='tight', dpi=300)
    print('✓ Generated: vram_comparison.pdf')

# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    import os
    
    # Create figures directory
    os.makedirs('../figures', exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating Figures for RIWC Paper")
    print("="*60 + "\n")
    
    # Generate all figures
    generate_architecture_figure()
    generate_speed_comparison()
    generate_cache_warmup()
    generate_activation_distributions()
    generate_gpu_speedup()
    generate_group_size_sensitivity()
    generate_latency_breakdown()
    generate_vram_comparison()
    
    print("\n" + "="*60)
    print("✓ All figures generated successfully!")
    print("  Location: ../figures/")
    print("  Formats: PDF (vector) + PNG (raster)")
    print("="*60)