"""
Script to generate SimpleCNNIoT architecture diagram matching Chapter 3 description.
Run: python generate_simplecnniot_diagram.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def draw_simplecnniot_architecture():
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Colors
    input_color = '#FFFFFF'
    conv_color = '#B3D9FF'  # Light blue
    pool_color = '#FFE4B3'  # Light orange
    classifier_color = '#E6D9F2'  # Light purple
    stats_color = '#F5F5F5'  # Light gray
    
    box_width = 8
    box_height = 1.2
    x_center = 5
    
    boxes = [
        (15, "Input (69 features)", input_color),
        (13.2, "Conv1d (6 channels, k=3, s=2) + ReLU\n+ Adaptive MaxPool + Dropout (0.1-0.15)", conv_color),
        (11.2, "Conv1d (16 channels, k=3, s=2) + ReLU\n+ Adaptive MaxPool + Dropout (0.1-0.15)", conv_color),
        (9.2, "Conv1d (16 channels, k=3, s=2) + ReLU\n+ Adaptive MaxPool + Dropout (0.1-0.15)", conv_color),
        (7.2, "Conv1d (16 channels, k=3, s=2) + ReLU\n+ Adaptive MaxPool + Dropout (0.1-0.15)", conv_color),
        (5.4, "Global Avg Pooling", pool_color),
        (3.8, "Quantized Classifier\n(100 → num_classes)", classifier_color),
        (1.8, "Model Stats: ~80K Params, ~3ms Inf., ~400KB Mem. (Quantized)", stats_color),
    ]
    
    # Draw boxes
    for y, text, color in boxes:
        height = 1.0 if '\n' not in text else 1.4
        rect = mpatches.FancyBboxPatch(
            (x_center - box_width/2, y - height/2),
            box_width, height,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor=color,
            edgecolor='#333333',
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x_center, y, text, ha='center', va='center', 
                fontsize=10, fontweight='normal', wrap=True)
    
    # Draw arrows
    arrow_ys = [14.3, 12.4, 10.4, 8.4, 6.6, 5.0, 3.2]
    for y in arrow_ys:
        ax.annotate('', xy=(x_center, y - 0.3), xytext=(x_center, y + 0.3),
                    arrowprops=dict(arrowstyle='->', color='#666666', lw=2))
    
    # Title
    ax.text(x_center, 15.8, "SimpleCNNIoT Architecture", 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = 'docs/simplecnniot_architecture.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Diagram saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    draw_simplecnniot_architecture()
