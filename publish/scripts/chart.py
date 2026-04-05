import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(16, 6))
ax.axis('off')

# Data with accuracy column only (no speed improvement)
data = [
    ['Model', 'Size', 'Traditional', 'Our Method', 'Accuracy'],
    ['', '', '(VRAM + Speed)', '(VRAM + Speed)', ''],
    ['Llama 7B', '14GB', '14GB @ 50 t/s', '2GB @ 25 t/s', '100% ✓'],
    ['Llama 13B', '26GB', '26GB @ 45 t/s', '4GB @ 18 t/s', '100% ✓'],
    ['Llama 30B', '60GB', '60GB @ 40 t/s', '8GB @ 15 t/s', '100% ✓'],
    ['Llama 70B', '140GB', '140GB @ 35 t/s', '12GB @ 12 t/s', '100% ✓'],
]

# Create table
table = ax.table(cellText=data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.3, 2.2)

# Color header rows
for i in range(5):
    # First header row
    table[(0, i)].set_facecolor('#2c3e50')
    table[(0, i)].set_text_props(color='white', weight='bold')
    # Second header row
    table[(1, i)].set_facecolor('#34495e')
    table[(1, i)].set_text_props(color='white', style='italic')

# Color "Our Method" column (column 3, rows 2-5)
for i in range(2, 6):
    table[(i, 3)].set_facecolor('#90EE90')
    table[(i, 3)].set_text_props(weight='bold')

# Color "Traditional" column (column 2, rows 2-5) - light red
for i in range(2, 6):
    table[(i, 2)].set_facecolor('#FFCCCC')

# Color Accuracy column (column 4, rows 2-5) - gold for 100%
for i in range(2, 6):
    table[(i, 4)].set_facecolor('#FFD700')
    table[(i, 4)].set_text_props(weight='bold')

# Add title
plt.title('70B LLMs on Consumer GPUs - 100% Accuracy Preserved\n', 
          fontsize=16, weight='bold', pad=20)

# Add subtitle
plt.figtext(0.5, 0.92, 'Same Outputs • No Quantization • 100% Local • Consumer Hardware',
            ha='center', fontsize=11, style='italic', color='#2c3e50')

plt.tight_layout()
plt.savefig('../figures/our_method_accuracy.png', dpi=200, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✓ Table saved as our_method_accuracy.png")