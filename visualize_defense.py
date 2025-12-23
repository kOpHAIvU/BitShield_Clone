import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_dig_cig():
    file_path = 'results/combined_metrics.xlsx'
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Please run attack_with_defense_updated.py first.")
        return

    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return

    # Filter for relevant defenses (ignore OBFUS or others if present)
    target_defenses = ['DIG', 'CIG']
    df_filtered = df[df['Defense Type'].isin(target_defenses)]

    if df_filtered.empty:
        print("No DIG/CIG data found in the Excel file.")
        return

    # Set style to a cleaner, professional look
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(12, 7))

    # Plot Detection Rate vs Attack Strength
    # Group by Defense Type and Strength to handle potential duplicates (take mean)
    df_agg = df_filtered.groupby(['Defense Type', 'Attack Strength'])['Detection Rate'].mean().reset_index()

    # Create Bar Plot
    chart = sns.barplot(
        x='Attack Strength', 
        y='Detection Rate', 
        hue='Defense Type', 
        data=df_agg, 
        palette='Set2',  # Cleaner, distinct colors
        edgecolor='black', # Add border to bars for clarity
        linewidth=1
    )

    plt.title('BitShield Effectiveness: DIG vs CIG', fontsize=20, weight='bold', pad=20)
    plt.xlabel('Attack Intensity (Noise Level)', fontsize=16, labelpad=15)
    plt.ylabel('Detection Rate (%)', fontsize=16, labelpad=15)
    plt.ylim(0, 115) # More space for labels
    
    # Improved Legend
    plt.legend(title='Defense Layer', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Add value labels with better positioning
    for container in chart.containers:
        chart.bar_label(container, fmt='%.1f%%', padding=5, fontsize=12, weight='bold')

    plt.tight_layout() # Adjust layout to prevent clipping
    output_file = 'results/defense_effectiveness.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")

if __name__ == "__main__":
    visualize_dig_cig()
