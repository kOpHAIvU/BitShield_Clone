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
    target_defenses = ['DIG', 'CIG', 'Combined (DIG+CIG)']
    df_filtered = df[df['Defense Type'].isin(target_defenses)]

    if df_filtered.empty:
        print("No DIG/CIG data found in the Excel file.")
        return

    # Set style
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Plot Detection Rate vs Attack Strength
    # Group by Defense Type and Strength to handle potential duplicates (take mean)
    df_agg = df_filtered.groupby(['Defense Type', 'Attack Strength'])['Detection Rate'].mean().reset_index()

    chart = sns.barplot(x='Attack Strength', y='Detection Rate', hue='Defense Type', data=df_agg, palette='viridis')

    plt.title('BitShield Component Effectiveness: DIG vs CIG', fontsize=16)
    plt.xlabel('Attack Strength (Noise Level)', fontsize=12)
    plt.ylabel('Detection Rate (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.legend(title='Defense Component')

    # Add value labels
    for container in chart.containers:
        chart.bar_label(container, fmt='%.1f%%', padding=3)

    output_file = 'results/defense_effectiveness.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")

if __name__ == "__main__":
    visualize_dig_cig()
