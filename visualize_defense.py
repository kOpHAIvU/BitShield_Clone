import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_dig_cig():
    """Visualize DIG and CIG defense effectiveness from attack_with_defense_extended.py results"""
    file_path = 'results/combined_metrics.xlsx'
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Please run attack_with_defense_extended.py first.")
        return

    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return

    print(f"Loaded {len(df)} rows from {file_path}")
    print(f"Available columns: {df.columns.tolist()}")
    print(f"Available Defense Types: {df['Defense Type'].unique().tolist() if 'Defense Type' in df.columns else 'N/A'}")

    # Filter for DIG and CIG defenses
    target_defenses = ['DIG', 'CIG']
    df_filtered = df[df['Defense Type'].isin(target_defenses)]

    if df_filtered.empty:
        print("No DIG/CIG data found in the Excel file.")
        print("Make sure to run attack_with_defense_extended.py with DIG/CIG modes.")
        return

    # Get unique models and datasets
    models = df_filtered['Model'].unique() if 'Model' in df_filtered.columns else ['Unknown']
    datasets = df_filtered['Dataset'].unique() if 'Dataset' in df_filtered.columns else ['Unknown']

    # Create visualization for each model/dataset combination
    for model in models:
        for dataset in datasets:
            df_subset = df_filtered[(df_filtered['Model'] == model) & (df_filtered['Dataset'] == dataset)]
            
            if df_subset.empty:
                continue

            # Set style
            sns.set_theme(style="whitegrid", context="talk")
            plt.figure(figsize=(14, 8))

            # Group by Defense Type and Attack Strength
            df_agg = df_subset.groupby(['Defense Type', 'Attack Strength'])['Detection Rate'].mean().reset_index()

            # Create Bar Plot
            chart = sns.barplot(
                x='Attack Strength', 
                y='Detection Rate', 
                hue='Defense Type', 
                data=df_agg, 
                palette={'CIG': '#4CAF50', 'DIG': '#FF9800'},
                edgecolor='black',
                linewidth=1
            )

            plt.title(f'BitShield Detection Rate: {model} on {dataset}', fontsize=20, weight='bold', pad=20)
            plt.xlabel('Attack Intensity (Noise Level)', fontsize=16, labelpad=15)
            plt.ylabel('Detection Rate (%)', fontsize=16, labelpad=15)
            plt.ylim(0, 115)
            
            plt.legend(title='Defense Layer', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

            # Add value labels
            for container in chart.containers:
                chart.bar_label(container, fmt='%.1f%%', padding=5, fontsize=11, weight='bold')

            plt.tight_layout()
            
            # Generate descriptive filename
            output_file = f'results/defense_{model}_{dataset}_dig_cig.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {output_file}")
            plt.close()


def visualize_all_defenses():
    """Visualize all defense mechanisms from attack_with_defense_extended.py results"""
    file_path = 'results/combined_metrics.xlsx'
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}.")
        return

    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return

    print(f"\n=== Defense Effectiveness Summary ===")
    print(f"Total experiments: {len(df)}")
    
    if 'Defense Type' in df.columns:
        for defense in df['Defense Type'].unique():
            df_def = df[df['Defense Type'] == defense]
            print(f"\n{defense}:")
            if 'Detection Rate' in df.columns:
                print(f"  Avg Detection Rate: {df_def['Detection Rate'].mean():.2f}%")
            if 'Accuracy After Attack' in df.columns:
                print(f"  Avg Accuracy After Attack: {df_def['Accuracy After Attack'].mean():.2f}%")
            if 'Accuracy Drop' in df.columns:
                print(f"  Avg Accuracy Drop: {df_def['Accuracy Drop'].mean():.2f}%")


if __name__ == "__main__":
    visualize_all_defenses()
    visualize_dig_cig()

