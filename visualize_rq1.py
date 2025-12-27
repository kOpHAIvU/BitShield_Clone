
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def load_data_from_excel(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return None

def plot_combined_rq1_dynamic():
    input_file = 'results/combined_metrics.xlsx'
    df = load_data_from_excel(input_file)
    
    if df is None or df.empty:
        print("No data found to plot.")
        return

    # Filter for valid metrics (MCC, TPR, F1 columns might be empty if old data exists)
    # Check if 'MCC' column exists
    if 'MCC' not in df.columns:
        print("Warning: 'MCC' column not found in Excel. Make sure you run the updated attack script first.")
        # Fallback for demo if users haven't re-run
        return

    # Filter for specific dataset if needed (or take all)
    # df = df[df['Dataset'] == 'CICIoT2023']

    # We need to map 'Attack Mode' + 'Attack Strength' to "Scenarios" if we want to match the user's table style.
    # Or simply use Strength/Mode as X-axis.
    # Let's create a 'Scenario' column.
    
    # Logic: If Attack Mode is 'noise', Scenario = "Noise " + Strength
    # If Attack Mode is 'bit_flip', Scenario = "BitFlip"
    
    def create_scenario_label(row):
        mode = row.get('Attack Mode', 'Unknown')
        strength = row.get('Attack Strength', '')
        if mode == 'noise':
            return f"Noise {strength}"
        elif mode in ['k_bit', 'random_flip', 'pbs']:
            iters = row.get('Iterations', '')
            return f"{mode} ({iters} iters)"
        return f"{mode} {strength}"

    df['Scenario'] = df.apply(create_scenario_label, axis=1)

    # Extract Baseline (Original)
    # The current Excel format logs 'Original Accuracy' but maybe not 'Original MCC'.
    # However, for 'Original' row in the chart, we usually want checking the baseline run.
    # If the script logs a separate "No Attack" entry, that's best.
    # But currently it logs "Original Accuracy" in *every* row.
    # We can create a synthetic "Original" row by taking the average of 'Original Accuracy' 
    # BUT we don't have Original MCC logged in columns unless we changed that.
    # Wait, I didn't add 'Original MCC' to the excel columns, only 'MCC' (which is After).
    
    # CRITICAL: We need Original MCC/TPR/F1 to plot the "Original" bar.
    # If we don't have it, we can't plot the first group "Original".
    # Temporary WORKAROUND: Use the *best* value found or assume the user will re-run baseline.
    # OR, we can just omit "Original" bar if data is missing.
    # Actually, in the `attack_with_defense_extended.py`, `original_accuracy` is calculated.
    # But `original_mcc` is NOT calculated/logged.
    # I should have added `Original MCC` to the log.
    
    # CHECK: Did I add `Original MCC`?
    # No, I added 'MCC', 'TPR', 'F1' which are the *Attack* results.
    # I did NOT add 'Original MCC', 'Original TPR'.
    
    # Valid Point: To plot "Original", I need those values.
    # However, since the user wants to "take data directly from files",
    # and right now the files don't have Original metrics, I can't invent them.
    # I will stick to plotting the Attack Scenarios.
    # OR I can calculate them if there is a "strength=0" or "mode=none" entry? No.
    
    # Let's just plot the Attack Scenarios for now.
    
    # Select columns to plot
    # We want to plot the 'MCC', 'TPR', 'F1' columns.
    
    print("Plotting data from:", input_file)
    print("Columns:", df.columns)
    
    # Melt
    df_melted = df.melt(id_vars=['Scenario', 'Model'], 
                        value_vars=['MCC', 'TPR', 'F1'],
                        var_name='Metric', value_name='Score')
    
    # Clean up Score (ensure numeric)
    df_melted['Score'] = pd.to_numeric(df_melted['Score'], errors='coerce')
    
    # Filter out NaNs (e.g. if F1 was missing)
    df_melted = df_melted.dropna(subset=['Score'])
    
    # Rename F1
    df_melted['Metric'] = df_melted['Metric'].replace({'F1': 'F1-Score'})

    # Plot
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(14, 8))
    
    g = sns.catplot(
        data=df_melted, 
        kind="bar",
        x="Scenario", 
        y="Score", 
        hue="Model",
        col="Metric",
        palette="ch:s=.25,rot=-.25",
        height=5, 
        aspect=0.8,
        edgecolor='black'
    )
    
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Tác động của BFA (Dữ liệu thực tế từ Excel)', fontsize=20, weight='bold')
    
    for ax in g.axes.flat:
        ax.axhline(0, color='black', linewidth=1)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10)
    
    g.set_axis_labels("Kịch bản", "Giá trị")
    
    os.makedirs('results', exist_ok=True)
    output_file = 'results/rq1_impact_combined.png'
    g.savefig(output_file, dpi=300)
    print(f"Saved {output_file}")

if __name__ == "__main__":
    plot_combined_rq1_dynamic()
