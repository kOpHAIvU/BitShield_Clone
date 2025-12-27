"""
Script to summarize all defense results into organized Excel file for report.
Creates separate sheets for each dataset with clean tables.
"""

import pandas as pd
import os

def summarize_results():
    file_path = 'results/combined_metrics.xlsx'
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return
    
    print(f"Loaded {len(df)} rows from {file_path}")
    
    # Filter for DIG and CIG
    df_filtered = df[df['Defense Type'].isin(['DIG', 'CIG'])]
    
    if df_filtered.empty:
        print("No DIG/CIG data found")
        return
    
    output_file = 'results/defense_summary_report.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        
        # Get unique datasets
        datasets = df_filtered['Dataset'].unique()
        
        for dataset in datasets:
            df_dataset = df_filtered[df_filtered['Dataset'] == dataset]
            
            # Create summary table: Model x Attack Strength x Defense Type -> Detection Rate
            summary_rows = []
            
            for model in df_dataset['Model'].unique():
                df_model = df_dataset[df_dataset['Model'] == model]
                
                for strength in sorted(df_model['Attack Strength'].unique()):
                    df_strength = df_model[df_model['Attack Strength'] == strength]
                    
                    cig_rate = df_strength[df_strength['Defense Type'] == 'CIG']['Detection Rate'].mean()
                    dig_rate = df_strength[df_strength['Defense Type'] == 'DIG']['Detection Rate'].mean()
                    
                    summary_rows.append({
                        'Model': model,
                        'Attack Strength': strength,
                        'CIG Detection Rate (%)': round(cig_rate, 2) if pd.notna(cig_rate) else 'N/A',
                        'DIG Detection Rate (%)': round(dig_rate, 2) if pd.notna(dig_rate) else 'N/A'
                    })
            
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_excel(writer, sheet_name=dataset[:31], index=False)
            
            print(f"Created sheet: {dataset}")
        
        # Create a combined overview sheet
        overview_rows = []
        for dataset in datasets:
            df_dataset = df_filtered[df_filtered['Dataset'] == dataset]
            for model in df_dataset['Model'].unique():
                df_model = df_dataset[df_dataset['Model'] == model]
                
                cig_avg = df_model[df_model['Defense Type'] == 'CIG']['Detection Rate'].mean()
                dig_avg = df_model[df_model['Defense Type'] == 'DIG']['Detection Rate'].mean()
                
                overview_rows.append({
                    'Dataset': dataset,
                    'Model': model,
                    'Avg CIG Detection Rate (%)': round(cig_avg, 2) if pd.notna(cig_avg) else 'N/A',
                    'Avg DIG Detection Rate (%)': round(dig_avg, 2) if pd.notna(dig_avg) else 'N/A'
                })
        
        overview_df = pd.DataFrame(overview_rows)
        overview_df.to_excel(writer, sheet_name='Overview', index=False)
        print("Created sheet: Overview")
    
    print(f"\n✓ Summary saved to: {output_file}")
    print("\nSheets created:")
    for dataset in datasets:
        print(f"  - {dataset}")
    print("  - Overview")


if __name__ == "__main__":
    summarize_results()
