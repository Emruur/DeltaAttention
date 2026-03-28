import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

# Hardcoded baseline path
BASELINE_PATH = "experiments/experiment_baseline_vibranium"

def load_experiment_data(exp_dir):
    """Crawls an experiment directory and compiles all task JSONs into a DataFrame."""
    print(f"\n[Data Load] Scanning directory: {exp_dir}")
    if not os.path.exists(exp_dir):
        print(f"[Warning] Directory not found: {exp_dir}")
        return pd.DataFrame()

    data = []
    for root, dirs, files in os.walk(exp_dir):
        if os.path.basename(root) == "tasks":
            for file in files:
                if file.endswith('.json'):
                    path = os.path.join(root, file)
                    try:
                        with open(path, 'r') as f:
                            d = json.load(f)
                            
                        # Extract core metrics (Raw values, rounding handled globally later)
                        row = {
                            'task': d.get('task_name', file.replace('.json', '')),
                            'accuracy': d.get('accuracy', 0.0),
                            'sparsity': d.get('sparsity', 0.0),
                            'total_benchmark_time_s': d.get('total_benchmark_time_s', 0.0)
                        }
                        
                        # Extract the correct latency metric
                        timings = d.get('timings', {})
                        # Fallback to root dict just in case it's not nested inside 'timings'
                        row['total_avg_ms'] = timings.get('total_avg_ms', d.get('total_avg_ms', 0.0))
                        
                        # Extract all parameters
                        params = d.get('parameters', {})
                        row.update(params)
                        
                        data.append(row)
                    except Exception as e:
                        print(f"[Error] Failed to read {path}: {e}")
                        
    df = pd.DataFrame(data)
    if not df.empty:
        print(f"[Data Load] Successfully loaded {len(df)} task results.")
    return df

def identify_variables(df):
    """Finds which parameters change across the experiment configs."""
    reserved_cols = ['task', 'accuracy', 'sparsity', 'total_benchmark_time_s', 'total_avg_ms']
    param_cols = [c for c in df.columns if c not in reserved_cols]
    
    variables = []
    for col in param_cols:
        if df[col].nunique() > 1:
            variables.append(col)
    return variables

def build_table_df(exp_df, baseline_df, x_param):
    """Builds the multi-index Pandas table structure (Pivot/Melt logic)."""
    metrics = ['accuracy', 'sparsity', 'total_avg_ms', 'total_benchmark_time_s']
    
    # 1. Calculate Averages for the Experiment runs
    exp_avg = exp_df.groupby([x_param])[metrics].mean().reset_index()
    exp_avg['task'] = 'Average'
    exp_full = pd.concat([exp_df, exp_avg], ignore_index=True)
    
    # 2. Inject Baseline Data (Forced to the top)
    if not baseline_df.empty:
        base_df = baseline_df.copy()
        base_df[x_param] = 'Baseline'
        
        base_avg = base_df[metrics].mean().to_frame().T
        base_avg['task'] = 'Average'
        base_avg[x_param] = 'Baseline'
        
        base_full = pd.concat([base_df, base_avg], ignore_index=True)
        combined = pd.concat([base_full, exp_full], ignore_index=True)
    else:
        combined = exp_full
        
    # 3. Melt the metrics into rows
    melted = combined.melt(id_vars=[x_param, 'task'], value_vars=metrics, var_name='Metric', value_name='Value')
    
    # 4. Pivot tasks to columns
    table_df = melted.pivot_table(index=[x_param, 'Metric'], columns='task', values='Value', aggfunc='first')
    
    # 5. Sorting and Formatting
    unique_vals = [v for v in table_df.index.get_level_values(0).unique() if str(v) != 'Baseline']
    sorted_idx = ['Baseline'] + sorted(unique_vals)
    table_df = table_df.reindex(sorted_idx, level=0)
    
    task_cols = sorted([c for c in table_df.columns if c != 'Average'])
    if 'Average' in table_df.columns:
        task_cols.append('Average')
    table_df = table_df[task_cols]
    
    # GLOBAL ROUNDING: Force exactly 3 decimal points for CSV and Terminal
    return table_df.round(3)

def save_table_as_png(df, title, out_path):
    """Renders a Pandas DataFrame as a high-quality Matplotlib table image."""
    print(f"  -> Rendering PNG image table...")

    # Define human-readable metric names for the image
    metric_map = {
        'accuracy': 'Acc (%)',
        'sparsity': 'Spars (%)',
        'total_avg_ms': 'Attn Fwd (ms)',
        'total_benchmark_time_s': 'Bench (s)'
    }

    render_df = df.reset_index()
    render_df['Metric'] = render_df['Metric'].map(metric_map)

    acc_rows = render_df['Metric'] == 'Acc (%)'
    spars_rows = render_df['Metric'] == 'Spars (%)'
    data_cols = [c for c in render_df.columns if c not in [render_df.columns[0], 'Metric']]

    for col in data_cols:
        # Convert numeric Acc/Spars to percentages for the image
        render_df.loc[acc_rows, col] *= 100
        render_df.loc[spars_rows, col] *= 100
        # Enforce 3 decimal rounding on the percentages
        render_df.loc[acc_rows | spars_rows, col] = render_df.loc[acc_rows | spars_rows, col].apply(lambda x: round(x, 3))

    # Merge duplicate parameter labels for a MultiIndex look in Matplotlib
    p_col = render_df.columns[0]
    last_val = None
    merged_p_col = []
    for val in render_df[p_col]:
        if val == last_val:
            merged_p_col.append('')
        else:
            merged_p_col.append(str(val))
            last_val = val
    render_df[p_col] = merged_p_col

    # Create canvas
    fig, ax = plt.subplots(figsize=(len(render_df.columns)*1.5, len(render_df)*0.3))
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Render table
    tbl = table(ax, render_df, loc='center', cellLoc='center', colWidths=[0.08]*2 + [0.1]*(len(render_df.columns)-2))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    
    cells = tbl.get_celld()
    for (row, col), cell in cells.items():
        cell.set_edgecolor('black')
        
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2f4f4f') 
        elif col == 0:
            cell.set_text_props(weight='bold')
            if cell.get_text().get_text() == 'Baseline':
                cell.set_facecolor('#ffcccc') 
            else:
                cell.set_facecolor('#f0f0f0') 
        elif row > 0:
            param_block_idx = render_df[p_col].iloc[:row].apply(lambda x: 1 if x != '' else 0).sum()
            if param_block_idx % 2 == 0:
                cell.set_facecolor('#e6f3ff') 
            
            if col == len(render_df.columns)-1:
                 cell.set_text_props(weight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [Success] PNG written to: {os.path.basename(out_path)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_folder", type=str, help="Path to the experiment folder")
    args = parser.parse_args()

    print("\n========================================")
    print(" BitNet Sparsity Table Generator v3.1")
    print("========================================")
    
    print("\nSTEP 1: Loading Baseline")
    baseline_df = load_experiment_data(BASELINE_PATH)
    
    print("\nSTEP 2: Loading Experiment Data")
    exp_df = load_experiment_data(args.exp_folder)
    
    if exp_df.empty:
        print("[Fatal] No data found. Exiting.")
        return

    print("\nSTEP 3: Parameter Analysis")
    variables = identify_variables(exp_df)
    print(f"[Analysis] Identified varying parameters: {variables}")

    if len(variables) == 0:
        print("\n[Generation] Single configuration detected (no changing parameters).")
        x_param = "Config"
        exp_df[x_param] = "Single_Run"
        
        final_table_df = build_table_df(exp_df, baseline_df, x_param)
        
        csv_path = os.path.join(args.exp_folder, "table_single_config.csv")
        final_table_df.to_csv(csv_path)
        print(f"\n[Success] CSV written to: {csv_path}")

        png_path = os.path.join(args.exp_folder, "table_single_config.png")
        exp_name = os.path.basename(os.path.normpath(args.exp_folder))
        title = f"{exp_name} | Single Configuration"
        save_table_as_png(final_table_df, title, png_path)
        
        print(f"\nTerminal Preview:\n")
        print(final_table_df.to_string())
        
    elif len(variables) == 1:
        x_param = variables[0]
        print(f"\n[Generation] Single variable detected: {x_param}")
        
        final_table_df = build_table_df(exp_df, baseline_df, x_param)
        
        csv_path = os.path.join(args.exp_folder, f"table_{x_param}.csv")
        final_table_df.to_csv(csv_path)
        print(f"\n[Success] CSV written to: {csv_path}")

        png_path = os.path.join(args.exp_folder, f"table_{x_param}.png")
        exp_name = os.path.basename(os.path.normpath(args.exp_folder))
        title = f"{exp_name} | Y-Axis: {x_param}"
        save_table_as_png(final_table_df, title, png_path)
        
        print(f"\nTerminal Preview:\n")
        print(final_table_df.to_string())
        
    else:
        print(f"\n[Interactive] Multiple parameters detected: {variables}")
        
        while True:
            x_param = input(f"-> Which parameter should group the Rows (Y-Axis)? {variables}: ").strip()
            if x_param in variables: break
            print("Invalid choice.")
            
        remaining_vars = [v for v in variables if v != x_param]
        
        while True:
            page_param = input(f"-> Which parameter splits the tables into different images? {remaining_vars}: ").strip()
            if page_param in remaining_vars: break
            print("Invalid choice.")
            
        print(f"\n[Generation] Generating 'Book' of table PNGs...")
        
        unique_page_vals = sorted(exp_df[page_param].unique().tolist())
        for val in unique_page_vals:
            page_df = exp_df[exp_df[page_param] == val]
            final_table_df = build_table_df(page_df, baseline_df, x_param)
            
            base_filename = f"table_{page_param}_{val}_vs_{x_param}"
            csv_path = os.path.join(args.exp_folder, f"{base_filename}.csv")
            png_path = os.path.join(args.exp_folder, f"{base_filename}.png")
            
            final_table_df.to_csv(csv_path)
            title = f"{os.path.basename(args.exp_folder)} (Fixed {page_param} = {val})"
            save_table_as_png(final_table_df, title, png_path)
            
            print(f"\n--- Preview: {page_param} = {val} ---")
            print(final_table_df.to_string())
            print("\n")

    print("========================================")
    print(" FINISHED. Files saved in experiment folder.")
    print("========================================")

if __name__ == "__main__":
    main()