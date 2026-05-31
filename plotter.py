import os
import json
import argparse
import pandas as pd
import textwrap
import matplotlib.pyplot as plt
from pandas.plotting import table

# Set to None (or pass --baseline) to run without a baseline column
BASELINE_PATH = None

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

                        # Delta-decoding-only metrics (only present in those experiments)
                        if 'avg_kv_compression_pct' in d:
                            row['avg_kv_compression_pct'] = d['avg_kv_compression_pct']
                        breakdown = timings.get('latency_breakdown_ms', {})
                        if 'time_decode_forward_total' in breakdown:
                            row['time_decode_forward_total'] = breakdown['time_decode_forward_total']
                        
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
    reserved_cols = ['task', 'accuracy', 'sparsity', 'total_benchmark_time_s', 'total_avg_ms',
                     'time_decode_forward_total', 'avg_kv_compression_pct']
    param_cols = [c for c in df.columns if c not in reserved_cols]
    
    variables = []
    for col in param_cols:
        non_null = df[col].dropna()
        has_nulls = df[col].isna().any()

        if not has_nulls:
            # No nulls: standard check
            if df[col].nunique() > 1:
                variables.append(col)
            continue

        # Has nulls: only vary if either
        #   (a) non-null values themselves differ (true grid sweep), or
        #   (b) non-null values are all boolean True — null means False/absent flag
        if non_null.nunique() > 1:
            variables.append(col)
        elif len(non_null) > 0 and set(non_null.unique()) <= {True, False}:
            # Boolean flag where null means False — treat as varying
            variables.append(col)
        # else: constant numeric/string absent in some configs (e.g. scale) — suppress
    return variables

def build_table_df(exp_df, baseline_df, x_param):
    """Builds the multi-index Pandas table structure (Pivot/Melt logic)."""
    metrics = ['accuracy', 'sparsity', 'total_avg_ms', 'total_benchmark_time_s']
    for optional in ('time_decode_forward_total', 'avg_kv_compression_pct'):
        if optional in exp_df.columns:
            metrics.append(optional)
    
    # Fill NaN x_param values with "False" so pivot_table doesn't drop them
    exp_df = exp_df.copy()
    if exp_df[x_param].isna().any():
        exp_df[x_param] = exp_df[x_param].fillna(False)

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
    all_level0 = table_df.index.get_level_values(0).unique()
    unique_vals = [v for v in all_level0 if str(v) != 'Baseline']
    has_baseline = 'Baseline' in all_level0
    sorted_idx = (['Baseline'] if has_baseline else []) + sorted(unique_vals)
    table_df = table_df.reindex(sorted_idx, level=0)
    
    task_cols = sorted([c for c in table_df.columns if c != 'Average'])
    if 'Average' in table_df.columns:
        task_cols.append('Average')
    table_df = table_df[task_cols]
    
    # GLOBAL ROUNDING: Force exactly 3 decimal points for CSV and Terminal
    return table_df.round(3)



def save_table_as_png(df, title, out_path):
    """Renders a Pandas DataFrame as a high-quality table image without an internal title."""
    print(f"  -> Rendering PNG image table...")

    # Define human-readable metric names
    metric_map = {
        'accuracy': 'Acc (%)',
        'sparsity': 'Spars (%)',
        'total_avg_ms': 'Attn Fwd (ms)',
        'total_benchmark_time_s': 'Bench (s)',
        'time_decode_forward_total': 'Decode Fwd (ms)',
        'avg_kv_compression_pct': 'KV Compress (%)',
    }

    render_df = df.reset_index()
    render_df['Metric'] = render_df['Metric'].map(metric_map)

    # --- Formatting Logic ---
    acc_rows = render_df['Metric'] == 'Acc (%)'
    spars_rows = render_df['Metric'] == 'Spars (%)'
    data_cols = [c for c in render_df.columns if c not in [render_df.columns[0], 'Metric']]

    for col in data_cols:
        render_df.loc[acc_rows, col] *= 100
        render_df.loc[spars_rows, col] *= 100
        render_df.loc[acc_rows | spars_rows, col] = render_df.loc[acc_rows | spars_rows, col].apply(
            lambda x: round(x, 3) if pd.notnull(x) else x
        )

    # Merge duplicate parameter labels
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

    # --- Header Wrapping ---
    wrapped_columns = [textwrap.fill(str(col), width=15) for col in render_df.columns]

    # --- Canvas Setup ---
    # Width multiplier for columns, height per row
    fig_width = len(render_df.columns) * 2.2
    fig_height = len(render_df) * 0.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    # Title removed per request - filename is sufficient

    # --- Render Table ---
    # Give first two columns (param/metric) fixed widths, distribute the rest
    n_data_cols = max(len(render_df.columns) - 2, 1)
    col_widths = [0.1, 0.1] + [0.8 / n_data_cols] * (len(render_df.columns) - 2)
    
    tbl = table(ax, render_df, loc='center', cellLoc='center', colWidths=col_widths)
    
    # Apply wrapped text to headers
    for i, col_text in enumerate(wrapped_columns):
        tbl.get_celld()[(0, i)].get_text().set_text(col_text)

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    
    # --- Styling ---
    cells = tbl.get_celld()
    for (row, col), cell in cells.items():
        cell.set_edgecolor('black')
        
        if row == 0:
            cell.set_height(0.15) # Tall header for wrapped text
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2f4f4f') 
        else:
            cell.set_height(0.08) # Standard data row height
            
        if col == 0:
            cell.set_text_props(weight='bold')
            if cell.get_text().get_text() == 'Baseline':
                cell.set_facecolor('#ffcccc') 
            else:
                cell.set_facecolor('#f0f0f0') 
        elif row > 0:
            # Row striping
            param_block_idx = render_df[p_col].iloc[:row].apply(lambda x: 1 if x != '' else 0).sum()
            if param_block_idx % 2 == 0:
                cell.set_facecolor('#e6f3ff') 
            
            if col == len(render_df.columns)-1:
                 cell.set_text_props(weight='bold')

    # tight_layout with small padding since title is gone
    plt.tight_layout(pad=1.0)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [Success] PNG written to: {os.path.basename(out_path)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_folder", type=str, help="Path to the experiment folder")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to a baseline experiment folder (overrides BASELINE_PATH)")
    parser.add_argument("--x_param", type=str, default=None,
                        help="Parameter to use as row grouping (skips interactive prompt)")
    parser.add_argument("--page_param", type=str, default=None,
                        help="Parameter to split into separate table images (skips interactive prompt)")
    args = parser.parse_args()

    print("\n========================================")
    print(" BitNet Sparsity Table Generator v3.1")
    print("========================================")

    baseline_path = args.baseline if args.baseline is not None else BASELINE_PATH

    print("\nSTEP 1: Loading Baseline")
    if baseline_path:
        baseline_df = load_experiment_data(baseline_path)
    else:
        print("[Info] No baseline path set — skipping baseline column.")
        baseline_df = pd.DataFrame()
    
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

        if args.x_param and args.x_param in variables:
            x_param = args.x_param
        else:
            while True:
                x_param = input(f"-> Which parameter should group the Rows (Y-Axis)? {variables}: ").strip()
                if x_param in variables: break
                print("Invalid choice.")

        remaining_vars = [v for v in variables if v != x_param]

        if args.page_param and args.page_param in remaining_vars:
            page_param = args.page_param
        else:
            while True:
                page_param = input(f"-> Which parameter splits the tables into different images? {remaining_vars}: ").strip()
                if page_param in remaining_vars: break
                print("Invalid choice.")
            
        print(f"\n[Generation] Generating 'Book' of table PNGs...")

        unique_page_vals = sorted(exp_df[page_param].dropna().unique().tolist())
        has_nan_page = exp_df[page_param].isna().any()
        if has_nan_page:
            unique_page_vals = [None] + unique_page_vals

        for val in unique_page_vals:
            if val is None:
                page_df = exp_df[exp_df[page_param].isna()]
            else:
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