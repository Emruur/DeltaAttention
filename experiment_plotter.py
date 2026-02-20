import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np

# ==========================================
# 1. Data Loading & Parameter Detection
# ==========================================
def load_and_detect_data(experiment_dir):
    """
    Scans the experiment directory, detects ALL parameters in config_params.json,
    and returns a flattened DataFrame plus a list of parameters that actually vary.
    """
    if not os.path.exists(experiment_dir):
        print(f"[Error] Directory not found: {experiment_dir}")
        return pd.DataFrame(), []

    print(f"[System] Scanning {experiment_dir}...")

    # 1. Walk through config folders
    config_dirs = [d for d in sorted(os.listdir(experiment_dir)) if d.startswith("config_")]
    
    all_param_keys = set()
    temp_records = []

    for item in config_dirs:
        config_path = os.path.join(experiment_dir, item)
        if not os.path.isdir(config_path): continue
            
        # A. Read Configuration Parameters
        params_file = os.path.join(config_path, "config_params.json")
        if not os.path.exists(params_file): continue
        
        try:
            with open(params_file, 'r') as f: 
                params = json.load(f)
        except Exception as e:
            print(f"[Warning] Skipped {params_file}: {e}")
            continue

        all_param_keys.update(params.keys())

        # B. Read Task Results
        tasks_dir = os.path.join(config_path, "tasks")
        if not os.path.exists(tasks_dir): continue

        for task_file in os.listdir(tasks_dir):
            if task_file.endswith(".json"):
                try:
                    with open(os.path.join(tasks_dir, task_file), 'r') as f:
                        res = json.load(f)
                    
                    record = {
                        "Config_ID": item,
                        "Accuracy": res.get("accuracy", 0.0),
                        "Sparsity": res.get("sparsity", 0.0)
                    }
                    record.update(params)
                    temp_records.append(record)
                except:
                    pass

    df = pd.DataFrame(temp_records)
    if df.empty: return df, []

    # 2. Identify Varying Parameters
    potential_params = list(all_param_keys)
    varying_params = []

    for col in potential_params:
        if col in df.columns:
            unique_vals = df[col].unique()
            if len(unique_vals) > 1:
                varying_params.append(col)

    print(f"[System] Loaded {len(df)} runs.")
    print(f"[System] Detected varying parameters: {varying_params}")
    
    return df, varying_params

# ==========================================
# 2. Plotting Logic
# ==========================================
def draw_split_cell(ax, x, y, width, height, acc_val, spar_val, norm_acc, cmap_acc, norm_spar, cmap_spar):
    """
    Draws a single rectangular cell split diagonally.
    """
    c_acc = cmap_acc(norm_acc(acc_val))
    c_spar = cmap_spar(norm_spar(spar_val))

    x0, y0 = x, y
    x1, y1 = x + width, y + height

    # Triangle 1 (Bottom-Left): Accuracy
    tri_acc = patches.Polygon([(x0, y0), (x1, y0), (x0, y1)], closed=True, color=c_acc, ec='white', lw=0.5)
    ax.add_patch(tri_acc)
    
    # Triangle 2 (Top-Right): Sparsity
    tri_spar = patches.Polygon([(x0, y1), (x1, y1), (x1, y0)], closed=True, color=c_spar, ec='white', lw=0.5)
    ax.add_patch(tri_spar)

    # Text Labels
    txt_c_acc = 'white' if np.mean(c_acc[:3]) < 0.5 else 'black'
    txt_c_spar = 'white' if np.mean(c_spar[:3]) < 0.5 else 'black'
    
    fs = 10 
    ax.text(x0 + width*0.25, y0 + height*0.3, f"{acc_val:.2f}", ha='center', va='center', color=txt_c_acc, fontsize=fs, fontweight='bold')
    ax.text(x0 + width*0.75, y0 + height*0.7, f"{spar_val:.2f}", ha='center', va='center', color=txt_c_spar, fontsize=fs, fontweight='bold')

def plot_1d_column(df, param_name, output_path):
    """
    Generates a vertical column of results for a single varying parameter.
    """
    # Aggregate and Sort
    summary = df.groupby(param_name)[['Accuracy', 'Sparsity']].mean().reset_index()
    summary = summary.sort_values(param_name)

    n_rows = len(summary)
    
    # Normalization
    norm_acc = mcolors.Normalize(vmin=summary['Accuracy'].min(), vmax=summary['Accuracy'].max())
    norm_spar = mcolors.Normalize(vmin=summary['Sparsity'].min(), vmax=summary['Sparsity'].max())
    cmap_acc = plt.get_cmap('viridis')
    cmap_spar = plt.get_cmap('plasma')

    # Figure Setup: Narrow width, height scales with number of items
    fig, ax = plt.subplots(figsize=(4, max(4, n_rows * 1.2)))
    ax.set_aspect('equal')

    # Draw Cells (Vertically stacked)
    # We iterate visually top-down, but coordinates are bottom-up
    y_labels = summary[param_name].tolist()
    
    for idx, (i, row) in enumerate(summary.iterrows()):
        acc = row['Accuracy']
        spar = row['Sparsity']
        
        # y coordinate: n_rows - 1 - idx puts the first item at the top
        draw_split_cell(ax, 0, n_rows - 1 - idx, 1, 1, acc, spar, norm_acc, cmap_acc, norm_spar, cmap_spar)

    # Axes Formatting
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows)
    
    ax.set_xticks([]) # Remove X ticks
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(y_labels[::-1], fontsize=11) # Reverse to match visual top-down
    
    ax.set_ylabel(param_name, fontsize=12, fontweight='bold')
    ax.set_title(f"Trade-off Summary", fontsize=14, pad=15)

    # Clean look
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(length=0)

    # Colorbars (Vertical placement for this narrow plot usually looks better, but keeping horizontal for consistency)
    cbar_ax_acc = fig.add_axes([0.15, 0.05, 0.7, 0.02]) 
    cb_acc = fig.colorbar(cm.ScalarMappable(norm=norm_acc, cmap=cmap_acc), cax=cbar_ax_acc, orientation='horizontal')
    cb_acc.set_label('Accuracy (Bottom-Left)', color='teal', fontsize=9, fontweight='bold')
    
    cbar_ax_spar = fig.add_axes([0.15, 0.01, 0.7, 0.02])
    cb_spar = fig.colorbar(cm.ScalarMappable(norm=norm_spar, cmap=cmap_spar), cax=cbar_ax_spar, orientation='horizontal')
    cb_spar.set_label('Sparsity (Top-Right)', color='purple', fontsize=9, fontweight='bold')

    plt.subplots_adjust(bottom=0.2, top=0.9, left=0.3) # More left margin for Y labels
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Output] Saved: {os.path.basename(output_path)}")

def plot_heatmap(df, x_param, y_param, title_suffix, output_path, global_norms=None):
    """
    Generates a single heatmap for 2 parameters (X and Y).
    """
    summary = df.groupby([x_param, y_param])[['Accuracy', 'Sparsity']].mean().reset_index()
    pivot_acc = summary.pivot(index=y_param, columns=x_param, values="Accuracy")
    pivot_spar = summary.pivot(index=y_param, columns=x_param, values="Sparsity")
    
    pivot_acc = pivot_acc.sort_index(ascending=True).sort_index(axis=1, ascending=True)
    pivot_spar = pivot_spar.sort_index(ascending=True).sort_index(axis=1, ascending=True)

    rows, cols = pivot_acc.shape

    if global_norms:
        norm_acc, norm_spar = global_norms
    else:
        norm_acc = mcolors.Normalize(vmin=summary['Accuracy'].min(), vmax=summary['Accuracy'].max())
        norm_spar = mcolors.Normalize(vmin=summary['Sparsity'].min(), vmax=summary['Sparsity'].max())

    cmap_acc = plt.get_cmap('viridis')
    cmap_spar = plt.get_cmap('plasma')

    fig, ax = plt.subplots(figsize=(max(6, cols*1.5), max(5, rows*1.5)))
    ax.set_aspect('equal')

    y_labels = pivot_acc.index.tolist()
    x_labels = pivot_acc.columns.tolist()

    for r_idx, r_label in enumerate(y_labels):
        for c_idx, c_label in enumerate(x_labels):
            acc = pivot_acc.loc[r_label, c_label]
            spar = pivot_spar.loc[r_label, c_label]
            if pd.isna(acc) or pd.isna(spar): continue
            draw_split_cell(ax, c_idx, rows - 1 - r_idx, 1, 1, acc, spar, norm_acc, cmap_acc, norm_spar, cmap_spar)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(cols) + 0.5)
    ax.set_yticks(np.arange(rows) + 0.5)
    ax.set_xticklabels(x_labels, fontsize=11, rotation=45 if len(str(x_labels[0])) > 5 else 0)
    ax.set_yticklabels(y_labels[::-1], fontsize=11)
    ax.set_xlabel(x_param, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_param, fontsize=12, fontweight='bold')
    ax.set_title(f"Trade-off Summary{title_suffix}", fontsize=14, pad=15)

    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(length=0)

    cbar_ax_acc = fig.add_axes([0.15, 0.02, 0.3, 0.03]) 
    cb_acc = fig.colorbar(cm.ScalarMappable(norm=norm_acc, cmap=cmap_acc), cax=cbar_ax_acc, orientation='horizontal')
    cb_acc.set_label('Accuracy (Bottom-Left)', color='teal', fontsize=10, fontweight='bold')
    cbar_ax_spar = fig.add_axes([0.55, 0.02, 0.3, 0.03])
    cb_spar = fig.colorbar(cm.ScalarMappable(norm=norm_spar, cmap=cmap_spar), cax=cbar_ax_spar, orientation='horizontal')
    cb_spar.set_label('Sparsity (Top-Right)', color='purple', fontsize=10, fontweight='bold')

    plt.subplots_adjust(bottom=0.15, top=0.9)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Output] Saved: {os.path.basename(output_path)}")

# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True, help="Experiment ID")
    args = parser.parse_args()

    base_dir = "experiments"
    exp_dir = os.path.join(base_dir, args.exp_name)
    
    # 1. Load Data
    df, params = load_and_detect_data(exp_dir)

    if df.empty or len(params) < 1:
        print("[Error] No varying parameters found.")
        exit()

    print("="*50)
    
    # 2. Logic for 1, 2, or 3 Parameters
    if len(params) == 1:
        # 1D Case: Column of results
        param_name = params[0]
        print(f"[Logic] 1 Parameter detected: {param_name}")
        output_file = os.path.join(exp_dir, "summary_tradeoff_1d.png")
        plot_1d_column(df, param_name, output_file)

    elif len(params) == 2:
        # 2D Case: Grid matrix
        x_param, y_param = params[0], params[1]
        print(f"[Logic] 2 Parameters detected: X={x_param}, Y={y_param}")
        output_file = os.path.join(exp_dir, "summary_tradeoff_matrix.png")
        plot_heatmap(df, x_param, y_param, "", output_file)

    elif len(params) == 3:
        # 3D Case: Book pages
        print(f"[Logic] 3 Parameters detected: {params}")
        print("Please select which parameter should be the 'Book Page' (the separate plots).")
        print("The other two will form the X/Y grid.")
        
        while True:
            selection = input(f"Enter parameter name for Book Page ({'/'.join(params)}): ").strip()
            if selection in params:
                page_param = selection
                break
            print("Invalid selection. Try again.")
        
        remaining = [p for p in params if p != page_param]
        x_param, y_param = remaining[0], remaining[1]
        
        print(f"[Config] Page={page_param} | Grid X={x_param}, Y={y_param}")

        g_norm_acc = mcolors.Normalize(vmin=df['Accuracy'].min(), vmax=df['Accuracy'].max())
        g_norm_spar = mcolors.Normalize(vmin=df['Sparsity'].min(), vmax=df['Sparsity'].max())
        global_norms = (g_norm_acc, g_norm_spar)

        unique_pages = sorted(df[page_param].unique())
        for val in unique_pages:
            page_df = df[df[page_param] == val].copy()
            filename = f"matrix_page_{page_param}_{val}.png"
            title_suffix = f"\n({page_param} = {val})"
            output_path = os.path.join(exp_dir, filename)
            plot_heatmap(page_df, x_param, y_param, title_suffix, output_path, global_norms)

    else:
        print(f"[Error] This script supports 1, 2, or 3 varying parameters. Found {len(params)}: {params}")