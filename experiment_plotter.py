import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np

def plot_1d_column(df, param_name, output_path, sparsity_col='Attn_Sparsity', sparsity_label='Attn Sparsity'):
    """
    Generates a vertical column of results for a single varying parameter.
    Dynamically tracks either Attention or MLP sparsity.
    """
    summary = df.groupby(param_name)[['Accuracy', sparsity_col]].mean().reset_index()
    summary = summary.sort_values(param_name)

    n_rows = len(summary)
    norm_acc = mcolors.Normalize(vmin=summary['Accuracy'].min(), vmax=summary['Accuracy'].max())
    norm_spar = mcolors.Normalize(vmin=summary[sparsity_col].min(), vmax=summary[sparsity_col].max())
    cmap_acc = plt.get_cmap('viridis')
    cmap_spar = plt.get_cmap('plasma')

    fig, ax = plt.subplots(figsize=(4, max(4, n_rows * 1.2)))
    ax.set_aspect('equal')

    y_labels = summary[param_name].tolist()
    
    for idx, (i, row) in enumerate(summary.iterrows()):
        acc = row['Accuracy']
        spar = row[sparsity_col]
        draw_split_cell(ax, 0, n_rows - 1 - idx, 1, 1, acc, spar, norm_acc, cmap_acc, norm_spar, cmap_spar)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([])
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(y_labels[::-1], fontsize=11)
    ax.set_ylabel(param_name, fontsize=12, fontweight='bold')
    ax.set_title(f"Trade-off Summary", fontsize=14, pad=15)

    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(length=0)

    cbar_ax_acc = fig.add_axes([0.15, 0.05, 0.7, 0.02]) 
    cb_acc = fig.colorbar(cm.ScalarMappable(norm=norm_acc, cmap=cmap_acc), cax=cbar_ax_acc, orientation='horizontal')
    cb_acc.set_label('Accuracy (Bottom-Left)', color='teal', fontsize=9, fontweight='bold')
    
    cbar_ax_spar = fig.add_axes([0.15, 0.01, 0.7, 0.02])
    cb_spar = fig.colorbar(cm.ScalarMappable(norm=norm_spar, cmap=cmap_spar), cax=cbar_ax_spar, orientation='horizontal')
    cb_spar.set_label(f'{sparsity_label} (Top-Right)', color='purple', fontsize=9, fontweight='bold')

    plt.subplots_adjust(bottom=0.2, top=0.9, left=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Output] Saved: {os.path.basename(output_path)}")

def plot_heatmap(df, x_param, y_param, title_suffix, output_path, global_norms=None, sparsity_col='Attn_Sparsity', sparsity_label='Attn Sparsity'):
    """
    Generates a single heatmap for 2 parameters (X and Y).
    Dynamically tracks either Attention or MLP sparsity.
    """
    summary = df.groupby([x_param, y_param])[['Accuracy', sparsity_col]].mean().reset_index()
    pivot_acc = summary.pivot(index=y_param, columns=x_param, values="Accuracy")
    pivot_spar = summary.pivot(index=y_param, columns=x_param, values=sparsity_col)
    
    pivot_acc = pivot_acc.sort_index(ascending=True).sort_index(axis=1, ascending=True)
    pivot_spar = pivot_spar.sort_index(ascending=True).sort_index(axis=1, ascending=True)

    rows, cols = pivot_acc.shape

    if global_norms:
        norm_acc, norm_spar = global_norms
    else:
        norm_acc = mcolors.Normalize(vmin=summary['Accuracy'].min(), vmax=summary['Accuracy'].max())
        norm_spar = mcolors.Normalize(vmin=summary[sparsity_col].min(), vmax=summary[sparsity_col].max())

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
    cb_spar.set_label(f'{sparsity_label} (Top-Right)', color='purple', fontsize=10, fontweight='bold')

    plt.subplots_adjust(bottom=0.15, top=0.9)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Output] Saved: {os.path.basename(output_path)}")
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
                    
                    # EXTENDED: Now tracking Task name and MLP Sparsity
                    record = {
                        "Config_ID": item,
                        "Task": res.get("task_name", task_file.replace('.json', '')),
                        "Accuracy": res.get("accuracy", 0.0),
                        "Attn_Sparsity": res.get("sparsity", 0.0),
                        "MLP_Sparsity": res.get("mlp_spars", 0.0)
                    }
                    record.update(params)
                    temp_records.append(record)
                except Exception as e:
                    print(f"[Warning] Failed to read task {task_file}: {e}")

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
# 2. Plotting Logic - Default (Heatmaps)
# ==========================================
def draw_split_cell(ax, x, y, width, height, acc_val, spar_val, norm_acc, cmap_acc, norm_spar, cmap_spar):
    c_acc = cmap_acc(norm_acc(acc_val))
    c_spar = cmap_spar(norm_spar(spar_val))

    x0, y0 = x, y
    x1, y1 = x + width, y + height

    tri_acc = patches.Polygon([(x0, y0), (x1, y0), (x0, y1)], closed=True, color=c_acc, ec='white', lw=0.5)
    ax.add_patch(tri_acc)
    
    tri_spar = patches.Polygon([(x0, y1), (x1, y1), (x1, y0)], closed=True, color=c_spar, ec='white', lw=0.5)
    ax.add_patch(tri_spar)

    txt_c_acc = 'white' if np.mean(c_acc[:3]) < 0.5 else 'black'
    txt_c_spar = 'white' if np.mean(c_spar[:3]) < 0.5 else 'black'
    
    fs = 10 
    ax.text(x0 + width*0.25, y0 + height*0.3, f"{acc_val:.2f}", ha='center', va='center', color=txt_c_acc, fontsize=fs, fontweight='bold')
    ax.text(x0 + width*0.75, y0 + height*0.7, f"{spar_val:.2f}", ha='center', va='center', color=txt_c_spar, fontsize=fs, fontweight='bold')

# ==========================================
# 3. Plotting Logic - Type 2 (CSV & Line Plot)
# ==========================================
# ==========================================
# 3. Plotting Logic - Type 2 (CSV & Visual Table)
# ==========================================
def process_type2(df, varying_param, output_dir):
    """
    Pivots the data so each task is a column, adds an average accuracy column,
    and includes both Attention and MLP sparsities. Saves to CSV and renders a PNG table.
    """
    print(f"[Type 2] Processing data for varying parameter: {varying_param}")

    # 1. Pivot tasks into columns for Accuracy
    acc_pivot = df.pivot_table(index=varying_param, columns='Task', values='Accuracy', aggfunc='mean')
    
    # 2. Calculate Average Accuracy across all tasks for each threshold
    acc_pivot['Avg_Accuracy'] = acc_pivot.mean(axis=1)
    
    # 3. Get Sparsities (Grouped by the varying param)
    spars_df = df.groupby(varying_param)[['Attn_Sparsity', 'MLP_Sparsity']].mean()
    
    # 4. Join them together
    final_df = acc_pivot.join(spars_df).reset_index()
    
    # Save to CSV (Always good to have the raw data)
    csv_path = os.path.join(output_dir, "type2_summary.csv")
    final_df.to_csv(csv_path, index=False)
    print(f"[Output] Saved tabular data to: {os.path.basename(csv_path)}")

    # 5. Generate Visual Table (PNG)
    # Create a display copy and round for visual cleanliness
    display_df = final_df.copy().round(4)
    
    # Dynamically size the image based on columns and rows
    fig_width = max(10, len(display_df.columns) * 1.5)
    fig_height = max(4, len(display_df) * 0.6 + 1)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table
    table = ax.table(cellText=display_df.values, 
                     colLabels=display_df.columns, 
                     cellLoc='center', 
                     loc='center')
    
    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2) # Stretch rows slightly for readability
    
    # Style the header row to make it pop
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#333333') # Dark grey header
        else:
            # Subtle alternating row colors
            if row % 2 == 0:
                cell.set_facecolor('#f3f4f6')

    plt.title(f"Performance Summary vs. {varying_param}", fontsize=14, fontweight='bold', pad=20)
    
    # Save the table image
    plot_path = os.path.join(output_dir, "type2_table.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Output] Saved visual table to: {os.path.basename(plot_path)}")
# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True, help="Experiment ID")
    parser.add_argument('--plot_type', type=str, default='default', choices=['default', 'type2'], 
                        help="Choose 'default' for heatmaps or 'type2' for multi-benchmark tables/lines.")
    args = parser.parse_args()

    base_dir = "experiments"
    exp_dir = os.path.join(base_dir, args.exp_name)
    
    # 1. Load Data
    df, params = load_and_detect_data(exp_dir)

    if df.empty or len(params) < 1:
        print("[Error] No varying parameters found.")
        exit()

    print("="*50)
    
    # ---------------------------------------------------------
    # ROUTING: Type 2
    # ---------------------------------------------------------
    if args.plot_type == 'type2':
        if len(params) > 1:
            print(f"[Warning] Type 2 expects exactly 1 varying parameter (the threshold). Found {len(params)}: {params}")
            print(f"Defaulting to using the first one: {params[0]}")
        process_type2(df, params[0], exp_dir)
        exit()

    # ---------------------------------------------------------
    # ROUTING: Default (Heatmaps & Splits)
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # ROUTING: Default (Heatmaps & Splits)
    # ---------------------------------------------------------
    if len(params) == 1:
        param_name = params[0]
        print(f"[Logic] 1 Parameter detected: {param_name}")
        
        # 1. Attention Plot
        out_attn = os.path.join(exp_dir, "summary_tradeoff_1d_attn.png")
        plot_1d_column(df, param_name, out_attn, sparsity_col='Attn_Sparsity', sparsity_label='Attn Sparsity')
        
        # 2. MLP Plot
        out_mlp = os.path.join(exp_dir, "summary_tradeoff_1d_mlp.png")
        plot_1d_column(df, param_name, out_mlp, sparsity_col='MLP_Sparsity', sparsity_label='MLP Sparsity')

    elif len(params) == 2:
        x_param, y_param = params[0], params[1]
        print(f"[Logic] 2 Parameters detected: X={x_param}, Y={y_param}")
        
        # 1. Attention Plot
        out_attn = os.path.join(exp_dir, "summary_tradeoff_matrix_attn.png")
        plot_heatmap(df, x_param, y_param, "", out_attn, sparsity_col='Attn_Sparsity', sparsity_label='Attn Sparsity')
        
        # 2. MLP Plot
        out_mlp = os.path.join(exp_dir, "summary_tradeoff_matrix_mlp.png")
        plot_heatmap(df, x_param, y_param, "", out_mlp, sparsity_col='MLP_Sparsity', sparsity_label='MLP Sparsity')

    else:
        print(f"[Error] Default script supports 1 or 2 varying parameters. Found {len(params)}.")
