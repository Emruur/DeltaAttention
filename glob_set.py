import torch
import globVR

from seaborn import histplot, boxplot, heatmap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def set_sample(n_data):
    globVR.n_sample = n_data

def update_latency(metric_name, dt_seconds):
    """
    Structured latency tracker. Stores metrics in a dictionary in globVR.
    """
    # If internal timing is disabled, do nothing.
    if not getattr(globVR, 'time_internal', False):
        return

    # Initialize the dictionary if it doesn't exist
    if not hasattr(globVR, 'latency_stats'):
        globVR.latency_stats = {}
    
    # Initialize the specific metric if it's new
    if metric_name not in globVR.latency_stats:
        globVR.latency_stats[metric_name] = {'time_ms': 0.0, 'calls': 0}
        
    # Accumulate
    globVR.latency_stats[metric_name]['time_ms'] += (dt_seconds * 1000)
    globVR.latency_stats[metric_name]['calls'] += 1

def queue_event_pair(metric_name, start_evt, end_evt):
    """
    Queues a pair of CUDA events for deferred latency measurement.
    """
    if not getattr(globVR, 'time_internal', False):
        return

    if not hasattr(globVR, 'latency_events'):
        globVR.latency_events = []
        
    globVR.latency_events.append((metric_name, start_evt, end_evt))

def resolve_latency_events():
    """
    Resolves all queued CUDA events and populates latency_stats.
    """
    if not getattr(globVR, 'time_internal', False) or not hasattr(globVR, 'latency_events'):
        return

    if not hasattr(globVR, 'latency_stats'):
        globVR.latency_stats = {}

    # Make sure all events are complete
    torch.cuda.synchronize()

    for metric_name, start_evt, end_evt in globVR.latency_events:
        if metric_name not in globVR.latency_stats:
            globVR.latency_stats[metric_name] = {'time_ms': 0.0, 'calls': 0}
            
        dt_ms = start_evt.elapsed_time(end_evt)
        globVR.latency_stats[metric_name]['time_ms'] += dt_ms
        globVR.latency_stats[metric_name]['calls'] += 1

    # Clear the queue
    globVR.latency_events.clear()



def compute_mlp_sparsity(input_tensor, keep_mask=None):
    
    # Calculate current sparsity
    if keep_mask is not None:
        current_spars = torch.sum(~keep_mask).item() / keep_mask.numel()
    else:
        # Fallback: scan the dense delta matrix for exact zeros
        current_spars = torch.sum(input_tensor == 0).item() / input_tensor.numel()
        
    # Update global tracking safely using Cumulative Moving Average
    if not hasattr(globVR, 'mlp_spars_count'):
        globVR.mlp_spars = 0.0
        globVR.mlp_spars_count = 0
        
    globVR.mlp_spars = (globVR.mlp_spars * globVR.mlp_spars_count + current_spars) / (globVR.mlp_spars_count + 1)
    globVR.mlp_spars_count += 1

def build_glob(activation, n_data, n_layer, max_len, d_model):
    activation[n_data] = torch.zeros(n_layer, max_len, d_model).cuda()
    #globVR.k_act[n_data] = torch.zeros(n_layer, max_len, d_model)
    #globVR.v_act[n_data] = torch.zeros(n_layer, max_len, d_model)
    # print("act:",activation[n_data].shape, n_sample)

def store_delta(activation, l, value, switch):
    # store the intermediate delta values by layer
    if switch == 1:
        activation.append(value)
        # print(f'check{activation}')


def delta_compute_sparsity(input):
    # compute the sparsity per layer
    delta_spars = {}
    avg_spars = 0.0
    layer_count = 0
    # seq_len = 
    for l, data in input.items():
        delta_spars[l] = torch.sum(data == 0)/torch.numel(data)
        avg_spars = avg_spars + delta_spars[l]
        layer_count += 1
    # avg_spars = avg_spars*(1-len_full_attn/seq_len) # consider the effect on the sparsity from the full attention sink/window
    return [avg_spars/layer_count, delta_spars]

def init_spars_head(x, n_head):
    for h in range(n_head):
        x.append(0.0)
    return x

def delta_compute_sparsity_head(input, n_head, switch, spars):
    # compute the sparsity per layer
    delta_spars = {}
    spars_update = {}
    if switch == 1:
        for h in range(n_head):
            head_in = input[:,h,:,:]
            delta_spars[h] = torch.sum(head_in == 0)/torch.numel(head_in)
            print(f"delta_spars: {delta_spars}")
            spars_update[h] = (spars[h] + delta_spars[h])/2
    return spars_update
    
def compute_sparsity(input,len,switch):
    if switch == 1:
        scale = 0
        if globVR.block_size < len:
            scale = 1-globVR.block_size/len
            globVR.count += 1
            # print(scale)
        if globVR.spars == 0.0:
            globVR.spars = torch.sum(input == 0)/torch.numel(input)*scale
        else:
            globVR.spars = (globVR.spars + torch.sum(input == 0)/torch.numel(input)*scale)/2
            # print(globVR.spars)
    return

def compute_sparsity_scale(input, scale, keep_mask=None):
    # 1. Calculate the current sparsity ratio
    if keep_mask is not None:
        # ~keep_mask inverts the boolean tensor (True becomes False).
        # Summing it counts exactly how many rows/elements were dropped.
        current_spars = (torch.sum(~keep_mask).item() / keep_mask.numel()) * (1 - scale)
    else:
        # Fallback: scan the dense delta matrix for exact zeros
        current_spars = (torch.sum(input == 0).item() / input.numel()) * (1 - scale)
        
    # 2. Update the global moving average
    if globVR.spars == 0.0:
        globVR.spars = current_spars
    else:
        globVR.spars = (globVR.spars + current_spars) / 2
        
    return

def store_attention(activation, l, value, switch):
    # store the intermediate delta values by layer
    if switch == 1:
        activation[l] = value.cpu()

def build_glob2(activation, n_layer):
    for l in range(n_layer):
        activation[l] = torch.tensor([])

def build_glob3(data, n_layer):
    for l in range(n_layer):
        data[l] = []

def set_glob2(activation, l, value, switch):
    # store the intermidiate activations by layer
    if switch == 1:
        activation[l] = torch.cat((activation[l],value.view(-1).cpu()), dim=0)

def set_glob_keep_dim(data, l, input, switch):
    if switch == 1:
        data[l] = torch.cat((data[l], input.cpu()), dim=0)

def set_glob3(data, l, input, switch):
    if switch == 1:
        data[l].append(input)
    
def set_glob_delta(delta, l, seq_len, n_head, value, switch):
    if switch == 1:
        delta[l] = torch.cat((delta[l], (value[:,:,0:seq_len-1,:].reshape(n_head,-1) - value[:,:,1:seq_len,:].reshape(n_head,-1))), dim=-1)
        #print(torch.sum(delta[l]==0)/torch.numel(delta[l]))

def draw_box(activation, n_layer, name):
    fig, axes = plt.subplots(1, n_layer, figsize=(100, 20), sharex=True, sharey=True)
    for l in range(n_layer):
        boxplot(data=activation[l], ax=axes[l])
    plt.tight_layout()
    plt.savefig(name)
    plt.close

def draw_hist(x, n_layer, head, path):
    fig, axes = plt.subplots(1, n_layer, figsize=(100, 20), sharex=True, sharey=True)
    for l, data in x.items():
        histplot(data=data[head,:].cpu(), bins = 30, ax=axes[l])
        axes[l].set_title(f"layer: {l}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close

def compute_std(x, n_layer, switch):
    interval = {}
    if switch == 1:
        for l in range(n_layer):
            std = torch.std(x[l])
            mean = torch.mean(x[l])
            interval[l] = (mean + 3*std, mean-3*std)
    return interval

def draw_heatmap_single_layer_attn(x, layer, head, size, path):
    begin = 0
    end = begin + size
    seq_len = x[layer].shape[-1]
    #print('input shape:', x[layer].shape)
    while begin < seq_len:
        if end > seq_len:
            end = seq_len
        plt.figure(figsize=(40, 40))  # Optional: Adjust the figure size
        #print('begin:end', begin, ':',end)
        #print('input:',x[layer][:,head,begin:end,begin:end].shape)
        len = end - begin
        x_in = x[layer][:,head,begin:end,begin:end].reshape(len,len)
        print('x_in:',x_in.shape)
        heatmap(x_in, annot=False, cmap="coolwarm",vmin=-1,vmax=1)  # Use your preferred colormap
        plt.xlabel("K")
        plt.ylabel("Q")
        plt.savefig(path+f'Token{begin}to{end}.png')
        plt.close()
        begin += size
        end += size

def draw_heatmap_all_layer_attn(x, n_layer, head, size, path):
    begin = 0
    size = x[0].shape[2]
    end = begin + size
    for l in range(n_layer):
        plt.figure(figsize=(size,size))  # Optional: Adjust the figure size
        #print('begin:end', begin, ':',end)
        #print('input:',x[layer][:,head,begin:end,begin:end].shape)
        x_in = x[l][:,head,begin:end,begin:end].reshape(size,size)
        print('x_in:',x_in.shape)
        ax = heatmap(x_in, annot=False, cmap="coolwarm",vmin=-1,vmax=1)  # Use your preferred colormap
        ax.set_xticks([])  # Remove x-axis numbers
        ax.set_yticks([])  # Remove y-axis numbers
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([])
        # plt.xlabel("K")
        # plt.ylabel("Q")
        plt.savefig(path+f'Attn_token{size}_layer{l}.png')
        plt.close()

def draw_heatmap_all_layer_query(x, n_layer, head, size, path):
    begin = 0
    end = begin + size
    for l in range(n_layer):
        plt.figure(figsize=(50, 50))  # Optional: Adjust the figure size
        print('input', x[l].shape)
        #print('input:',x[l][:,head,begin:end,begin:end].shape)
        x_in = x[l][:,head,begin:end,begin:end].reshape(size,size)
        #print('x_in:',x_in.shape)
        heatmap(x_in, annot=False, cmap="coolwarm")  # Use your preferred colormap
        plt.xlabel("Dimension")
        plt.ylabel("Token")
        plt.savefig(path+f'Q_token{size}_layer{l}.png')
        plt.close()
      


    

# def draw_heatmap(x:dict, n_layer:int, trunc_size: int):
#     fig, axes = plt.subplots(1, n_layer, figsize=(200, 20), sharey=True, sharex=True)
#     l = 0
#     for l in range(n_layer):
#         # plt.figure(figsize=(30,30))
#         # plt.imshow(tensors[k], interpolation='nearest')
#         heatmap(x[l][:trunc_size,:trunc_size], ax=axes[l], cmap="coolwarm", annot=True, fmt = ".2f")

#         # plt.savefig('result_seaborn2.jpg')
#         l = l + 1
#     plt.tight_layout()
#     plt.savefig('v_spars_hist_.png')
#     plt.close()