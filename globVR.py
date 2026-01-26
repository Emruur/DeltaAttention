# import torch
act = {}
act_q = {}
act_k = {}
act_v = {}
act_o = {}
act_attn = {}
delta_head = []
delta_key = []
switch_act = 0
switch_attn = 0
switch_ffn = 0
switch_down = 0
act_up = {}
act_down_in = {}
act_down_out = {}
act_gate = {}
layer_in = {}
n_sample = 0

direct_prune = 0
direct_thresh = 0
direct_spars = 0.0

count = 0
spars = 0.0
scale = 0.05

window_size = 0
sink_size = 0
block_size = 0
refresh_rate = 0

attn_pattern = 0
delta_pattern_thresh = 0
quant_on = 0

delta_query_on = 0
delta_query_thresh = 0
collect_delta_query = 0

delta_pf_key_on = 1
delta_pf_key_thresh = 0.6
collect_delta_pf_key = 1

delta_key_on = 0
delta_key_thresh = 0
collect_delta_key = 0

delta_attn_on = 0
collect_delta_attn = 0
delta_attn_thresh = 0

collect_delta_attn_mat = 0
collect_delta_q_mat = 0

delta_q = {}
delta_k = {}
delta_pf_k = {}
delta_attn = {}
# q_pre = torch.tensor([]).cuda()
# attn_pre = torch.tensor([]).cuda()
