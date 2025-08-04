import copy
import math
import torch
from torch import nn
import numpy as np

from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from torch_geometric.utils import k_hop_subgraph

# GAT LAYER THAT RETURN AN ATTENTION MATRIX
class GAT_Layer(nn.Module):
    def __init__(self, in_channels=6, embedding_dim=64, out_channels=64, heads=1, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(in_channels,embedding_dim)
        self.gat_1 = GATConv(embedding_dim, out_channels // heads, heads=heads, dropout=dropout)
        self.gat_2 = GATConv(out_channels, out_channels, heads=1, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,edge_index):
        x_linear = self.linear(x) # [B,embedding_dim]
        
        x_gat1 = self.gat_1(x_linear, edge_index)

        x_gat1 = F.leaky_relu(x_gat1)
        x_gat1 = self.dropout(x_gat1)
        
        out = self.gat_2(x_gat1, edge_index)
        out = F.leaky_relu(out)
        out = self.dropout(out)
        
        return out
    
if __name__ == "__main__":
    import sys
    import os
    from torch_geometric.data import Data, Batch 
    # Get the parent directory of the current file (adjust path if needed)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))  # one level up
    sys.path.append(project_root)
    from utils.prepare import test_info_all, MulT_TTE_collate_func, Datadict, DataLoader, BatchSampler
    import numpy as np
    from torch_geometric.utils import k_hop_subgraph
    def collate_fn(data, info_all):
        edgeinfo, nodeinfo, indexinfo, scaler, scaler2 = info_all
        time = torch.Tensor([d[-1] for d in data])
        linkids = []
        dateinfo = []
        inds = []
        for ind, l in enumerate(data):
            linkids.append(np.asarray(l[1]))
            dateinfo.append(l[2:5])
            inds.append(l[0])
        lens = np.asarray([len(k) for k in linkids], dtype=np.int16)
        
        routes_coords = []
        mappings = []
        edge_index_list = []
        offset = 0
        global_edge_index = indexinfo 
        segment_lens = []
        
        for route in linkids:
            route = torch.tensor(route).long()
            # route --> tensor of 1 route
            # routeset --> k_hop of 1 route
            # route_edge_index --> local map edge_index [0..N-1]
            # mapping --> index of route in newly create route_edge_index#
            routeset, route_edge_index, mapping, _ = k_hop_subgraph(
                node_idx=route,  # node(s) to center the subgraph on
                num_hops=2,
                edge_index=global_edge_index,  # this MUST be a tensor of shape [2, num_edges]
                relabel_nodes=True
            )

            assert route.size(0) == mapping.size(0), "Route size and mapping size must match"   
            
            sub_node_coords = []       
            for segment in routeset:
                segment = segment.item()
                start_node,end_node = edgeinfo[segment][2:4]
                start_coord = nodeinfo[start_node]
                end_coord = nodeinfo[end_node]
                sub_node_coords.append(start_coord + end_coord)
            routes_coords.append(torch.tensor(sub_node_coords, dtype=torch.float32))
                    
            adjusted_route_edge_index = route_edge_index + offset
            edge_index_list.append(adjusted_route_edge_index)
            
            adjusted_mapping = mapping + offset
            mappings.append(adjusted_mapping)
            
            offset += routeset.size(0)
            segment_lens.append(mapping.size(0))
        
        routes_tensor = torch.cat(routes_coords, dim=0)
        edge_index_tensor = torch.cat(edge_index_list, dim=-1)
        mappings_tensor = torch.cat(mappings, dim=-1)    
        return {'edgeindex': edge_index_tensor,'routes':routes_tensor,'mappings' : mappings_tensor, 'segment_lens': segment_lens}, time      

    test_info_all()
    tdata = np.load('mydata/train.npy', allow_pickle=True)
    info_all = test_info_all()
    loader = DataLoader(Datadict(tdata), batch_sampler=BatchSampler(tdata, 48),
                                       collate_fn=lambda x: collate_fn(x, info_all),
                                       pin_memory=True)
    model = GAT_Layer(6)
    features, truth_data  = loader.__iter__().__next__()  
    print(features['routes'])
    print(features['edgeindex'])
    out = model(features['routes'], features['edgeindex'])
    out = out[features['mappings']]
    print(out.shape)
    B,T = 48, max(features['segment_lens'])
    print(B,T)
    D = out.shape[-1]
    start = 0
    out_padded = torch.zeros(B, T, D, dtype=out.dtype, device=out.device)
    for i in range(B):
        seg_len = features['segment_lens'][i]
        end = start + seg_len
        print(f"Segment {i}: start={start}, end={end}, seg_len={seg_len}, actual_len={end-start}")
        out_padded[i, :seg_len, :] = out[start:end]
        start = end
    print(out_padded.shape)
    # it should go back to B*T*D --> num_segments_in_a_batch, dim --> need to separate and pad
    # global_edge_index = torch.load("mydata/porto_edge_index.pt").long()
    # global_edge_index_moved = global_edge_index + 1
    # dataset = 
    # print(global_edge_index_moved.min(), global_edge_index_moved.max())
    # num_edge = global_edge_index_moved.max().item()
    # print(num_edge)
    # model = GAT_Layer(in_channels=num_edge, embedding_dim=64, out_channels=64, heads=2, dropout=0.6)
    
    # linkids_list = list(map(lambda x: np.asarray(x), [[1, 2, 5], [2, 4]]))
    # tensor_list = [torch.tensor(l).long() + 1 for l in linkids_list]
    # padded_linkids = torch.nn.utils.rnn.pad_sequence(tensor_list, batch_first=True, padding_value=0)
    # x = padded_linkids.flatten()
    # print(x)
    # # [B,T,D]
    # out = model(global_edge_index_moved)
    # print(out.shape)
    # out_route = out[x]
    # B,T = padded_linkids.shape
    # D = out_route.shape[-1]
    # out_route = out_route.view(B,T,D)
    # # out_route = out[x_remapped]
    # # B,T = padded_linkids.shape
    # # D = out_route.shape[1]
    # # segment_id_to_output_index = {int(seg.item()): i for i, seg in enumerate(x)}
    # # indices = torch.tensor([segment_id_to_output_index[int(seg.item())] for seg in flat_linkids_tensor])
    # # gathered_out = out_route[indices]
    # # print("Gathered output shape:", gathered_out.shape)
    # # out_route = gathered_out.view(B, T, D)
    # # print(segment_id_to_output_index)
    # # print("Output shape:", out_route.shape)
