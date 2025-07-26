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
    def __init__(self, in_channels, embedding_dim=64, out_channels=64, heads=1, dropout=0.2):
        super().__init__()
        self.embeddings = nn.Embedding(in_channels,embedding_dim, padding_idx=0)
        self.gat_1 = GATConv(embedding_dim, out_channels // heads, heads=heads, dropout=dropout)
        self.gat_2 = GATConv(out_channels, out_channels, heads=1, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self,edge_index):
        num_nodes = self.embeddings.num_embeddings
        device = edge_index.device
        x_embed = self.embeddings(torch.arange(num_nodes, device=device))
        
        x_gat1 = self.gat_1(x_embed, edge_index)
        # x_cat = torch.cat([x_embed,x_gat1], dim=-1)
        x_gat1 = F.leaky_relu(x_gat1)
        x_gat1 = self.dropout(x_gat1)
        
        out = self.gat_2(x_gat1, edge_index)
        out = F.leaky_relu(out)
        out = self.dropout(out)
        
        return out
        
if __name__ == "__main__":
    global_edge_index = torch.load("mydata/porto_edge_index.pt").long()
    global_edge_index_moved = global_edge_index + 1
    print(global_edge_index_moved.min(), global_edge_index_moved.max())
    num_edge = global_edge_index_moved.max().item()
    print(num_edge)
    model = GAT_Layer(in_channels=num_edge, embedding_dim=64, out_channels=64, heads=2, dropout=0.6)
    
    linkids_list = list(map(lambda x: np.asarray(x), [[1, 2, 5], [2, 4]]))
    tensor_list = [torch.tensor(l).long() + 1 for l in linkids_list]
    padded_linkids = torch.nn.utils.rnn.pad_sequence(tensor_list, batch_first=True, padding_value=0)
    x = padded_linkids.flatten()
    print(x)
    # flat_linkids_tensor = padded_linkids.flatten()
    # print(flat_linkids_tensor)
    # print(x)
    # sub_segment, sub_edge_index, _, edge_mask = k_hop_subgraph(
    #     x, num_hops=2, edge_index=global_edge_index,relabel_nodes=True
    # )
    # unique_sub_segment, inverse_indices = torch.unique(sub_segment, return_inverse=True)
    # segment_id_to_subgraph_index = {int(n.item()): i for i, n in enumerate(unique_sub_segment)}
    # x_remapped = torch.tensor([segment_id_to_subgraph_index[int(n.item())] for n in x])
    # [B,T,D]
    out = model(global_edge_index_moved)
    print(out.shape)
    out_route = out[x]
    B,T = padded_linkids.shape
    D = out_route.shape[-1]
    out_route = out_route.view(B,T,D)
    # out_route = out[x_remapped]
    # B,T = padded_linkids.shape
    # D = out_route.shape[1]
    # segment_id_to_output_index = {int(seg.item()): i for i, seg in enumerate(x)}
    # indices = torch.tensor([segment_id_to_output_index[int(seg.item())] for seg in flat_linkids_tensor])
    # gathered_out = out_route[indices]
    # print("Gathered output shape:", gathered_out.shape)
    # out_route = gathered_out.view(B, T, D)
    # print(segment_id_to_output_index)
    # print("Output shape:", out_route.shape)
