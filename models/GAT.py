import copy
import math
import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from torch_geometric.utils import k_hop_subgraph

# GAT LAYER THAT RETURN AN ATTENTION MATRIX
class GAT_Layer(nn.Module):
    def __init__(self, in_channels, embedding_dim=64, out_channels=64, heads=1, dropout=0.6):
        super().__init__()
        self.embeddings = nn.Embedding(in_channels,embedding_dim)
        self.gat_conv = GATConv(embedding_dim, out_channels // heads, heads=heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        
        x_embed = self.embeddings(x)
        x_gat, (gat_edge_index, attn_weights) = self.gat_conv(x_embed, edge_index,return_attention_weights=True)
        x_cat = torch.cat([x_embed,x_gat], dim=-1)
        out = F.leaky_relu(x_gat)
        out = self.dropout(out)
        return out, attn_weights #[num_edges, dim]
        
if __name__ == "__main__":
    global_edge_index = torch.load("mydata/porto_edge_index.pt")
    num_edge = global_edge_index.unique().shape[0]
    model = GAT_Layer(in_channels=num_edge, embedding_dim=64, out_channels=64, heads=2, dropout=0.6)
    linkids_tensor = torch.tensor([
        [1, 2, 5],   # Route 0
        [2, 4, 6],   # Route 1
    ])  # shape: [B=2, T=3]
    x = linkids_tensor.flatten().unique() 
    flat_linkids_tensor = linkids_tensor.flatten()
    print(flat_linkids_tensor)
    print(x)
    sub_segment, sub_edge_index, _, edge_mask = k_hop_subgraph(
        x, num_hops=1, edge_index=global_edge_index,relabel_nodes=True
    )
    unique_sub_segment, inverse_indices = torch.unique(sub_segment, return_inverse=True)
    segment_id_to_subgraph_index = {int(n.item()): i for i, n in enumerate(unique_sub_segment)}
    x_remapped = torch.tensor([segment_id_to_subgraph_index[int(n.item())] for n in x])
    # [B,T,D]
    out, attn_weights = model(sub_segment, sub_edge_index)
    out_route = out[x_remapped]
    B,T = linkids_tensor.shape
    D = out_route.shape[1]
    segment_id_to_output_index = {int(seg.item()): i for i, seg in enumerate(x)}
    indices = torch.tensor([segment_id_to_output_index[int(seg.item())] for seg in flat_linkids_tensor])
    gathered_out = out_route[indices]
    print("Gathered output shape:", gathered_out.shape)
    out_route = gathered_out.view(B, T, D)
    print(segment_id_to_output_index)
    print("Output shape:", out_route.shape)
    print()
    print("Attention weights shape:", attn_weights.shape)
