import torch
from torch import nn

from models.LayerNormGRU import LayerNormGRU
class Discriminator(nn.Module):
    def __init__(self, input_dim,seq_hidden_dim,seq_layer,classifier_hidden_dim):
        ...
        super().__init__()

        # # time-aware sematic embedding
        # self.bert_config = BertConfig(num_attention_heads = bert_attention_heads, hidden_size = bert_hiden_size, pad_token_id=pad_token_id,
        #                               vocab_size=vocab_size, num_hidden_layers = bert_hidden_layers)
        # self.seg_embedding_learning = BertForMaskedLM(self.bert_config)
        # # spatial encoder
        # self.gps_rep = nn.Linear(4, 16)
        # self.spatial_encoder = nn.Sequential(
        #     nn.Linear(input_dim, seq_input_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(seq_input_dim, seq_input_dim)
        # )
        # # attribute feature encoder
        # self.highwayembed = nn.Embedding(15, 5, padding_idx=0)
        # self.weekembed = nn.Embedding(8, 3)
        # self.dateembed = nn.Embedding(367, 10)
        # self.timeembed = nn.Embedding(1441, 20)
        # multi-faceted Sequential Encoder
        # self.sequence = LayerNormGRU(input_dim, seq_hidden_dim, seq_layer)
        self.classifier = Classifier(seq_hidden_dim + 1,classifier_hidden_dim, 1)
        
    def forward(self,spatio_temporal_features, t_fake: torch.Tensor,t_real, seq_len=None):
        # spatiotemporal_features: [batch_size, seq_len, input_dim]
        spatio_temporal_features = spatio_temporal_features.permute(1,0,2)
        t_fake_tensor = t_fake.unsqueeze(-1) # [batch_size, 1, 1]
        
        t_fake_tensor = t_fake_tensor.expand(-1,spatio_temporal_features.size(1),-1) # [batch_size, seq_len, 1]
        assert t_fake_tensor.dim() == 3, f"Expected 3D tensor, got {t_real_tensor.shape} tensor"
        t_real_tensor = t_real.unsqueeze(-1).unsqueeze(-1) # [batch_size, 1, 1]
        # print(t_real_tensor)
        assert t_real_tensor.dim() == 3, f"Expected 3D tensor, got {t_real_tensor.shape} tensor"
        t_real_tensor = t_real_tensor.expand(-1,spatio_temporal_features.size(1),-1) # [batch_size, seq_len, 1]
        
        assert spatio_temporal_features.shape[:2] == t_fake_tensor.shape[:2]
        
        classifier_fake_in = torch.concat([spatio_temporal_features, t_fake_tensor], dim=-1) # [batch_size, seq_len, input_dim + 1]
        classifier_real_in = torch.concat([spatio_temporal_features, t_real_tensor], dim=-1) # [batch_size, seq_len, input_dim + 1]
        
        pooled_fake = classifier_fake_in.mean(dim=1) # [batch_size, input_dim + 1]
        pooled_real = classifier_real_in.mean(dim=1) # [batch_size, input_dim + 1]

        out_fake = self.classifier(pooled_fake) # [batch_size, 1]
        out_real = self.classifier(pooled_real) # [batch_size, 1]
        return out_fake, out_real
        
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
class WGANCritic(nn.Module):
    def __init__(self,input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def pooling_sum(self, hiddens, lens):
        lens = lens.to(hiddens.device)
        lens = torch.autograd.Variable(torch.unsqueeze(lens, dim=1), requires_grad=False)
        batch_size = range(hiddens.shape[0])
        for i in batch_size:
            hiddens[i, 0] = torch.sum(hiddens[i, :lens[i]], dim=0)
        return hiddens[list(batch_size), 0]
    
    def forward(self, spatio_temporal_features, t: torch.Tensor | None, lens):
        if t is not None:
            spatio_temporal_features = spatio_temporal_features.permute(1,0,2) # [batch_size, seq_len, input_dim]
            t_tensor = t.unsqueeze(-1) # [batch_size, 1, 1]
            # print("t_tensor shape: ", t_tensor.shape)
            t_tensor = t_tensor.expand(t_tensor.size(0),spatio_temporal_features.size(1),t_tensor.size(-1)) # [batch_size, seq_len, 1]
            d_input = torch.concat([spatio_temporal_features, t_tensor], dim=-1) # [batch_size, seq_len, input_dim + 1]        
        else:
            d_input = spatio_temporal_features
        # assert d_input.size(-1) == self.input_dim + 1, f"Expected input dim {self.input_dim + 1}, got {d_input.shape[2]}"
        output = self.model(d_input) # [batch_size, seq_len, 1]
        
        pooled_output = self.pooling_sum(output,lens) # [batch_size,1]
        # print(pooled_output.shape)
        # exit()
        return pooled_output, d_input