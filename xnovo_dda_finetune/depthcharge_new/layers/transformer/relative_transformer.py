
from typing import Optional
import torch
import torch.nn
import torch.nn.functional as F
from .transformer import ActivationFunction
from .multi_head_relative_pos_attention import FixedRelativeMultiheadAttention, AttentionMask, SpectraMultiheadAttention, PeptideMultiheadAttention
from .multi_head_attention import MultiHeadAttention
from .transformer import Transformer, TransformerEncoderWithLayer, TransformerDecoderWithLayer


class RelativeTransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation: ActivationFunction = F.relu):
        super().__init__()
        self.self_attn = FixedRelativeMultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = activation
        self.reset_parameters()

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src2 = self.self_attn(src, src, AttentionMask(mask, None))
        src = src + self.dropout1(src2)
        src = self.norm1(src) 
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight, gain=torch.nn.init.calculate_gain('relu')
                                      if self.activation is F.relu else 1.0)
        torch.nn.init.xavier_uniform_(self.linear2.weight)

class SpectraTransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation: ActivationFunction = F.relu):
        super().__init__()
        self.self_attn = SpectraMultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = activation

        self.linear1_edge = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout_edge = torch.nn.Dropout(dropout)
        self.linear2_edge = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1_edge = torch.nn.LayerNorm(d_model)
        self.norm2_edge = torch.nn.LayerNorm(d_model)
        self.dropout1_edge = torch.nn.Dropout(dropout)
        self.dropout2_edge = torch.nn.Dropout(dropout)

        self.activation_edge = activation
        self.reset_parameters()

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                mz_abs_embedding=None, mz_rel_embedding=None) -> torch.Tensor:
        """
        src:[B, L, C]
        mask: [B,cL]
        mz_abs_embedding: [B, 1, L]
        mz_rel_embedding: [B, L, C]
        """
        """node features."""
        B, L, C = src.shape
        src2, attn_scores = self.self_attn(src, src, mz_abs_embedding=mz_abs_embedding,
                                mz_rel_embedding=mz_rel_embedding,
                                mask=AttentionMask(mask, None))
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
        """edge features.""" 
        edge = mz_rel_embedding + self.dropout1_edge(attn_scores)
        edge = self.norm1_edge(edge)
        edge_2 = self.linear2_edge(self.dropout_edge(self.activation_edge(self.linear1_edge(edge))))
        edge = edge + self.dropout2_edge(edge_2)
        edge = self.norm2_edge(edge)
        return src, edge

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight, gain=torch.nn.init.calculate_gain('relu')
                                      if self.activation is F.relu else 1.0)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        # torch.nn.init.xavier_uniform_(self.linear1_edge.weight, gain=torch.nn.init.calculate_gain('relu')
        #                               if self.activation_edge is F.relu else 1.0)
        # torch.nn.init.xavier_uniform_(self.linear2_edge.weight)

class RelativeTransformerDecoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation: ActivationFunction = F.relu):
        super().__init__()

        self.self_attn = FixedRelativeMultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

        self.activation = activation
        self.reset_parameters()

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                full_target: Optional[torch.Tensor] = None, pos_offset: int = 0) -> torch.Tensor:
        assert pos_offset == 0 or tgt_mask is None
        B,L,C = tgt.shape
        tgt2 = self.self_attn(tgt, tgt if full_target is None else full_target, mask=AttentionMask(None, tgt_mask),
                              pos_offset=pos_offset)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, scores = self.multihead_attn(tgt, memory, mask=AttentionMask(memory_key_padding_mask, None))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, scores

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight, gain=torch.nn.init.calculate_gain('relu')
                                      if self.activation is F.relu else 1.0)
        torch.nn.init.xavier_uniform_(self.linear2.weight)


class RelativeTransformer(Transformer):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: ActivationFunction = F.relu):

        super().__init__(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, activation,
                         TransformerEncoderWithLayer(RelativeTransformerEncoderLayer),
                         TransformerDecoderWithLayer(RelativeTransformerDecoderLayer))



class PeptideTransformerDecoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation: ActivationFunction = F.relu):
        super().__init__()

        self.self_attn = PeptideMultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

        self.activation = activation
        self.reset_parameters()

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                full_target: Optional[torch.Tensor] = None, peps_matrix = None) -> torch.Tensor:
        tgt2 = self.self_attn(tgt, tgt if full_target is None else full_target, peps_matrix=peps_matrix, mask=AttentionMask(None, tgt_mask))
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, mask=AttentionMask(memory_key_padding_mask, None))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight, gain=torch.nn.init.calculate_gain('relu')
                                      if self.activation is F.relu else 1.0)
        torch.nn.init.xavier_uniform_(self.linear2.weight)