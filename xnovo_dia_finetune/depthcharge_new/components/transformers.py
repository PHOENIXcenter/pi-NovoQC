"""Base Transformer models for working with mass spectra and peptides"""
import re
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from depthcharge_new.components.encoders import MassEncoder, PeakEncoder, PositionalEncoder
from depthcharge_new.masses import PeptideMass
from depthcharge_new import utils
from depthcharge_new.layers.transformer.relative_transformer import RelativeTransformerEncoderLayer, RelativeTransformerDecoderLayer
    
class SpectrumEncoder(torch.nn.Module):
    def __init__(
        self,
        dim_model=512,
        n_head=8,
        dim_feedforward=1024,
        n_layers=2,
        dropout=0.1,
        peak_encoder=True,
        dim_intensity=None,
        max_charge=5,
    ):
        """Initialize a SpectrumEncoder"""
        super().__init__()

        self.latent_spectrum = torch.nn.Parameter(torch.zeros(1, 1, dim_model))
        self.layer_num = n_layers

        # peak encoder
        if peak_encoder:
            self.peak_encoder = PeakEncoder(
                dim_model,
                dim_intensity=dim_intensity,
            )
        else:
            self.peak_encoder = torch.nn.Linear(2, dim_model)
        
        # The Transformer layers:
        layer = RelativeTransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
            
        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
        )
        
        # predicted precursors encoder (m/z, charge)
        self.mass_encoder = MassEncoder(dim_model)
        self.charge_encoder = torch.nn.Embedding(max_charge+1, dim_model, padding_idx=0)
    
    def inference(self, spectra):
        # peaks mask
        mask = ~spectra.sum(dim=2).bool() 
        # get peaks embedding
        _, peaks = self.peak_encoder(spectra)   
        # Feed through model
        features = peaks
        for l in self.transformer_encoder.layers:
            features =l(features, mask)
        return features, mask
    
    def forward(self, spectra, precursors):
        # peaks mask
        zeros = ~spectra.sum(dim=2).bool() 
        # get peaks embedding
        _, peaks = self.peak_encoder(spectra)   

        mz_start = torch.ones((spectra.size(0),1,1)).to(self.device) * (18.0106 + 1.0078) 
        start_masses = self.mass_encoder(mz_start)
        peaks, mask = self.get_latent(zeros, peaks, spectra, start_masses)
        
        features, mask = self.relative_attn_encoder(peaks, mask, spectra, precursors)
        return features, mask
    
    def relative_attn_encoder(self, peaks, mask, spectra, precursors):
        # prepare mass and charge
        masses = self.mass_encoder(precursors[:, None, [0]])
        charges = self.charge_encoder(precursors[:, 1].int())
        precursors_encoding = masses + charges[:, None, :]
        peaks, mask = self.get_new_peaks(peaks, precursors_encoding, mask, add_len=1)

        # Feed through model
        features = peaks
        for l in self.transformer_encoder.layers:
            features =l(features, mask)
        return features, mask
    
    def get_latent(self, zeros, peaks, spectra, start_encoding):
        mask = [
            torch.tensor([[False]] * spectra.shape[0]).type_as(zeros),
            zeros,
        ]
        mask = torch.cat(mask, dim=1)    
        # Add the spectrum representation to each input:
        latent_spectra = self.latent_spectrum.repeat(peaks.shape[0], 1, 1)
        start_encoding += latent_spectra
        peaks = torch.cat([start_encoding, peaks], dim=1)
        return peaks, mask
    
    def get_new_peaks(self, peaks, precursors, mask, add_len=1):
        B, L, C = peaks.shape
        new_peaks = torch.zeros((B, L + add_len, C), device=self.device)
        new_mask = torch.ones((B, L + add_len), dtype=torch.bool, device=self.device)
        for i in range(B):
            element_length = peaks[i,~mask[i]].size(0) + add_len
            new_peaks[i,:element_length,:] = torch.cat((peaks[i,~mask[i],:], precursors[i,:,:]), dim=0)   
            new_mask[i,:element_length] = False
        return new_peaks, new_mask

    @property
    def device(self):
        """The current device for the model"""
        return next(self.parameters()).device

class PrecursorDecoder(torch.nn.Module):
    def __init__(
        self,
        dim_model=512,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0,
        max_charge=5,
        max_length=5,
    ):
        super(PrecursorDecoder, self).__init__()
        # precursors encoder (m/z, charge)
        self.max_c = max_charge
        self.max_l = max_length
        self.fragment_predictor = torch.nn.Linear(n_head * n_layers, 20)
        
        self.peak_encoder = PeakEncoder(
                dim_model,
                dim_intensity=None,
            )
        self.latent_prec = torch.nn.Parameter(torch.zeros(1, self.max_l, dim_model))
        self.mass_encoder = MassEncoder(dim_model)
        self.charge_encoder = torch.nn.Embedding(max_charge+1, dim_model)
        self.length_encoder = torch.nn.Embedding(max_length+1, dim_model, padding_idx=0)
        self.pos_encoder = PositionalEncoder(dim_model)

        layer = RelativeTransformerDecoderLayer(
                d_model=dim_model,
                nhead=n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
        self.transformer_decoder = torch.nn.TransformerDecoder(
            layer,
            num_layers=n_layers,
        )
        
        self.prec_mass = torch.nn.Linear(dim_model, 1)
        self.prec_charge = torch.nn.Linear(dim_model, max_charge+1)
        self.mass_loss_fn = nn.MSELoss()  
        self.charge_loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # weight=torch.tensor([0, 0.2504, 0.0165, 0.0457, 0.6761, 0.0114]))  
        
    def xeforward(self, mz_labels, sequences, prec_seqs, isolation_windows, memory, memory_key_padding_mask, is_inference=False):
        # prepare isolation window centor
        windows = self.mass_encoder(isolation_windows[:,:,[0]])
        tgt = windows.repeat(1, self.max_l, 1) + self.latent_prec.repeat(windows.shape[0], 1, 1)
        encoded, tgt = self.pos_encoder(tgt)
        
        stop_token = torch.tensor([self.max_c]).int().to(self.device)
        if prec_seqs is not None:
            target_masses = prec_seqs[:,:,0]
            zeros = ~prec_seqs.sum(dim=2).bool()  
            masses = self.mass_encoder(prec_seqs[:, :, [0]]) 
            charges = self.charge_encoder(prec_seqs[:, :, 1].int()) 
            _precursors = masses + charges
            if not is_inference:
                B, L, C = _precursors.shape
                target_charges = torch.zeros((B,L+1)).to(self.device)
                precursors = torch.zeros((B,L+1, C)).to(self.device)
                for i in range(B):
                    element_length = torch.sum(~zeros[i]).item() + 1
                    precursors[i, :element_length, :] = torch.cat((_precursors[i, ~zeros[i], :], self.charge_encoder(stop_token)), dim=0)
                    target_charges[i, :element_length] = torch.cat((prec_seqs[i,~zeros[i], 1].int(), stop_token))
            padding_length = self.max_l - target_masses.size(1) # [B,L]
            padding_length_c = self.max_l - target_charges.size(1) # [B,L]
            if padding_length > 0:
                target_masses = F.pad(target_masses, (0, padding_length), value=0)  
                target_charges = F.pad(target_charges, (0, padding_length_c), value=0)  
                precursors = F.pad(precursors, (0, 0, 0, padding_length_c), value=0)
        else:
            target_charges = torch.tensor([[]]).to(self.device)
            new_peptides = None
            
        if not is_inference:
            bsz, output_len = target_charges.size()
            final_preds, scores_list = self.decode(memory, tgt, memory_key_padding_mask)
            pred_masses = self.prec_mass(final_preds) 
            pred_charges = self.prec_charge(final_preds)     
            
            pred_frags = self.fragment_predictor(scores_list.permute(0,2,3,1)) # [B, 4, mz, 20]
            loss_mass = self.mass_loss_fn(pred_masses[:,:,0], target_masses / 10000)
            loss_charge = self.charge_loss_fn(pred_charges.reshape(-1, self.max_c+1), target_charges.flatten().long())
            return pred_masses, pred_charges, scores_list.mean(dim=1), sequences, mz_labels, target_charges, loss_mass, loss_charge, pred_frags
        else:
            preds, scores_list = self.decode(memory, tgt, memory_key_padding_mask)
            pred_masses = self.prec_mass(preds)  
            pred_charges = self.prec_charge(preds)  
            return pred_masses, pred_charges, scores_list.mean(dim=1), None, None, target_charges


    def decode(self, memory, tgt, memory_key_padding_mask):
        preds, output_list, scores_list = tgt, [], []
        for i in range(len(self.transformer_decoder.layers)):
            preds, scores = self.transformer_decoder.layers[i].forward(
                tgt=preds,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask.to(self.device),
            )
            scores_list.append(scores)
        scores_list = torch.cat(scores_list, dim=1) # .mean(dim=1)
        return preds, scores_list

    @property
    def device(self):
        """The current device for the model"""
        return next(self.parameters()).device
    
class _PeptideTransformer(torch.nn.Module):
    def __init__(
        self,
        dim_model,
        residues,
        max_charge,
    ):
        super().__init__()
        self.reverse = False
        self._peptide_mass = PeptideMass(residues=residues)
        self._amino_acids = list(self._peptide_mass.masses.keys()) + ["$"]
        self._idx2aa = {i + 1: aa for i, aa in enumerate(self._amino_acids)}
        self._aa2idx = {aa: i for i, aa in self._idx2aa.items()}

        self.pos_encoder = PositionalEncoder(dim_model)
        self.charge_encoder = torch.nn.Embedding(max_charge+1, dim_model, padding_idx=0)
        self.aa_encoder = torch.nn.Embedding(
            len(self._amino_acids) + 1,
            dim_model,
            padding_idx=0,
        )

    def tokenize(self, sequence, partial=False):
        if not isinstance(sequence, str):
            return sequence  # Assume it is already tokenized.

#         sequence = sequence.replace("I", "L")
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)
        if self.reverse:
            sequence = list(reversed(sequence))

        if not partial:
            sequence += ["$"]

        tokens = [self._aa2idx[aa] for aa in sequence]
        tokens = torch.tensor(tokens, device=self.device)
        return tokens

    def detokenize(self, tokens):
        sequence = [self._idx2aa.get(i.item(), "") for i in tokens]
        if "$" in sequence:
            idx = sequence.index("$")
            sequence = sequence[: idx + 1]

        if self.reverse:
            sequence = list(reversed(sequence))

        return sequence

    @property
    def vocab_size(self):
        """Return the number of amino acids"""
        return len(self._aa2idx)

    @property
    def device(self):
        """The current device for the model"""
        return next(self.parameters()).device


class PeptideEncoder(_PeptideTransformer):
    def __init__(
        self,
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0,
        residues="canonical",
        max_charge=5,
    ):
        """Initialize a PeptideEncoder"""
        super().__init__(
            dim_model=dim_model,
            pos_encoder=pos_encoder,
            residues=residues,
            max_charge=max_charge,
        )

        # The Transformer layers:
        layer = torch.nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
        )

    def forward(self, sequences, charges):

        sequences = utils.listify(sequences)
        tokens = [self.tokenize(s) for s in sequences]
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        encoded = self.aa_encoder(tokens)

        # Encode charges
        charges = self.charge_encoder(charges - 1)[:, None]
        encoded = torch.cat([charges, encoded], dim=1)

        # Create mask
        mask = ~encoded.sum(dim=2).bool()

        # Add positional encodings
        encoded = self.pos_encoder(encoded)

        # Run through the model:
        latent = self.transformer_encoder(encoded, src_key_padding_mask=mask)
        return latent, mask


class PeptideDecoder(_PeptideTransformer):

    def __init__(
        self,
        dim_model=512,
        n_head=8,
        dim_feedforward=1024,
        n_layers=2,
        dropout=0,
        reverse=True,
        residues="canonical",
        max_charge=5,
        max_length=5,
        num_iterations=3, 
        aa_max_length=50,
    ):
        """Initialize a PeptideDecoder"""
        super().__init__(
            dim_model=dim_model,
            residues=residues,
            max_charge=max_charge,
        )
        self.reverse = reverse
        self.max_l = max_length
        self.max_aa_l = aa_max_length
        # builf model
        self.mass_encoder = MassEncoder(dim_model)
        layer = RelativeTransformerDecoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_decoder = torch.nn.TransformerDecoder(
            layer,
            num_layers=n_layers,
        )

        self.final = torch.nn.Linear(dim_model, len(self._amino_acids) + 1)
        
        # info for precursors decoding
        self.max_c = max_charge
        self.num_iterations = num_iterations  
        self.prec_mass = torch.nn.Linear(dim_model, 1)
        self.prec_charge = torch.nn.Linear(dim_model, max_charge+1)
        self.mass_loss_fn = nn.MSELoss() 
        self.charge_loss_fn = nn.CrossEntropyLoss()  
        self.ctcloss = nn.CTCLoss(blank=self.max_l, zero_infinity=True)
        
    def forward(self, isolation_windows, sequences, 
                memory, memory_key_padding_mask, 
                train_signal=False, mask_ratio=0.0):
        # Prepare sequences
        if sequences is not None:
            sequences = utils.listify(sequences)
            tokens = [self.tokenize(s) for s in sequences]
            tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        else:
            tokens = torch.tensor([[]]).to(self.device)
        new_tokens = tokens.clone()
        
        if train_signal:
            train_tokens = torch.zeros(tokens.shape).to(self.device)
            for i in range(tokens.size(0)):
                mask = tokens[i,:] == 0
                mask_len = tokens.size(1) - torch.sum(mask)
                rand_mask = torch.rand(mask_len).to(self.device)
                mask[:mask_len] = (rand_mask < mask_ratio).float() + mask[:mask_len].float()
                tokens[i,mask.float()!=0] = 0
        
        precursors = self.mass_encoder(isolation_windows[:,:,[0]])
        # Feed through model:
        if sequences is None:
            tgt = precursors
        else:
            tgt = torch.cat([precursors, self.aa_encoder(tokens)], dim=1)
        
        encoded, tgt = self.pos_encoder(tgt)
        
        tgt_mask = generate_tgt_mask(tgt.shape[1]).type_as(precursors).bool()
        
        preds = tgt
        score_list = []
        for i in range(len(self.transformer_decoder.layers)):
            preds, scores = self.transformer_decoder.layers[i].forward(tgt=preds, 
            memory=memory,
            tgt_mask=tgt_mask.bool(),
            memory_key_padding_mask=memory_key_padding_mask.to(self.device),
            )
            score_list.append(scores)
        scores = torch.cat(score_list, dim=1)
        return self.final(preds), new_tokens, score_list

    
def extract_sub_spectra(memory, attn, prec_c, windows, sequences=None, topk=150, stop_class=5):
    stop_mask = (torch.argmax(prec_c, dim=-1) == stop_class).int()
    cumsum_stop_mask = stop_mask.cumsum(dim=1)
    stop_indices = torch.argmax((cumsum_stop_mask == 1).int(), dim=1)
    attn_shape = attn.size()
    k = min(topk, attn_shape[-1] - 1)  
    _, topk_indices = torch.topk(attn, k=k, dim=-1)

    memory_expanded = memory.unsqueeze(1).expand(-1, topk_indices.size(1), -1, -1)  # [B, topk, N, C]
    memory_repeated = torch.gather(memory_expanded, 2, topk_indices.unsqueeze(-1).expand(-1, -1, -1, memory.size(-1)))

    sub_spectra = []
    stop_token_idxs = []
    new_windows = []
    new_peps = []
    bs, seq_len, _ = topk_indices.size()
    for bs_idx in range(bs):
        stop_idx = stop_indices[bs_idx].item()
        if cumsum_stop_mask[bs_idx, stop_idx] == 0:
            stop_idx = seq_len - 1
        elif stop_idx == 0:  
            stop_idx = 1
        sub_spectrum = memory_repeated[bs_idx, :stop_idx, ...]
        sub_spectra.append(sub_spectrum)
        new_windows.append(windows[[bs_idx],:].repeat(stop_idx, 1, 1))
        if sequences is not None:
            if len(sequences[bs_idx]) < stop_idx:
                new_peps += sequences[bs_idx] + ['$'] * (stop_idx - len(sequences[bs_idx]))
            else:
                new_peps += sequences[bs_idx][:stop_idx]        
    new_windows = torch.cat(new_windows, dim=0) 
    
    sub_spectra = torch.cat(sub_spectra, dim=0)
    sorted_indices = torch.argsort(sub_spectra[:, :, 0], dim=1, descending=False)
    sorted_tensor = torch.gather(sub_spectra, dim=1, index=sorted_indices.unsqueeze(-1).expand(-1, -1, 2))
    return sorted_tensor, new_windows, new_peps

def generate_sub_spectra_precursors(prec_m, prec_c, stop_token_idxs, sequences=None):
    total_length_m = sum(k for k in stop_token_idxs)
    total_length_c = total_length_m  
    _prec_c = torch.argmax(prec_c, dim=-1)
    
    prec_m_token = torch.zeros(total_length_m, device=prec_m.device)
    prec_c_token = torch.zeros(total_length_c, dtype=torch.long, device=prec_c.device)

    index_m = 0
    new_peps = []
    for i, k in enumerate(stop_token_idxs):
        next_index_m = index_m + k
        prec_m_token[index_m:next_index_m] = prec_m[i, :k, 0].detach()
        prec_c_token[index_m:next_index_m].copy_(_prec_c[i, :k].detach())
        index_m = next_index_m
        if sequences is not None:
            if len(sequences[i]) < k:
                new_peps += sequences[i] + ['$'] * (k - len(sequences[i]))
            else:
                new_peps += sequences[i][:k]
                
    prec_m_token *= 1000
    return prec_m_token, prec_c_token, new_peps

def generate_tgt_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask