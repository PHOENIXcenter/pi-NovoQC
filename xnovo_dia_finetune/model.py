"""A de novo peptide sequencing model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import numpy as np
import depthcharge_new.masses
from depthcharge_new.components.transformers import SpectrumEncoder, PrecursorDecoder, PeptideDecoder, extract_sub_spectra, generate_sub_spectra_precursors
from depthcharge_new.masses import PeptideMass
from depthcharge_new.components.encoders import MassEncoder

import evaluate

import torch

def normalize_intensities(tensor):
    intensities = tensor[..., 1]
    max_val = torch.max(intensities, dim=-1, keepdim=True).values
    normalized_intensities = torch.where(max_val != 0, intensities / (max_val+1e-10), intensities)
    tensor_normalized = tensor.clone()
    tensor_normalized[..., 1] = normalized_intensities
    return tensor_normalized


class Spec2Pep(nn.Module):
    def __init__(self,
        dim_model = 512,
        n_head = 8,
        dim_feedforward = 1024,
        n_layers = 9,
        dropout = 0.0,
        dim_intensity = None,
        max_length = 100,
        prec_length = 5,
        residues = "canonical",
        max_charge = 5,
        precursor_mass_tol = 50,
        isotope_error_range = (0, 1),
        train_signal=False,
        train_mask_ratio=0.0,
        sub_peaks=50,
    ):
        super(Spec2Pep, self).__init__()
        self.dim_model  =dim_model
        self.mask_ratio = train_mask_ratio
        self.train_signal = train_signal
        self.max_c = max_charge
        self.sub_peaks = sub_peaks
        
        # Build the model.
        self.spec_enc = SpectrumEncoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            dim_intensity=dim_intensity,
            max_charge=max_charge,
        )
        self.spec_enc_sub = SpectrumEncoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            dim_intensity=dim_intensity,
            max_charge=max_charge,
        )
        self.prec_dec = PrecursorDecoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            max_charge=max_charge,
            max_length=prec_length,
        )
        self.pep_dec = PeptideDecoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            residues=residues,
            max_charge=max_charge,
        )
        
        self.softmax = torch.nn.Softmax(dim=2)
        self.loss_m = torch.nn.SmoothL1Loss() # MSELoss()
        self.loss_c = torch.nn.CrossEntropyLoss(weight=None, ignore_index=0)
        self.loss_frag = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.0000, 1.8000, 1.8000, 1.8000, 1.8000, 1.8000, 1.8000, 1.8000, 1.8000,
                                                                        1.8000, 1.8000, 1.8000, 1.8000, 1.8000, 1.8000, 1.8000, 1.8000, 1.8000,
                                                                        1.8000, 0.01000]), ignore_index=0)
        self.loss_pep = torch.nn.CrossEntropyLoss(weight=None, ignore_index=0) 
        
        # Data properties.
        self.max_length = max_length
        self.prec_length = prec_length
        self.residues = residues
        self.precursor_mass_tol = precursor_mass_tol
        self.isotope_error_range = isotope_error_range
        self.peptide_mass_calculator = depthcharge_new.masses.PeptideMass(
            self.residues
        )
        self.stop_token = self.pep_dec._aa2idx["$"]
    
    
    def forward(self, spectra, windows):
        aa_scores, tokens = self.greedy_decode(
            spectra.to(self.spec_enc.device),
            windows.to(self.pep_dec.device),
        )
        return [self.pep_dec.detokenize(t) for t in tokens], aa_scores

    def greedy_decode(self, spectra, windows):

        memories, mem_masks = self.spec_enc.inference(normalize_intensities(spectra))
        # predicted precursor information
        # Start with the first amino acid predictions.
        pred_prec_m, pred_prec_c, prec_attn_weights, _, _, _ = self.prec_dec.xeforward(None, None, None, windows, memories, mem_masks, is_inference=True)
        # Extract subspectra.
        sub_spectra, sub_windows, _ = extract_sub_spectra(spectra, prec_attn_weights, pred_prec_c, windows, topk=self.sub_peaks, stop_class=self.max_c)
        new_memories, new_mem_masks = self.spec_enc.inference(normalize_intensities(sub_spectra))

        # predicted peptides
        # Initialize the scores.
        scores = torch.zeros(
            new_memories.shape[0], self.max_length + 1, self.pep_dec.vocab_size + 1
        ).type_as(sub_spectra)
        # Start with the first amino acid predictions.
        scores[:, :1, :], _, _ = self.pep_dec(sub_windows, None, new_memories, new_mem_masks)
        # print(scores)
        tokens = torch.argmax(scores, axis=2)
        # Keep predicting until a stop token is predicted or max_length is
        # reached.
         #The stop token does not count towards max_length.

        for i in range(2, self.max_length + 2):
            decoded = (tokens == self.pep_dec._aa2idx["$"]).any(axis=1)
            if decoded.all():
                break
            scores[~decoded, :i, :], _, _ = self.pep_dec(
                sub_windows[~decoded, :, :], 
                tokens[~decoded, : (i - 1)], 
                new_memories[~decoded, :, :],
                new_mem_masks[~decoded, :],
            )
            tokens = torch.argmax(scores, axis=2)
        _, _, pep_attn_weights = self.pep_dec(
                                    sub_windows,
                                    tokens,
                                    new_memories,
                                    new_mem_masks,
                                )            
        return self.softmax(scores), tokens
    
    def _forward_step(self, spectra, precs, windows, sequences, mz_labels):

        truth_length = [len(seq) for seq in sequences]
        memories, mem_masks = self.spec_enc.inference(normalize_intensities(spectra))
        pred_prec_m, pred_prec_c, prec_attn_weights, _, truth_mz_labels, truth_prec_c, loss_m, loss_c, pred_frags = self.prec_dec.xeforward(mz_labels, sequences, precs, windows, memories, mem_masks)
        sub_spectra, new_windows, new_peps = extract_sub_spectra(spectra, prec_attn_weights, pred_prec_c, windows, sequences=sequences, topk=self.sub_peaks, stop_class=self.max_c)
        new_memories, new_mem_masks = self.spec_enc.inference(normalize_intensities(sub_spectra))
        pred_pep, truth_pep, pep_attn_weights = self.pep_dec(new_windows, new_peps, new_memories, new_mem_masks, train_signal=self.train_signal, mask_ratio=self.mask_ratio)
        return pred_pep, truth_pep, new_peps, prec_attn_weights, loss_m, loss_c, pred_frags, truth_mz_labels
    
    def training_step(self, spectra, precs, windows, sequences, mz_labels):
        pred_pep, truth_pep, sub_peptides, prec_attn_weights, loss_m, loss_c, pred_frags, truth_mz_labels = self._forward_step(spectra, precs, windows, sequences, mz_labels)
        pep_tokens = torch.argmax(pred_pep, axis=2)
        aa_scores = self.softmax(pred_pep)
        peptides_pred_raw = [self.pep_dec.detokenize(t) for t in pep_tokens]
        aa_precision, aa_recall, pep_recall, _ = self.get_review(peptides_pred_raw, sub_peptides)
        loss_frag = self.loss_frag(pred_frags.reshape(-1, 20), truth_mz_labels.flatten().long())
        loss_pep = self.loss_pep(pred_pep[:,:-1, :].reshape(-1, self.pep_dec.vocab_size+1), truth_pep.flatten())
        return (loss_m, loss_c, loss_frag, loss_pep), torch.tensor(aa_precision), torch.tensor(aa_recall), torch.tensor(pep_recall), sub_peptides

    def validation_step(self, spectra, precs, windows, sequences, mz_labels):
        # Record the loss.
        loss, _, _, _, sub_peptides = self.training_step(spectra, precs, windows, sequences, mz_labels)

        # Calculate and log amino acid and peptide match evaluation metrics from
        # the predicted peptides.
        peptides_pred_raw, _ = self.forward(spectra, windows)
        aa_precision, aa_recall, pep_recall, peptides_collocet = self.get_review(peptides_pred_raw, sub_peptides)
        peptides_collocet = self.collect_peptides(peptides_pred_raw, sub_peptides)
        return loss, torch.tensor(aa_precision), torch.tensor(aa_recall), torch.tensor(pep_recall), peptides_collocet, sub_peptides
    
#     @profile
    def get_review(self, pred, truth):
        # FIXME: Temporary fix to skip predictions with multiple stop tokens.
        peptides_pred, peptides_true = [], []
        peptides_collect = []
        for peptide_pred, peptide_true in zip(pred, truth):
            if len(peptide_pred) > 0:
                if peptide_pred[0] == "$":
                    peptide_pred = peptide_pred[1:]  # Remove stop token.
                if "$" not in peptide_pred and len(peptide_pred) > 0:
                    peptides_pred.append(peptide_pred)
                    peptides_true.append(peptide_true)
                    peptides_collect.append([''.join(peptide_pred), peptide_true])
        aa_precision, aa_recall, pep_recall = evaluate.aa_match_metrics(
            *evaluate.aa_match_batch(
                peptides_pred, peptides_true, self.pep_dec._peptide_mass.masses
            )
        )
        return aa_precision, aa_recall, pep_recall, peptides_collect
    
    def collect_peptides(self, pred, truth):
        # FIXME: Temporary fix to skip predictions with multiple stop tokens.
        peptides_pred, peptides_true = [], []
        peptides_collect = []
        for peptide_pred, peptide_true in zip(pred, truth):
            if peptide_pred[0] == "$":
                peptide_pred = peptide_pred[1:]  # Remove stop token.
            peptides_pred.append(peptide_pred)
            peptides_true.append(peptide_true)
            peptides_collect.append([''.join(peptide_pred), peptide_true])
        return peptides_collect
class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters, loader_len, start_epoch, max_epochs, old=False):
        self.warmup, self.max_iters = warmup, max_iters
        self.loader_len = loader_len
        self.start_epoch = start_epoch
        self.max_epochs = max_epochs
        self.old = old
        if self.old:
            print("use old schedular!")
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        epoch += self.start_epoch * self.loader_len
        lr_factor_1 = 0.5 * (1 + np.cos(np.pi * epoch / self.max_iters)) * 0.4
        lr_factor_2 = epoch / self.warmup
        lr_factor = min(lr_factor_1, lr_factor_2)
        if self.old:
            if epoch < self.warmup:
                lr_factor = lr_factor_2
            else:
                lr_factor = lr_factor_1
        return lr_factor

# Bulid Model
def build_model(args, config):

    model = Spec2Pep(
    dim_model=config["dim_model"],
    n_head=config["n_head"],
    dim_feedforward=config["dim_feedforward"],
    n_layers=config["n_layers"],
    dropout=config["dropout"],
    dim_intensity=config["dim_intensity"],
    max_length=config["max_length"],
    prec_length=config['prec_length'],
    residues=config["residues"],
    max_charge=config["max_charge"],
    precursor_mass_tol=config["precursor_mass_tol"],
    isotope_error_range=config["isotope_error_range"],
    train_signal=config['train_signal'],
    train_mask_ratio=config['train_mask_ratio'],
    sub_peaks=config['sub_peaks'],
    )
    if args.multiprocessing == True:
        loc = 'cuda:{}'.format(args.local_rank)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False, find_unused_parameters=True)
        
        # TimsTOF DDA pretrain model
        if args.pretrain_model == 'TimsTOF_DDA':
            cpt_massive = torch.load('pretrained_DDA_weights/TimsTOF_DDA_pretrained.pth',
                            map_location=loc)['state_dict']
        elif args.pretrain_model == 'MassIVE_KB':
        # MassIVE pretrain model
            cpt_massive = torch.load('pretrained_DDA_weights/MassIVE_KB_trained_xNovo_DDA.pth',
                            map_location=loc)['state_dict']
        enc_stat_dict = {}
        for k,v in cpt_massive.items():
            if 'encoder.latent_spectrum' in k:
                enc_stat_dict[k.replace('encoder.latent_spectrum','spec_enc.latent_spectrum')] = v
                enc_stat_dict[k.replace('encoder.latent_spectrum','spec_enc_sub.latent_spectrum')] = v         
            if 'encoder.peak' in k:
                enc_stat_dict[k.replace('encoder.peak','spec_enc.peak')] = v
                enc_stat_dict[k.replace('encoder.peak','spec_enc_sub.peak')] = v   
                enc_stat_dict[k.replace('encoder.peak','prec_dec.peak')] = v   
            if 'encoder.transformer_encoder' in k :
                enc_stat_dict[k.replace('encoder.t','spec_enc.t')] = v
                enc_stat_dict[k.replace('encoder.t','spec_enc_sub.t')] = v
            if 'decoder.transformer_decoder' in k :
                enc_stat_dict[k.replace('decoder.t','prec_dec.t')] = v
                enc_stat_dict[k.replace('decoder.t','pep_dec.t')] = v
                
           # decoder
            if 'decoder.transformer_decoder' in k :
                enc_stat_dict[k.replace('decoder.t','prec_dec.t')] = v
                enc_stat_dict[k.replace('decoder.t','pep_dec.t')] = v
            if 'decoder.aa_encoder' in k:
                enc_stat_dict[k.replace('decoder.aa_encoder','pep_dec.aa_encoder')] = v
            if 'decoder.final' in k:
                enc_stat_dict[k.replace('decoder.final','pep_dec.final')] = v
            if 'decoder.mass_encoder' in k:
                enc_stat_dict[k.replace('decoder.mass_encoder','pep_dec.mass_encoder')] = v    
        model.module.load_state_dict(cpt_massive, strict=False)
        print('Loaded pretrain model!!!')
        model.eval()
        if config['modelname'] is not False:
            model.module.load_state_dict(torch.load(config['modelname'], map_location=loc)['state_dict'])
            model.eval()
    else:
        # TimsTOF DDA pretrain model
        if args.pretrain_model == 'TimsTOF_DDA':
            cpt_massive = torch.load('pretrained_DDA_weights/TimsTOF_DDA_pretrained.pth',
                            map_location=loc)['state_dict']
        elif args.pretrain_model == 'MassIVE_KB':
        # MassIVE pretrain model
            cpt_massive = torch.load('pretrained_DDA_weights/MassIVE_KB_trained_xNovo_DDA.pth',
                            map_location=loc)['state_dict']
        enc_stat_dict = {}
        for k,v in cpt_massive.items():
            if 'encoder.latent_spectrum' in k:
                enc_stat_dict[k.replace('encoder.latent_spectrum','spec_enc.latent_spectrum')] = v
                enc_stat_dict[k.replace('encoder.latent_spectrum','spec_enc_sub.latent_spectrum')] = v         
            if 'encoder.peak' in k:
                enc_stat_dict[k.replace('encoder.peak','spec_enc.peak')] = v
                enc_stat_dict[k.replace('encoder.peak','spec_enc_sub.peak')] = v   
                enc_stat_dict[k.replace('encoder.peak','prec_dec.peak')] = v   
            if 'encoder.transformer_encoder' in k :
                enc_stat_dict[k.replace('encoder.t','spec_enc.t')] = v
                enc_stat_dict[k.replace('encoder.t','spec_enc_sub.t')] = v
            if 'decoder.transformer_decoder' in k :
                enc_stat_dict[k.replace('decoder.t','prec_dec.t')] = v
                enc_stat_dict[k.replace('decoder.t','pep_dec.t')] = v
                
           # decoder
            if 'decoder.transformer_decoder' in k :
                enc_stat_dict[k.replace('decoder.t','prec_dec.t')] = v
                enc_stat_dict[k.replace('decoder.t','pep_dec.t')] = v
            if 'decoder.aa_encoder' in k:
                enc_stat_dict[k.replace('decoder.aa_encoder','pep_dec.aa_encoder')] = v
            if 'decoder.final' in k:
                enc_stat_dict[k.replace('decoder.final','pep_dec.final')] = v
            if 'decoder.mass_encoder' in k:
                enc_stat_dict[k.replace('decoder.mass_encoder','pep_dec.mass_encoder')] = v    
        model.load_state_dict(cpt_massive, strict=False)
        print('Loaded pretrain model!!!')
        
        model.to(args.device)
        if config['modelname'] is not False:
            model.load_state_dict(torch.load(config['modelname'], map_location=args.device)['state_dict'])
            model.eval()
    return model



