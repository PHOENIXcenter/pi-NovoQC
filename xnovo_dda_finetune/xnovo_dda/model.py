"""π-xNovo model definition (DDA version).

The core architecture is implemented via a Transformer-based
:class:`depthcharge_new.components.transformers.SpectrumEncoder` and
:class:`depthcharge_new.components.transformers.PeptideDecoder`.

Important
---------
This code depends on a local / forked package named ``depthcharge_new``.
For public release, make sure to:
  1) vendor your fork into this repository, or
  2) publish it as an installable package, or
  3) refactor imports to the upstream ``depthcharge`` package.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from depthcharge_new.components.transformers import PeptideDecoder, SpectrumEncoder

from . import evaluate

class Spec2Pep(nn.Module):

    def __init__(self,
        dim_model = 512,
        n_head = 8,
        dim_feedforward = 1024,
        n_layers = 9,
        dropout = 0.0,
        dim_intensity = None,
        max_length = 100,
        residues = "canonical",
        max_charge = 5,
        use_by_mask=True,
        train_mask_ratio=0.0,
        train_signal=False,
    ):
        super(Spec2Pep, self).__init__()
        self.dim_model  =dim_model
        self.use_by_mask = use_by_mask
        self.mask_ratio = train_mask_ratio
        self.train_signal = train_signal

        # Build the model.
        self.encoder = SpectrumEncoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            dim_intensity=dim_intensity,
            max_charge=max_charge,
        )
        self.decoder = PeptideDecoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            residues=residues,
            max_charge=max_charge,
        )
        self.softmax = torch.nn.Softmax(dim=2)
        self.celoss = torch.nn.CrossEntropyLoss(weight=None, ignore_index=0)
        self.celoss_2 = torch.nn.MSELoss(reduction='none')
        # Data properties.
        self.max_length = max_length
        
        self.final_layer_1 = nn.Linear(dim_model, 256)
        self.final_layer_2 = nn.Linear(256, 128)
        self.final_layer_3 = nn.Linear(128, 64)
        self.final_layer_4 = nn.Linear(64,2)
        self.final_mass = nn.Linear(dim_model,1)

        self.adaptiveAvgPool_layer = nn.AdaptiveAvgPool1d(1)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, spectra, precursors):
        aa_scores, tokens = self.greedy_decode(
            spectra.to(self.encoder.device),
            precursors.to(self.decoder.device),
        )
        return [self.decoder.detokenize(t) for t in tokens], aa_scores

    def greedy_decode(self, spectra, precursors):
        memories, mem_masks = self.encoder(spectra, precursors)
        # Initialize the scores.
        scores = torch.zeros(
            spectra.shape[0], self.max_length + 1, self.decoder.vocab_size + 1
        ).type_as(spectra)
        # Start with the first amino acid predictions.
        scores[:, :1, :], _ = self.decoder(
            None, precursors, memories, mem_masks
        )
        tokens = torch.argmax(scores, axis=2)
        # Keep predicting until a stop token is predicted or max_length is
        # reached.
        # The stop token does not count towards max_length.
        for i in range(2, self.max_length + 2):
            decoded = (tokens == self.decoder._aa2idx["$"]).any(axis=1)
            if decoded.all():
                break
            scores[~decoded, :i, :], _ = self.decoder(
                tokens[~decoded, : (i - 1)],
                precursors[~decoded, :],
                memories[~decoded, :, :],
                mem_masks[~decoded, :],
            )

            tokens = torch.argmax(scores, axis=2)

        return self.softmax(scores), tokens

    def _forward_step(self, spectra, precursors, sequences):
        if self.use_by_mask:
            """half mz used to predict peptide"""
            # prec_mass prediction
            memories, mem_mask = self.encoder(spectra, precursors) 
            B, L, C = memories.shape
            # x_4 = self.final_layer_1(memories)
            x_1 = F.relu(self.final_layer_1(memories))      # [B,L,512]->[B,L,256]                                
            x_2 = F.relu(self.final_layer_2(x_1))      # [B,L,256]->[B,L,128]                                 
            x_3 = F.relu(self.final_layer_3(x_2))      # [B,L,128]->[B,L,64]
            x_4 = self.final_layer_4(x_3)              # [B,L,64] ->[B,L,2]
            by_mask_pred = self.softmax(x_4)         # [B,L,2]
            by_mask_pred = by_mask_pred.ge(0.5)
            by_mask_pred[mem_mask.int() == 0] = 0    

            # For C-terminal decoding, one may need to reverse the peptide
            # sequence (kept here for historical reasons; not used by default).
            sequences_reverse = []
            for i in range(len(sequences)):
                sequences_reverse.append(''.join(reversed(sequences[i])))
            return  *self.decoder(sequences, precursors, memories, by_mask_pred[...,0], train_signal=self.train_signal, mask_ratio=self.mask_ratio), *self.decoder(sequences, precursors, memories, by_mask_pred[...,1], train_signal=self.train_signal, mask_ratio=self.mask_ratio) 
        else:
            """original prediction."""
            return self.decoder(sequences, precursors, *self.encoder(spectra, precursors), train_signal=self.train_signal, mask_ratio=self.mask_ratio)

    def training_step(self, batch):
        if self.use_by_mask:
            """half mz used to predict peptide"""
            pred, truth, pred_2, truth_2 = self._forward_step(*batch)
            tokens = torch.argmax(pred, axis=2)
            aa_scores = self.softmax(pred)
            peptides_pred_raw = [self.decoder.detokenize(t) for t in tokens]
            aa_precision, aa_recall, pep_recall, _ = self.get_review(peptides_pred_raw, batch[2])
            pred = pred[:, :-1, :].reshape(-1, self.decoder.vocab_size + 1)
            pred_2 = pred_2[:, :-1, :].reshape(-1, self.decoder.vocab_size + 1)
            loss_1 = 0.5 * (self.celoss(pred, truth.flatten()) + self.celoss(pred_2, truth.flatten()))
            return loss_1 , torch.tensor(aa_precision), torch.tensor(aa_recall), torch.tensor(pep_recall)
        else:
            """original prediction."""
            pred, truth = self._forward_step(*batch)
            tokens = torch.argmax(pred, axis=2)
            aa_scores = self.softmax(pred)
            peptides_pred_raw = [self.decoder.detokenize(t) for t in tokens]
            aa_precision, aa_recall, pep_recall, _ = self.get_review(peptides_pred_raw, batch[2])
            pred = pred[:, :-1, :].reshape(-1, self.decoder.vocab_size + 1)
            loss_1 = self.celoss(pred, truth.flatten())
            return loss_1 , torch.tensor(aa_precision), torch.tensor(aa_recall), torch.tensor(pep_recall)

    def validation_step(self, batch):
        # Record the loss.
        loss, _, _, _ = self.training_step(batch)

        # Calculate and log amino acid and peptide match evaluation metrics from
        # the predicted peptides.
        peptides_pred_raw, _ = self.forward(batch[0], batch[1])
        aa_precision, aa_recall, pep_recall, peptides_collocet = self.get_review(peptides_pred_raw, batch[2])
        return loss, torch.tensor(aa_precision), torch.tensor(aa_recall), torch.tensor(pep_recall), peptides_collocet

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
                peptides_pred, peptides_true, self.decoder._peptide_mass.masses
            )
        )
        return aa_precision, aa_recall, pep_recall, peptides_collect

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
        lr_factor_1 = 0.5 * (1 + np.cos(np.pi * epoch / self.max_iters)) 
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
    residues=config["residues"],
    max_charge=config["max_charge"],
    use_by_mask=config['use_by_mask'],
    train_mask_ratio=config['train_mask_ratio'],
    train_signal=config['train_signal'],
    )
    if args.multiprocessing == True:
        loc = 'cuda:{}'.format(args.local_rank)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False, find_unused_parameters=True)
        if config['modelname'] is not False:
            model.module.load_state_dict(torch.load(config['modelname'], map_location=loc)['state_dict'], strict=False)
            model.eval()
    else:
        model.eval()
        model.to(args.device)
        if config['modelname'] is not False:
            print(config['modelname'])
            model.load_state_dict(torch.load(config['modelname'], map_location=args.device)['state_dict'], strict=False)
            model.eval()
    return model



