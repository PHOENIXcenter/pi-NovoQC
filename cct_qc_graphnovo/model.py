"""A de novo peptide sequencing model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import numpy as np
import depthcharge_new.masses
from depthcharge_new.components.transformers import PeptideDecoder, SpectrumEncoder
from depthcharge_new.masses import PeptideMass
from depthcharge_new.components.encoders import MassEncoder

import evaluate
import time

class Spec2Pep(nn.Module):
    """
    A Transformer model for de novo peptide sequencing.

    Use this model in conjunction with a pytorch-lightning Trainer.

    Parameters
    ----------
    dim_model : int
        The latent dimensionality used by the transformer model.
    n_head : int
        The number of attention heads in each layer. ``dim_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int
        The dimensionality of the fully connected layers in the transformer
        model.
    n_layers : int
        The number of transformer layers.
    dropout : float
        The dropout probability for all layers.
    dim_intensity : Optional[int]
        The number of features to use for encoding peak intensity. The remaining
        (``dim_model - dim_intensity``) are reserved for encoding the m/z value.
        If ``None``, the intensity will be projected up to ``dim_model`` using a
        linear layer, then summed with the m/z encoding for each peak.
    custom_encoder : Optional[Union[SpectrumEncoder, PairedSpectrumEncoder]]
        A pretrained encoder to use. The ``dim_model`` of the encoder must be
        the same as that specified by the ``dim_model`` parameter here.
    max_length : int
        The maximum peptide length to decode.
    residues: Union[Dict[str, float], str]
        The amino acid dictionary and their masses. By default ("canonical) this
        is only the 20 canonical amino acids, with cysteine carbamidomethylated.
        If "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    max_charge : int
        The maximum precursor charge to consider.
    precursor_mass_tol : float, optional
        The maximum allowable precursor mass tolerance (in ppm) for correct
        predictions.
    isotope_error_range : Tuple[int, int]
        Take into account the error introduced by choosing a non-monoisotopic
        peak for fragmentation by not penalizing predicted precursor m/z's that
        fit the specified isotope error: `abs(calc_mz - (precursor_mz - isotope * 1.00335 / precursor_charge)) < precursor_mass_tol`
    warmup_iters: int
        The number of warm up iterations for the learning rate scheduler.
    max_iters: int
        The total number of iterations for the learning rate scheduler.
    **kwargs : Dict
        Additional keyword arguments passed to the Adam optismizer.
    """

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
        """
        Predict peptide sequences for a batch of MS/MS spectra.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all of the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.

        Returns
        -------
        peptides : List[str]
            The predicted peptide sequences for each spectrum.
        aa_scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The individual amino acid scores for each prediction.
        """
        aa_scores, tokens, attn_scores, final_feats = self.greedy_decode(
            spectra.to(self.encoder.device),
            precursors.to(self.decoder.device),
        )
        return [self.decoder.detokenize(t) for t in tokens], aa_scores, attn_scores, final_feats

    def greedy_decode(self, spectra, precursors):
        """
        Greedy decoding of the spectrum predictions.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all of the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.

        Returns
        -------
        scores : torch.Tensor of shape (n_spectra, max_length, n_amino_acids)
            The individual amino acid scores for each prediction.
        tokens : torch.Tensor of shape (n_spectra, max_length, n_amino_acids)
            The predicted tokens for each spectrum.
        """
        memories, mem_masks = self.encoder(spectra, precursors)
        # Initialize the scores.
        scores = torch.zeros(
            spectra.shape[0], self.max_length + 1, self.decoder.vocab_size + 1
        ).type_as(spectra)
        # Start with the first amino acid predictions.
        scores[:, :1, :], _, attn_scores, final_feats = self.decoder(
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
            scores[~decoded, :i, :], _, attn_scores, final_feats = self.decoder(
                tokens[~decoded, : (i - 1)],
                precursors[~decoded, :],
                memories[~decoded, :, :],
                mem_masks[~decoded, :],
            )

            tokens = torch.argmax(scores, axis=2)
#             print(tokens[~decoded, : (i - 1)].shape, precursors[~decoded, :].shape, memories[~decoded, :, :].shape, mem_masks[~decoded, :].shape)
        _, _, attn_scores, _ = self.decoder(
            tokens,
            precursors,
            memories,
            mem_masks,
        )
#         print(attn_scores[0].shape)
        return self.softmax(scores), tokens, attn_scores, final_feats

    def _forward_step(self, spectra, precursors, sequences, eigh_L=None, peps_matrix=None):
        """
        The forward learning step.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all of the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.
        sequences : List[str] of length n_spectra
            The partial peptide sequences to predict.

        Returns
        -------
        scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The individual amino acid scores for each prediction.
        tokens : torch.Tensor of shape (n_spectra, length)
            The predicted tokens for each spectrum.
        """
        
        if self.use_by_mask:
            """half mz used to predict peptide"""
            # prec_mass prediction
            memories, mem_mask = self.encoder(spectra, precursors) 
            B, L, C = memories.shape
            pred_prec_y = torch.zeros((B,1)).to(self.encoder.device)
            pred_prec_b = torch.zeros((B,1)).to(self.encoder.device)
            # x_4 = self.final_layer_1(memories)
            x_1 = F.relu(self.final_layer_1(memories))      # [B,L,512]->[B,L,256]                                
            x_2 = F.relu(self.final_layer_2(x_1))      # [B,L,256]->[B,L,128]                                 
            x_3 = F.relu(self.final_layer_3(x_2))      # [B,L,128]->[B,L,64]
            x_4 = self.final_layer_4(x_3)              # [B,L,64] ->[B,L,2]
            by_mask_pred = self.softmax(x_4)         # [B,L,2]
            by_mask_pred = by_mask_pred.ge(0.5)
            by_mask_pred[mem_mask.int() == 0] = 0    

            y_prec_mass = precursors[:,[0]] + 1.007276
            b_prec_mass = precursors[:,[0]] + 1.007276 - 18.0106

            sequences_reverse = []
            for i in range(len(sequences)):
                sequences_reverse.append(''.join(reversed(sequences[i])))
            return  *self.decoder(sequences, precursors, memories, by_mask_pred[...,0], train_signal=self.train_signal, mask_ratio=self.mask_ratio), *self.decoder(sequences, precursors, memories, by_mask_pred[...,1], train_signal=self.train_signal, mask_ratio=self.mask_ratio) 
        else:
            """original prediction."""
            return self.decoder(sequences, precursors, *self.encoder(spectra, precursors), train_signal=self.train_signal, mask_ratio=self.mask_ratio)

    def training_step(self, batch):
        """
        A single training step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            peptide sequences as torch Tensors.
        mode : str
            Logging key to describe the current stage.

        Returns
        -------
        torch.Tensor
            The loss of the training step.
        """

        if self.use_by_mask:
            """half mz used to predict peptide"""
            pred, truth, attn_scores, final_feats, pred_2, truth_2, _, _ = self._forward_step(*batch)
            tokens = torch.argmax(pred, axis=2)
            aa_scores = self.softmax(pred)
            peptides_pred_raw = [self.decoder.detokenize(t) for t in tokens]
            aa_precision, aa_recall, pep_recall, peptides_collocet = self.get_review(peptides_pred_raw, batch[2])
            pred = pred[:, :-1, :].reshape(-1, self.decoder.vocab_size + 1)
            pred_2 = pred_2[:, :-1, :].reshape(-1, self.decoder.vocab_size + 1)
            loss_1 = 0.5 * (self.celoss(pred, truth.flatten()) + self.celoss(pred_2, truth.flatten()))
            return loss_1, torch.tensor(aa_precision), torch.tensor(aa_recall), torch.tensor(pep_recall), aa_scores, peptides_collocet, attn_scores, final_feats
        else:
            """original prediction."""
            pred, truth, attn_scores, final_feats = self._forward_step(*batch)
            tokens = torch.argmax(pred, axis=2)
            aa_scores = self.softmax(pred)
            peptides_pred_raw = [self.decoder.detokenize(t) for t in tokens]
            aa_precision, aa_recall, pep_recall, peptides_collocet = self.get_review(peptides_pred_raw, batch[2])
            pred = pred[:, :-1, :].reshape(-1, self.decoder.vocab_size + 1)
            loss_1 = self.celoss(pred, truth.flatten())
            return loss_1 , torch.tensor(aa_precision), torch.tensor(aa_recall), torch.tensor(pep_recall), aa_scores, peptides_collocet, attn_scores, final_feats

    def validation_step(self, batch):
        """
        A single validation step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            peptide sequences.

        Returns
        -------
        torch.Tensor
            The loss of the validation step.
        """
        # Record the loss.
        loss, _, _, _, _, _, _, _ = self.training_step(batch)

        # Calculate and log amino acid and peptide match evaluation metrics from
        # the predicted peptides.
        peptides_pred_raw, aa_score, attn_scores, final_feats = self.forward(batch[0], batch[1])
        aa_precision, aa_recall, pep_recall, peptides_collocet = self.get_review(peptides_pred_raw, batch[2])
        return loss, torch.tensor(aa_precision), torch.tensor(aa_recall), torch.tensor(pep_recall), aa_score, peptides_collocet, attn_scores, final_feats

    def denovo_step(self, batch):
        """
        A single validation step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            peptide sequences.

        Returns
        -------
        torch.Tensor
            The loss of the validation step.
        """
        # Calculate and log amino acid and peptide match evaluation metrics from
        # the predicted peptides.
        peptides_pred_raw, _, attn_scores, final_feats = self.forward(batch[0], batch[1])
        peptides_collocet = self.denovo_review(peptides_pred_raw, batch[2])
        return peptides_collocet, attn_scores, final_feats
   
    def denovo_review(self, pred, truth):
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

        return peptides_collect
    
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
    """
    Learning rate scheduler with linear warm up followed by cosine shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup : int
        The number of warm up iterations.
    max_iters : torch.optim
        The total number of iterations.
    """

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
            model.module.load_state_dict(torch.load(config['modelname'], map_location=loc)['state_dict'])
            model.eval()
    else:
        model.to(args.device)
        if config['modelname'] is not False:
            model.load_state_dict(torch.load(config['modelname'], map_location=args.device)['state_dict'])
            model.eval()
    return model



