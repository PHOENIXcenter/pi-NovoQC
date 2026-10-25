"""pi-xNovo inference model with checkpoint-compatible layer names."""
import torch
from torch import nn
from .depthcharge_new.components.transformers import PeptideDecoder, SpectrumEncoder

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
        _, _, attn_scores, _ = self.decoder(
            tokens,
            precursors,
            memories,
            mem_masks,
        )
        return self.softmax(scores), tokens, attn_scores, final_feats

