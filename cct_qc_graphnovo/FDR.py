"""CCT / TDC-style FDR utilities used in the GraphNovo demo.

This module contains helper functions for:
- generating candidate matches from a de novo model,
- building target/decoy feature pairs,
- fitting scoring models (strict / relaxed / combined),
- estimating FDR and applying interval adjustment.

The code is research-oriented and optimized for the accompanying notebook.
"""

import torch
import re
import os 
import math
import yaml
import numpy as np
import subprocess
import pandas as pd
from tqdm import tqdm
from pyteomics import mgf, mzml
from pyteomics_master.pyteomics import mass, mgf
import numpy as np
from scipy.stats import entropy, spearmanr
from evaluate import aa_match_batch

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
import xgboost as xgb
import csv, json
import numpy as np
import random

from typing import List
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from collections import defaultdict
import tempfile

from torch.utils.data import Dataset, DataLoader

def generate_decoy_fasta(target_fasta_path, decoy_fasta_path, decoy_prefix="DECOY_"):
    """Generate a reverse decoy protein database"""
    records = []
    for record in SeqIO.parse(target_fasta_path, "fasta"):
        rev_seq = record.seq[::-1]  
        rev_record = SeqRecord(rev_seq, id=decoy_prefix + record.id, description="decoy")
        records.append(rev_record)
    SeqIO.write(records, decoy_fasta_path, "fasta")
    return decoy_fasta_path

def build_blast_db(fasta_path, db_name, conda_env="blast"):
    """Build a BLAST database"""
    cmd = [
        "conda", "run", "-n", conda_env,
        "makeblastdb",
        "-in", fasta_path,
        "-dbtype", "prot",
        "-out", db_name
    ]
    print(f"[✓] Running BLAST:\n{' '.join(cmd)}")
    
    subprocess.run(cmd, check=True)


def deduplicate_and_merge_fastas(fasta_paths: List[str], output_merged_fasta: str):
    """
    Merge multiple target FASTA files and deduplicate protein sequences while retaining protein names and  description information.
    """
    seen_sequences = set()
    unique_records = []

    for fasta_path in fasta_paths:
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq_str = str(record.seq)
            if seq_str not in seen_sequences:
                seen_sequences.add(seq_str)
                unique_records.append(record)  

    SeqIO.write(unique_records, output_merged_fasta, "fasta")
    print(f"[✓] Deduplicated target proteins have been written to {output_merged_fasta}，with a total of {len(unique_records)} sequences.")
    return output_merged_fasta


def generate_and_merge_decoy_fastas(fasta_paths: List[str], output_merged_decoy_fasta: str, decoy_prefix="DECOY_"):
    """
    Generate a decoy FASTA file (with reverse sequences) for each FASTA file, then merge and deduplicate them into a single decoy FASTA file. Retain the protein names and descriptions, adding a "decoy_" prefix.
    """
    seen_sequences = set()
    decoy_records = []

    for fasta_path in fasta_paths:
        for record in SeqIO.parse(fasta_path, "fasta"):
            rev_seq = str(record.seq)[::-1]
            if rev_seq not in seen_sequences:
                seen_sequences.add(rev_seq)
                rev_record = SeqRecord(
                    Seq(rev_seq),
                    id=decoy_prefix + record.id,
                    description=decoy_prefix + record.description
                )
                decoy_records.append(rev_record)

    SeqIO.write(decoy_records, output_merged_decoy_fasta, "fasta")
    print(f"[✓] Deduplicated decoy proteins have been written to {output_merged_decoy_fasta}，with a total of  {len(decoy_records)} sequences.")
    return output_merged_decoy_fasta

def merge_target_and_decoy(target_fasta: str, decoy_fasta: str, output_combined_fasta: str):
    """
    Merge the deduplicated target and decoy FASTA files and write them as the final combined FASTA.
    """
    with open(output_combined_fasta, 'w') as out_f:
        with open(target_fasta, 'r') as f1:
            out_f.write(f1.read())
        with open(decoy_fasta, 'r') as f2:
            out_f.write(f2.read())
    print(f"[✓] The merged database has been generated. {output_combined_fasta}")
    return output_combined_fasta

# basic useful dictionary
ion_type_dict = {
    '0': 'a',
    '1': 'b',
    '2': 'y',
    '3': 'noise',
}
stat_type_dict = {
    '0': '1_',
    '1': '1_NH3',
    '2': '1_H2O',
    '3': '2_',
    '4': '2_NH3',
    '5': '2_H2O',
#     '8': '2_H2O_NH3',
#     '9': '2_H2O_H2O',
}

def get_theo_mz(peptide):
    """
    ion_type_dict = {
        '0': 'a',
        '1': 'b',
        '2': 'y',
    }
    stat_type_dict = {
        '0': '1_',
        '1': '1_NH3',
        '2': '1_H2O',
        '3': '1_H2O_NH3',
        '4': '1_H2O_H2O',
        '5': '2_',
        '6': '2_NH3',
        '7': '2_H2O',
        '8': '2_H2O_NH3',
        '9': '2_H2O_H2O',
    }
    """
    # Define charge states
    charges = [1, 2]
    # Generate b-ions with NH3 and H2O loss for each charge state
    a_ions = []
    b_ions = []
    y_ions = []
    peptide = re.split(r"(?<=.)(?=[A-Z])", peptide)
    for i in range(1, len(peptide)):
        for charge in charges:
            ion_mass = mass.fast_mass(peptide[:i], ion_type='a', charge=charge)
            a_ions.append(ion_mass)
            ion_mass_nh3 = ion_mass - mass.Composition('NH3').mass()
            a_ions.append(ion_mass_nh3)
            ion_mass_h2o = ion_mass - mass.Composition('H2O').mass()
            a_ions.append(ion_mass_h2o)
#             ion_mass_h2o_nh3 = ion_mass - mass.Composition('H2O').mass() - mass.Composition('NH3').mass()
#             a_ions.append(ion_mass_h2o_nh3)
#             ion_mass_h2o_h2o = ion_mass - mass.Composition('H2O').mass() - mass.Composition('H2O').mass()
#             a_ions.append(ion_mass_h2o_h2o)

    for i in range(1, len(peptide)):
        for charge in charges:
            ion_mass = mass.fast_mass(peptide[:i], ion_type='b', charge=charge)
            b_ions.append(ion_mass)
            ion_mass_nh3 = ion_mass - mass.Composition('NH3').mass()
            b_ions.append(ion_mass_nh3)
            ion_mass_h2o = ion_mass - mass.Composition('H2O').mass()
            b_ions.append(ion_mass_h2o)
#             ion_mass_h2o_nh3 = ion_mass - mass.Composition('H2O').mass() - mass.Composition('NH3').mass()
#             b_ions.append(ion_mass_h2o_nh3)
#             ion_mass_h2o_h2o = ion_mass - mass.Composition('H2O').mass() - mass.Composition('H2O').mass()
#             b_ions.append(ion_mass_h2o_h2o)

        # Generate y-ions with NH3 and H2O loss for each charge state
    for i in range(1, len(peptide)):
        for charge in charges:
            ion_mass = mass.fast_mass(peptide[-i:], ion_type='y', charge=charge)
            y_ions.append(ion_mass)
            ion_mass_nh3 = ion_mass - mass.Composition('NH3').mass()
            y_ions.append(ion_mass_nh3)
            ion_mass_h2o = ion_mass - mass.Composition('H2O').mass()
            y_ions.append(ion_mass_h2o)
#             ion_mass_h2o_nh3 = ion_mass - mass.Composition('H2O').mass() - mass.Composition('NH3').mass()
#             y_ions.append(ion_mass_h2o_nh3)
#             ion_mass_h2o_h2o = ion_mass - mass.Composition('H2O').mass() - mass.Composition('H2O').mass()
#             y_ions.append(ion_mass_h2o_h2o)
    return np.array([a_ions, b_ions, y_ions])

def get_aby_labels(seqs, exp_mz, theo_mz):
    aby_label = []
    for i in range(len(exp_mz)):
        if np.min(abs(theo_mz - exp_mz[i])) <= 0.05:
            ion_type_idx, stat_type_idx = np.where(abs(theo_mz - exp_mz[i])==np.min(abs(theo_mz - exp_mz[i])))
            peak_ion_type = ion_type_dict[str(ion_type_idx[0])]
            ion_idx, stat_idx = int(stat_type_idx[0]/len(stat_type_dict))+1, stat_type_idx[0]%len(stat_type_dict)
            peak_stat_type = stat_type_dict[str(stat_idx)]
            aby_label.append(f'{peak_ion_type}{ion_idx}-{peak_stat_type}')
        else:
            aby_label.append('N')
    return aby_label

def get_internal_labels(seqs, exp_mz, internal_mz, internal_label):
    aby_label = []
    for i in range(len(exp_mz)):
        label = None
        for j in range(len(internal_mz)):
            if np.min(abs(internal_mz[j] - exp_mz[i])) <= 0.05:
                idx = np.where(abs(internal_mz[j] - exp_mz[i]) == np.min(abs(internal_mz[j] - exp_mz[i])))
                label = internal_label[j][idx].tolist()[0]
                continue
        if label == None:
            label = 'N'
        aby_label.append(label)
    return aby_label

class PeptideMass:
    """A simple class for calculating peptide masses

    Parameters
    ----------
    residues: Dict or str {"massivekb", "canonical"}, optional
        The amino acid dictionary and their masses. By default this is only
        the 20 canonical amino acids, with cysteine carbamidomethylated. If
        "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    """

    canonical = {
        "G": 57.021463735,
        "A": 71.037113805,
        "S": 87.032028435,
        "P": 97.052763875,
        "V": 99.068413945,
        "T": 101.047678505,
        "C+57.021": 103.009184505 + 57.02146,
        "L": 113.084064015,
        "I": 113.084064015,
        "N": 114.042927470,
        "D": 115.026943065,
        "Q": 128.058577540,
        "K": 128.094963050,
        "E": 129.042593135,
        "M": 131.040484645,
        "H": 137.058911875,
        "F": 147.068413945,
        # "U": 150.953633405,
        "R": 156.101111050,
        "Y": 163.063328575,
        "W": 186.079312980,
        # "O": 237.147726925,
    }

    # Modfications found in MassIVE-KB
    massivekb = {
        # N-terminal mods:
        "+42.011": 42.010565,  # Acetylation
        "+43.006": 43.005814,  # Carbamylation
        "-17.027": -17.026549,  # NH3 loss
        "+43.006-17.027": (43.006814 - 17.026549),
        # AA mods:
        "M+15.995": canonical["M"] + 15.994915,  # Met Oxidation
        "N+0.984": canonical["N"] + 0.984016,  # Asn Deamidation
        "Q+0.984": canonical["Q"] + 0.984016,  # Gln Deamidation
    }

    # Constants
    hydrogen = 1.007825035
    oxygen = 15.99491463
    h2o = 2 * hydrogen + oxygen
    proton = 1.00727646688

    def __init__(self, residues="canonical"):
        """Initialize the PeptideMass object"""
        if residues == "canonical":
            self.masses = self.canonical
        elif residues == "massivekb":
            self.masses = self.canonical
            self.masses.update(self.massivekb)
        else:
            self.masses = residues

    def __len__(self):
        """Return the length of the residue dictionary"""
        return len(self.masses)

    def mass(self, seq, charge=None):
        """Calculate a peptide's mass or m/z.

        Parameters
        ----------
        seq : list or str
            The peptide sequence, using tokens defined in ``self.residues``.
        charge : int, optional
            The charge used to compute m/z. Otherwise the neutral peptide mass
            is calculated

        Returns
        -------
        float
            The computed mass or m/z.
        """
        if isinstance(seq, str):
            seq = re.split(r"(?<=.)(?=[A-Z])", seq)

        calc_mass = sum([self.masses[aa] for aa in seq]) + self.h2o
        if charge is not None:
            calc_mass = (calc_mass / charge) + self.proton

        return calc_mass
    
def normalize_intensities(tensor):
    intensities = tensor[..., 1]
    max_val = torch.max(intensities, dim=-1, keepdim=True).values
    normalized_intensities = torch.where(max_val != 0, intensities / max_val, intensities)
    tensor_normalized = tensor.clone()
    tensor_normalized[..., 1] = normalized_intensities
    return tensor_normalized

class ValidSpectrumDataset(Dataset):
    
    def __init__(self, cfg):
        super().__init__()
        self.path = cfg['mgf_file']
        self.species_list = os.listdir(self.path)
        self.valid_species = cfg['valid_species']
        self.spec_step = cfg['spec_step'] 
        self.cfg = cfg
        self.species = self.valid_species
        self.filename = None

        self.spectra_num_dic = dict()
        for species in self.species_list:
            stepfiles = os.listdir(os.path.join(self.path, species))
            spectra_number = np.max([int(step.split('_')[1][:-4]) for step in stepfiles])
            self.spectra_num_dic[species] = spectra_number

        # get species and spectrum numbers for validation
        self.valid_spectra_num = self.spectra_num_dic[self.valid_species]    

        self.mass_items = list(self.cfg['residues'].values())
        self.aa_2_masses = []
        for i in range(len(self.mass_items)):
            for j in range(len(self.mass_items)):
                self.aa_2_masses.append(self.mass_items[i]+self.mass_items[j])
        self.spectrum = None
        
    def __getitem__(self, idx):
        mgf_path, spectrum_idx = self.getfile(idx)
        spectra = mgf.read(mgf_path, read_ions=True)
        try:
            spectrum = spectra[spectrum_idx]
            precursor_mz = spectrum['params']['pepmass'][0]
            precursor_charge = int(spectrum['params']['charge'][0])
            peptide = spectrum['params']['seq']
            mz_array = spectrum['m/z array']
            intensity_array = spectrum['intensity array']  
            self.spectrum = spectrum
        except:
            print(mgf_path, spectrum_idx)
            spectrum = spectra[spectrum_idx-1]
            precursor_mz = spectrum['params']['pepmass'][0]
            precursor_charge = int(spectrum['params']['charge'][0])
            peptide = spectrum['params']['seq']
            mz_array = spectrum['m/z array']
            intensity_array = spectrum['intensity array']  
            self.spectrum = spectrum
        mz_intensity = torch.from_numpy(np.array([mz_array, intensity_array])).T.float()
        
        return mz_intensity, precursor_mz, precursor_charge, peptide, idx
    
    def __len__(self):
        return self.valid_spectra_num

    def getinfo(self, idx):
        return self.getfile(idx)

    def getfile(self, idx):
        step = idx // self.spec_step
        if (step+1) * self.spec_step <= self.valid_spectra_num:
            self.filename = f'{step*self.spec_step}_{(step+1)*self.spec_step}.mgf'
        else:
            self.filename = f'{step*self.spec_step}_{self.valid_spectra_num}.mgf'
        mgf_path = os.path.join(self.path, self.valid_species, self.filename)
        spectrum_idx = idx - step * self.spec_step
        return mgf_path, int(spectrum_idx)
    
def prepare_batch(batch):
    mz_intensity, precursor_mz, precursor_charge, peptide, scan_index = list(zip(*batch))
    mz_intensity = torch.nn.utils.rnn.pad_sequence(mz_intensity, batch_first=True)
    precursor_mz = torch.tensor(precursor_mz)
    precursor_charge = torch.tensor(precursor_charge)
    precursor_mass = (precursor_mz - 1.007276) * precursor_charge + 1.007276
    precursors = torch.vstack([precursor_mass, precursor_charge, precursor_mz]).T.float()
    return mz_intensity, precursors, peptide, scan_index
    
def compute_fused_score(attn_score, intensity_rank, alpha=0.3, beta=0.7):
    eps = 1e-8
    return alpha * attn_score + beta * intensity_rank  # or np.log(attn_score+eps) + np.log(intensity_rank+eps)

def get_rank_percentile(intensity_list):
    sorted_vals = np.sort(intensity_list)[::-1]
    total = len(sorted_vals)
    rank_map = {v: i for i, v in enumerate(sorted_vals)}
    return [1.0 - rank_map[val] / total for val in intensity_list]

def compute_aa_confidences(attention_matrix, theor_frags, pred_pep, mz, intens, top_k=5):
    """
    attention_matrix: Rows from top to bottom correspond to the peptide sequence from right to left (C‑terminus to N‑terminus).
    """
    """
    Calculate the confidence score for each amino acid in the sequence and mark anchor positions (high attention with matched fragment peaks).
    - peptide_sequence: Peptide sequence string.
    - spectrum_peaks: [(mz, intensity), ...] List of spectral peaks.
    - cross_attention: 2D array or tensor of shape [L, M] representing the attention distribution from each sequence position to the spectral input features.
    - tolerance_ppm: Mass tolerance (in ppm) for theoretical peak matching.
    - top_k: For each amino acid, examine the top‑k attention peaks for checking.

    Returns: (aa_confidences, aa_details)
        aa_confidences: A list of length L, each element is the confidence score (0~1) of the corresponding amino acid.
        aa_details: A list of length L, each element is a dictionary containing information about that amino acid site (index, aa, is_anchor, confidence, matched_peak_mz, matched_peak_intensity, relative_intensity, attention_score).
    """
    pred_pep_reversed = list(reversed(re.split(r"(?<=.)(?=[A-Z])", pred_pep)))
    L = len(pred_pep_reversed)
    aa_confidences = [0.0] * L
    aa_details = []
    intens_rank = get_rank_percentile(intens)
    for i in range(attention_matrix.shape[0]): 
        top_k_indices = np.argpartition(attention_matrix[i], -top_k)[-top_k:]
        aa_info = {
            "index": i,
            "aa": pred_pep_reversed[i],
            'theor_frag_y': f'y{i+1}',
            'theor_frag_b': f'b{len(pred_pep_reversed)-(i+1+1)}',
            "is_anchor": False,
            "confidence": 0.0,
            "matched_peak_mz": None,
            "matched_peak_intensity": None,
            "relative_intensity": None,
            "attention_score": None
        }
        confidence_score = 0.0
        best_match_score = 0.0
        for idx in top_k_indices:
            peak_mz = mz[idx]
            peak_intensity = intens[idx]
            peak_intensity_rank = intens_rank[idx]
            attn_weight = attention_matrix[i][idx]     
            match_b = False
            match_y = False
            theor_b_mz = None
            theor_y_mz = None
            if theor_frags[idx] != 'N':
                if 'y' in theor_frags[idx]:
                    if abs(int(theor_frags[idx].split('-')[0][1:])-i) <= 2:
                        match_y = True
                if 'b' in theor_frags[idx]:
                    if abs(int(theor_frags[idx].split('-')[0][1:]) - (len(pred_pep_reversed)-(i+1))) <= 2:
                        match_b = True
            if match_b or match_y:
                if aa_info["matched_peak_mz"] is None:
                    aa_info["matched_peak_mz"] = peak_mz
                    aa_info["matched_peak_intensity"] = peak_intensity
                    aa_info["intensity_rank"] = peak_intensity_rank
                    aa_info["attention_score"] = attn_weight
                score = compute_fused_score(attn_weight, peak_intensity_rank)
                confidence_score += score       
                aa_info["is_anchor"] = True
        aa_conf = confidence_score
        aa_info["confidence"] = aa_conf
        aa_confidences[i] = aa_conf
        aa_details.append(aa_info)
    return aa_confidences, aa_details

def compute_global_metrics(peptide_sequence, theor_frags, pred_pep, mz, intens):
    """
    Calculate peptide‑level quality control metrics: fragment coverage, number of complementary ion pairs, and explained intensity.
    Return a dictionary {coverage, complement_fraction, explained_intensity}.
    """
    pred_pep_reversed = list(reversed(re.split(r"(?<=.)(?=[A-Z])", pred_pep)))
    seq_length = len(pred_pep_reversed)
    total_theoretical = len(theor_frags)
    num_matched = sum(np.array(theor_frags) != 'N')
    coverage = num_matched / total_theoretical if total_theoretical > 0 else 0.0
    total_intensity = sum(intens)
    explained_intensity = sum(intens[np.array(theor_frags) != 'N']) / total_intensity
    
    return {
        "coverage": coverage,
        "explained_intensity": explained_intensity
    }

def evaluate_peptide_sequence(attention_matrix, theor_frags, peptide_sequence, mz, intens, top_k=3, global_factor=0.1,
                              masses=None, precursor_mass=None, precursor_charge=None, precursor_tol_ppm=20, 
                              rt=None, rt_diff_threshold=None):
    """
    Evaluate the matching quality between a given peptide sequence and the spectrum, returning a result dictionary containing multi-level quality control information.

    Parameters:
      - peptide_sequence (str): The de novo predicted peptide sequence.
      - spectrum_peaks (list of (mz, intensity)): List of spectrum peak tuples.
      - cross_attention (ndarray or tensor): Transformer cross-attention matrix with shape [L, M].
      - precursor_mass (float): Experimental monoisotopic precursor m/z (optional).
      - precursor_tol_ppm (float): Precursor mass tolerance in ppm, used for mass deviation judgment.
      - rt (float): Experimental retention time of the spectrum (optional).
      - rt_diff_threshold (float): Retention time tolerance threshold (optional, if predicted RT model results are available).

    Returns:
      A dictionary containing quality control assessments at both the amino acid and peptide levels, suitable for JSON serialization.
    """
    # 1. Calculate confidence scores and anchor information for each amino acid
    aa_confidences, aa_details = compute_aa_confidences(attention_matrix, theor_frags, peptide_sequence, mz, intens, top_k)
    # 2. Calculate global quality control metrics and preliminary global confidence score
    metrics = compute_global_metrics(peptide_sequence, theor_frags, peptide_sequence, mz, intens)
    coverage = metrics["coverage"]
    explained_intensity = metrics["explained_intensity"]
    global_confidence = (coverage + explained_intensity) / 2.0
    metrics["global_confidence_initial"] = global_confidence
    # 3. Check precursor mass match and penalize the confidence score if it exceeds the threshold
    mass_pass = True
    if precursor_mass is not None:
        seq = re.split(r"(?<=.)(?=[A-Z])", peptide_sequence)
        cal_seq = [seq[0]]
        for i in range(1, len(seq)):
            cal_seq.append(seq[i].replace('+42.011', ''))
        predicted_mz = masses.mass(''.join(cal_seq), precursor_charge)
        
        mass_error = 0.0
        if predicted_mz != 0:
            mass_error = abs(predicted_mz - precursor_mass) / predicted_mz * 1e6  
        if mass_error > precursor_tol_ppm:
            mass_pass = False
            global_confidence = global_confidence * global_factor
            metrics["mass_error_ppm"] = mass_error
        else:
            metrics["mass_error_ppm"] = mass_error
    result = {
        "sequence": peptide_sequence,
        "aa_confidences": np.array(aa_confidences),
#         "aa_details": aa_details,
        "metrics": metrics,
        "global_confidence": global_confidence,
        "mass_pass": mass_pass
    }
    if rt is not None:
        result["rt"] = rt
        result["rt_pass"] = rt_pass
    return result

def compute_attention_features(attn_matrix, theor_mz_list, mz_array, intensity_array, attention_threshold=0.05):
    """
    Calculate a set of attention features at both the amino acid and peptide levels.

    Parameters:
        attn_matrix: np.ndarray of shape [L_aa, L_ms], the attention from each amino acid in the protein sequence to each spectral peak.
        theor_mz_list: list of theoretical fragment m/z values, length L_aa.
        mz_array: array of experimental m/z values, shape [L_ms].
        intensity_array: array of intensity values, shape [L_ms].
        attention_threshold: float, threshold for determining significant attention spectral peaks.

    Returns:
        features: dict, containing all statistical features at both the amino acid and peptide levels.
    """
    L_aa, L_ms = attn_matrix.shape
    features = {
        "aa_entropy": [],
        "aa_concentration": [],
        "aa_offset": [],
        "aa_center_drift": [],
        "aa_span": [],
    }

    intensity_array = np.array(intensity_array)
    mz_array = np.array(mz_array)

    aa_peak_indices = []  

    for i in range(L_aa):
        attn_row = attn_matrix[i]
        attn_row = attn_row / (attn_row.sum() + 1e-8)  

        # entropy
        ent = entropy(attn_row)
        features["aa_entropy"].append(ent)

        # Concentration Ratio (Maximum Attention Proportion)
        max_attn = attn_row.max()
        features["aa_concentration"].append(max_attn)

        # Matching Offset (m/z difference between the maximum peak and theoretical fragment m/z)
        peak_idx = np.argmax(attn_row)
        aa_peak_indices.append(peak_idx)
        offset = abs(mz_array[peak_idx] - theor_mz_list[i]) if theor_mz_list[i] is not None else None
        features["aa_offset"].append(offset)

        # Center Shift (Difference between Weighted Center and Theoretical m/z)
        weighted_center = np.sum(attn_row * mz_array)
        center_drift = abs(weighted_center - theor_mz_list[i]) if theor_mz_list[i] is not None else None
        features["aa_center_drift"].append(center_drift)

        # Span (Standard Deviation)
        std_dev = np.sqrt(np.sum(attn_row * (mz_array - weighted_center)**2))
        features["aa_span"].append(std_dev)

    # peptide-level feature
    attn_sim = []
    for i in range(L_aa - 1):
        sim = np.dot(attn_matrix[i], attn_matrix[i+1]) / (
            np.linalg.norm(attn_matrix[i]) * np.linalg.norm(attn_matrix[i+1]) + 1e-8)
        attn_sim.append(sim)
    features["avg_attention_similarity"] = np.mean(attn_sim)
    peak_mzs = mz_array[aa_peak_indices]
    rho, _ = spearmanr(list(range(L_aa)), peak_mzs)
    features["fragment_alignment_consistency"] = rho
    return features

def compute_psm_features_with_attention(evaluate_results, attn_matrix, theor_mz_list, mz_array, intensity_array):
    """
    Parameters:
        evaluate_results: Output from the original evaluate_peptide_sequence function
        attn_matrix: Cross-attention matrix [L_aa, L_ms]
        theor_mz_list: list, m/z of theoretical fragments for each amino acid
        mz_array: Experimental spectrum m/z array
        intensity_array: Experimental spectrum intensity array

    Returns:
        feature_dict: dict, containing all merged features
    """
    base_features = {}
    aa_conf_list = evaluate_results['aa_confidences']
    # 1. List of amino acid confidence scores (e.g., with length equal to the peptide length)
    base_features["aa_conf_mean"] = sum(aa_conf_list) / len(aa_conf_list)
    base_features["aa_conf_var"] = np.var(aa_conf_list)
    base_features["aa_conf_min"] = min(aa_conf_list)
    base_features["aa_conf_max"] = max(aa_conf_list)
    base_features['num_low_conf'] = sum([1 for c in aa_conf_list if c < 0.2])
    base_features['num_high_conf'] = sum([1 for c in aa_conf_list if c > 0.8])
    # 2. Global quality control score
    base_features["global_qc_score"] = evaluate_results['global_confidence']
    # 3. Fragment coverage calculation
    base_features["fragment_coverage"] = evaluate_results['metrics']['coverage']
    # 4. Explained intensity percentage
    base_features["explained_intensity"] = evaluate_results['metrics']['explained_intensity']
    # 5. Precursor mass deviation (ppm)  (not for DIA)
    base_features["precursor_error_ppm"] = evaluate_results['metrics']['mass_error_ppm']
    # Anchor statistics (temporarily using high confidence as a proxy for anchor points)
    base_features['anchor_ratio'] = base_features['num_high_conf'] / len(aa_conf_list)

    # Add attention-derived features
    attn_features = compute_attention_features(
        attn_matrix, theor_mz_list, mz_array, intensity_array
    )

    full_features = {**base_features}
    for k, v in attn_features.items():
        if isinstance(v, list):
            full_features[f"{k}_mean"] = np.mean(v)
            full_features[f"{k}_std"] = np.std(v)
            full_features[f"{k}_min"] = np.min(v)
            full_features[f"{k}_max"] = np.max(v)
        else:
            full_features[k] = v
    return full_features

def del_wrong_premodification(label):
    pred = re.split(r"(?<=.)(?=[A-Z])", label)
    pred_new = []
    for i in range(0, len(pred)):
        if i == 0:
            if pred[i] == '+42.011':
                pred_new.append('+42.011')
            else:
                pred_new.append(pred[i].replace('+42.011', ''))
        else:    
            if '+42.011' in pred[i]:
                pred_new.append(pred[i].replace('+42.011', ''))
            else:
                pred_new.append(pred[i])                        
    pred = ''.join(pred_new)    
    return pred

def model_test(model, val_loader, decoy_peptides=False, top_k=10, attn_type='max', device='cuda:0'):
    model.eval()
    conf_path = 'config_xNovo_DDA.yaml'
    with open(conf_path) as f:
        config = yaml.safe_load(f)
        
    masses = PeptideMass(residues=config['residues'])
    wrong_spec = 0
    tot_loss, tot_aa_precision, tot_aa_recall, tot_pep_recall = 0, 0, 0, 0
    peptides_collect_list = []
    peptide_bool_list = []
    eval_results_list, psm_feature_list = [], []
    spec_idx_list = []
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as pbar:
            for i, inputs in enumerate(val_loader):
                mz_intensity, precursors, peptide, scan_index = inputs
                mz_intensity = mz_intensity.to(device).type(torch.cuda.FloatTensor)
                precursors = precursors.to(device).type(torch.cuda.FloatTensor)
                if decoy_peptides:
                    peptide = tuple([seq for seq in decoy_peptides[i*256:(i+1)*256]])
                    loss, aa_precision, aa_recall, pep_recall, aa_scores, peptides_collect, attn_scores, final_feats = \
                    model.training_step([mz_intensity, precursors, peptide])
                else:
                    loss, aa_precision, aa_recall, pep_recall, aa_scores, peptides_collect, attn_scores, final_feats = \
                    model.validation_step([mz_intensity, precursors, peptide])
                    
                tot_loss += loss.item() * mz_intensity.size(0)
                tot_aa_precision += aa_precision.item() * mz_intensity.size(0)
                tot_aa_recall += aa_recall.item() * mz_intensity.size(0)
                tot_pep_recall += pep_recall.item() * mz_intensity.size(0)
                
                # compute attention matrix
                if attn_type == 'max':
                    attn_scores = torch.cat(attn_scores, dim=1).max(dim=1)[0].detach().cpu().numpy()
                elif attn_type == 'mean':
                    attn_scores = torch.cat(attn_scores, dim=1).mean(dim=1).detach().cpu().numpy()
                
                # Check if the predicted peptide matches the database search results
                aa_matches_batch, _, _ = aa_match_batch([seq[0] for seq in peptides_collect], [seq[1] for seq in peptides_collect], 
                                                model.decoder._peptide_mass.masses) 
                B, _, _ = attn_scores.shape
                prec_mzs = precursors[:, 2].detach().cpu().numpy().tolist()
                prec_cs = precursors[:, 1].detach().cpu().numpy().tolist()
                
                # Compute peptide-spectrum match features based on the attention matrix
                for k in range(B):

                    mzs, intens = mz_intensity[k,:,0].detach().cpu().numpy(),  mz_intensity[k,:,1].detach().cpu().numpy()
                    mzs, intens = mzs[mzs!=0], intens[intens!=0]
                    mzs, intens = np.array([0] + mzs.tolist() + [precursors[k, 2].item()]), np.array([0.0] + intens.tolist() + [0.0])
                    
                    try:
                        pred, label = peptides_collect[k]
                        if decoy_peptides:
                            pred = del_wrong_premodification(label)
                        else:
                            pred = del_wrong_premodification(pred)
                        # Calculate theoretical ions
                        pred_theo_mz = get_theo_mz(pred)
                        pred_aby_list = get_aby_labels(pred, mzs, pred_theo_mz)
                        pred_matched_aby_list = [] 
                        for m in range(len(pred_aby_list)):
                            if pred_aby_list[m] != 'N':
                                pred_matched_aby_list.append(pred_aby_list[m])
                            else:
                                pred_matched_aby_list.append(pred_aby_list[m])
                        pred_matched_aby_list = ['0'] + pred_matched_aby_list[1:-1] + ['prec']

                        pep_len = len(re.split(r"(?<=.)(?=[A-Z])", pred)) 
                        if decoy_peptides:
                            attention_matrix = attn_scores[k, 1:1+pep_len, :len(mzs)] 
                        else:
                            attention_matrix = attn_scores[k, :pep_len, :len(mzs)]
                        qc_results = evaluate_peptide_sequence(attention_matrix, pred_matched_aby_list, pred, mzs, intens,
                                                               masses=masses, precursor_mass=prec_mzs[k], 
                                                               precursor_charge=prec_cs[k], top_k=top_k)  
                        psm_features = compute_psm_features_with_attention(qc_results, attention_matrix, 
                                                                           pred_theo_mz.reshape(-1), mzs, intens)
                        eval_results_list.append(qc_results)
                        psm_feature_list.append(psm_features)
                        spec_idx_list.append(scan_index[k])
                        peptide_bool_list.append(aa_matches_batch[k][1])
                        peptides_collect_list.append([pred, label])
                    
                    except:
                        wrong_spec += 1
                        eval_results_list.append([])
                        psm_feature_list.append([])
                        spec_idx_list.append(scan_index[k])
                        peptide_bool_list.append(False)
                        peptides_collect_list.append(['STQPPEPTIDE', 'STQPPEPTIDE'])
                        continue
                    
                pbar.update(1)
    tot_loss /= len(val_loader.dataset)
    tot_aa_precision /= len(val_loader.dataset)
    tot_aa_recall /= len(val_loader.dataset)
    tot_pep_recall /= len(val_loader.dataset)
    return peptides_collect_list, peptide_bool_list, eval_results_list, psm_feature_list, spec_idx_list

import matplotlib.pyplot as plt
import pandas as pd
import random

def del_blank_preds(peptides_collect_list, peptide_bool_list, eval_results_list, psm_feature_list, spec_idx_list, del_index):
    peptide_bool_list = [peptide_bool_list[i] for i in range(len(peptides_collect_list)) if not i in del_index]
    eval_results_list = [eval_results_list[i].copy() for i in range(len(peptides_collect_list)) if not i in del_index]
    psm_feature_list = [psm_feature_list[i].copy() for i in range(len(peptides_collect_list)) if not i in del_index]
    spec_idx_list = [spec_idx_list[i] for i in range(len(peptides_collect_list)) if not i in del_index]
    peptides_collect_list = [peptides_collect_list[i] for i in range(len(peptides_collect_list)) if not i in del_index]
    return peptides_collect_list, peptide_bool_list, eval_results_list, psm_feature_list, spec_idx_list

def denovo_and_evaluate_psm(model, val_loader, top_k=10, attn_type='max', device='cuda:0', xgb_ratio=1):
    target_peptides, target_bools, target_eval_results, target_features, target_spec_idxs = model_test(model, val_loader, top_k=top_k, attn_type=attn_type, device=device)
    pred_peptides = [seq[0] for seq in target_peptides]
    db_peptides = [seq[1] for seq in target_peptides]
    decoy_peptides = pred_peptides

    decoy_peptides, decoy_bools, decoy_eval_results, decoy_features, decoy_spec_idxs = model_test(model, val_loader, decoy_peptides=decoy_peptides, 
                                                          top_k=top_k, attn_type=attn_type, device=device)
     # delete blank peptides
    del_index_target = [i for i in range(len(target_features)) if type(target_features[i]) == list]
    del_index_decoy = [i for i in range(len(decoy_features)) if type(decoy_features[i]) == list]
    del_index = list(sorted(list(set(del_index_target+del_index_decoy))))
    
    target_peptides, target_bools, target_eval_results, target_features, target_spec_idxs = del_blank_preds(target_peptides, target_bools, target_eval_results, target_features, target_spec_idxs, del_index)
    decoy_peptides, decoy_bools, decoy_eval_results, decoy_features, decoy_spec_idxs = del_blank_preds(decoy_peptides, decoy_bools, decoy_eval_results, decoy_features, decoy_spec_idxs, del_index)
    pred_peptides = [seq[0] for seq in target_peptides]
    db_peptides = [seq[1] for seq in target_peptides]
    decoy_peptides = [seq[0] for seq in decoy_peptides]
    
    target_results = [{'features':target_features[i], 
                  'aa_confidence': target_eval_results[i]['aa_confidences'].tolist(),
                  'target_bool': target_bools[i], 
                  'sequence': pred_peptides[i],
                  'spectrum_idx': target_spec_idxs[i],
                  'psm_id': i} for i in range(len(target_spec_idxs))]
    decoy_results = [{'features':decoy_features[i], 
                  'aa_confidence': decoy_eval_results[i]['aa_confidences'].tolist(),
                  'target_bool': False, 
                  'sequence': decoy_peptides[i],
                  'spectrum_idx': target_spec_idxs[i],
                  'psm_id': i} for i in range(len(target_spec_idxs))]
    return target_results, decoy_results 

def convert_results_to_csv(results):
    features = [result['features'] for result in results]
    sequence = [result['sequence'] for result in results]
    target_bool = [result['target_bool'] for result in results]
    spectrum_idx = [result['spectrum_idx'] for result in results]
    psm_id = [result['psm_id'] for result in results]
    aa_confidence = [result['aa_confidence'] for result in results]
    
    df = pd.DataFrame(features)
    df['sequence'] = sequence
    df['spectrum_idx'] = spectrum_idx
    df['psm_id'] = psm_id
    df['target_bool'] = target_bool
    df['aa_confidence'] = aa_confidence
    return df

def convert_csv_to_dict(features):
    target_bools = features['target_bool'].tolist()
    target_seqs = features['sequence'].tolist()
    target_spec_idxs = features['spectrum_idx'].tolist()
    target_psm_ids = features['psm_id'].tolist()
    target_aa_confidence = features['aa_confidence'].tolist()
    columns_to_exclude = ['target_bool', 'sequence', 'spectrum_idx', 'psm_id', 'aa_confidence']
    features_df = features.drop(columns=columns_to_exclude)
    target_features = features_df.to_dict('records')
    return target_bools, target_seqs, target_spec_idxs, target_psm_ids, target_aa_confidence, target_features

# psm_mlp.py  ───────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import torch.backends.cudnn as cudnn
import random

random.seed(454)
np.random.seed(454)
torch.manual_seed(454)
torch.cuda.manual_seed_all(454)
cudnn.deterministic = True
cudnn.benchmark = False

# ────────────────────────────────────────────────────
# 1. feature dataset
# ────────────────────────────────────────────────────
class PsmDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ────────────────────────────────────────────────────
# 2. MLP (33→256→128→64→1)
# ────────────────────────────────────────────────────
class MLPClassifier(nn.Module):
    def __init__(self, in_dim=33, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 1),
            nn.Sigmoid()            
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ────────────────────────────────────────────────────
# 3. train and evaluation scoring model
# ────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(X)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)
    
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with tqdm(total=len(loader)) as pbar:
        for X, y in loader:
            X = X.to(device)
            y_hat = model(X).cpu()
            y_true.append(y)
            y_pred.append(y_hat)
            pbar.update(1)
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    auc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, (y_pred > 0.5))
    return auc, acc, y_true, y_pred

# ────────────────────────────────────────────────────
# 4. main function of scoring model training 
# ────────────────────────────────────────────────────
def train_mlp(X, y,
         lr=1e-4, batch_size=256, epochs=50, dropout=0.1,
         test_size=0.2, random_state=42, patience=8, device='cuda:0', ckpt_path=None):
    
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)

    # 4.3 DataLoader
    train_loader = DataLoader(PsmDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(PsmDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False)

    # 4.4 mode;, optimizer, loss
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(in_dim=np.array(X).shape[1], dropout=dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()

    # 4.5 train loop (early stopping)
    best_auc, epochs_no_improve = 0, 0
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        auc, acc, _, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | Loss {loss:.4f} | Val AUC {auc:.4f} | Val Acc {acc:.4f}")

        # early stopping
        if auc > best_auc:
            best_auc = auc
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(),
                        "scaler": scaler}, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # 4.6 load best ckpt and report 
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    scaler = ckpt["scaler"]
    _, _, y_true, y_pred = evaluate(model, val_loader, device)
    print("\nFinal classification report (Val):")
    print(classification_report(y_true, (y_pred > 0.5)))
    return model

def return_X_and_y(features, pred_bools=False):
    X = [list(dict(d).values()) for d in features]
    df_ = pd.DataFrame(X, columns=features[0].keys()).fillna(0) 
    X = df_.to_numpy().tolist()
    if pred_bools:
        return X, pred_bools
    else:
        return X, [1 if i > (int(len(X)) / 2) else 0 for i in range(len(X))]

def test_mlp(X, y, batch_size=256, ckpt_path="best_psm_mlp.pt", device='cuda:0'):
    ckpt = torch.load(ckpt_path, map_location=device)
    scaler = ckpt["scaler"]
    model = MLPClassifier(in_dim=np.array(X).shape[1]).to(device)
    model.load_state_dict(ckpt["model"])
    criterion = nn.BCELoss()
    
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    new_X = scaler.transform(X) 
    val_loader   = DataLoader(PsmDataset(new_X, y),   batch_size=batch_size, shuffle=False)
    model.eval()
    with torch.no_grad():
        _, _, y_true, y_pred = evaluate(model, val_loader, device)
        print("\nFinal classification report (Val):")
        print(classification_report(y_true, (y_pred > 0.5)))
    return y_pred

def gamma_fdr(df_fdr, fdr_thresh=0.05, min_gamma=0.01, max_gamma=100.0, eps=1e-4,
              conservative_fdr_ratio = 0.5, interval_margin = 0.1):
    """calculation of adjustment factor gamma."""
    true_fdr = np.array(df_fdr['observed_fdp'].tolist())
    estimated_fdr = np.array(df_fdr['estimated_fdr'].tolist())

    target_idx_list = np.where(np.abs(true_fdr - estimated_fdr)<=eps)[0] # score threshold (s_c) of conservatism starting point 
    if len(target_idx_list) == 0:
        target_idx = 0
    else:
        target_idx = max(target_idx_list)
    conservative_start_point = estimated_fdr[target_idx]  
    print(f"conservative start point: {conservative_start_point:.4f}, target fdr: {fdr_thresh:.4f}, start idx: {target_idx}")
    
    if conservative_start_point <= fdr_thresh * conservative_fdr_ratio or max(true_fdr)<=fdr_thresh:
        # Interval conservativeness theorem
        expected_interval_low = fdr_thresh * (1 - interval_margin)
        expected_interval_high = fdr_thresh * (1 + interval_margin)
        low_idx = np.argmin(np.abs(true_fdr - expected_interval_low))
        high_idx = np.argmin(np.abs(true_fdr - expected_interval_high))
        gamma_low = true_fdr[low_idx] / estimated_fdr[low_idx]
        gamma_high = true_fdr[high_idx] / estimated_fdr[high_idx]
        gamma = max([gamma_low, gamma_high])

        gamma = np.clip(gamma, min_gamma, max_gamma)
        df_fdr['estimated_fdr_update'] = gamma * df_fdr['estimated_fdr']
        if gamma == gamma_low:
            optimal_idx = high_idx
        else:
            optimal_idx = low_idx
        exact_low, exact_high = true_fdr[low_idx], true_fdr[high_idx]
        print(f"gamma of interval: interval[{expected_interval_low:.4f}, {expected_interval_high:.4f}], gamma={gamma:.4f}")
        print((f"conservative interval: interval[{exact_low:.4f}, {exact_high:.4f}], gamma={gamma:.4f}"))
    else:
        # Adjustment-factor theorem
        target_idx_list = np.where(np.abs(true_fdr - fdr_thresh)<=eps)[0] 
        if len(target_idx_list) == 0:
            target_idx = 0
        else:
            target_idx = max(target_idx_list)
        gamma = true_fdr[target_idx] / estimated_fdr[target_idx]  
        print(f"according to Adjustment-factor theorem: gamma={gamma:.4f} (nearby target fdr)")
        gamma = np.clip(gamma, min_gamma, max_gamma)
        df_fdr['estimated_fdr_update'] = gamma * df_fdr['estimated_fdr']
        optimal_idx = max(np.where(np.abs(df_fdr['estimated_fdr_update'] - fdr_thresh)<=eps)[0])
        exact_low, exact_high = true_fdr[optimal_idx], true_fdr[-1]
        print((f"conservative interval: interval[{exact_low:.4f}, {exact_high:.4f}], gamma={gamma:.4f}"))
    return df_fdr, gamma, true_fdr[optimal_idx], optimal_idx, [exact_low, exact_high]
    
def calculate_fdr(df_file, score_name):
    df_sorted_psm = df_file.copy()
    df_sorted_psm = df_sorted_psm.sort_values(by=score_name, ascending=False).reset_index(drop=True)
    cum_target = (df_sorted_psm['is_decoy'] == 0).cumsum()
    df_sorted_psm['cum_decoy']  = (df_sorted_psm['is_decoy'] == 1).cumsum()
    df_sorted_psm['cum_decoy_true'] = ((df_sorted_psm['is_decoy'] == 1) & (df_sorted_psm['denovo_label'] == 1)).cumsum()
    df_sorted_psm['cum_decoy_false'] = ((df_sorted_psm['is_decoy'] == 1) & (df_sorted_psm['denovo_label'] == 0)).cumsum()
    df_sorted_psm['cum_true'] = ((df_sorted_psm['is_decoy'] == 0) & (df_sorted_psm['denovo_label'] == 1)).cumsum()
    df_sorted_psm['cum_false']  = ((df_sorted_psm['is_decoy'] == 0) & (df_sorted_psm['denovo_label'] == 0)).cumsum()
    df_sorted_psm['estimated_fdr'] = df_sorted_psm['cum_decoy'] / np.maximum(cum_target, 1)
    df_sorted_psm['observed_fdp']  = df_sorted_psm['cum_false'] / np.maximum((df_sorted_psm['cum_true']+df_sorted_psm['cum_false']), 1)
    return df_sorted_psm

def get_corss_point_fdr_curve(df_file, fdr_threshold=0.05, eps=1e-4, 
                              est_fdr_key='estimated_fdr', fdp_key='observed_fdp', score_name='pep_score'):
    cross_mark_target_fdr = df_file.loc[abs(df_file[est_fdr_key] - fdr_threshold)<eps]   # taget fdr cross
    cross_mark_candidate = df_file.loc[abs(df_file[est_fdr_key] - df_file[fdp_key])<eps]  # candidate fdr corss
    
    if est_fdr_key=='estimated_fdr_update': # target FDR
        if len(cross_mark_target_fdr) != 0:
            est_fdr = cross_mark_target_fdr[est_fdr_key].tolist()[-1]
            score_thresh = cross_mark_target_fdr[score_name].tolist()[-1]
            true_target_number = cross_mark_target_fdr['cum_true'].tolist()[-1]
        else:
            est_fdr = df_file[est_fdr_key].tolist()[-1]
            score_thresh = df_file[score_name].tolist()[-1]
            true_target_number = df_file['cum_true'].tolist()[-1]
    elif est_fdr_key=='estimated_fdr': # conservative FDR start point 
        if len(cross_mark_candidate) != 0:  
            est_fdr = cross_mark_candidate[est_fdr_key].tolist()[-1]
            score_thresh = cross_mark_candidate[score_name].tolist()[-1]
            true_target_number = cross_mark_candidate['cum_true'].tolist()[-1]   
        else:
            est_fdr = df_file[est_fdr_key].tolist()[0]
            score_thresh = df_file[score_name].tolist()[0]
            true_target_number = df_file['cum_true'].tolist()[0]
    return est_fdr, score_thresh, true_target_number

def extract_indb_and_snp_peptides(denovo_name, query_length_thresh=6, evalue=10, mutation_num=3):
    blast_results = pd.read_csv(f'{denovo_name}.csv', sep=',')
    blast_results = blast_results.loc[blast_results['iden_info'] != 'iden_info']
    
    blast_results = blast_results[blast_results['align_evalue'] <= evalue]
    blast_results = blast_results[blast_results['query_length'] >= query_length_thresh]
    
    blast_results = blast_results.drop_duplicates('query_sequence').reset_index()
    indices = blast_results.loc[blast_results['iden_ratio']==1.0].index
    selected_rows = blast_results.loc[indices].drop_duplicates('query_info', keep=False)
    in_db_peps = selected_rows['query_sequence'].tolist()
    
    iden_info_list = blast_results['iden_info'].tolist()
    query_length = blast_results['query_length'].tolist()
    iden_matches = [int(iden_info_list[i].split('/')[0]) for i in range(len(iden_info_list))]
    iden_stds = [int(iden_info_list[i].split('/')[1]) for i in range(len(iden_info_list))]
    aa_diff = np.where((np.array(iden_stds) - np.array(iden_matches) > 0) & (np.array(iden_stds) - np.array(iden_matches) <= mutation_num) & (np.array(iden_stds) == np.array(query_length)))[0]
    aa_diff_peps = blast_results.loc[aa_diff]['query_sequence'].tolist()
    return in_db_peps, aa_diff_peps

def parse_peptide_proteins(name, output_path, query_length_thresh=6, evalue=10, mutation_num=3):
    blast_results = pd.read_csv(f'{name}.csv', sep=',')
    blast_results = blast_results.loc[blast_results['iden_info'] != 'iden_info']
    blast_results['align_evalue'] = pd.to_numeric(blast_results['align_evalue'], errors='coerce')
    blast_results['query_length'] = pd.to_numeric(blast_results['query_length'], errors='coerce')
    
    blast_results = blast_results[blast_results['align_evalue'] <= evalue]
    blast_results = blast_results[blast_results['query_length'] >= query_length_thresh].reset_index()    
    
    iden_info_list = blast_results['iden_info'].tolist()
    query_length = blast_results['query_length'].tolist()
    iden_matches = [int(iden_info_list[i].split('/')[0]) for i in range(len(iden_info_list))]
    iden_stds = [int(iden_info_list[i].split('/')[1]) for i in range(len(iden_info_list))]
    if mutation_num != 0:
        indices = np.where((np.array(iden_stds) - np.array(iden_matches) > 0) & (np.array(iden_stds) - np.array(iden_matches) <= mutation_num) & (np.array(iden_stds) == np.array(query_length)))[0]
    else:
        indices = np.where((np.array(iden_stds) - np.array(iden_matches) == 0) & (np.array(iden_stds) == np.array(query_length)))[0]
    blast_results = blast_results.loc[indices].reset_index()
    
    subject_id = blast_results['subject_id'].tolist()
    PSM_ID_list = [int(idx[8:]) for idx in blast_results['query_info'].tolist()]
    pep_protein_score_list = [float(i.split(' ')[0]) for i in blast_results['align_score'].tolist()]
    pep_list = blast_results['query_sequence'].tolist()
    data = {
        'PSMId': PSM_ID_list,
        'Score': pep_protein_score_list,
        'peptide': pep_list,
        'proteinIds': subject_id,
        'Label': [0 if 'DECOY' in subject_id[i] else 1 for i in range(len(subject_id))]
           }
    df = pd.DataFrame(data)
    df["proteinIds"] = df["proteinIds"].apply(lambda x: x.split(" ")[0])    
    df.to_csv(output_path, sep="\t", index=False)    
    return blast_results, df

def run_protein_inference_cli(params, params_path, psm_file, fasta_file, output_dir):
    with open(params_path, "w") as f:
        yaml.dump(params, f)
        
    cmd = [
        "protein_inference_cli.py",
        "-f", psm_file,
        "-db", fasta_file,
        '-o', output_dir,
        "-y", params_path,
    ]
    
    subprocess.run(cmd, check=True)
    
    print(f"[✓] finished， results output to directory: {output_dir}")
    
def merge_inferred_peptides_with_psm_id(inferred_peptides_path, original_psm_path):
    """
    Parameters:
        inferred_peptides_path (str): Path to the inference_peptides.tsv file exported by protein inference
        original_psm_path (str): Path to the original PSM data file (containing peptide sequence and psm_id)
        output_path (str): Path for the merged output file
    """
    df_inferred = pd.read_csv(inferred_peptides_path, sep=',')
    df_inferred = df_inferred.rename(columns={"Peptides": "peptide"})
    df_psm = pd.read_csv(original_psm_path, sep='\t')
    peptide_col = 'peptide' if 'peptide' in df_inferred.columns else 'Peptide'
    if 'sequence' in df_psm.columns:
        df_psm = df_psm.rename(columns={'sequence': 'peptide'})
    df_merged = pd.merge(df_inferred, df_psm[['peptide', 'PSMId']], on='peptide', how='left')
    return df_merged
