"""Dataset and feature utilities for π-xNovo-DIA.

This module defines:
- Spectrum/peptide preprocessing helpers
- Dataset classes and PyTorch DataLoaders
- Mass/fragment utilities used by the Transformer model

The implementation is aligned with the π-NovoQC training pipeline.
"""

# -*- coding: utf-8 -*-
import torch
import torch.utils.data.distributed as dist
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import os 
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from pyteomics import mgf
import re
import numpy as np
from pyteomics_master.pyteomics import mass, mgf


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
        "+229.163": 229.163,
        # AA mods:
        "M+15.995": canonical["M"] + 15.994915,  # Met Oxidation
        "N+0.984": canonical["N"] + 0.984016,  # Asn Deamidation
        "Q+0.984": canonical["Q"] + 0.984016,  # Gln Deamidation
        "K+229.163": 357.257963,
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

def read_spectrum_by_line_offset(file_path, line_offset, n_peaks):    
    with open(file_path, 'r') as f:
        f.seek(line_offset)
        spectrum = {
            'params': {
                'title': '',
                'scans': '',
                'pepmass': [],
                'charge': [],
                'rtinseconds': [],
                'seq': []
            },
            'm/z array': [],
            'intensity array': [],
        }
        for line in f:
            line = line.strip()
            if line.startswith("BEGIN IONS"):
                # Begin new spectrum parsing
                continue
            elif line.startswith("TITLE="):
                spectrum['params']['title'] = line.split("=", 1)[1].strip()
            elif line.startswith("SCANS="):
                spectrum['params']['scans'] = line.split("=", 1)[1].strip()
            elif line.startswith("PEPMASS="):
                pepmass_list = line.split("=", 1)[1].strip().split(", ")
                _pepmass_list = []
                for mass in pepmass_list:
                    _pepmass_list.append(float(mass.strip("'\"")))
                spectrum['params']['pepmass'] = _pepmass_list # list(map(float, line.split("=", 1)[1].strip().split(", ")))
            elif line.startswith("CHARGE="):
                charge_list = line.split("=", 1)[1].strip().split(", ")
                _charge_list = []
                for mass in charge_list:
                    _charge_list.append(int(mass.strip("'\"")))
                spectrum['params']['charge'] = _charge_list # list(map(int, line.split("=", 1)[1].strip().split(", ")))
            elif line.startswith("RTINSECONDS="):
                rt_list = line.split("=", 1)[1].strip().split(", ")
                _rt_list = []
                for mass in rt_list:
                    _rt_list.append(float(mass.strip("'\"")))
                spectrum['params']['rtinseconds'] = _rt_list # list(map(float, line.split("=", 1)[1].strip().split(", ")))
            elif line.startswith("SEQ="):
                spectrum['params']['seq'] = line.split("=", 1)[1].strip("\n").split(",")
            elif line.startswith("END IONS"):
                break
            else:
                if line: 
#                     try:
                    mz, intensity = map(float, line.split())
                    spectrum['m/z array'].append(mz)
                    spectrum['intensity array'].append(intensity)

        mz_array = spectrum['m/z array']
        intensity_array = spectrum['intensity array'] 
        sorted_intensity_idxs = np.argsort(np.array(intensity_array))[-n_peaks:]
        intensities = np.array(intensity_array)[sorted_intensity_idxs]  
        mz_array = np.array(mz_array)[sorted_intensity_idxs]
        
        sorted_mz_idxs = np.argsort(mz_array)
        intensity_array = intensities[sorted_mz_idxs]
        mz_array = mz_array[sorted_mz_idxs]
                                                       
        spectrum['m/z array'] = mz_array
        spectrum['intensity array'] = intensity_array
        return spectrum


class SpectrumDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.prec_length = cfg['prec_length']
        self.path = cfg['mgf_file']
        self.psm_path = cfg['psm_path']
        self.lookup_path = cfg['lookup_file']
        self.species_list = os.listdir(self.psm_path)
        self.species_list = [filename[:-4] for filename in self.species_list] 
        print(self.species_list)
        self.valid_species = cfg['valid_species']
        self.spec_step = cfg['spec_step'] 
        self.cfg = cfg
        self.species = None
        self.filename = None
        self.n_peaks = cfg['n_peaks']
        self.sample_num = cfg['sample_num']
        self.spec_num = cfg['spec_num']

        self.psms_list = []
        with tqdm(total=len(self.species_list)) as pbar:
            for species in self.species_list:
                psms = pd.read_csv(os.path.join(self.psm_path, f'{species}.csv'), sep=',')
                self.psms_list.append(psms)
                pbar.update(1)
                
        self.spectra_num_dic = dict()
        for species in self.species_list:
            stepfiles = os.listdir(os.path.join(self.path, species))
            spectra_number = np.max([int(step.split('_')[1][:-4]) for step in stepfiles])
            self.spectra_num_dic[species] = spectra_number

        # get species and spectrum numbers for training
        self.train_species = []
        self.train_spectra_num = []
        for i in range(len(self.species_list)):
            if self.species_list[i] != self.valid_species:
                self.train_species.append(self.species_list[i])
                self.train_spectra_num.append(self.spectra_num_dic[self.species_list[i]])  
                
                
        self.train_pep_num_idx = list(range(np.sum(self.train_spectra_num)))
        self.mass_items = list(self.cfg['residues'].values())
        print(self.mass_items)
    
    def __getitem__(self, idx):
        try:
            mgf_path, spectrum_idx = self.getfile(idx)
            species, filename = mgf_path.split('/')[-2:]
            abs_id = int(filename.split('_')[0]) + spectrum_idx
            lookup_table_path = os.path.join(self.lookup_path, species, f'{filename[:-4]}.csv')    
            psms_idx = self.species_list.index(species)
            psms = self.psms_list[psms_idx].loc[[abs_id]]
            window_center, lower_offset, upper_offset = psms['window_center'].tolist()[0], psms['lower_offset'].tolist()[0], psms['upper_offset'].tolist()[0]

            looup_table = pd.read_csv(lookup_table_path, sep=',')
            line_offset = looup_table.loc[[spectrum_idx]]['position'].tolist()[0]
            spectrum = read_spectrum_by_line_offset(mgf_path, line_offset, self.n_peaks)

            precursor_mz = spectrum['params']['pepmass']
            precursor_charge = [int(c) for c in spectrum['params']['charge']]
            peptide = spectrum['params']['seq']
            rt = spectrum['params']['rtinseconds']
        except:
            mgf_path, spectrum_idx = self.getfile(idx-1)
            species, filename = mgf_path.split('/')[-2:]
            abs_id = int(filename.split('_')[0]) + spectrum_idx
            lookup_table_path = os.path.join(self.lookup_path, species, f'{filename[:-4]}.csv')    
            psms_idx = self.species_list.index(species)
            psms = self.psms_list[psms_idx].loc[[abs_id]]
            window_center, lower_offset, upper_offset = psms['window_center'].tolist()[0], psms['lower_offset'].tolist()[0], psms['upper_offset'].tolist()[0]

            looup_table = pd.read_csv(lookup_table_path, sep=',')
            line_offset = looup_table.loc[[spectrum_idx]]['position'].tolist()[0]
            spectrum = read_spectrum_by_line_offset(mgf_path, line_offset, self.n_peaks)

            precursor_mz = spectrum['params']['pepmass']
            precursor_charge = [int(c) for c in spectrum['params']['charge']]
            peptide = psms['pep'].tolist()[0].split(',') # spectrum['params']['seq']
            rt = spectrum['params']['rtinseconds']            
        
        sample_idx_list = [i for i in range(len(precursor_mz))]
        if len(precursor_mz) >= self.sample_num:
            sample_idx_list = random.sample(sample_idx_list, self.sample_num)
        else:
            padding_len = self.sample_num - len(precursor_mz)
            sample_idx_list = sample_idx_list + random.choices(sample_idx_list, k=padding_len)
            
        precursor_mz = [precursor_mz[idx] for idx in sample_idx_list]
        _precursor_charge = [precursor_charge[idx] for idx in sample_idx_list]
        precursor_charge = []
        for i in range(len(_precursor_charge)):
            if int(_precursor_charge[i]) >= 5:
                precursor_charge.append(4)
            else:
                precursor_charge.append(_precursor_charge[i])
                
        peptide = [peptide[idx] for idx in sample_idx_list]
        rt = [rt[idx] for idx in sample_idx_list]
    
        sorted_charge_idxs = np.argsort(np.array(precursor_charge))
        precursor_charge = np.array(precursor_charge)[sorted_charge_idxs]  
        precursor_mz = np.array(precursor_mz)[sorted_charge_idxs]  
        rt = np.array(rt)[sorted_charge_idxs]
        peptide = [peptide[idx] for idx in sorted_charge_idxs]
    
        mz_array = spectrum['m/z array']
        intensity_array = spectrum['intensity array']  
        mz_labels = [self.get_aby_labels(seq, mz_array) for seq in peptide]
        mz_labels = np.array(mz_labels)
        mz_labels = torch.from_numpy(mz_labels)
        
        padding_length = self.prec_length - mz_labels.size(0)
        mz_labels = F.pad(mz_labels, (0, self.n_peaks-len(mz_array), 0, padding_length), value=0)  
        
        window_center = torch.tensor(window_center)
        lower_offset = torch.tensor(lower_offset)
        upper_offset = torch.tensor(upper_offset)
        window = torch.vstack([window_center, window_center-lower_offset, window_center+upper_offset]).T.float()
        
        precursor_mz = torch.tensor(precursor_mz)
        precursor_charge = torch.tensor(precursor_charge)
        rt = torch.tensor(rt)
        precursor_mass = (precursor_mz - 1.007276) * precursor_charge + 1.007276
        precursors = torch.vstack([precursor_mass, precursor_charge, precursor_mz]).T.float()
        mz_intensity = torch.from_numpy(np.array([mz_array, intensity_array])).T.float()
        mz_intensity = F.pad(mz_intensity, (0, 0, 0, self.n_peaks-len(mz_array)), value=0)
        return window, precursors, peptide, rt, mz_intensity, mz_labels

    def __len__(self):
        return np.sum(self.train_spectra_num)

    def getinfo(self, idx):
        return self.getfile(idx)

    def getfile(self, idx):
        for i in range(len(self.train_spectra_num)):
            if idx < np.sum(np.array(self.train_spectra_num)[:i+1]):
                species_idx = i
                break
        self.species = self.train_species[species_idx]
        if species_idx == 0:
            rel_idx = idx
        else:
            rel_idx = idx - np.sum(np.array(self.train_spectra_num)[:species_idx])
        step = rel_idx // self.spec_step
        if (step+1) * self.spec_step < self.train_spectra_num[species_idx]:
            self.filename = f'{step*self.spec_step}_{(step+1)*self.spec_step}.mgf'
        else:

            self.filename = f'{step*self.spec_step}_{self.train_spectra_num[species_idx]}.mgf'
        mgf_path = os.path.join(self.path, self.species, self.filename)
        spectrum_idx = rel_idx - step * self.spec_step
        return mgf_path, int(spectrum_idx)

    def get_aby_labels(self, seqs, exp_mz):
        theo_mz = get_theo_mz(seqs)
        aby_list = []
        aby_label = []
        for i in range(len(exp_mz)):
            if np.min(abs(theo_mz - exp_mz[i])) <= 0.05:
                ion_type_idx, stat_type_idx = np.where(abs(theo_mz - exp_mz[i])==np.min(abs(theo_mz - exp_mz[i])))
                peak_ion_type = ion_type_dict[str(ion_type_idx[0])]
                ion_idx, stat_idx = int(stat_type_idx[0]/len(stat_type_dict))+1, stat_type_idx[0]%len(stat_type_dict)
                peak_stat_type = stat_type_dict[str(stat_idx)]
                aby_label.append(ion_type_idx[0] * len(stat_type_dict) + stat_idx + 1)
            else:
                aby_label.append(3 * len(stat_type_dict) + 1)
        return aby_label

class ValidSpectrumDataset(Dataset):
    
    def __init__(self, cfg):
        super().__init__()
        self.prec_length = cfg['prec_length']
        self.path = cfg['mgf_file']
        self.psm_path = cfg['psm_path']
        self.lookup_path = cfg['lookup_file']
        self.species_list = os.listdir(self.psm_path)
        self.species_list = [filename[:-4] for filename in self.species_list] 
        self.valid_species = cfg['valid_species']
        self.spec_step = cfg['spec_step'] 
        self.cfg = cfg
        self.species = self.valid_species
        self.filename = None
        self.n_peaks = cfg['n_peaks']
        self.sample_num = cfg['sample_num']
        
        self.psms = pd.read_csv(os.path.join(self.psm_path, f'{self.valid_species}.csv'), sep=',')
                
        self.spectra_num_dic = dict()
        for species in self.species_list:
            stepfiles = os.listdir(os.path.join(self.path, species))
            spectra_number = np.max([int(step.split('_')[1][:-4]) for step in stepfiles])
            self.spectra_num_dic[species] = spectra_number

        # get species and spectrum numbers for validation
        self.valid_spectra_num = self.spectra_num_dic[self.valid_species]    

        self.mass_items = list(self.cfg['residues'].values())
    
    def __getitem__(self, idx):
        mgf_path, spectrum_idx = self.getfile(idx)
        species, filename = mgf_path.split('/')[-2:]
        abs_id = int(filename.split('_')[0]) + spectrum_idx
        lookup_table_path = os.path.join(self.lookup_path, species, f'{filename[:-4]}.csv')  
        psms = self.psms.loc[[abs_id]]
        window_center, lower_offset, upper_offset = psms['window_center'].tolist()[0], psms['lower_offset'].tolist()[0], psms['upper_offset'].tolist()[0]
        looup_table = pd.read_csv(lookup_table_path, sep=',')
        line_offset = looup_table.loc[[spectrum_idx]]['position'].tolist()[0]

        spectrum = read_spectrum_by_line_offset(mgf_path, line_offset, self.n_peaks)

        precursor_mz = spectrum['params']['pepmass']
        precursor_charge = [int(c) for c in spectrum['params']['charge']]
        peptide = spectrum['params']['seq']
        rt = spectrum['params']['rtinseconds']
        
        sample_idx_list = [i for i in range(len(precursor_mz))]
        if len(precursor_mz) >= self.sample_num:
            sample_idx_list = random.sample(sample_idx_list, self.sample_num)
        else:
            padding_len = self.sample_num - len(precursor_mz)
            sample_idx_list = sample_idx_list + random.choices(sample_idx_list, k=padding_len)
        precursor_mz = [precursor_mz[idx] for idx in sample_idx_list]
        _precursor_charge = [precursor_charge[idx] for idx in sample_idx_list]
        precursor_charge = []
        for i in range(len(_precursor_charge)):
            if int(_precursor_charge[i]) >= 5:
                precursor_charge.append(4)
            else:
                precursor_charge.append(_precursor_charge[i])
        peptide = [peptide[idx] for idx in sample_idx_list]
        rt = [rt[idx] for idx in sample_idx_list]
    
        # Reorder by precursor charge.
        sorted_charge_idxs = np.argsort(np.array(precursor_charge))
        precursor_charge = np.array(precursor_charge)[sorted_charge_idxs]  
        precursor_mz = np.array(precursor_mz)[sorted_charge_idxs]  
        rt = np.array(rt)[sorted_charge_idxs]
        peptide = [peptide[idx] for idx in sorted_charge_idxs]
        
        mz_array = spectrum['m/z array']
        intensity_array = spectrum['intensity array']  

        mz_labels = [self.get_aby_labels(seq, mz_array) for seq in peptide]
        mz_labels = np.array(mz_labels)
        mz_labels = torch.from_numpy(mz_labels)
        
        padding_length = self.prec_length - mz_labels.size(0)
        mz_labels = F.pad(mz_labels, (0, self.n_peaks-len(mz_array), 0, padding_length), value=0)  
        window_center = torch.tensor(window_center)
        lower_offset = torch.tensor(lower_offset)
        upper_offset = torch.tensor(upper_offset)
        window = torch.vstack([window_center, window_center-lower_offset, window_center+upper_offset]).T.float()
        
        precursor_mz = torch.tensor(precursor_mz)
        precursor_charge = torch.tensor(precursor_charge)
        rt = torch.tensor(rt)
        precursor_mass = (precursor_mz - 1.007276) * precursor_charge + 1.007276
        precursors = torch.vstack([precursor_mass, precursor_charge, precursor_mz]).T.float()
        mz_intensity = torch.from_numpy(np.array([mz_array, intensity_array])).T.float()    
        mz_intensity = F.pad(mz_intensity, (0, 0, 0, self.n_peaks-len(mz_array)), value=0)
        return window, precursors, peptide, rt, mz_intensity, mz_labels
    

    def __len__(self):
#         return 20000
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

    def get_aby_labels(self, seqs, exp_mz):
        theo_mz = get_theo_mz(seqs)
        aby_list = []
        aby_label = []
        for i in range(len(exp_mz)):
            if np.min(abs(theo_mz - exp_mz[i])) <= 0.05:
                ion_type_idx, stat_type_idx = np.where(abs(theo_mz - exp_mz[i])==np.min(abs(theo_mz - exp_mz[i])))
                peak_ion_type = ion_type_dict[str(ion_type_idx[0])]
                ion_idx, stat_idx = int(stat_type_idx[0]/len(stat_type_dict))+1, stat_type_idx[0]%len(stat_type_dict)
                peak_stat_type = stat_type_dict[str(stat_idx)]
                aby_label.append(ion_type_idx[0] * len(stat_type_dict) + stat_idx + 1)
            else:
                aby_label.append(3 * len(stat_type_dict) + 1)
        return aby_label

def prepare_batch(batch):
    windows, precs, peptide, rt, mz_intensity, mz_labels = list(zip(*batch))
    windows = torch.nn.utils.rnn.pad_sequence(windows, batch_first=True)
    mz_intensity = torch.nn.utils.rnn.pad_sequence(mz_intensity, batch_first=True)
    precs = torch.nn.utils.rnn.pad_sequence(precs, batch_first=True)
    mz_labels = torch.nn.utils.rnn.pad_sequence(mz_labels, batch_first=True)
    rt = torch.nn.utils.rnn.pad_sequence(rt, batch_first=True).unsqueeze(dim=2)
    return windows, precs, peptide, rt, mz_intensity, mz_labels


def bulid_loader(args, cfg):
    if args.multiprocessing == True:
        # train
        train_dataset = SpectrumDataset(cfg)
        train_sampler = dist.DistributedSampler(train_dataset)                
        train_loader = DataLoader(train_dataset, batch_size=cfg['train_batch_size'], collate_fn=prepare_batch,
                                shuffle=(train_sampler is None), num_workers=cfg['n_workers'],
                                sampler=train_sampler, drop_last=True)
        # validation
        val_dataset = ValidSpectrumDataset(cfg)
        val_sampler = dist.DistributedSampler(val_dataset)
        val_loader = DataLoader(val_dataset, batch_size=cfg['val_batch_size'],
                                shuffle=(val_sampler is None), num_workers=cfg['n_workers'], collate_fn=prepare_batch,
                                sampler=val_sampler, drop_last=False)
        return train_sampler, train_loader, val_loader
    else:
        # train
        train_dataset = SpectrumDataset(cfg)
        train_loader = DataLoader(train_dataset, batch_size=cfg['train_batch_size'], collate_fn=prepare_batch,
                                shuffle=True, num_workers=cfg['n_workers'], drop_last=True)
        # validation
        val_dataset = ValidSpectrumDataset(cfg)
        val_loader = DataLoader(val_dataset, batch_size=cfg['val_batch_size'],
                                shuffle=False, num_workers=cfg['n_workers'], collate_fn=prepare_batch, drop_last=False)
        return train_loader, val_loader
