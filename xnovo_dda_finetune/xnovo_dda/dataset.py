"""Dataset utilities for π-xNovo (DDA).

This module provides a minimal MGF-based dataset + collate function used by
`main_xNovo.py`.

Expected input layout
---------------------
The training/validation data are expected to be stored as *chunks* of MGF files
under a root directory, e.g.::

    data/mgf_dda/
      SAMPLE_A_DDA_.../
        0_999.mgf
        1000_1999.mgf
        ...
      SAMPLE_B_DDA_.../
        0_999.mgf
        ...

Each MGF entry should contain at least:
  - m/z array
  - intensity array
  - params.pepmass (precursor m/z)
  - params.charge
  - params.seq (peptide sequence string; modified residues are allowed)

Notes
-----
* This loader keeps only the top-N most intense peaks (default 150) per
  spectrum, then sorts them by m/z.
* Intensities are normalized to the max intensity within each spectrum.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch
import torch.utils.data.distributed as dist
from pyteomics import mgf
from torch.utils.data import DataLoader, Dataset

def normalize_intensities(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize peak intensities within each spectrum.

    Parameters
    ----------
    tensor:
        Tensor of shape (n_peaks, 2) or (batch, n_peaks, 2), where the last
        dimension is (m/z, intensity).

    Returns
    -------
    torch.Tensor
        Tensor with intensities rescaled to max(intensity)=1 within each
        spectrum.
    """
    intensities = tensor[..., 1]
    max_val = torch.max(intensities, dim=-1, keepdim=True).values
    normalized_intensities = torch.where(max_val != 0, intensities / max_val, intensities)
    tensor_normalized = tensor.clone()
    tensor_normalized[..., 1] = normalized_intensities
    return tensor_normalized

class SpectrumDataset(Dataset):
    
    def __init__(self, cfg):
        super().__init__()
        self.path = cfg['mgf_file']
        # Number of peaks to keep per spectrum (top-N by intensity).
        self.n_peaks = int(cfg.get('n_peaks', 150))
        self.species_list = os.listdir(self.path)
        print(len(self.species_list))
        self.valid_species = cfg['valid_species']
        self.spec_step = cfg['spec_step'] 
        self.cfg = cfg
        self.species = None
        self.filename = None
        
        self.spectra_num_dic = dict()
        for species in self.species_list:
            stepfiles = os.listdir(os.path.join(self.path, species))
            stepfiles = [step for step in stepfiles if step[-4:] == '.mgf']
            spectra_number = np.max([int(step.split('_')[1][:-4]) for step in stepfiles])
            self.spectra_num_dic[species] = spectra_number

        # get species and spectrum numbers for training
        self.train_species = []
        self.train_spectra_num = []
        for i in range(len(self.species_list)):
            if self.species_list[i] != self.valid_species:
                self.train_species.append(self.species_list[i])
                self.train_spectra_num.append(self.spectra_num_dic[self.species_list[i]])  
        print(len(self.train_spectra_num))
        self.mass_items = list(self.cfg['residues'].values())
        print(self.mass_items)
        self.aa_2_masses = []
        for i in range(len(self.mass_items)):
            for j in range(len(self.mass_items)):
                self.aa_2_masses.append(self.mass_items[i]+self.mass_items[j])

        self.spectrum = None
        self.spectrum_path = None

    def __getitem__(self, idx):
        mgf_path, spectrum_idx = self.getfile(idx)
        spectra = mgf.read(mgf_path)
        try:
            spectrum = spectra[spectrum_idx]
        except:
            spectrum = spectra[spectrum_idx-1]
        precursor_mz = spectrum['params']['pepmass'][0]
        precursor_charge = int(spectrum['params']['charge'][0])

        peptide = spectrum['params']['seq'].replace('M(+15.99)', 'M+15.995').replace('C(+57.02)', 'C+57.021')
        mz_array = spectrum['m/z array']
        intensity_array = spectrum['intensity array'] 

        sorted_intensity_idxs = np.argsort(np.array(intensity_array))[-self.n_peaks:]
        intensities = np.array(intensity_array)[sorted_intensity_idxs]  
        mz_array = np.array(mz_array)[sorted_intensity_idxs]
        
        sorted_mz_idxs = np.argsort(mz_array)
        intensity_array = intensities[sorted_mz_idxs]
        mz_array = mz_array[sorted_mz_idxs]
        mz_intensity = torch.from_numpy(np.array([mz_array, intensity_array])).T.float()
        
        return normalize_intensities(mz_intensity), precursor_mz, precursor_charge, peptide

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
            self.filename = f'{step*self.spec_step}_{(step+1)*self.spec_step-1}.mgf'
        else:

            self.filename = f'{step*self.spec_step}_{self.train_spectra_num[species_idx]}.mgf'
        mgf_path = os.path.join(self.path, self.species, self.filename)
        spectrum_idx = rel_idx - step * self.spec_step
        return mgf_path, int(spectrum_idx)

        
class ValidSpectrumDataset(Dataset):
    
    def __init__(self, cfg):
        super().__init__()
        self.path = cfg['mgf_file']
        # Keep consistent preprocessing between train/valid.
        self.n_peaks = int(cfg.get('n_peaks', 150))
        self.species_list = os.listdir(self.path)
        self.valid_species = cfg['valid_species']
        self.spec_step = cfg['spec_step'] 
        self.cfg = cfg
        self.species = self.valid_species
        self.filename = None

        self.spectra_num_dic = dict()
        for species in self.species_list:
            stepfiles = os.listdir(os.path.join(self.path, species))
            stepfiles = [step for step in stepfiles if step[-4:] == '.mgf']
            spectra_number = np.max([int(step.split('_')[1][:-4]) for step in stepfiles])

            self.spectra_num_dic[species] = spectra_number

        # get species and spectrum numbers for validation
        self.valid_spectra_num = self.spectra_num_dic[self.valid_species]    

        self.mass_items = list(self.cfg['residues'].values())
        self.aa_2_masses = []
        for i in range(len(self.mass_items)):
            for j in range(len(self.mass_items)):
                self.aa_2_masses.append(self.mass_items[i]+self.mass_items[j])

    def __getitem__(self, idx):
        mgf_path, spectrum_idx = self.getfile(idx)
        spectra = mgf.read(mgf_path, read_ions=True)
        try:
            spectrum = spectra[spectrum_idx]
        except:
            spectrum = spectra[spectrum_idx-1]
        precursor_mz = spectrum['params']['pepmass'][0]
        precursor_charge = int(spectrum['params']['charge'][0])
        peptide = spectrum['params']['seq'].replace('M(+15.99)', 'M+15.995').replace('C(+57.02)', 'C+57.021') # .replace('C', 'C+57.021')
        mz_array = spectrum['m/z array']
        intensity_array = spectrum['intensity array']     
        sorted_intensity_idxs = np.argsort(np.array(intensity_array))[-self.n_peaks:]
        intensities = np.array(intensity_array)[sorted_intensity_idxs]  
        mz_array = np.array(mz_array)[sorted_intensity_idxs]
        
        sorted_mz_idxs = np.argsort(mz_array)
        intensity_array = intensities[sorted_mz_idxs]
        mz_array = mz_array[sorted_mz_idxs]
        mz_intensity = torch.from_numpy(np.array([mz_array, intensity_array])).T.float()
        return normalize_intensities(mz_intensity), precursor_mz, precursor_charge, peptide
    

    def __len__(self):
        return self.valid_spectra_num

    def getinfo(self, idx):
        return self.getfile(idx)

    def getfile(self, idx):
        step = idx // self.spec_step
        if (step+1) * self.spec_step <= self.valid_spectra_num:
            self.filename = f'{step*self.spec_step}_{(step+1)*self.spec_step-1}.mgf'
        else:
            self.filename = f'{step*self.spec_step}_{self.valid_spectra_num}.mgf'
        mgf_path = os.path.join(self.path, self.valid_species, self.filename)
        spectrum_idx = idx - step * self.spec_step
        return mgf_path, int(spectrum_idx)

def prepare_batch(batch):
    """Collate function used by PyTorch DataLoader.

    This pads variable-length peak lists to a common length (within the batch)
    and constructs the precursor feature matrix.
    """
    mz_intensity, precursor_mz, precursor_charge, peptide = list(zip(*batch))
    mz_intensity = torch.nn.utils.rnn.pad_sequence(mz_intensity, batch_first=True)
    precursor_mz = torch.tensor(precursor_mz)
    precursor_charge = torch.tensor(precursor_charge)
    precursor_mass = (precursor_mz - 1.007276) * precursor_charge + 1.007276
    precursors = torch.vstack([precursor_mass, precursor_charge, precursor_mz]).T.float()
    return mz_intensity, precursors, peptide 


def bulid_loader(args, cfg):
    """Backward-compatible alias.

    The original internal code misspelled this helper name as
    ``bulid_loader``. It is kept here to avoid breaking existing scripts.
    Please use :func:`build_loader` in new code.
    """
    return build_loader(args, cfg)


def build_loader(args, cfg):
    """Build train/validation dataloaders.

    In single-GPU mode, returns (train_loader, val_loader).
    In distributed mode (args.multiprocessing=True), returns
    (train_sampler, train_loader, val_loader).
    """
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
