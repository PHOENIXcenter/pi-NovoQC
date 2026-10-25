# -*- coding: utf-8 -*-
"""Build spectrum–peptide associations for DIA training.

Many DIA identification tools report IDs at the *run level* rather than the
*spectrum level*. This script assigns each identified peptide to MS/MS spectra
using retention time (RT) and isolation-window metadata.

Outputs:
  - One CSV per run with spectrum indices and the assigned peptide (used later by
    `s2_train_data_generation.py` and for training).

Notes:
  - This script is research code and may require small adaptations to match the
    exact column names produced by your database-search tool.
"""

import os
import re
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from pyteomics import mzml

parser = argparse.ArgumentParser('')
# args.local_rank = int(os.getenv('LOCAL_RANK', 0))
# Distributed training parameters
parser.add_argument('--id_start', default=0, type=int,
                    help='number of distributed processes')
parser.add_argument('--id_end', default=10, type=int,
                    help='number of distributed processes')
parser.add_argument('--rt_thresh', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--db_results_dir', default='data/db_results_sep', type=str,
                    help='Directory containing per-run DB-search results (CSV/TSV).')
parser.add_argument('--mzml_dir', default='data/mzml', type=str,
                    help='Directory containing DIA mzML files.')
parser.add_argument('--out_dir', default='data/psm_info', type=str,
                    help='Output directory for generated spectrum–peptide mappings (CSV).')
parser.add_argument('--ms2_cache_dir', default='data/cache/mzml_ms2_idxs', type=str,
                    help='Cache directory for extracted MS2 spectra indices (optional speed-up).')
parser.add_argument('--pep_step', default=100, type=int,
                    help='Chunk size for processing peptides per run.')
args = parser.parse_args()

ms2_cache_dir = args.ms2_cache_dir
os.makedirs(ms2_cache_dir, exist_ok=True)


db_results_dir = args.db_results_dir
bat_mzml_dir = args.mzml_dir
filenames = os.listdir(db_results_dir)
filenames = [filename[:-4] for filename in filenames][args.id_start: args.id_end]

std_aa_mass = {
  "G": 57.021464,
  "A": 71.037114,
  "S": 87.032028,
  "P": 97.052764,
  "V": 99.068414,
  "T": 101.047670,
  "C+57.021": 160.030649, # 103.009185 + 57.021464
  "L": 113.084064,
  "I": 113.084064,
  "N": 114.042927,
  "D": 115.026943,
  "Q": 128.058578,
  "K": 128.094963,
  "E": 129.042593,
  "M": 131.040485,
  "H": 137.058912,
  "F": 147.068414,
  "R": 156.101111,
  "Y": 163.063329,
  "W": 186.079313,
  # Amino acid modifications.
  "+42.011": 42.010565,  # Acetylation
  "M+15.995": 147.035400,    # Met oxidation:   131.040485 + 15.994915
  "N+0.984": 115.026943,   # Asn deamidation: 114.042927 +  0.984016
  "Q+0.984": 129.042594,
  "+229.163": 229.163,
  # AA mods:
  "M+15.995": 131.040485 + 15.994915,  # Met Oxidation
  "N+0.984": 114.042927 + 0.984016,  # Asn Deamidation
  "Q+0.984": 128.058578 + 0.984016,  # Gln Deamidation
  "K+229.163": 357.257963,}    # Gln deamidation: 128.058578 +  0.984016}
"""A dictionary with monoisotopic masses of the twenty standard
amino acid residues, selenocysteine and pyrrolysine.
"""

device = 'cuda:0'
ms2_step = 10000
pep_step = args.pep_step
rt_thresh = args.rt_thresh
psm_info_dir = args.out_dir
os.makedirs(psm_info_dir, exist_ok=True)
os.makedirs(psm_info_dir, exist_ok=True)

for filename in filenames:
    print(filename)
    spectra = mzml.read(os.path.join(bat_mzml_dir, f'{filename}.mzML'), use_index=True)
    pep_files_ = pd.read_csv(os.path.join(db_results_dir, f'{filename}.csv'), sep=',')
    key_list = pep_files_.keys().tolist()
    query_key = [key for key in key_list if 'raw.EG.PTMSites [Oxidation (M)]' in key][0]
    raw_key_num = query_key.split(' ')[0]
    pep_files_ = pep_files_[pep_files_[query_key] != 'Filtered'].reset_index(drop=False)
    
    if os.path.exists(os.path.join(ms2_cache_dir, f'{filename}.npy')):
        ms2_list = np.load(os.path.join(ms2_cache_dir, f'{filename}.npy'), allow_pickle=True).tolist()
        pep_start, pep_end = 0, int(len(pep_files_) / pep_step)
    else:
        ms2_list = []
        pep_start, pep_end = 0, int(len(pep_files_) / pep_step)
        with tqdm(total=len(spectra)) as pbar:
            for spec_idx in range(len(spectra)):
                if spectra[spec_idx]['ms level'] == 2:
                    ms2_list.append(spectra[spec_idx])
                pbar.update(1)
        np.save(os.path.join(ms2_cache_dir, f'{filename}.npy'), np.array(ms2_list))            
    # Extract MS2 retention times and isolation windows.
    ms2_rt_list = []
    with tqdm(total=len(ms2_list)) as pbar:
        for spec in ms2_list:
            cur_rt = 60 * spec['scanList']['scan'][0]['scan start time']
            ms2_rt_list.append(cur_rt)
            pbar.update(1)
     
    ms2_peaks_num_list = []
    with tqdm(total=len(ms2_list)) as pbar:
        for spec in ms2_list:
            ms2_peaks_num_list.append(len(spec['m/z array']))
            pbar.update(1)    
    
    ms2_prec_lower_window_list = []
    ms2_prec_upper_window_list = []
    with tqdm(total=len(ms2_list)) as pbar:
        for spec in ms2_list:
            window = spec['precursorList']['precursor'][0]['isolationWindow']
            window_center = window['isolation window target m/z']
            lower_offset = window['isolation window lower offset']
            upper_offset = window['isolation window upper offset']
            prec_lower = window_center - lower_offset
            prec_upper = window_center + upper_offset
            ms2_prec_lower_window_list.append(prec_lower)
            ms2_prec_upper_window_list.append(prec_upper) 
            pbar.update(1)

    ms2_rt = torch.from_numpy(np.array(ms2_rt_list)).unsqueeze(dim=0).to(device).detach()
    ms2_peaks_num = torch.from_numpy(np.array(ms2_peaks_num_list)).unsqueeze(dim=0).to(device).detach()
    ms2_prec_lower = torch.from_numpy(np.array(ms2_prec_lower_window_list)).unsqueeze(dim=0).to(device).detach()
    ms2_prec_upper = torch.from_numpy(np.array(ms2_prec_upper_window_list)).unsqueeze(dim=0).to(device).detach()
    ms2_peaks_num = ms2_peaks_num > 10
    # Find the matching PSM indices.
    pep_indices = []
    ms2_indices = []
    with tqdm(total=pep_end) as pbar:
        for i in range(pep_start, pep_end-1):
            if i != pep_end-1:
                pep_rt = pep_files_[f'{raw_key_num} {filename}.raw.EG.ApexRT'].tolist()[i*pep_step:(i+1)*pep_step]
                pep_c = pep_files_['EG.PrecursorId'].tolist()[i*pep_step:(i+1)*pep_step]
                pep_c = [int(prec.split('.')[-1]) for prec in pep_c]
                peps = pep_files_['EG.ModifiedSequence'].tolist()[i*pep_step:(i+1)*pep_step]
                peps = [pep[1:-1].replace('C', 'C+57.021').replace('[Oxidation (M)]', '+15.995').replace('[Acetyl (Protein N-term)]', '+42.011') for pep in peps]
                pep_prec_mz = [mass.fast_mass(re.split(r"(?<=.)(?=[A-Z])", peps[i]), ion_type='M', charge=int(pep_c[i]), aa_mass=std_aa_mass) for i in range(len(peps))]
            else:
                pep_rt = pep_files_[f'{raw_key_num} {filename}.raw.EG.ApexRT'].tolist()[i*pep_step:len(pep_files_[f'{raw_key_num} {filename}.raw.EG.ApexRT'])]
                pep_c = pep_files_['EG.PrecursorId'].tolist()[i*pep_step:len(pep_files_['EG.PrecursorId'])]
                pep_c = [int(prec.split('.')[-1]) for prec in pep_c]
                peps = pep_files_['EG.ModifiedSequence'].tolist()[i*pep_step:len(pep_files_['EG.PrecursorId'])]
                peps = [pep[1:-1].replace('C', 'C+57.021').replace('[Oxidation (M)]', '+15.995').replace('[Acetyl (Protein N-term)]', '+42.011') for pep in peps]
                pep_prec_mz = [mass.fast_mass(re.split(r"(?<=.)(?=[A-Z])", peps[i]), ion_type='M', charge=int(pep_c[i]), aa_mass=std_aa_mass) for i in range(len(peps))]
            pep_rt = [float(i) for i in pep_rt]
            pep_prec_mz = [float(i) for i in pep_prec_mz]
            pep_rt = torch.from_numpy(np.array(pep_rt)).unsqueeze(dim=1).to(device).detach() * 60
            pep_prec_mz = torch.from_numpy(np.array(pep_prec_mz)).unsqueeze(dim=1).to(device).detach()

            rt_diff = pep_rt - ms2_rt
            prec_lower_diff = pep_prec_mz - ms2_prec_lower
            prec_upper_diff = ms2_prec_upper - pep_prec_mz
            rt_bool = torch.abs(rt_diff) < rt_thresh
            prec_lower_bool = prec_lower_diff > 0
            prec_upper_bool = prec_upper_diff > 0
            
            # Combine the conditions element by element.
            combined = rt_bool & prec_lower_bool & prec_upper_bool & ms2_peaks_num

            # Find indices that satisfy every condition.
            indices = torch.nonzero(combined)
            indices[:,0] = indices[:,0] + i * pep_step
            pep_indices += indices[:,0].cpu().numpy().tolist()
            ms2_indices += indices[:,1].cpu().numpy().tolist()
            pbar.update(1)

    # Write PSMs in MS2-index order.
    sorted_sm2_idxs = np.argsort(np.array(ms2_indices))
    ms2_indices = np.array(ms2_indices)[sorted_sm2_idxs]
    pep_indices = np.array(pep_indices)[sorted_sm2_idxs]

    ms2_indices_list = list(set(ms2_indices))
    rows = ['ms2_idx', 'ms2_rt', 'window_center', 'lower_offset', 'upper_offset',
            'pep_idx', 'pep', 'pep_rt', 'pep_cahrge', 'pep_mz']
    with open(f'{psm_info_dir}/{filename}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(rows)
        for idx in ms2_indices_list:
            _ms2_idx = ms2_list[idx]['index']
            _ms2_rt = 60 * ms2_list[idx]['scanList']['scan'][0]['scan start time']
            _window_center = ms2_list[idx]['precursorList']['precursor'][0]['isolationWindow']['isolation window target m/z']
            _ms2_prec_lower = ms2_list[idx]['precursorList']['precursor'][0]['isolationWindow']['isolation window lower offset']
            _ms2_prec_upper = ms2_list[idx]['precursorList']['precursor'][0]['isolationWindow']['isolation window upper offset']

            _peps_idx = str(pep_indices[ms2_indices==idx].tolist())[1:-1]
            _peps_rt = str(pep_files_.loc[pep_indices[ms2_indices==idx]][f'{raw_key_num} {filename}.raw.EG.ApexRT'].tolist())[1:-1]
            pep_c = pep_files_.loc[pep_indices[ms2_indices==idx]]['EG.PrecursorId'].tolist()
            pep_c = [int(prec.split('.')[-1]) for prec in pep_c]
            _peps_charge = str(pep_c)[1:-1]
            peps = pep_files_.loc[pep_indices[ms2_indices==idx]]['EG.ModifiedSequence'].tolist()
            peps = [pep[1:-1].replace('C', 'C+57.021').replace('[Oxidation (M)]', '+15.995').replace('[Acetyl (Protein N-term)]', '+42.011') for pep in peps]
            _peps  = ','.join(peps)
            pep_prec_mz = [mass.fast_mass(re.split(r"(?<=.)(?=[A-Z])", peps[i]), ion_type='M', charge=int(pep_c[i]), aa_mass=std_aa_mass) for i in range(len(peps))]
            _peps_mz = str(pep_prec_mz)[1:-1]
            
            row = [_ms2_idx, _ms2_rt, _window_center, _ms2_prec_lower, _ms2_prec_upper, 
                  _peps_idx, _peps, _peps_rt, _peps_charge, _peps_mz]
            writer.writerow(row)
