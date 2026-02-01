# -*- coding: utf-8 -*-
"""Generate training MGF shards from DIA mzML + matched PSMs.

This script reads:
  - DIA mzML files (raw MS/MS spectra)
  - CSV files with spectrum–peptide mappings (output of `s1_psm_match.py`)

and writes MGF shards that can be consumed by `main_xNovo.py`.

Outputs:
  - `train_data_dir/<run_name>/<start>_<end>.mgf`

Notes:
  - The script assumes the CSV contains at least the following columns:
    `ms2_idx`, `pep`, `pep_rt`, `pep_mz`, and `pep_cahrge` (typo preserved from
    the original experiment; rename if needed).
"""

import os
import math
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
parser.add_argument('--db_results_dir', default='data/db_results_filtered', type=str,
                    help='Directory used to enumerate run names (one file per run).')
parser.add_argument('--mzml_dir', default='data/mzml', type=str,
                    help='Directory containing DIA mzML files.')
parser.add_argument('--psm_info_dir', default='data/psm_info', type=str,
                    help='Directory containing spectrum–peptide mapping CSVs from s1_psm_match.py.')
parser.add_argument('--train_data_dir', default='data/train_data', type=str,
                    help='Output directory for generated training MGF shards.')
parser.add_argument('--spec_step', default=1000, type=int,
                    help='Number of spectrum–peptide pairs per output MGF shard.')
args = parser.parse_args()

db_results_dir = args.db_results_dir
bat_mzml_dir = args.mzml_dir
generated_psms_dirs = args.psm_info_dir
train_data_dir = args.train_data_dir
spec_step = args.spec_step

os.makedirs(train_data_dir, exist_ok=True)



def write_mgf(spectra, output_file):
    with open(output_file, 'w') as f:
        for spectrum in spectra:
            params = spectrum['params']
            f.write('BEGIN IONS\n')
            f.write(f"TITLE={params['title']}\n")
            f.write(f"SCANS={params['scans']}\n")
            f.write(f"PEPMASS={params['pepmass']}\n")
            f.write(f"CHARGE={params['charge']}\n")
            f.write(f"RTINSECONDS={params['rtinseconds']}\n")
            f.write(f"SEQ={params['seq']}\n")
            m_z_array = spectrum['m/z array']
            intensity_array = spectrum['intensity array']
            for mz, intensity in zip(m_z_array, intensity_array):
                f.write(f"{mz} {intensity}\n")
            f.write('END IONS\n\n')

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

for k in range(len(filenames)):
    blank_mz = 0
    filename = filenames[k]
    print(filename)
    if not os.path.exists(os.path.join(generated_psms_dirs, f'{filename}.csv')):
        print('Not exists!!!!')
        continue
    psms = pd.read_csv(os.path.join(generated_psms_dirs, f'{filename}.csv'), sep=',')
    spectra = mzml.read(os.path.join(bat_mzml_dir, f'{filename}.mzML'), use_index=True)
    mgf_save_dir = os.path.join(train_data_dir, f'{filename}') 
    os.makedirs(mgf_save_dir, exist_ok=True)

    ms2_idxs = psms['ms2_idx'].tolist()
    pep_idxs = psms['pep_idx'].tolist()
    peps = psms['pep'].tolist()
    pep_rts = psms['pep_rt'].tolist()
    pep_mzs = psms['pep_mz'].tolist()
    pep_charges = psms['pep_cahrge'].tolist()
    spectra_num = math.ceil(len(ms2_idxs)/spec_step)
    print(len(ms2_idxs))
    with tqdm(total=spectra_num) as pbar:
        for j in range(spectra_num):
            if (j+1) * spec_step < len(ms2_idxs):
                start_step, end_step = j * spec_step, (j+1) * spec_step
            else:
                start_step, end_step = j * spec_step, len(ms2_idxs)
            save_path = os.path.join(mgf_save_dir, f'{start_step}_{end_step}.mgf')
            if os.path.exists(save_path):
                pbar.update(1)
                continue  

            iden_spec_list = []
#             print(start_step, end_step)
            for i in range(start_step, end_step):
                spec_idx, pep_idx, pep, prec_rt, prec_mz, prec_c = \
                ms2_idxs[i], pep_idxs[i], peps[i].split(','), pep_rts[i], pep_mzs[i], pep_charges[i]
                pep = ','.join([_pep.replace('(UniMod:4)', '+57.021') for _pep in pep])
                spectrum = spectra[spec_idx]
                if len(spectrum['m/z array'].tolist()) <= 10:
                    raise ValueError
#                 q = i
#                 for m in range(start_step, end_step):
#                     if len(spectrum['m/z array'].tolist()) <= 10:
#                         print(len(spectrum['m/z array'].tolist()))
#                         print(spectrum['m/z array'].tolist())
#                         q = q + 1
#                         print(q, i)
#                         spec_idx, pep_idx, pep, prec_rt, prec_mz, prec_c = \
#                         ms2_idxs[q], pep_idxs[q], peps[q].split(','), pep_rts[q], pep_mzs[q], pep_charges[q]
#                         pep = ','.join([_pep.replace('(UniMod:4)', '+57.021') for _pep in pep])
#                         spectrum = spectra[spec_idx]  
#                     else:
#                         blank_mz += 1
#                         break
                params = {'title':pep_idx, 
                  'seq':pep, 
                  'pepmass':prec_mz, 
                  'charge':prec_c, 
                  'rtinseconds':prec_rt,
                  'scans':spec_idx}
                spec = {'params':params, 'm/z array':spectrum['m/z array'], 'intensity array':spectrum['intensity array']}
                iden_spec_list.append(spec)
            write_mgf(iden_spec_list, save_path)
#             print(blank_mz) 
            pbar.update(1)
    print(f'blank mz is {blank_mz}')