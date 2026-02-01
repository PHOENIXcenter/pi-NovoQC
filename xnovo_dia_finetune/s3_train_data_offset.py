"""Build per-run spectrum index lookups for faster training.

This script scans the generated MGF/TXT training shards and writes lightweight
CSV index files that map scan identifiers to line offsets (used as a speed-up
during training).

Inputs:
  - `spec_dir`      : output of `s2_train_data_generation.py`
  - `psm_dir`       : output of `s1_psm_match.py`
  - `db_results_dir`: directory used to enumerate run names (one file per run)

Outputs:
  - `out_dir/<run_name>/*.csv`
"""

import os 
import csv
import pandas as pd
import numpy as np

def read_txt(file_path):
    spectra = []
    with open(file_path, 'r') as f:
        spectrum = None
        for line in f:
            line = line.strip()
            if line.startswith("BEGIN IONS"):
                if spectrum:
                    spectra.append(spectrum)
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
                spectrum['params']['seq'] = line.split("=", 1)[1].strip().split(",")
            elif line.startswith("END IONS"):
                spectra.append(spectrum)
                spectrum = None
            else:
                if line:  # 忽略空行
                    try:
                        mz, intensity = map(float, line.split())
                        spectrum['m/z array'].append(mz)
                        spectrum['intensity array'].append(intensity)
                    except ValueError:
                        # 如果解析失败，打印错误信息或采取其他措施
                        print(f"Warning: Line could not be parsed: {line}")
    if spectrum:
        spectra.append(spectrum)
    return spectra

def create_index(file_path):
    index = {}
    with open(file_path, 'r') as f:
        current_pos = 0
        begin_pos = 0
        for line in f:
            if line.startswith("BEGIN IONS"):
                begin_pos = current_pos
            elif line.startswith("SCANS="):
                scans = line.split("=", 1)[1].strip()
                index[scans] = begin_pos
            current_pos += len(line)
    return index

def save_index_to_csv(index, csv_file):
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['scans', 'position'])
        for scans, position in index.items():
            writer.writerow([scans, position])

def read_spectrum_by_line_offset(file_path, line_offset):    
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
                spectrum['params']['pepmass'] = list(map(float, line.split("=", 1)[1].strip().split(", ")))
            elif line.startswith("CHARGE="):
                spectrum['params']['charge'] = line.split("=", 1)[1].strip().split(", ")
            elif line.startswith("RTINSECONDS="):
                spectrum['params']['rtinseconds'] = list(map(float, line.split("=", 1)[1].strip().split(", ")))
            elif line.startswith("SEQ="):
                spectrum['params']['seq'] = line.split("=", 1)[1].strip().split(",")
            elif line.startswith("END IONS"):
                break
            else:
                if line:  # 忽略空行
                    try:
                        mz, intensity = map(float, line.split())
                        spectrum['m/z array'].append(mz)
                        spectrum['intensity array'].append(intensity)
                    except ValueError:
                        print(f"Warning: Line could not be parsed: {line}")
        return spectrum
    
import os 
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser('')
# args.local_rank = int(os.getenv('LOCAL_RANK', 0))
# Distributed training parameters
parser.add_argument('--id_start', default=0, type=int,
                    help='number of distributed processes')
parser.add_argument('--id_end', default=10, type=int,
                    help='number of distributed processes')
parser.add_argument('--db_results_dir', default='data/db_results_filtered', type=str,
                    help='Directory used to enumerate run names (one file per run).')
parser.add_argument('--psm_dir', default='data/psm_info', type=str,
                    help='Directory containing spectrum–peptide mapping CSVs from s1_psm_match.py.')
parser.add_argument('--spec_dir', default='data/train_data', type=str,
                    help='Directory containing training shards from s2_train_data_generation.py.')
parser.add_argument('--out_dir', default='data/train_data_offset', type=str,
                    help='Output directory for generated index CSVs.')
args = parser.parse_args()

db_results_dir = args.db_results_dir
psm_dir = args.psm_dir
spec_dir = args.spec_dir
index_dir = args.out_dir
os.makedirs(index_dir, exist_ok=True)
filenames = os.listdir(db_results_dir)   
filenames = [filename[:-4] for filename in filenames][args.id_start: args.id_end]
print(filenames)
with tqdm(total=len(filenames)) as pbar22:
    for file_id in range(len(filenames)):
        if not os.path.exists(os.path.join(spec_dir, filenames[file_id])):
            continue
        psms = pd.read_csv(os.path.join(psm_dir, f'{filenames[file_id]}.csv'), sep=',')
        spectra_files = os.listdir(os.path.join(spec_dir, filenames[file_id]))
        csv_dir = os.path.join(index_dir, filenames[file_id])
        os.makedirs(csv_dir, exist_ok=True)
        filename = filenames[file_id]
        with tqdm(total=len(spectra_files)) as pbar:
            for spec_id in range(len(spectra_files)):
                csv_file = os.path.join(csv_dir, f'{spectra_files[spec_id][:-4]}.csv')
                if os.path.exists(csv_file):
                    pbar.update(1)
                    continue
                spectra = read_txt(os.path.join(spec_dir, filenames[file_id], spectra_files[spec_id]))
                spec_scans_idxs = create_index(os.path.join(spec_dir, filenames[file_id], spectra_files[spec_id]))
                save_index_to_csv(spec_scans_idxs, csv_file)
                pbar.update(1)
        pbar22.update(1)