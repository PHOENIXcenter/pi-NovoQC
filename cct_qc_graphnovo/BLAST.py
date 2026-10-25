"""BLAST helpers for peptide-to-protein mapping.

This file provides lightweight wrappers around NCBI BLAST+ for:
- converting peptide lists to FASTA,
- running short-peptide BLAST searches,
- parsing BLAST outputs.

Note: the default implementation calls BLAST via `conda run -n blast ...`.
Edit the `conda_env` argument if you manage BLAST differently.
"""

import re
import csv
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import subprocess

def read_fasta(file_path):
    sequences = {}
    with open(file_path, 'r') as file:
        sequence_id = ""
        sequence = ""
        for line in file:
            if line.startswith(">"):
                if sequence_id:
                    sequences[sequence_id] = sequence
                sequence_id = line.strip()[1:]
                sequence = ""
            else:
                sequence += line.strip()
        if sequence_id:
            sequences[sequence_id] = sequence
    return sequences

def peptides_to_fasta(seqs, spec_id, output_file):
    if len(seqs) != len(spec_id):
        raise ValueError
    with open(output_file, 'w') as file:
        for i in range(len(seqs)):
            file.write(f">peptide_{spec_id[i]}\n")
            file.write(seqs[i] + "\n")

def blast_peptides(query_name, db_name, output_name, evalue=10, conda_env="blast"):
    """
    Align peptides to the target database using blastp.
    Run blastp in the specified conda environment via conda run.
    """
    cmd = [
        "conda", "run", "-n", conda_env,
        "blastp",
        "-task", "blastp-short",
        "-query", query_name,
        "-db", db_name,
        "-evalue", str(evalue),
        "-outfmt", "0",
        "-out", output_name
    ]
    print(f"[✓] Running BLAST:\n{' '.join(cmd)}")
    
    subprocess.run(cmd, check=True)



def parse_and_combine_sequences(blast_output):
    lines = blast_output.strip().split("\n")
    
    # Accumulate sequences and alignment coordinates across blocks.
    query_sequence = ""
    align_sequence = ""
    subject_sequence = ""
    query_start = None
    query_end = None
    subject_start = None
    subject_end = None
    
    for line in lines:
        if line.startswith("Query"):
            parts = line.split()
            if not query_start:
                query_start = int(parts[1])
            query_end = int(parts[3])
            query_sequence += parts[2]
            align_sequence += ' ' * (len(parts[1]) + len(parts[3]) + 4)  # Pad the alignment line.
        elif line.startswith("Sbjct"):
            parts = line.split()
            if not subject_start:
                subject_start = int(parts[1])
            subject_end = int(parts[3])
            subject_sequence += parts[2]
        else:
            # Collect the alignment match line.
            align_sequence += line.strip()

    # Remove spaces from the alignment match string.
    align_sequence = align_sequence.replace(" ", "")

    # Assemble the full alignment sequences.
    full_query_sequence = f"{query_sequence}"
    full_subject_sequence = f"{subject_sequence}"
    full_align_sequence = align_sequence

    return {
        'query_sequence': full_query_sequence,
        'align_sequence': full_align_sequence,
        'subject_sequence': full_subject_sequence,
        'query_idx':(query_start, query_end),
        'subject_idx':(subject_start, subject_end)
    }

def parse_blast_output_to_dict(blast_text):
    # Parse query metadata and collect its alignments.
    lines = blast_text.split("\n\n\n")
    query_info, query_length = lines[0].split('\n')[0][7:], int(lines[0].split('\n')[2].split('=')[-1])
    alignments = []
    for i in range(len(lines)):
        if i != (len(lines) - 1) and i != 0:
            if 'Lambda' in lines[i]:  # Stop at the trailing BLAST statistics.
                break
            subject_info, align_info, align_results = lines[i].split('\n\n')[0], lines[i].split('\n\n')[1], '\n\n'.join(lines[i].split('\n\n')[2:])
            if 'Score' in subject_info: # wrong match, without protein accession id
                continue
            # subject info
            subject_info = subject_info.split('\nLength=')
            for word in subject_info:
                if word.startswith(">"):
                    subject_id = word[1:].replace('\n', '')
#                 elif word.startswith("Length="):    
#                     subject_length = int(word.split('=')[-1])
                else:
                    subject_length = int(word)
            # align info
#             try:
            if 'Score' in align_info:
                align_info, align_metrics = align_info.split('\n')[0].split(','), align_info.split('\n')[1].split(',')
                align_score, align_evalue, align_method = align_info[0][9:], float(align_info[1][11:]), align_info[2][9:-2]
#             except:
#                 continue
            else:
#                 print(align_info) 
#                 print('Wrong align info!!!!!')
                raise ValueError
#             print(f'*************the {i}th match')
#             print(align_info)
            iden_info, iden_ratio = align_metrics[0][14:].split(' ')[0], round(int(align_metrics[0][14:].split(' ')[1].split('%')[0][1:])/100, 2)
            pos_info, pos_ratio = align_metrics[1][13:].split(' ')[0], round(int(align_metrics[1][13:].split(' ')[1].split('%')[0][1:])/100, 2)
            gap_info, gap_ratio = align_metrics[2][8:].split(' ')[0], round(int(align_metrics[2][8:].split(' ')[1].split('%')[0][1:])/100, 2)
            # align_seqs
#             print("****************parse before:")
#             print(align_results, type(align_results))
            align_results = parse_and_combine_sequences(align_results)
#             print(align_results)
#             print(align_results['query_idx'][1], align_results['query_idx'][0], query_length)
            if (align_results['query_idx'][1] - align_results['query_idx'][0] + 1) != query_length:
                continue
            alignment = {
                         'query_info':query_info, 'query_length':query_length,
                         'subject_id':subject_id, 'subject_length':subject_length, 
                         'align_score':align_score, 'align_evalue':align_evalue, 'align_method':align_method, 
                         'iden_info':iden_info, 'iden_ratio':iden_ratio, 
                         'pos_info':pos_info, 'pos_ratio':pos_ratio,
                         'gap_info':gap_info, 'gap_ratio':gap_ratio,
                         'query_sequence': align_results['query_sequence'],
                         'align_sequence': align_results['align_sequence'],
                         'subject_sequence': align_results['subject_sequence'],
                         'query_idx': align_results['query_idx'],
                         'subject_idx': align_results['subject_idx'],
                        }
            alignments.append(alignment)
#         except:
    return alignments

def process_db_results(file_path, final_results_path):
#     file_path = "results.txt"
    title = file_path.split('/')[-1][:-4]
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Split the output into peptide query blocks.
    separator = 'Query= peptide'
    contigs = content.split(separator)

    # Restore each query header and skip the text before the first query.
    processed_contigs = [separator + contig for contig in contigs[1:]]
    # Report the number of peptide queries.
    print(len(processed_contigs))
    
    all_alignments = []
    no_hits = 0
    with tqdm(total=len(processed_contigs)) as pbar:
        for i in range(len(processed_contigs)):
            if 'No hits found' in processed_contigs[i]:
                no_hits += 1 
                pbar.update(1)
                continue
            else:    
#                 try:
                all_alignments += parse_blast_output_to_dict(processed_contigs[i])
#                 except:
    #                 print(processed_contigs[i])
#                 break
                pbar.update(1)
    print(no_hits, len(all_alignments), len(processed_contigs))    


    # root_dir = '/code/'
#     title = 'PEAKS'
    with open(f'{final_results_path}.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(list(all_alignments[0].keys()))
        with tqdm(total=len(all_alignments)) as pbar:
            for k in range(len(all_alignments)):
                head = all_alignments[k].values()
                writer.writerow(head) 
                pbar.update(1)     
