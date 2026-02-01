"""Training/validation utilities for π-xNovo (DDA).

This module contains:
  - A light-weight Config wrapper (YAML -> Python object)
  - Training/validation loops used by `main_xNovo.py`

Design notes
------------
* The training loop logs per-iteration and per-epoch metrics to TensorBoard
  via `tensorboardX.SummaryWriter`.
* For public release, the code is intentionally kept simple and explicit.
"""

from __future__ import annotations

import os
from pprint import pformat

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from tqdm import tqdm

class Config:
    """
    Configuration namespace. Convert dictionary to members.
    """

    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, (list, tuple)):
                setattr(self, k, [Config(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()


def read_cfg(path):
    with open(path) as f:
        cfgs = yaml.safe_load(f)
    return Config(cfgs)

def metric_average(val):
    """Compute an all-reduce average of a scalar across distributed workers.

    Parameters
    ----------
    val:
        A Python scalar or a 0-d tensor convertible to float.

    Returns
    -------
    float
        The mean value across all ranks.
    """
    tensor = torch.tensor(val).cuda().type(torch.cuda.FloatTensor)
    tensor_list = [torch.zeros(1).cuda() for _ in range(dist.get_world_size())]
    
    dist.all_gather(tensor_list, tensor)
    avg_tensor = 0
    for i in range(len(tensor_list)):
        avg_tensor += tensor_list[i].item()
    
    return avg_tensor / len(tensor_list)

def train_one_epoch(args, epoch, model, loader, optimizer, scheduler, writer):
    """Run one epoch of fine-tuning.

    This function logs per-iteration loss/metrics and returns the epoch-averaged
    training loss.
    """
    model.train()
    train_loss, train_aa_precision, train_aa_recall, train_pep_recall  = 0, 0, 0, 0
    with tqdm(total=len(loader)) as pbar:
        for i, inputs in enumerate(loader):
            mz_intensity, precursors, peptide = inputs
            mz_intensity = mz_intensity.to(args.device, dtype=torch.float32)
            precursors = precursors.to(args.device, dtype=torch.float32)
            loss, aa_precision, aa_recall, pep_recall = model.training_step([mz_intensity, precursors, peptide])
            print(i, loss.item(), aa_precision.item(), aa_recall.item(), pep_recall.item())
            train_loss += loss.item() * mz_intensity.size(0)
            train_aa_precision += aa_precision.item() * mz_intensity.size(0)
            train_aa_recall += aa_recall.item() * mz_intensity.size(0)
            train_pep_recall += pep_recall.item() * mz_intensity.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('train_iter_loss', loss.item(), epoch*len(loader)+i)
            writer.add_scalar('iter_aa_precision', aa_precision.item(), epoch*len(loader)+i)
            writer.add_scalar('iter_aa_recall', aa_recall.item(), epoch*len(loader)+i)
            writer.add_scalar('iter_peptide_recall', pep_recall.item(), epoch*len(loader)+i)
            if scheduler is not False:
                scheduler.step()
                lr = scheduler.get_lr()
                writer.add_scalar('lr', lr[0], epoch*len(loader)+i)
            pbar.update(1)
        train_loss /= len(loader.dataset)
        train_aa_precision /= len(loader.dataset)
        train_aa_recall /= len(loader.dataset)
        train_pep_recall /= len(loader.dataset)
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_aa_precision', train_aa_precision, epoch)
        writer.add_scalar('train_aa_recall', train_aa_recall, epoch)
        writer.add_scalar('train_pep_recall', train_pep_recall, epoch)
    return train_loss, scheduler

def valid_inter_epoch(args, epoch, model, loader, writer):
    model.eval()
    tot_loss, tot_aa_precision, tot_aa_recall, tot_pep_recall  = 0, 0, 0, 0

    peptides_collect_list = []

    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for i, inputs in enumerate(loader):
                mz_intensity, precursors, peptide = inputs
                mz_intensity = mz_intensity.to(args.device, dtype=torch.float32)
                precursors = precursors.to(args.device, dtype=torch.float32)

                loss, aa_precision, aa_recall, pep_recall, peptides_collect = model.validation_step([mz_intensity, precursors, peptide])
                print(i, loss.item(), aa_precision.item(), aa_recall.item(), pep_recall.item())

                tot_loss += loss.item() * mz_intensity.size(0)
                tot_aa_precision += aa_precision.item() * mz_intensity.size(0)
                tot_aa_recall += aa_recall.item() * mz_intensity.size(0)
                tot_pep_recall += pep_recall.item() * mz_intensity.size(0)

                peptides_collect_list = peptides_collect_list + peptides_collect

                writer.add_scalar('inter_iter_loss', loss.item(), epoch*len(loader)+i)
                writer.add_scalar('inter_iter_aa_precision', aa_precision.item(), epoch*len(loader)+i)
                writer.add_scalar('inter_iter_aa_recall', aa_recall.item(), epoch*len(loader)+i)
                writer.add_scalar('inter_iter_peptide_recall', pep_recall.item(), epoch*len(loader)+i)
                pbar.update(1)
    tot_loss /= len(loader.dataset)
    tot_aa_precision /= len(loader.dataset)
    tot_aa_recall /= len(loader.dataset)
    tot_pep_recall /= len(loader.dataset)

    writer.add_scalar('inter_valid_loss', tot_loss, epoch)
    writer.add_scalar('inter_aa_precision', tot_aa_precision, epoch)
    writer.add_scalar('inter_aa_recall', tot_aa_recall, epoch)
    writer.add_scalar('inter_pep_recall', tot_pep_recall, epoch)
    model.train()
    return tot_loss, tot_aa_precision, tot_aa_recall, tot_pep_recall

def valid_one_epoch(args, epoch, model, loader, writer, config):
    """Evaluate on the validation set for one epoch."""
    model.eval()
    tot_loss, tot_aa_precision, tot_aa_recall, tot_pep_recall  = 0, 0, 0, 0

    peptides_collect_list = []

    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for i, inputs in enumerate(loader):
                mz_intensity, precursors, peptide = inputs
                mz_intensity = mz_intensity.to(args.device, dtype=torch.float32)
                precursors = precursors.to(args.device, dtype=torch.float32)

                loss, aa_precision, aa_recall, pep_recall, peptides_collect = model.validation_step([mz_intensity, precursors, peptide])

                tot_loss += loss.item() * mz_intensity.size(0)
                tot_aa_precision += aa_precision.item() * mz_intensity.size(0)
                tot_aa_recall += aa_recall.item() * mz_intensity.size(0)
                tot_pep_recall += pep_recall.item() * mz_intensity.size(0)
                
                peptides_collect_list = peptides_collect_list + peptides_collect

                writer.add_scalar('validation_iter_loss', loss.item(), epoch*len(loader)+i)
                writer.add_scalar('validation_iter_aa_precision', aa_precision.item(), epoch*len(loader)+i)
                writer.add_scalar('validation_iter_aa_recall', aa_recall.item(), epoch*len(loader)+i)
                writer.add_scalar('validation_iter_peptide_recall', pep_recall.item(), epoch*len(loader)+i)
                pbar.update(1)
    tot_loss /= len(loader.dataset)
    tot_aa_precision /= len(loader.dataset)
    tot_aa_recall /= len(loader.dataset)
    tot_pep_recall /= len(loader.dataset)
            
    writer.add_scalar('valid_loss', tot_loss, epoch)
    writer.add_scalar('aa_precision', tot_aa_precision, epoch)
    writer.add_scalar('aa_recall', tot_aa_recall, epoch)
    writer.add_scalar('pep_recall', tot_pep_recall, epoch)

    list1 = peptides_collect_list
    names = np.arange(len(list1))
    test = pd.DataFrame(index=names, data=list1)
    csv_name = config['valid_species'] + '_' + config['title'] + '.csv'
    test_save_path = os.path.join('test_results', csv_name)
    test.to_csv(test_save_path)
    return tot_loss, tot_aa_precision, tot_aa_recall, tot_pep_recall
