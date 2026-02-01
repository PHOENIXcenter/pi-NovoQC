"""Training utilities for π-xNovo-DIA.

Includes:
- training loops
- evaluation helpers
- logging utilities

Research code: designed for reproducibility rather than minimal dependencies.
"""

import torch
import torch.distributed as dist

import os
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pformat
from model import normalize_intensities
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

def metric_average(val,):
    tensor = torch.tensor(val).cuda().type(torch.cuda.FloatTensor)
    tensor_list = [torch.zeros(1).cuda() for _ in range(dist.get_world_size())]
    
    dist.all_gather(tensor_list, tensor)
    avg_tensor = 0
    for i in range(len(tensor_list)):
        avg_tensor += tensor_list[i].item()
    
    return avg_tensor / len(tensor_list)

def train_one_epoch(args, epoch, model, loader, optimizer, scheduler, writer):
    model.train()
    loc = 'cuda:{}'.format(args.local_rank)
    total_psms = 0
    train_loss, train_aa_precision, train_aa_recall, train_pep_recall = 0, 0, 0, 0
    train_loss_m, train_loss_c, train_loss_pep = 0, 0, 0
    with tqdm(total=len(loader)) as pbar:
        for i, inputs in enumerate(loader):
            windows, precs, peptide, rt, mz_intensity, mz_labels = inputs
            windows = windows.cuda().type(torch.cuda.FloatTensor)
            mz_intensity = mz_intensity.cuda().type(torch.cuda.FloatTensor)
            mz_labels = mz_labels.cuda().type(torch.cuda.FloatTensor)                
            precs = precs.cuda().type(torch.cuda.FloatTensor)
            (loss_frag, loss_c, loss_m, loss_pep), aa_precision, aa_recall, pep_recall, sub_peptides = model.module.training_step(mz_intensity, precs, windows, peptide, mz_labels)
            total_psms += len(sub_peptides)
            total_loss = loss_pep + loss_m + loss_c + loss_frag

            print(loc, i, loss_pep.item(), loss_m.item(), loss_c.item(), loss_frag.item(), total_psms)
            print(loc, i, aa_precision.item(), aa_recall.item(), pep_recall.item())
            train_loss += total_loss.item() * len(sub_peptides)
            train_loss_m += loss_m.item() * len(sub_peptides)
            train_loss_c += loss_c.item() * len(sub_peptides)
            train_loss_pep += loss_pep.item() * len(sub_peptides)
            train_aa_precision += aa_precision.item() * len(sub_peptides)
            train_aa_recall += aa_recall.item() * len(sub_peptides)
            train_pep_recall += pep_recall.item() * len(sub_peptides)
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()

            writer.add_scalar('train_iter_loss', total_loss.item(), epoch*len(loader)+i)
            writer.add_scalar('train_iter_prec_m', loss_m.item(), epoch*len(loader)+i)
            writer.add_scalar('train_iter_prec_c', loss_c.item(), epoch*len(loader)+i)
            writer.add_scalar('train_iter_pep', loss_pep.item(), epoch*len(loader)+i)

            writer.add_scalar('iter_aa_precision', aa_precision.item(), epoch*len(loader)+i)
            writer.add_scalar('iter_aa_recall', aa_recall.item(), epoch*len(loader)+i)
            writer.add_scalar('iter_peptide_recall', pep_recall.item(), epoch*len(loader)+i)

            if scheduler is not False:
                scheduler.step()
                lr = scheduler.get_lr()
                writer.add_scalar('lr', lr, epoch*len(loader)+i)
            pbar.update(1)
        train_loss = metric_average(train_loss / total_psms)
        train_loss_m = metric_average(train_loss_m / total_psms)
        train_loss_c = metric_average(train_loss_c / total_psms)
        train_loss_pep = metric_average(train_loss_pep / total_psms)
        train_aa_precision = metric_average(train_aa_precision / total_psms)
        train_aa_recall = metric_average(train_aa_recall / total_psms)
        train_pep_recall = metric_average(train_pep_recall / total_psms)
        print(loc, train_loss, train_loss_pep, train_loss_m, train_loss_c, train_pep_recall, train_aa_precision, train_aa_recall)
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_prec_m', train_loss_m, epoch)
        writer.add_scalar('train_prec_c', train_loss_c, epoch)
        writer.add_scalar('train_pep', train_loss_pep, epoch)
        writer.add_scalar('train_aa_precision', train_aa_precision, epoch)
        writer.add_scalar('train_aa_recall', train_aa_recall, epoch)
        writer.add_scalar('train_pep_recall', train_pep_recall, epoch)
    return train_loss, scheduler

def valid_one_epoch(args, epoch, model, loader, writer, config):
    model.eval()
    loc = 'cuda:{}'.format(args.local_rank)
    total_psms = 0
    valid_loss, tot_aa_precision, tot_aa_recall, tot_pep_recall  = 0, 0, 0, 0
    valid_loss_m, valid_loss_c, valid_loss_pep = 0, 0, 0
    peptides_collect_list = []

    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for i, inputs in enumerate(loader):
                windows, precs, peptide, rt, mz_intensity, mz_labels = inputs
                windows = windows.cuda().type(torch.cuda.FloatTensor)
                mz_intensity = mz_intensity.cuda().type(torch.cuda.FloatTensor)
                mz_labels = mz_labels.cuda().type(torch.cuda.FloatTensor)
                precs = precs.cuda().type(torch.cuda.FloatTensor)
                (loss_frag, loss_c, loss_m, loss_pep), aa_precision, aa_recall, pep_recall, peptides_collect, sub_peptides = model.module.validation_step(mz_intensity, precs, windows, peptide, mz_labels)
                total_psms += len(sub_peptides)
                total_loss = loss_m + loss_c + loss_pep + loss_frag
                print(loc, i, loss_pep.item(), loss_m.item(), loss_c.item(), loss_frag.item(), total_psms)
                print(loc, i, aa_precision.item(), aa_recall.item(), pep_recall.item())
                
                valid_loss += total_loss.item() * len(sub_peptides)
                valid_loss_m += loss_m.item() * len(sub_peptides)
                valid_loss_c += loss_c.item() * len(sub_peptides)
                valid_loss_pep += loss_pep.item() * len(sub_peptides)

                tot_aa_precision += aa_precision.item() * len(sub_peptides)
                tot_aa_recall += aa_recall.item() * len(sub_peptides)
                tot_pep_recall += pep_recall.item() * len(sub_peptides)

                peptides_collect_list = peptides_collect_list + peptides_collect

                writer.add_scalar('validation_iter_loss', total_loss.item(), epoch*len(loader)+i)
                writer.add_scalar('validation_iter_prec_m', loss_m.item(), epoch*len(loader)+i)
                writer.add_scalar('validation_iter_prec_c', loss_c.item(), epoch*len(loader)+i)
                writer.add_scalar('validation_iter_pep', loss_pep.item(), epoch*len(loader)+i)

                writer.add_scalar('validation_iter_aa_precision', aa_precision.item(), epoch*len(loader)+i)
                writer.add_scalar('validation_iter_aa_recall', aa_recall.item(), epoch*len(loader)+i)
                writer.add_scalar('validation_iter_peptide_recall', pep_recall.item(), epoch*len(loader)+i)
                pbar.update(1)
    valid_loss = metric_average(valid_loss / total_psms)
    valid_loss_m = metric_average(valid_loss_m / total_psms)
    valid_loss_c = metric_average(valid_loss_c / total_psms)
    valid_loss_pep = metric_average(valid_loss_pep / total_psms)
    tot_aa_precision = metric_average(tot_aa_precision / total_psms)
    tot_aa_recall = metric_average(tot_aa_recall / total_psms)
    tot_pep_recall = metric_average(tot_pep_recall / total_psms)
            
    writer.add_scalar('valid_loss', valid_loss, epoch)
    writer.add_scalar('valid_prec_m', valid_loss_m, epoch)
    writer.add_scalar('valid_prec_c', valid_loss_c, epoch)
    writer.add_scalar('valid_pep', valid_loss_pep, epoch)
    writer.add_scalar('valid_aa_precision', tot_aa_precision, epoch)
    writer.add_scalar('valid_aa_recall', tot_aa_recall, epoch)
    writer.add_scalar('valid_pep_recall', tot_pep_recall, epoch)

    list1 = peptides_collect_list
    names = np.arange(len(list1))
    test = pd.DataFrame(index=names, data=list1)
    csv_name = config['title'] + f'_{args.local_rank}.csv'
    test_save_path = os.path.join('test_results', csv_name)
    test.to_csv(test_save_path)
    return valid_loss, tot_aa_precision, tot_aa_recall, tot_pep_recall
