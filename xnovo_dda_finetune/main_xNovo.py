"""Entry point for π-xNovo-DDA fine-tuning.

This script is intentionally kept as a *single-command runner* so that users can
fine-tune with:

    python main_xNovo.py --config configs/xnovo_dda_finetune.yaml

The underlying training code lives in :mod:`xnovo_dda`.

Notes
-----
* The original internal code supported distributed training via
  ``torch.distributed.launch``. That launcher is deprecated; for multi-GPU runs
  prefer ``torchrun`` (PyTorch >= 1.9).
* This public version focuses on readability and reproducibility.
"""

from __future__ import annotations

import argparse
import os
import random
import time

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import yaml

# TensorBoard logging: prefer PyTorch built-in SummaryWriter, fall back to
# tensorboardX if needed.
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    from tensorboardX import SummaryWriter

from xnovo_dda.dataset import build_loader
from xnovo_dda.model import CosineWarmupScheduler, build_model
from xnovo_dda.utils import train_one_epoch, valid_one_epoch

# distribution command
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=11111 main_xNovo.py

parser = argparse.ArgumentParser('Pytorch Training')
# Distributed training parameters
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int, 
                    help='node rank for distributed training')  
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')
parser.add_argument('--multiprocessing', default=False, type=bool, 
                    help='whether multiprocessing')  
parser.add_argument('--n_gpus', default=1, type=int, 
                    help='whether multiprocessing')  
parser.add_argument(
    '--config',
    default='configs/xnovo_dda_finetune.yaml',
    type=str,
    help='Path to the YAML config file.',
)
parser.add_argument('--species', default='P0031_TOF4_DDA_20240614_Z439_2_150ng_80min_RD8_1_7655', type=str, 
                    help='step learning rate for schedular')       
parser.add_argument('--device', default='cuda:0', type=str, 
                    help='step learning rate for schedular')                                    
parser.add_argument('--train_mask_ratio', default=0.0, type=float)
parser.add_argument('--dropout', default=0.0, type=float)
parser.add_argument('--max_epoch', default=10, type=int)
parser.add_argument('--pretrain_model', default='False', type=str, help='False/MassIVE_KB/TimsTOF_DDA')
args = parser.parse_args()
os.makedirs('config_file',exist_ok=True)
os.makedirs('checkpoints',exist_ok=True)
os.makedirs('test_results',exist_ok=True)
def main(args):
    print(args)
    # Read parameters from the config file.
    with open(args.config) as f:
        config = yaml.safe_load(f)
    # Override a few config fields from CLI for quick experiments.
    # (YAML remains the single source of truth for most settings.)
#     if args.pretrain_model == 'False':
#         pass
#     elif args.pretrain_model == 'TimsTOF_DDA':
#         config['modelname'] = 'weights/TimsTOF_DDA_pretrained.pth'
#     elif args.pretrain_model == 'MassIVE_KB':
#         config['modelname'] = 'weights/MassIVE_KB_trained_xNovo_DDA.pth'
    if args.pretrain_model == 'False':
        pass
    elif args.pretrain_model == 'TimsTOF_DDA':
        config['modelname'] = '/code/xNovo_DIA_finetune/TimsTOF_DDA_pretrained.pth'
    elif args.pretrain_model == 'MassIVE_KB':
        config['modelname'] = '/code/xNovo_DDA_github/MassIVE_KB_trained_xNovo_DDA.pth'
        
    config['max_epochs'] = args.max_epoch
    config['title'] = config['title'] + f'_{args.max_epoch}epoch_{args.dropout}'
    config['dropout'] = args.dropout
    config['train_mask_ratio'] = args.train_mask_ratio
    if config["random_seed"] is not None:
        random.seed(config["random_seed"])
        torch.manual_seed(config["random_seed"])
        torch.cuda.manual_seed_all(config["random_seed"])
        cudnn.deterministic = True
    print(config)
    # Load datasets.
    # NOTE: the original internal code used a misspelled helper name
    # ("bulid_loader"). In this public repo we expose the correct
    # ``build_loader`` API.
    train_loader, val_loader = build_loader(args, config)
    print(len(train_loader.dataset), len(val_loader.dataset))
    # model, loss, optimizer
    model = build_model(args, config)
    # log
    title = config['title']
    writer = SummaryWriter(f'log/{title}', purge_step=0)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    if config['scheduler'] is not False:
        if config['old_schedular']:
            config['max_iters'] = len(train_loader) * config['max_epochs'] 
        else:
            config['warmup_iters'] = len(train_loader) * config['max_epochs'] * 0.5
            config['max_iters'] = len(train_loader) * config['max_epochs'] * 1.1
        scheduler = CosineWarmupScheduler(optimizer, warmup=config['warmup_iters'], max_iters=config['max_iters'], 
                                        start_epoch=config['start_epoch'], old=config['old_schedular'],
                                        loader_len=len(train_loader), max_epochs=config['max_epochs'])
    else:
        scheduler = False
    torch.backends.cudnn.benchmark = True

    with open(f'config_file/{title}_{time.time()}.yaml', 'w') as file:
        file.write(yaml.safe_dump(config))
    # Loop
    best_loss = 5
    for epoch in range(config['start_epoch'], config['max_epochs']):
        print(f'Current epoch is: {epoch}')
        train_loss, scheduler = train_one_epoch(args, epoch, model, train_loader, 
                                    optimizer, scheduler, writer)
        # if epoch % 15 == 14 or epoch == (config['max_epochs'] - 1):        
        val_loss, aa_precision, aa_recall, pep_recall = \
            valid_one_epoch(args, epoch, model, val_loader, writer, config)
        print(val_loss, aa_precision, aa_recall, pep_recall)
        # save model
        if train_loss <= best_loss:
            best_loss = train_loss
            obj = {
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'best_rtre': best_loss,
            'optimizer' : optimizer.state_dict()
            }
            os.makedirs(config['cpt_path'], exist_ok=True)
            cpt_path = os.path.join(config['cpt_path'], f'best_{title}.pth')
            torch.save(obj, cpt_path)

if __name__ == '__main__':
    # initial 
    # dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(args.local_rank)
    main(args)        
        
