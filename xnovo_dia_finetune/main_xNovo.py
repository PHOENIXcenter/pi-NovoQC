"""Main entry point for π-xNovo-DIA fine-tuning.

This script reads a YAML config (see `config_xNovo.yaml`) and fine-tunes the model on
spectrum–peptide pairs prepared from DIA data.

Typical usage:
    python main_xNovo.py --config config_xNovo.yaml

For multi-GPU runs, prefer `torchrun`:
    torchrun --nproc_per_node 2 main_xNovo.py --multiprocessing True --n_gpus 2 --config config_xNovo.yaml

Outputs (checkpoints and logs) are written to:
    - checkpoints/
    - test_results/
    - config_file/

Note: This is research code released for reproducibility.
"""

import torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
import os 
import yaml
import time
import random
import argparse
from tensorboardX import SummaryWriter

from dataset import bulid_loader
from model import build_model, CosineWarmupScheduler
from utils import train_one_epoch, valid_one_epoch

# distributed command
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=11111 --use_env main_xNovo.py --multiprocessing=True --n_gpus=2 --n_peaks=600 --sub_peaks=150 --n_layers=9 --sample_num=4 --bs=32 --max_epochs=6 --lr=5e-4 --prec_length=5
parser = argparse.ArgumentParser('')
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
parser.add_argument('--config', default='configs/xnovo_dia_finetune.yaml', type=str, 
                    help='path of config')  
parser.add_argument('--device', default='cuda:0', type=str, 
                    help='step learning rate for schedular')                                    
parser.add_argument('--n_peaks', default=600, type=int)
parser.add_argument('--sub_peaks', default=150, type=int)
parser.add_argument('--n_layers', default=9, type=int)
parser.add_argument('--dropout', default=0.0, type=float) 
parser.add_argument('--sample_num', default=4, type=int) 
parser.add_argument('--prec_length', default=5, type=int) 
parser.add_argument('--bs', default=32, type=int) 
parser.add_argument('--n_workers', default=4, type=int)
parser.add_argument('--lr', default=5e-4, type=float)
parser.add_argument('--max_epochs', default=10, type=int)
parser.add_argument('--train_signal', default=False, type=bool)
parser.add_argument('--train_mask_ratio', default=0.0, type=float)
parser.add_argument('--add', default='astraldia', type=str) 
parser.add_argument('--valid_species', default='Hh_24h_500_04', type=str)
parser.add_argument('--pretrain_model', default='MassIVE_KB', type=str, help='/MassIVE_KB/TimsTOF_DDA')

args = parser.parse_args()
args.local_rank = int(os.getenv('LOCAL_RANK', 0))

os.makedirs('config_file',exist_ok=True)
os.makedirs('checkpoints',exist_ok=True)
os.makedirs('test_results',exist_ok=True)
def main(args):
    print(args)
    """
    ================================================================================
    """
    # Read parameters from the config file.
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config['n_peaks'] = args.n_peaks
    config['sub_peaks'] = args.sub_peaks
    config['n_layers'] = args.n_layers
    config['dropout'] = args.dropout
    config['sample_num'] = args.sample_num
    config['prec_length'] = args.prec_length
    config['train_batch_size'] = args.bs
    config['val_batch_size'] = args.bs
    config['n_workers'] = args.n_workers
    config['lr'] = args.lr
    config['max_epochs'] = args.max_epochs
    config['train_signal'] = args.train_signal
    config['train_mask_ratio'] = args.train_mask_ratio
    config['valid_species'] = args.valid_species
    config['title'] = f'2enc_peaks{args.n_peaks}_{args.sub_peaks}_layers{args.n_layers}_drop{args.dropout}_sample{args.sample_num}_bs{args.bs}{args.lr}_epochs{args.max_epochs}_mask{args.train_signal}{args.train_mask_ratio}_{args.add}_len{args.prec_length}'
    print(config['title'])
    if config["random_seed"] is not None:
        random.seed(config["random_seed"])
        torch.manual_seed(config["random_seed"])
        cudnn.deterministic = True
    # load dataset
    train_sampler, train_loader, val_loader = bulid_loader(args, config)
    print(len(train_loader), len(train_loader.dataset))
    print(len(val_loader), len(val_loader.dataset))
    # model, loss, optimizer
    model = build_model(args, config)
    loc = 'cuda:{}'.format(args.local_rank)
    
    # log
    title = config['title']
    if config['modelname'] is not False:
        loc = 'cuda:{}'.format(args.local_rank)
        config['start_epoch'] = torch.load(config['modelname'], map_location=loc)['epoch']

        writer = SummaryWriter(f'log/{title}', purge_step=config['start_epoch'] * len(train_loader))
    else:
        writer = SummaryWriter(f'log/{title}', purge_step=0)
   
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    if config['modelname'] is not False:
        optimizer.load_state_dict(torch.load(config['modelname'], map_location=loc)['optimizer'])
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

    with open(f'config_file/{config["valid_species"]}_{title}_{time.time()}.yaml', 'w') as file:
        file.write(yaml.safe_dump(config))
    # Loop
    best_loss = 5
    for epoch in range(config['start_epoch'], config['max_epochs']):
        print(epoch)
        train_sampler.set_epoch(epoch)
        train_loss, scheduler = train_one_epoch(args, epoch, model, train_loader, 
                                    optimizer, scheduler, writer)
        val_loss, aa_precision, aa_recall, pep_recall = \
            valid_one_epoch(args, epoch, model, val_loader, writer, config)
        print(val_loss, aa_precision, aa_recall, pep_recall)
        # save model
        if args.local_rank == 0:
            if train_loss <= best_loss:
                best_loss = train_loss
                obj = {
                'epoch': epoch+1,
                'state_dict': model.module.state_dict(),
                'best_rtre': best_loss,
                'optimizer' : optimizer.state_dict()
                }
                os.makedirs(config['cpt_path'], exist_ok=True)
                cpt_path = os.path.join(config['cpt_path'], f'best_{title}.pth')
                torch.save(obj, cpt_path)

if __name__ == '__main__':
    # initial 
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    main(args) 
