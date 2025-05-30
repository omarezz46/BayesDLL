import os, sys
import logging
import time
from datetime import datetime
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import wandb  # Add wandb import

import networks
import datasets
import utils

parser = argparse.ArgumentParser()

# method and hparams
parser.add_argument('--method', type=str, default='mc_dropout', help='approximate posterior inference method')
parser.add_argument('--hparams', type=str, default='', help='all hparams specific to the method (comma-separated, =-assigned forms)')

# finetuning of pretrained model or training from the scratch (None)
parser.add_argument('--pretrained', type=str, default=None, help='path or url to the pretrained model')

# dataset and backbone network
parser.add_argument('--dataset', type=str, default='mnist', help='dataset name')
parser.add_argument('--backbone', type=str, default='mlp', help='backbone name')
parser.add_argument('--val_heldout', type=float, default=0.1, help='validation set heldout proportion')

# error calibration
parser.add_argument('--ece_num_bins', type=int, default=15, help='number of bins for error calibration')

# cyclical specific params
parser.add_argument('--num_cycles', type=int, default=1, help='number of cycles')
parser.add_argument('--proportion_exploration', type=float, default=0.5, help='proportion of exploration phase in each cycle')
parser.add_argument('--full_sample', type=bool, default=False, help='full sample in the exploration phase')

# other optim hparams
parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lr_head', type=float, default=None, help='learning rate for head')
parser.add_argument('--momentum', type=float, default=0.5, help='momentum')

parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--log_dir', type=str, default='results', help='root folder for saving logs')
parser.add_argument('--test_eval_freq', type=int, default=1, help='do test evaluation (every this epochs)')

# wandb arguments
parser.add_argument('--use_wandb', action='store_true', help='use wandb for experiment tracking')
parser.add_argument('--wandb_project', type=str, default='bayesdll', help='wandb project name')
parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity/username')
parser.add_argument('--wandb_tags', type=str, default='', help='comma-separated tags for wandb')
parser.add_argument('--wandb_name', type=str, default=None, help='run name (defaults to timestamp-based)')

args = parser.parse_args()


if torch.cuda.is_available():
    args.device = torch.device('cuda')
elif torch.backends.mps.is_available():
    args.device = torch.device('mps')
else:
    args.device = torch.device('cpu')

# args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
args.use_cuda = torch.cuda.is_available()

# random seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.mps.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
np.random.seed(args.seed)

# parse hparams
hparams = args.hparams
hparams = hparams.replace('"', '')
hpstr = hparams.replace(',', '_')
opts = hparams.split(',')
hparams = {}
for opt in opts:
    if '=' in opt:
        key, val = opt.split('=')
        hparams[key] = val  # note: string valued
args.hparams = hparams

if args.lr_head is None:
    args.lr_head = args.lr

# set directory for saving results
pretr = 1 if args.pretrained is not None else 0
main_dir = f'{args.dataset}_val_heldout{args.val_heldout}/'
main_dir += f'{args.backbone}/{args.method}_{hpstr}_pretr{pretr}/' 
main_dir += f'ep{args.epochs}_bs{args.batch_size}_lr{args.lr}_lrh{args.lr_head}_mo{args.momentum}/'
main_dir += f'seed{args.seed}_' + datetime.now().strftime('%Y_%m%d_%H%M%S')
args.log_dir = os.path.join(args.log_dir, main_dir)
utils.mkdir(args.log_dir)

# Initialize wandb if requested
if args.use_wandb:
    # Set up wandb configuration
    wandb_config = {
        'method': args.method,
        'dataset': args.dataset,
        'backbone': args.backbone,
        'val_heldout': args.val_heldout,
        'pretrained': args.pretrained,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'lr_head': args.lr_head,
        'momentum': args.momentum,
        'epochs': args.epochs,
        'seed': args.seed,
        'num_cycles': args.num_cycles if hasattr(args, 'num_cycles') else 1,
        'proportion_exploration': args.proportion_exploration if hasattr(args, 'proportion_exploration') else 0.5,
        'device': str(args.device)
    }
    
    # Add method-specific hyperparameters
    for key, val in hparams.items():
        wandb_config[f'hparam_{key}'] = val
    
    # Parse tags
    tags = [tag.strip() for tag in args.wandb_tags.split(',') if tag.strip()]
    
    # Generate run name if not provided
    run_name = args.wandb_name if args.wandb_name else main_dir.replace('/', '_')
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=wandb_config,
        name=run_name,
        tags=tags,
        dir=args.log_dir
    )
    
    # Save command line for reproducibility
    wandb.config.update({'command': " ".join(sys.argv)})
    
    # Make wandb accessible to runners
    args.wandb = wandb
else:
    args.wandb = None

# create logger
logging.basicConfig(
    handlers=[
        logging.FileHandler(os.path.join(args.log_dir, 'logs.txt')), 
        logging.StreamHandler()
    ], 
    format='[%(asctime)s,%(msecs)03d %(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger()
cmd = " ".join(sys.argv)
logger.info(f"Command :: {cmd}\n")
if args.use_wandb:
    logger.info(f"WandB run: {wandb.run.name} (ID: {wandb.run.id})")

# prepare data
logger.info('Preparing data...')
train_loader, val_loader, test_loader, args.ND = datasets.prepare(args)  # ND = train set size

# create backbone (skeleton)
logger.info('Creating an underlying backbone network (skeleton)...')
net = networks.create_backbone(args)
logger.info('Total params in the backbone: %.2fM' % (net.get_nb_parameters() / 1000000.0))
logger.info('Backbone modules:\n%s' % (net.get_module_names()))

# log model architecture to wandb
if args.use_wandb:
    wandb.config.update({'total_params_M': net.get_nb_parameters() / 1000000.0})
    wandb.config.update({'model_architecture': net.get_module_names()})

# load pretrained backbone (with zero'ed final prediction module)
if args.pretrained is not None:
    logger.info('Load pretrained backbone network...')
    net0 = networks.load_pretrained_backbone(args)  # feat-ext params = pretrained, head = zero
    net = networks.load_pretrained_backbone(args, zero_head=False)  # feat-ext params = pretrained, head = random
else:
    logger.info('No pretrained backbone network provided.')
    net0 = None

try:
    if args.method == 'vanilla':
        from methods.vanilla import Runner
        runner = Runner(net, net0, args, logger)
        results = runner.train(train_loader, val_loader, test_loader)

    elif args.method == 'vi':
        from methods.vi import Runner
        runner = Runner(net, net0, args, logger)
        results = runner.train(train_loader, val_loader, test_loader)

    elif args.method == 'mc_dropout':
        from methods.mc_dropout import Runner
        runner = Runner(net, net0, args, logger)
        results = runner.train(train_loader, val_loader, test_loader)

    elif args.method == 'sgld':
        from methods.sgld import Runner
        runner = Runner(net, net0, args, logger)
        results = runner.train(train_loader, val_loader, test_loader)

    elif args.method == 'la':
        from methods.la import Runner
        runner = Runner(net, net0, args, logger)
        results = runner.train(train_loader, val_loader, test_loader)

    elif args.method == 'csgld':
        from methods.csgld import Runner
        runner = Runner(net, net0, args, logger)
        results = runner.train(train_loader, val_loader, test_loader)

    elif args.method == 'csghmc':
        from methods.csghmc import Runner
        runner = Runner(net, net0, args, logger)
        results = runner.train(train_loader, val_loader, test_loader)
    
    elif args.method == 'sghmc':
        from methods.sghmc import Runner
        runner = Runner(net, net0, args, logger)
        results = runner.train(train_loader, val_loader, test_loader)
    
    elif args.method == 'adam_sghmc':
        from methods.adam_sghmc import Runner
        runner = Runner(net, net0, args, logger)
        results = runner.train(train_loader, val_loader, test_loader)

    elif args.method == 'adam_csghmc':
        from methods.adam_csghmc import Runner
        runner = Runner(net, net0, args, logger)
        results = runner.train(train_loader, val_loader, test_loader)
    else:
        raise NotImplementedError

    # Log final results to wandb
    if args.use_wandb and results:
        # Log summary metrics
        summary = {
            'final_train_loss': float(results['losses_train'][-1]),
            'final_train_error': float(results['errors_train'][-1]),
            'final_test_loss': float(results['losses_test'][-1]),
            'final_test_error': float(results['errors_test'][-1]),
            'best_test_loss': float(min(results['losses_test'])),
            'best_test_error': float(min(results['errors_test']))
        }
        
        if results['losses_val'] is not None:
            summary.update({
                'final_val_loss': float(results['losses_val'][-1]),
                'final_val_error': float(results['errors_val'][-1]),
                'best_val_loss': float(min(results['losses_val'])),
                'best_val_error': float(min(results['errors_val']))
            })
            
        wandb.summary.update(summary)
        
        # # Log final artifacts
        # model_artifact = wandb.Artifact(f"{wandb.run.name}-model", type="model")
        # model_artifact.add_dir(args.log_dir)
        # wandb.log_artifact(model_artifact)

except Exception as e:
    logger.exception(f"Training failed with error: {e}")
    if args.use_wandb:
        wandb.finish(exit_code=1)
    raise

finally:
    # Properly finish wandb run
    if args.use_wandb:
        wandb.finish()