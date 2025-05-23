
import os
import sys
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make sure BayesDLL modules are in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import networks
import datasets

# Import various method runners
from methods.csgld import Runner as CSGLDRunner
from methods.sgld import Runner as SGLDRunner
from methods.csghmc import Runner as CSGHMCRunner
from methods.sghmc import Runner as SGHMCRunner
from methods.vi import Runner as VIRunner
from methods.mc_dropout import Runner as MCDropoutRunner
from methods.la import Runner as LARunner
from methods.adam_sghmc import Runner as AdamSGHMCRunner

try:
    import wandb
except ImportError:
    wandb = None

def parse_args():
    parser = argparse.ArgumentParser(description='Pre-training ResNet101 using Bayesian methods')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='imagenet', help='Dataset for pre-training (imagenet, cifar100)')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of classes (1000 for ImageNet)')
    parser.add_argument('--val_heldout', type=float, default=0.1, help='Validation set holdout proportion')

    # Model and Method arguments
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['resnet101'], help='Backbone network')
    parser.add_argument('--method', type=str, default='csgld', 
                      choices=['csgld', 'sgld', 'csghmc', 'sghmc', 'vi', 'mc_dropout', 'la', 'adam_sghmc'],
                      help='Bayesian inference method')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--lr_head', type=float, default=None, help='Learning rate for classifier head')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    
    # Cyclical methods (cSGLD, cSGHMC) specific hyperparameters
    parser.add_argument('--num_cycles', type=int, default=2, help='Number of sampling cycles for cyclical methods')
    parser.add_argument('--proportion_exploration', type=float, default=0.5, help='Proportion of exploration phase in each cycle')
    parser.add_argument('--full_sample', type=bool, default=False, help='full sample in the exploration phase')
    
    # Method-specific hyperparameters (combined as comma-separated key=value pairs)
    parser.add_argument('--hparams', type=str, default=None, help='Comma-separated hyperparameters for the chosen method')
    
    # Default hyperparameters for each method
    parser.add_argument('--default_hparams', action='store_true', help='Use default hyperparameters for the chosen method')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save results')
    parser.add_argument('--log_dir', type=str, default=None, help='Directory to save logs (defaults to output_dir)')
    parser.add_argument('--ckpt_freq', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--test_eval_freq', type=int, default=5, help='Evaluate on test set every N epochs')
    parser.add_argument('--ece_num_bins', type=int, default=15, help='Number of bins for error calibration')
    
    # Device and parallelization
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='BayesDLL_Pretrain_ResNet101', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (username or team)')
    
    args = parser.parse_args()

    args.use_cuda = torch.cuda.is_available()
    
    # Set default output directory based on method
    if args.output_dir is None:
        args.output_dir = f'./output/pretrain_resnet101_{args.method}'
    
    # Set log_dir to output_dir if not specified
    if args.log_dir is None:
        args.log_dir = args.output_dir
        
    # Set default learning rate for classifier head if not specified
    if args.lr_head is None:
        args.lr_head = args.lr
    
    # Set default hyperparameters based on method if requested
    if args.default_hparams or args.hparams is None:
        args.hparams = get_default_hparams(args.method)
    
    # Parse hparams string into dictionary
    hparams = args.hparams
    hparams = hparams.replace('"', '')
    hpstr = hparams.replace(',', '_')
    opts = hparams.split(',')
    hparams_dict = {}
    for opt in opts:
        if '=' in opt:
            key, val = opt.split('=')
            hparams_dict[key] = val  # string valued
    args.hparams = hparams_dict
    
    return args

def get_default_hparams(method):
    """Return default hyperparameters for each method"""
    hparams = {
        'csgld': 'prior_sig=1.0,Ninflate=1.0,nd=0.01,burnin=50,thin=10,bias=informative,nst=5,temp=1.0',
        'sgld': 'prior_sig=1.0,Ninflate=1e3,nd=1.0,burnin=50,thin=10,bias=informative,nst=5,temp=1.0',
        'csghmc': 'prior_sig=1.0,Ninflate=1.0,nd=0.01,burnin=50,momentum_decay=0.18,thin=10,bias=informative,nst=5,temp=1.0',
        'sghmc': 'prior_sig=1.0,Ninflate=1e3,nd=1.0,burnin=5,momentum_decay=0.18,thin=1,bias=informative,nst=5,temp=1.0',
        'vi': 'prior_sig=1.0,kld=1.0,bias=gaussian,nst=5,temp=1.0',
        'mc_dropout': 'prior_sig=1.0,p_drop=0.1,kld=1e-3,bias=gaussian,nst=5,temp=1.0',
        'la': 'prior_sig=1.0,bias=informative,nst=5,temp=1.0',
        'adam_sghmc': 'prior_sig=1.0,Ninflate=1e3,nd=1.0,momentum_decay=0.05,burnin=50,thin=10,bias=informative,nst=5,temp=1.0,beta1=0.9,beta2=0.999'
    }
    return hparams.get(method, '')

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.log_dir, 'pretrain_log.txt')
    logging.basicConfig(
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        format='[%(asctime)s,%(msecs)03d %(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )
    logger = logging.getLogger()
    
    # Log the command line arguments
    cmd = " ".join(sys.argv)
    logger.info(f"Command :: {cmd}\n")
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    
    # Initialize WandB if requested
    if args.use_wandb and wandb is not None:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"pretrain_resnet101_{args.method}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        logger.info(f"WandB run: {wandb.run.name} (ID: {wandb.run.id})")
    else:
        args.use_wandb = False
    
    # Check for dataset availability first
    logger.info(f"Checking dataset: {args.dataset}")
    
    # Prepare data loaders
    logger.info(f"Preparing dataset: {args.dataset}")
    if args.dataset == 'imagenet':
        train_loader, val_loader, test_loader, args.ND = datasets.prepare(args, data_root=args.data_path)
    elif args.dataset == 'cifar100':
        train_loader, val_loader, test_loader, args.ND = datasets.prepare(args, data_root=args.data_path)
        args.num_classes = 100  # Override num_classes for CIFAR-100
    elif args.dataset == 'cifar10':
        train_loader, val_loader, test_loader, args.ND = datasets.prepare(args, data_root=args.data_path)
        args.num_classes = 10
    else:
        logger.error(f"Unknown dataset: {args.dataset}")
        sys.exit(1)
    
    # Create ResNet101 model
    logger.info("Creating ResNet101 model...")
    args.pretrained = None  # Ensure we're not loading pretrained weights for pre-training
    net = networks.create_backbone(args)
    logger.info(f"Total parameters: {net.get_nb_parameters() / 1000000.0:.2f}M")
    net = net.to(args.device)
    
    # Initialize zero-prior model for methods that require it
    logger.info("Initializing zero prior...")
    net0 = networks.create_backbone(args)
    with torch.no_grad():
        for pn, p in net0.named_parameters():
            p.copy_(torch.zeros_like(p))
    net0 = net0.to(args.device)
    
    # Initialize the appropriate runner based on the selected method
    logger.info(f"Initializing {args.method.upper()} runner...")
    runner = get_runner(args.method, net, net0, args, logger)
    
    # Train the model
    logger.info(f"Starting pre-training with {args.method.upper()}...")
    results = runner.train(train_loader, val_loader, test_loader)
    
    # Log final results
    logger.info("Pre-training completed.")
    if results and args.use_wandb:
        wandb.log({
            'final_train_loss': float(results['losses_train'][-1]),
            'final_train_error': float(results['errors_train'][-1]),
            'final_test_loss': float(results['losses_test'][-1]),
            'final_test_error': float(results['errors_test'][-1]),
            'best_test_loss': float(min(results['losses_test'])),
            'best_test_error': float(min(results['errors_test'])),
        })
    
    if args.use_wandb:
        wandb.finish()

def get_runner(method, net, net0, args, logger):
    """
    Return the appropriate runner based on the method
    """
    runners = {
        'csgld': CSGLDRunner,
        'sgld': SGLDRunner,
        'csghmc': CSGHMCRunner,
        'sghmc': SGHMCRunner,
        'vi': VIRunner,
        'mc_dropout': MCDropoutRunner,
        'la': LARunner,
        'adam_sghmc': AdamSGHMCRunner
    }
    
    if method in runners:
        return runners[method](net=net, net0=net0, args=args, logger=logger)
    else:
        raise ValueError(f"Unknown method: {method}")

if __name__ == '__main__':
    main()


