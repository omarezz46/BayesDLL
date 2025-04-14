import torch
import torch.nn as nn
import numpy as np
import math
import os
import logging
import time
from tqdm.notebook import tqdm
import torch.nn.functional as F
from bayesdll import calibration

class CyclicalSGMCMC:
    """Abstract wrapper to add cyclical learning rate scheduling to any SG MCMC method."""
    
    def __init__(self, 
                 base_lr,  # α0, initial stepsize 
                 nbr_of_cycles,  # M, number of cycles
                 epochs,  # Used to calculate K (total iterations)
                 proportion_exploration=0.5):  # β, proportion of exploration stage
        
        self.base_lr = base_lr
        self.number_of_cycles = nbr_of_cycles
        self.epochs = epochs
        self.proportion_exploration = proportion_exploration
        self.current_epoch = 0  # Track current epoch
        self.sample_at_bottom = True  # Sample in exploitation phase

    
    def calculate_lr(self, epoch, batch, batches_per_epoch):
        """Calculate the learning rate based on current position in cycle."""

        K = self.epochs * batches_per_epoch
        cycle_length = K // self.number_of_cycles
        k = epoch * batches_per_epoch + batch + 1
        
        # normalized cycle position
        cycle_pos = ((k - 1) % cycle_length) / cycle_length
        
        return self.base_lr * (1 + np.cos(cycle_pos * np.pi)) / 2

    
    def should_sample(self, epoch, batch, batches_per_epoch):
        """Determine if a sample should be collected at this point."""
        if not self.sample_at_bottom:
            return True
            
        K = self.epochs * batches_per_epoch
        cycle_length = K / self.number_of_cycles
        k = epoch * batches_per_epoch + batch + 1
        
        cycle_pos = ((k - 1) % cycle_length) / cycle_length
        
        return cycle_pos >= self.proportion_exploration
    
    def last_in_cycle(self, epoch, batch, batches_per_epoch):
        """Check if this is the last batch in the current cycle."""
        K = self.epochs * batches_per_epoch
        cycle_length = K / self.number_of_cycles
        k = epoch * batches_per_epoch + batch + 1
        
        return (k % cycle_length) == 0
        
    def get_cycle_number(self, epoch, batch, batches_per_epoch):
        """Calculate the current cycle number."""
        K = self.epochs * batches_per_epoch
        cycle_length = K / self.number_of_cycles
        k = epoch * batches_per_epoch + batch + 1
        return int((k - 1) // cycle_length) + 1