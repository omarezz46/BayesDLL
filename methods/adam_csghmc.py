import os
import copy
import time
from tqdm import tqdm
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim.lr_scheduler
from methods.cyclical import CyclicalSGMCMC

import calibration


class Runner:

    def __init__(self, net, net0, args, logger):

        '''
        Args:
            net = randomly initialized backbone (on cpu)
            net0 = pretrained backbone or None (on cpu)
        '''
        
        self.args = args
        self.logger = logger
        
        # prepare prior backbone (either 0 or pretrained)
        if args.pretrained is None:  # if no pretrained backbone provided, zero out all params
            self.net0 = copy.deepcopy(net)
            with torch.no_grad():
                for pn, p in self.net0.named_parameters():
                    p.copy_(torch.zeros_like(p))
        else:  # pretrained backbone available
            self.net0 = net0
        self.net0 = self.net0.to(args.device)

        # workhorse network (current Adam-SGHMC sample is maintained in here)
        self.net = net.to(args.device)

        # create Adam Hamiltonian mcmc model (nn.Module with actually no parameters)
        hparams = args.hparams
                # Add temperature parameter
        self.temperature = float(hparams.get('temperature', 1.0))

        self.perform_cold_restarts = str(hparams.get('perform_cold_restarts', False)).lower() == 'true'
        if self.perform_cold_restarts:
            self.logger.info("Performing cold restarts: re-initializing network parameters with fresh random weights at the start of each cycle.")
        else:
            self.logger.info("Cold restarts disabled: keeping network parameters across cycles.")
        
        self.model = Model(
            ND=args.ND, prior_sig=float(hparams['prior_sig']), 
            bias=str(hparams['bias']), 
            momentum_decay=float(hparams['momentum_decay']),
            beta1=float(hparams.get('beta1', 0.9)),
            beta2=float(hparams.get('beta2', 0.999)),
            epsilon=float(hparams.get('epsilon', 1e-8)),
            temperature=self.temperature  # Pass temperature to model
        ).to(args.device)


        # create optimizer (for workhorse network -- to update SGHMC sample)
        # self.optimizer = torch.optim.SGD(
        #     self.net.parameters(), 
        #     lr = args.lr, momentum = args.momentum, weight_decay = 0
        # )
        # Force SGD optimizer momentum to 0, since momentum is handled in Adam-SGHMC
        self.optimizer = torch.optim.SGD(
            [{'params': [p for pn, p in self.net.named_parameters() if self.net.readout_name not in pn], 'lr': args.lr},
             {'params': [p for pn, p in self.net.named_parameters() if self.net.readout_name in pn], 'lr': args.lr_head}],
            momentum = 0, weight_decay = 0
        )

        self.cyclical_scheduler = CyclicalSGMCMC(
            base_lr=args.lr,
            nbr_of_cycles=args.num_cycles if hasattr(args, 'num_cycles') else 10,
            epochs=args.epochs,
            proportion_exploration=args.proportion_exploration if hasattr(args, 'proportion_exploration') else 0.5
        )

        self.criterion = torch.nn.CrossEntropyLoss()

        self.Ninflate = float(hparams['Ninflate'])  # N inflation factor (factor in data augmentation)
        self.nd = float(hparams['nd'])  # noise discount
        self.burnin = int(hparams['burnin'])  # burnin periord (in epochs)
        self.thin = int(hparams['thin'])  # thinning steps (in batch iters)
        self.nst = int(hparams['nst'])  # number of samples at test time

        # Initialize sample collection variables
        self.samples_collected = 0
        self.current_cycle = 0
        self.samples_per_cycle = {}

        self.cycle_theta_mom1 = {}  # Stores mean parameter vector for each cycle
        self.cycle_theta_mom2 = {}   # Stores variance parameter vector for each cycle
        self.cycle_likelihoods = {}  # Stores likelihoods for each sample in each cycle
        self.cycle_states = {}  # Stores the state of the model for each cycle

    def _reinitialize_network_fresh(self):
        """
        Re-initializes the network parameters with fresh random weights.
        Uses standard PyTorch initialization schemes for different layer types.
        """
        def fresh_weight_init(m):
            if isinstance(m, nn.Linear):
                # Xavier/Glorot initialization for linear layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                # Kaiming initialization for conv layers (good for ReLU)
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                # Standard initialization for batch norm
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif hasattr(m, 'reset_parameters'):
                # Fallback to module's own reset_parameters if available
                m.reset_parameters()

        self.net.apply(fresh_weight_init)
        self.logger.info("Network parameters re-initialized with fresh random weights for cold restart.")

    def _reset_optimizer_states(self):
        """
        Reset all Adam-SGHMC optimizer states (momentum, m, v, t).
        """
        if hasattr(self.model, 'momentum_buffer'):
            for name in self.model.momentum_buffer:
                self.model.momentum_buffer[name].zero_()
                if hasattr(self.model, 'm') and name in self.model.m:
                    self.model.m[name].zero_()
                if hasattr(self.model, 'v') and name in self.model.v:
                    self.model.v[name].zero_()
        self.model.t = 0  # Reset Adam time step
        self.logger.info("All optimizer states (momentum, m, v, t) reset for new cycle.")

    def train(self, train_loader, val_loader, test_loader):
        '''
        Train the model using Cyclical SGLD.
        '''
        args = self.args
        logger = self.logger

        logger.info('Start training with Cyclical SGHMC...')

        losses_train = np.zeros(args.epochs)
        errors_train = np.zeros(args.epochs)
        if val_loader is not None:
            losses_val = np.zeros(args.epochs)
            errors_val = np.zeros(args.epochs)
        losses_test = np.zeros(args.epochs)
        errors_test = np.zeros(args.epochs)

        best_loss = np.inf

        tic0 = time.time()
        for ep in range(args.epochs):
            # Reset collection states for each epoch
            self.cyclical_scheduler.current_epoch = ep
            
            tic = time.time()
            losses_train[ep], errors_train[ep], cycle_updated = self.train_one_epoch(train_loader)
            toc = time.time()
            
            prn_str = f'[Epoch {ep}/{args.epochs}] Training summary: '
            prn_str += f'loss = {losses_train[ep]:.4f}, prediction error = {errors_train[ep]:.4f} '
            prn_str += f'(time: {toc-tic:.4f} seconds)'
            logger.info(prn_str)
            
            # Test evaluation at the end of each cycle or if a cycle was completed during the epoch
            if cycle_updated : #or (ep % args.test_eval_freq == 0 and ep > 0)
                # test on validation set (if available)
                if val_loader is not None:
                    tic = time.time()
                    losses_val[ep], errors_val[ep], targets_val, logits_val, logits_all_val = \
                        self.evaluate(val_loader)
                    #evaluate point estimation on validation set
                    point_loss, point_error = self.evaluate_simple(val_loader)
                    #evaluate cycle theta_mom1 on validation set
                    logger.info(f'Point estimation on validation set: loss = {point_loss:.4f}, error = {point_error:.4f}')
                    toc = time.time()
                    prn_str = f'(Epoch {ep}) Validation summary: '
                    prn_str += f'loss = {losses_val[ep]:.4f}, prediction error = {errors_val[ep]:.4f} '
                    prn_str += f'(time: {toc-tic:.4f} seconds)'
                    logger.info(prn_str)

                # test on test set
                tic = time.time()
                losses_test[ep], errors_test[ep], targets_test, logits_test, logits_all_test = \
                    self.evaluate(test_loader)
                toc = time.time()
                prn_str = f'(Epoch {ep}) Test summary: '
                prn_str += f'loss = {losses_test[ep]:.4f}, prediction error = {errors_test[ep]:.4f} '
                prn_str += f'(time: {toc-tic:.4f} seconds)'
                logger.info(prn_str)

                loss_now = losses_val[ep] if val_loader is not None else losses_test[ep]
                if loss_now < best_loss:
                    best_loss = loss_now
                    logger.info(f'Best evaluation loss so far! @epoch {ep}: loss = {loss_now}')
                    
                    # save logits and labels
                    if val_loader is not None:
                        fname = self.save_logits(
                            targets_val, logits_val, logits_all_val, suffix='val'
                        )  # save prediction logits on validation set
                        logger.info(f'Logits on val set saved at {fname}')
                        
                    fname = self.save_logits(
                        targets_test, logits_test, logits_all_test, suffix='test'
                    )  # save prediction logits on test set
                    logger.info(f'Logits on test set saved at {fname}')

                    # perform error calibration (ECE, MCE, reliability plot, etc.)
                    ece_no_ts, mce_no_ts, nll_no_ts = calibration.analyze(
                        targets_test, logits_test, num_bins = args.ece_num_bins, 
                        plot_save_path = os.path.join(args.log_dir, 'reliability_T1.png'), 
                        temperature = 1
                    )  # calibration with default temperature (T=1)
                    logger.info(
                        f'[Calibration - Default T=1] ECE = {ece_no_ts:.4f}, MCE = {mce_no_ts:.4f}, NLL = {nll_no_ts:.4f}'
                    )
                    
                    if val_loader is not None:  # perform temperature scaling
                        Topt, success = calibration.find_optimal_temperature(
                            targets_val, logits_val, 
                            plot_save_path = os.path.join(args.log_dir, 'temp_scale_optim_curve.png'), 
                        )  # find optimal temperature on val set
                        if success:
                            ece_ts, mce_ts, nll_ts = calibration.analyze(
                                targets_test, logits_test, num_bins = args.ece_num_bins, 
                                plot_save_path = os.path.join(args.log_dir, 'reliability_Topt.png'), 
                                temperature = Topt
                            )  # calibration with optimal temperature
                            logger.info(
                                f'[Calibration - Temp-scaled Topt={Topt[0]:.4f}] ECE = {ece_ts:.4f}, MCE = {mce_ts:.4f}, NLL = {nll_ts:.4f}'
                            )
                        else:
                            logger.info('!! Temperature scaling optimization failed !!')

        toc0 = time.time()
        logger.info(f'Training done! Total time = {toc0-tic0:.4f} (average per epoch = {(toc0-tic0)/args.epochs:.4f}) seconds')
        logger.info(f'Total samples collected: {self.samples_collected} across {self.current_cycle} cycles')
        
        return {
            'losses_train': losses_train,
            'errors_train': errors_train,
            'losses_val': losses_val if val_loader is not None else None,
            'errors_val': errors_val if val_loader is not None else None,
            'losses_test': losses_test,
            'errors_test': errors_test,
            'samples_per_cycle': self.samples_per_cycle
        }


    def train_one_epoch(self, train_loader):
        '''
        Run SGLD steps for one epoch using cyclical learning rates.
        
        Returns:
            loss = averaged NLL loss
            err = averaged classification error
            cycle_updated = whether a new cycle was completed
        '''
        args = self.args
        logger = self.logger

        self.net.train()
        
        loss, error, nb_samples = 0, 0, 0
        cycle_updated = False
        batches_per_epoch = len(train_loader)
        
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch_idx, (x, y) in enumerate(tepoch):
                current_lr = self.cyclical_scheduler.calculate_lr(
                    epoch=self.cyclical_scheduler.current_epoch,
                    batch=batch_idx,
                    batches_per_epoch=batches_per_epoch
                )
        
                should_sample = self.cyclical_scheduler.should_sample(
                    epoch=self.cyclical_scheduler.current_epoch,
                    batch=batch_idx,
                    batches_per_epoch=batches_per_epoch
                ) and batch_idx % self.thin == 0
                
                last_in_cycle = self.cyclical_scheduler.last_in_cycle(
                    epoch=self.cyclical_scheduler.current_epoch,
                    batch=batch_idx,
                    batches_per_epoch=batches_per_epoch
                )
                
                # Update optimizer learning rates for both body and head
                for i, param_group in enumerate(self.optimizer.param_groups):
                    # The second parameter group (index 1) is the head, based on initialization order
                    if i == 1:  # Head parameters group
                        param_group['lr'] = current_lr * (args.lr_head / args.lr)
                    else:  # Body parameters group
                        param_group['lr'] = current_lr
                            
                # Process the batch
                x, y = x.to(args.device), y.to(args.device)
                
                # Evaluate SGLD updates for the batch
                loss_, out = self.model(
                    x, y, self.net, self.net0, self.criterion, 
                    [pg['lr'] for pg in self.optimizer.param_groups],
                    self.Ninflate, self.nd
                )
                if hasattr(args, 'clip_grad') and args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), args.clip_grad)

                self.optimizer.step()  # update self.net (ie, theta)
                
                # Prediction on training
                pred = out.data.max(dim=1)[1]
                err = pred.ne(y.data).sum()
                
                loss += loss_ * len(y)
                error += err.item()
                nb_samples += len(y)
                

                if should_sample :
                    # Get current cycle number
                    cycle_number = self.cyclical_scheduler.get_cycle_number(
                        epoch=self.cyclical_scheduler.current_epoch,
                        batch=batch_idx,
                        batches_per_epoch=batches_per_epoch
                    )
                    # Log when we're in sampling phase
                    if batch_idx % 50 == 0:  # Only log occasionally to avoid spam
                        logger.info(f'Sampling phase: collecting posterior sample at lr={current_lr:.6f}')
                    
                    # Update running moments for this cycle
                    with torch.no_grad():
                        theta_vec = nn.utils.parameters_to_vector(self.net.parameters())
                        
                        # Initialize or update cycle-specific moments
                        if cycle_number not in self.cycle_theta_mom1:
                            self.cycle_theta_mom1[cycle_number] = theta_vec.clone()
                            self.cycle_theta_mom2[cycle_number] = theta_vec**2
                        else:
                            cycle_count = self.samples_per_cycle.get(cycle_number, 0) + 1
                            self.cycle_theta_mom1[cycle_number] = (theta_vec + 
                                (cycle_count-1) * self.cycle_theta_mom1[cycle_number]) / cycle_count
                            self.cycle_theta_mom2[cycle_number] = (theta_vec**2 + 
                                    (cycle_count-1) * self.cycle_theta_mom2[cycle_number]) / cycle_count
                    
                    self.samples_collected += 1
                    self.samples_per_cycle[cycle_number] = self.samples_per_cycle.get(cycle_number, 0) + 1

                else:
                    # Log when we're in exploration phase
                    if batch_idx % 50 == 0 and not should_sample:  # Only log occasionally to avoid spam
                        logger.info(f'Exploration phase: lr={current_lr:.6f}')
                    
                    tepoch.set_postfix(loss=loss/nb_samples, error=error/nb_samples, lr=current_lr)
                
                                    # Handle end of cycle
                if last_in_cycle:
                    # Get current cycle number
                    self.logger.info(f'Resetting momentum states for new cycle {cycle_number}')
                    if hasattr(self.model, 'momentum_buffer'):
                        for name in self.model.momentum_buffer:
                            self.model.momentum_buffer[name].zero_()
                            self.model.m[name].zero_()
                            self.model.v[name].zero_()
                    self.model.t = 0  # Reset Adam time step
                
                    cycle_number = self.cyclical_scheduler.get_cycle_number(
                        epoch=self.cyclical_scheduler.current_epoch,
                        batch=batch_idx,
                        batches_per_epoch=batches_per_epoch
                    )
                    
                    self.cycle_states[cycle_number] = copy.deepcopy(self.net.state_dict())

                    if cycle_number > self.current_cycle:
                        cycle_updated = True
                        self.current_cycle = cycle_number
                        logger.info(f'Completed cycle {cycle_number}')
                        
                        # Calculate full batch likelihood for this cycle's model
                        likelihood = np.array(self.full_batch_likelihoods(train_loader))
                        
                        # Store the likelihood for this cycle
                        self.cycle_likelihoods[cycle_number] = likelihood
                        logger.info(f'Cycle {cycle_number} full batch likelihood: {likelihood.mean():.6e}')

                        # Save parameter vector for this cycle
                        with torch.no_grad():
                            self.save_ckpt(epoch=self.cyclical_scheduler.current_epoch)
                        
                        # Prepare for next cycle - always reset optimizer states
                        self._reset_optimizer_states()
                    
                        # Optionally perform cold restart (fresh random weights)
                        if self.perform_cold_restarts and cycle_number >= 1:  # Don't restart after cycle 0
                            logger.info(f'Performing COLD RESTART: Fresh random weight initialization for cycle {cycle_number + 1}')
                            self._reinitialize_network_fresh()
                            # Optimizer states are already reset above
                        else:
                            logger.info(f'Standard cycle transition: keeping weights, optimizer states reset for cycle {cycle_number + 1}')
        
        return loss/nb_samples, error/nb_samples, cycle_updated


    def evaluate(self, test_loader):
        '''
        Prediction using the Gaussian Mixture Model from all cycles.
        Each cycle contributes a Gaussian component with weight based on the likelihoods.
        
        Returns:
            loss = averaged test CE loss
            err = averaged test error
            targets = all groundtruth labels
            logits = all prediction logits (after weighted averaging)
            logits_all = all prediction logits (before averaging across components)
        '''
        args = self.args
        
        # Calculate GMM weights based on cycle likelihoods
        gmm_weights = self.calculate_gmm_weights()
        
        # Log the weights for insight
        self.logger.info(f"GMM component weights: {gmm_weights}")
        
        # Create networks for each cycle's statistics
        cycle_networks = {}
        for cycle in self.cycle_theta_mom1.keys():
            # Create a copy of the network for this cycle
            net_c = copy.deepcopy(self.net)
            cycle_networks[cycle] = net_c
            
            # Load the mean parameters for this cycle
            with torch.no_grad():
                nn.utils.vector_to_parameters(self.cycle_theta_mom1[cycle], net_c.parameters())
        
        # Evaluate on test data
        loss, error, nb_samples = 0, 0, 0
        targets, logits, logits_all = [], [], []
        
        with tqdm(test_loader, unit="batch") as tepoch:
            for x, y in tepoch:
                x, y = x.to(args.device), y.to(args.device)
                
                batch_logits_all = []
                batch_logits = None
                
                # For each cycle in the mixture
                for cycle, net_c in cycle_networks.items():
                    net_c.eval()
                    weight = gmm_weights.get(cycle, 0.0)
                    

                    if weight < 1e-10:
                        continue
                    
                    component_logits = []
                    
                    if self.nst == 0:
                        # Just use the mean parameters
                        with torch.no_grad():
                            out = net_c(x)
                        component_logits.append(out)
                    else:
                        param_vars = copy.deepcopy(net_c)
                        param_means = copy.deepcopy(net_c)
                        with torch.no_grad():
                            if cycle in self.cycle_theta_mom2:
                                ratio = self.samples_per_cycle.get(cycle, 0) / (self.samples_per_cycle.get(cycle, 0) - 1)
                                if self.samples_per_cycle.get(cycle, 0) > 1:
                                    cycle_variance = ratio * (self.cycle_theta_mom2[cycle] - self.cycle_theta_mom1[cycle]**2)
                                else:
                                    cycle_variance = self.cycle_theta_mom2[cycle] - self.cycle_theta_mom1[cycle]**2
                                cycle_variance.clamp_(min=1e-12)
                                nn.utils.vector_to_parameters(cycle_variance, param_vars.parameters())
                                nn.utils.vector_to_parameters(self.cycle_theta_mom1[cycle], param_means.parameters())
                        for _ in range(self.nst):
                            with torch.no_grad():
                                net_sample = copy.deepcopy(net_c)
                                
                                for p, p_mean, p_var in zip(net_sample.parameters(), param_means.parameters(), param_vars.parameters()):
                                    eps = torch.randn_like(p)
                                    p.copy_(p_mean + p_var.sqrt() * eps)  # Now shapes match
                                    # p.copy_(p_mean)
                                    
                                out = net_sample(x)
                            component_logits.append(out)
                    
                    # Stack outputs from this cycle
                    component_logits = torch.stack(component_logits, dim=2)
                    
                    # Average the predictions for this component
                    if self.nst == 0:
                        component_out = component_logits.squeeze(2)
                    else:
                        component_out = F.log_softmax(component_logits, dim=1).logsumexp(-1) - np.log(self.nst)
                    
                    # Store for later mixture
                    batch_logits_all.append(component_logits)
                    
                    # Add this component's output to the mixture (weighted)
                    if batch_logits is None:
                        batch_logits = weight * component_out
                    else:
                        batch_logits += weight * component_out
                
                # Stack all components for later analysis
                batch_logits_all = torch.stack(batch_logits_all, dim=3) if batch_logits_all else torch.zeros((x.size(0), args.num_classes, 1, 1), device=args.device)
                
                # Calculate loss and error
                loss_batch = self.criterion(batch_logits, y)
                pred = batch_logits.data.max(dim=1)[1]
                err = pred.ne(y.data).sum()
                
                # Collect statistics
                targets.append(y.cpu().detach().numpy())
                logits.append(batch_logits.cpu().detach().numpy())
                logits_all.append(batch_logits_all.cpu().detach().numpy())
                loss += loss_batch.item() * len(y)
                error += err.item()
                nb_samples += len(y)
                
                tepoch.set_postfix(loss=loss/nb_samples, error=error/nb_samples)
        
        # Concatenate results across batches
        targets = np.concatenate(targets, axis=0)
        logits = np.concatenate(logits, axis=0)
        logits_all = np.concatenate(logits_all, axis=0)
        
        return loss/nb_samples, error/nb_samples, targets, logits, logits_all

    def evaluate_simple(self, test_loader):
        """
        Simple evaluation using current network state (before burn-in period).
        No Bayesian averaging, just deterministic forward pass.
        """
        args = self.args
        
        self.net.eval()
        
        loss, error, nb_samples = 0, 0, 0
        with torch.no_grad():
            with tqdm(test_loader, unit="batch", desc="Val") as tepoch:
                for x, y in tepoch:
                    x, y = x.to(args.device), y.to(args.device)
                    
                    # Simple forward pass with current network
                    logits_ = self.net(x)
                    loss_ = self.criterion(logits_, y)
                    
                    # Predictions
                    pred = logits_.data.max(dim=1)[1]
                    err = pred.ne(y.data).sum()
                    
                    loss += loss_.item() * len(y)
                    error += err.item()
                    nb_samples += len(y)
                    
                    tepoch.set_postfix(loss=loss/nb_samples, error=error/nb_samples)
        
        self.net.train()  # Switch back to training mode
        
        return loss/nb_samples, error/nb_samples

    def save_logits(self, targets, logits, logits_all, suffix=None):

        suffix = '' if suffix is None else f'_{suffix}'
        fname = os.path.join(self.args.log_dir, f'logits{suffix}.pkl')
        
        with open(fname, 'wb') as ff:
            pickle.dump(
                {'targets': targets, 'logits': logits, 'logits_all': logits_all}, 
                ff, protocol=pickle.HIGHEST_PROTOCOL
            )

        return fname


    def save_ckpt(self, epoch):
        fname = os.path.join(self.args.log_dir, f"{self.current_cycle}_ckpt.pt")

        torch.save(
            {   'last_theta': nn.utils.parameters_to_vector(self.net.parameters()),
                'cycle_theta_mom1': self.cycle_theta_mom1,
                'cycle_theta_mom2': self.cycle_theta_mom2,
                'cycle_likelihoods': self.cycle_likelihoods,
                'cycle_states': self.cycle_states,
                # 'prior_sig': self.model.prior_sig, 
                # 'optimizer': self.optimizer.state_dict(),
                'epoch': epoch,
                'current_cycle': self.current_cycle,
                # 'samples_collected': self.samples_collected,
                'samples_per_cycle': self.samples_per_cycle
            },
            fname
        )

        return fname


    def load_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.args.device, weights_only=False)
        
        # Load GMM components
        self.cycle_theta_mom1 = ckpt.get('cycle_theta_mom1', {})
        self.cycle_theta_mom2 = ckpt.get('cycle_theta_mom2', {})
        self.cycle_likelihoods = ckpt.get('cycle_likelihoods', {})
        
        # self.model.prior_sig = ckpt['prior_sig']
        # self.optimizer.load_state_dict(ckpt['optimizer'])
        self.current_cycle = ckpt.get('current_cycle', 0)
        # self.samples_collected = ckpt.get('samples_collected', 0)
        self.samples_per_cycle = ckpt.get('samples_per_cycle', {})

        return ckpt['epoch']
    
    def full_batch_likelihoods(self, train_loader):
        """
        Calculate full-batch log-likelihood for self.nst samples based on the current model.
        Returns a list of likelihood values (not negative log-likelihood).
        """
        self.logger.info(f"Calculating full-batch likelihood for current cycle using {self.nst} samples...")
        likelihoods = []
        
        # Get current parameters as mean
        with torch.no_grad():
            param_mean = self.cycle_theta_mom1[self.current_cycle] 
            # Calculate parameter variance based on current cycle's samples
            cycle_variance = None
            if self.current_cycle in self.cycle_theta_mom2 and self.current_cycle in self.cycle_theta_mom1:
                n_samples = self.samples_per_cycle.get(self.current_cycle, 0)
                if n_samples > 1:
                    ratio = n_samples / (n_samples - 1)
                    cycle_variance = ratio * (self.cycle_theta_mom2[self.current_cycle] - 
                                            self.cycle_theta_mom1[self.current_cycle]**2)
                    cycle_variance.clamp_(min=1e-12)  # Prevent negative variance
        
        # If self.nst is 0, just use the mean parameters
        samples_to_evaluate = max(1, self.nst)
        for sample_idx in range(samples_to_evaluate):
            # Create a copy of the network for this sample
            net_sample = copy.deepcopy(self.net)
            param_means = copy.deepcopy(self.net)
            param_vars = copy.deepcopy(self.net)
            nn.utils.vector_to_parameters(param_mean, param_means.parameters())
            nn.utils.vector_to_parameters(cycle_variance, param_vars.parameters())

            # If using multiple samples, add Gaussian noise to parameters
            if self.nst > 0 and cycle_variance is not None:
                with torch.no_grad():
                    for p, p_mean, p_var in zip(net_sample.parameters(), param_means.parameters(), param_vars.parameters()):
                        eps = torch.randn_like(p)
                        p.copy_(p_mean + p_var.sqrt() * eps)
                        # p.copy_(p_mean)
            
            net_sample.eval()  # Set model to evaluation mode
            
            with torch.no_grad():  # Disable gradient calculation
                loss, error, nb_samples = 0, 0, 0
                with tqdm(train_loader, unit="batch") as tepoch:
                    for x, y in tepoch:

                        x, y = x.to(self.args.device), y.to(self.args.device)

                        logits_all_ = []
                        logits_ = net_sample(x)

                        loss_ = self.criterion(logits_, y)

                        pred = logits_.data.max(dim=1)[1]
                        err = pred.ne(y.data).sum()

                        loss += loss_.item() * len(y)
                        error += err.item()
                        nb_samples += len(y)
            
            avg_loss = loss / nb_samples
            likelihood = np.exp(-avg_loss)  # Convert loss to likelihood
            likelihoods.append(likelihood)
            
            self.logger.info(f"Sample {sample_idx+1} - Full batch average loss: {avg_loss:.6f}, likelihood: {likelihood:.6e}")
        
        return likelihoods

    
    def calculate_gmm_weights(self):
        """
        Calculate weights for each cycle in the Gaussian Mixture Model.
        Uses the formula: w_c = [ (1/K_c) * sum(1/p(D|θ_j^c)) ]^-1
        """
        weights = {}
        
        # Return early if no cycles collected
        if not self.cycle_likelihoods:
            return {0: 1.0}
        
        # Calculate weights for each cycle
        for cycle, likelihoods in self.cycle_likelihoods.items():
            # if not likelihoods:
            #     weights[cycle] = 0.0
            #     continue
                
            # Compute inverse average inverse likelihood
            avg_inv_likelihood = np.mean([1.0/likelihood for likelihood in likelihoods])
            weights[cycle] = 1.0 / avg_inv_likelihood
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {c: w/total_weight for c, w in weights.items()}
        else:
            # If all weights are zero, use uniform weights
            weights = {c: 1.0/len(weights) for c in weights}
        
        return weights

    
class Model(nn.Module):

    '''
    Adam-SGHMC sampler model.

    Actually no parameters involved.
    '''

    def __init__(self, ND, prior_sig=1.0, bias='informative', momentum_decay=0.05, beta1=0.9, beta2=0.999, epsilon=1e-8, temperature=1.0):

        '''
        Args:
            ND = training data size
            prior_sig = prior Gaussian sigma
            bias = how to treat bias parameters:
                "informative": -- the same treatment as weights
                "uninformative": uninformative bias prior
            momentum_decay = momentum decay parameter (alpha in SGHMC papers)
            beta1 = Adam beta1 parameter (exponential decay rate for first moment)
            beta2 = Adam beta2 parameter (exponential decay rate for second moment)
            epsilon = small constant for numerical stability
        '''

        super().__init__()

        self.ND = ND
        self.prior_sig = prior_sig
        self.bias = bias
        self.momentum_decay = momentum_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.temperature = temperature  # Temperature parameter for scaling
        self.t = 0  # Initialize time step counter

    def forward(self, x, y, net, net0, criterion, lrs, Ninflate=1.0, nd=1.0):
        '''
        Evaluate minibatch Adam-SGHMC updates for a given batch.
        Args:
            x, y = batch input, output
            net = workhorse network (its parameters will be filled in)
            net0 = prior mean parameters
            criterion = loss function
            lrs = learning rates in list (adjusted to "eta" in SGHMC)
            Ninflate = inflate N by this order of magnitude
            nd = noise discount factor
        Returns:
            loss = NLL loss on the batch
            out = class prediction on the batch
        Effects:
            net has .grad fields filled with Adam-SGHMC updates
        '''
        
        bias = self.bias
        N = self.ND * Ninflate  # inflated training data size (accounting for data augmentation, etc.)
        if len(lrs) == 1:
            lr_body, lr_head = lrs[0], lrs[0]
        else:
            lr_body, lr_head = lrs[0], lrs[1]
        
        # Initialize Adam moment estimates and momentum if not already done
        if not hasattr(self, 'momentum_buffer'):
            self.momentum_buffer = {}
            self.m = {}  # First moment estimate
            self.v = {}  # Second moment estimate
            for name, param in net.named_parameters():
                self.momentum_buffer[name] = torch.zeros_like(param)
                self.m[name] = torch.zeros_like(param)
                self.v[name] = torch.zeros_like(param)
        
        # Increment time step
        self.t += 1
        
        # Forward pass with theta
        out = net(x)

        # Evaluate NLL loss
        loss = criterion(out, y)

        # Gradient d{loss_nll}/d{theta}
        net.zero_grad()
        loss.backward()
        
        # Compute and set Adam-SGHMC updates
        with torch.no_grad():
            for (pname, p), p0 in zip(net.named_parameters(), net0.parameters()):
                if p.grad is not None:
                    if net.readout_name not in pname:
                        lr = lr_body
                    else:
                        lr = lr_head
                    
                    # Get momentum and moment estimates for this parameter
                    v_momentum = self.momentum_buffer[pname]
                    m = self.m[pname]
                    v = self.v[pname]
                    
                    # Compute gradient term including prior
                    if 'bias' in pname and bias == 'uninformative':
                        grad_U = p.grad / self.temperature # Only data likelihood gradient
                    else:
                        grad_U = ( p.grad / self.temperature ) + (p - p0) / (self.prior_sig**2) / N  # Prior + likelihood gradient
                    
                    # Update biased first moment estimate (Adam)
                    m = self.beta1 * m + (1 - self.beta1) * grad_U
                    
                    # Update biased second moment estimate (Adam)
                    v = self.beta2 * v + (1 - self.beta2) * (grad_U * grad_U)
                    
                    # Compute bias-corrected first and second moment estimates
                    m_hat = m / (1 - self.beta1 ** self.t)
                    v_hat = v / (1 - self.beta2 ** self.t)
                    
                    # Compute preconditioned gradient
                    precond_grad = m_hat / (torch.sqrt(v_hat) + self.epsilon)
                    
                    # Noise term for SGHMC with preconditioning
                    precond_term = 1.0 / (torch.sqrt(v_hat) + self.epsilon)
                    noise_scale = nd * torch.sqrt(2 * self.momentum_decay * precond_term / N )
                    noise = noise_scale * torch.randn_like(p)
                    
                    # Update momentum (v) using Adam-SGHMC update rule
                    v_momentum = v_momentum * (1 - self.momentum_decay) + lr * precond_grad + noise
                    
                    # Store updated momentum and moment estimates
                    self.momentum_buffer[pname] = v_momentum
                    self.m[pname] = m
                    self.v[pname] = v
                    
                    # Update gradient using momentum
                    # p.grad = p.grad + v_momentum.clone()
                    p.grad = v_momentum.clone()
        
        return loss.item(), out.detach()
