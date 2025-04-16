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

        # workhorse network (current SGLD sample is maintained in here)
        self.net = net.to(args.device)

        hparams = args.hparams
        self.model = Model(
            ND=args.ND, prior_sig=float(hparams['prior_sig']), bias=str(hparams['bias'])
        ).to(args.device)

        self.optimizer = torch.optim.SGD(
            [{'params': [p for pn, p in self.net.named_parameters() if self.net.readout_name not in pn], 'lr': args.lr},
             {'params': [p for pn, p in self.net.named_parameters() if self.net.readout_name in pn], 'lr': args.lr_head}],
            momentum = args.momentum, weight_decay = 0
        )

        # Initialize cyclical scheduler
        self.cyclical_scheduler = CyclicalSGMCMC(
            base_lr=args.lr,
            nbr_of_cycles=args.num_cycles if hasattr(args, 'num_cycles') else 10,
            epochs=args.epochs,
            proportion_exploration=args.proportion_exploration if hasattr(args, 'proportion_exploration') else 0.5
        )

        self.criterion = torch.nn.CrossEntropyLoss()

        self.Ninflate = float(hparams['Ninflate'])  # N inflation factor (factor in data augmentation)
        self.nd = float(hparams['nd'])  # noise discount
        self.nst = int(hparams['nst'])  # number of samples at test time
        
        # Initialize sample collection variables
        self.samples_collected = 0
        self.current_cycle = 0
        self.samples_per_cycle = {}

        self.cycle_theta_means = {}  # Stores mean parameter vector for each cycle
        self.cycle_theta_vars = {}   # Stores variance parameter vector for each cycle
        self.cycle_likelihoods = {}  # Stores likelihoods for each sample in each cycle

    def train(self, train_loader, val_loader, test_loader):
        '''
        Train the model using Cyclical SGLD.
        '''
        args = self.args
        logger = self.logger

        logger.info('Start training with Cyclical SGLD...')

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
                    
                    # save checkpoint
                    fname = self.save_ckpt(ep)  # save checkpoint
                    logger.info(f'Checkpoint saved at {fname}')

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
        in_sampling_phase = False
        just_entered_sampling = False
        cycle_updated = False
        batches_per_epoch = len(train_loader)
        
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch_idx, (x, y) in enumerate(tepoch):
                
                # Calculate learning rate for this batch based on cycle position
                current_lr = self.cyclical_scheduler.calculate_lr(
                    epoch=self.cyclical_scheduler.current_epoch,
                    batch=batch_idx,
                    batches_per_epoch=batches_per_epoch
                )
                
                # Check if we should collect samples based on cycle position
                should_sample = self.cyclical_scheduler.should_sample(
                    epoch=self.cyclical_scheduler.current_epoch,
                    batch=batch_idx,
                    batches_per_epoch=batches_per_epoch
                )
                
                # Check if this is the last batch in the current cycle
                last_in_cycle = self.cyclical_scheduler.last_in_cycle(
                    epoch=self.cyclical_scheduler.current_epoch,
                    batch=batch_idx,
                    batches_per_epoch=batches_per_epoch
                )
                
                # Detect transition into sampling phase
                if should_sample and not in_sampling_phase:
                    just_entered_sampling = True
                    in_sampling_phase = True
                    logger.info('Entering sampling phase - initializing posterior sample averages')
                    # Initialize posterior sample aggregation
                    with torch.no_grad():
                        theta_vec = nn.utils.parameters_to_vector(self.net.parameters())
                        self.post_theta_mom1 = theta_vec * 1.0  # 1st moment
                        if self.nst > 0:  # need to maintain sample variances as well
                            self.post_theta_mom2 = theta_vec**2  # 2nd moment
                        self.post_theta_cnt = 1
                elif not should_sample:
                    in_sampling_phase = False
                
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
                
                self.optimizer.step()  # update self.net (ie, theta)
                
                # Prediction on training
                pred = out.data.max(dim=1)[1]
                err = pred.ne(y.data).sum()
                
                loss += loss_ * len(y)
                error += err.item()
                nb_samples += len(y)
                

                if should_sample and not just_entered_sampling:
                    # Get current cycle number
                    cycle_number = self.cyclical_scheduler.get_cycle_number(
                        epoch=self.cyclical_scheduler.current_epoch,
                        batch=batch_idx,
                        batches_per_epoch=batches_per_epoch
                    )
                    
                    # Determine if we should collect this sample (based on max samples per cycle)
                    collect_this_sample = True

                    if collect_this_sample:
                        if batch_idx % 50 == 0:  # Log occasionally to reduce spam
                            logger.info(f'Sampling phase: collecting posterior sample at lr={current_lr:.6f}')
                        
                        # Compute likelihood of the entire dataset (using the batch as a proxy)
                        # This is proportional to exp(-loss * dataset_size/batch_size)
                        likelihood = np.exp(-loss_ * self.args.ND / len(y))
                        
                        # Store likelihood for this sample
                        if cycle_number not in self.cycle_likelihoods:
                            self.cycle_likelihoods[cycle_number] = []
                        self.cycle_likelihoods[cycle_number].append(likelihood)
                        
                        # Update running moments for this cycle
                        with torch.no_grad():
                            theta_vec = nn.utils.parameters_to_vector(self.net.parameters())
                            
                            # Initialize or update cycle-specific moments
                            if cycle_number not in self.cycle_theta_means:
                                self.cycle_theta_means[cycle_number] = theta_vec.clone()
                                if self.nst > 0:
                                    self.cycle_theta_vars[cycle_number] = theta_vec**2
                                cycle_count = 1
                            else:
                                cycle_count = self.samples_per_cycle.get(cycle_number, 0) + 1
                                self.cycle_theta_means[cycle_number] = (theta_vec + 
                                    (cycle_count-1) * self.cycle_theta_means[cycle_number]) / cycle_count
                                if self.nst > 0:
                                    self.cycle_theta_vars[cycle_number] = (theta_vec**2 + 
                                        (cycle_count-1) * self.cycle_theta_vars[cycle_number]) / cycle_count
                            
                            # Also update the global moments for backward compatibility
                            self.post_theta_mom1 = (theta_vec + self.post_theta_cnt*self.post_theta_mom1) / (self.post_theta_cnt+1)
                            if self.nst > 0:  # need to maintain sample variances as well
                                self.post_theta_mom2 = (theta_vec**2 + self.post_theta_cnt*self.post_theta_mom2) / (self.post_theta_cnt+1)
                        
                        # Update counters
                        self.post_theta_cnt += 1
                        self.samples_collected += 1
                        self.samples_per_cycle[cycle_number] = self.samples_per_cycle.get(cycle_number, 0) + 1

                        # Handle end of cycle
                        if last_in_cycle:
                            # Get current cycle number
                            cycle_number = self.cyclical_scheduler.get_cycle_number(
                                epoch=self.cyclical_scheduler.current_epoch,
                                batch=batch_idx,
                                batches_per_epoch=batches_per_epoch
                            )
                            
                            if cycle_number > self.current_cycle:
                                cycle_updated = True
                                self.current_cycle = cycle_number
                                logger.info(f'Completed cycle {cycle_number}')
                                
                                # Save parameter vector for this cycle
                                with torch.no_grad():
                                    self.save_ckpt(epoch=self.cyclical_scheduler.current_epoch)  
                else:
                        # Reset flag after first batch in sampling phase
                    just_entered_sampling = False
                        
                        # Log when we're in exploration phase
                    if batch_idx % 50 == 0 and not should_sample:  # Only log occasionally to avoid spam
                        logger.info(f'Exploration phase: lr={current_lr:.6f}')
                    
                    tepoch.set_postfix(loss=loss/nb_samples, error=error/nb_samples, lr=current_lr)
                

        
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
        for cycle in self.cycle_theta_means.keys():
            # Create a copy of the network for this cycle
            net_c = copy.deepcopy(self.net)
            cycle_networks[cycle] = net_c
            
            # Load the mean parameters for this cycle
            with torch.no_grad():
                nn.utils.vector_to_parameters(self.cycle_theta_means[cycle], net_c.parameters())
        
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
                    
                    # For each cycle, evaluate either with:
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
                            if cycle in self.cycle_theta_vars:
                                nn.utils.vector_to_parameters(self.cycle_theta_vars[cycle], param_vars.parameters())
                                nn.utils.vector_to_parameters(self.cycle_theta_means[cycle], param_means.parameters())
                        for _ in range(self.nst):
                            with torch.no_grad():
                                net_sample = copy.deepcopy(net_c)
                                
                                for p, p_mean, p_var in zip(net_sample.parameters(), param_means.parameters(), param_vars.parameters()):
                                    eps = torch.randn_like(p)
                                    p.copy_(p_mean + p_var.sqrt() * eps)  # Now shapes match
                                    
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
        fname = os.path.join(self.args.log_dir, f"ckpt.pt")

        torch.save(
            {
                'post_theta_mom1': self.post_theta_mom1 if hasattr(self, 'post_theta_mom1') else None,  
                'post_theta_mom2': self.post_theta_mom2 if hasattr(self, 'post_theta_mom2') and self.nst > 0 else None, 
                'post_theta_cnt': self.post_theta_cnt if hasattr(self, 'post_theta_cnt') else 0,
                'cycle_theta_means': self.cycle_theta_means,
                'cycle_theta_vars': self.cycle_theta_vars if self.nst > 0 else None,
                'cycle_likelihoods': self.cycle_likelihoods,
                'prior_sig': self.model.prior_sig, 
                'optimizer': self.optimizer.state_dict(),
                'epoch': epoch,
                'current_cycle': self.current_cycle,
                'samples_collected': self.samples_collected,
                'samples_per_cycle': self.samples_per_cycle
            },
            fname
        )

        return fname

    def load_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.args.device)

        if ckpt['post_theta_mom1'] is not None:
            self.post_theta_mom1 = ckpt['post_theta_mom1']
        if ckpt['post_theta_mom2'] is not None:
            self.post_theta_mom2 = ckpt['post_theta_mom2']
        self.post_theta_cnt = ckpt.get('post_theta_cnt', 0)
        
        # Load GMM components
        self.cycle_theta_means = ckpt.get('cycle_theta_means', {})
        self.cycle_theta_vars = ckpt.get('cycle_theta_vars', {})
        self.cycle_likelihoods = ckpt.get('cycle_likelihoods', {})
        
        self.model.prior_sig = ckpt['prior_sig']
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.current_cycle = ckpt.get('current_cycle', 0)
        self.samples_collected = ckpt.get('samples_collected', 0)
        self.samples_per_cycle = ckpt.get('samples_per_cycle', {})

        return ckpt['epoch']
    
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
            if not likelihoods:
                weights[cycle] = 0.0
                continue
                
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
    SGLD sampler model.

    Actually no parameters involved.
    '''

    def __init__(self, ND, prior_sig=1.0, bias='informative'):

        '''
        Args:
            ND = training data size
            prior_sig = prior Gaussian sigma
            bias = how to treat bias parameters:
                "informative": -- the same treatment as weights
                "uninformative": uninformative bias prior
        '''

        super().__init__()

        self.ND = ND
        self.prior_sig = prior_sig
        self.bias = bias


    def forward(self, x, y, net, net0, criterion, lrs, Ninflate=1.0, nd=1.0):

        '''
        Evaluate minibatch SGLD updates for a given a batch.

        Args:
            x, y = batch input, output
            net = workhorse network (its parameters will be filled in)
            net0 = prior mean parameters
            criterion = loss function
            lrs = learning rates in list (adjusted to "eta" in SGLD)
            Ninflate = inflate N by this order of magnitude
            nd = noise discount factor

        Returns:
            loss = NLL loss on the batch
            out = class prediction on the batch

        Effects:
            net has .grad fields filled with SGLD updates
        '''

        bias = self.bias

        N = self.ND * Ninflate  # inflated training data size (accounting for data augmentation, etc.)

        if len(lrs) == 1:
            lr_body, lr_head = lrs[0], lrs[0]
        else:
            lr_body, lr_head = lrs[0], lrs[1]

        # fwd pass with theta
        out = net(x)

        # evaluate nll loss
        loss = criterion(out, y)

        # gradient d{loss_nll}/d{theta}
        net.zero_grad()
        loss.backward()

        # compute and set: grad = -(1/N) * d{logp(th)}/d{th} + d{loss}/d{th} + sqrt{2/(N*lr)} * N(0,I)
        with torch.no_grad():
            for (pname, p), p0 in zip(net.named_parameters(), net0.parameters()):
                if p.grad is not None:
                    if net.readout_name not in pname:
                        lr = lr_body
                    else:
                        lr = lr_head
                    if 'bias' in pname and bias == 'uninformative':
                        p.grad = p.grad + (
                            nd * np.sqrt(2/(N*lr)) * torch.randn_like(p)  # sqrt{2/(N*lr)} * N(0,I)
                        )
                    else:
                        p.grad = p.grad + (
                            (p-p0)/(self.prior_sig**2)/N +  # scaled negative log-prior grad = -(1/N) * d{logp(th)}/d{th}
                            nd * np.sqrt(2/(N*lr)) * torch.randn_like(p)  # sqrt{2/(N*lr)} * N(0,I)
                        )

        return loss.item(), out.detach()