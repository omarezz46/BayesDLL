import os
import copy
import time
from tqdm import tqdm
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        if args.pretrained is None:  # if no pretrained backbone provided, zero out all params
            self.net0 = copy.deepcopy(net)
            with torch.no_grad():
                for _, p in self.net0.named_parameters():
                    p.copy_(torch.zeros_like(p))
        else:  
            self.net0 = net0
        self.net0 = self.net0.to(args.device)

        self.net = net.to(args.device)

        hparams = args.hparams
        self.model = Model(
            ND=args.ND, prior_sig=float(hparams['prior_sig']), runner=self, bias=str(hparams['bias']), momentum_decay=float(hparams['momentum_decay'])
        ).to(args.device)

        self.perform_cold_restarts = str(hparams.get('perform_cold_restarts', False)).lower() == 'true'
        if self.perform_cold_restarts:
            self.logger.info("Performing cold restarts: re-initializing network parameters with fresh random weights at the start of each cycle.")
        else:
            self.logger.info("Cold restarts disabled: keeping network parameters across cycles.")

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

        self.cycle_last_models_metadata = {}  # Stores metadata about the last 3 models per cycle
        self.all_model_metadata = []  # List of all collected model metadata
        self.model_counter = 0  # Counter for unique model IDs
        
        # Create models directory for disk storage
        self.models_dir = os.path.join(args.log_dir, 'collected_models')
        os.makedirs(self.models_dir, exist_ok=True)
        logger.info(f"Model storage directory created at: {self.models_dir}")

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

            if val_loader is not None:
                if ep % 5 == 0 or ep == args.epochs - 1:
                    tic_pe_val = time.time()
                    pe_loss_val, pe_err_val = self.evaluate_point_estimate(val_loader, self.net, desc_prefix=f"PE Val (Cycle {self.current_cycle} Mean)")
                    toc_pe_val = time.time()
                    prn_str_pe_val = f'(Epoch {ep}) Point Estimate Val (Cycle {self.current_cycle} Mean): '
                    prn_str_pe_val += f'loss = {pe_loss_val:.4f}, prediction error = {pe_err_val:.4f} '
                    prn_str_pe_val += f'(time: {toc_pe_val-tic_pe_val:.4f} seconds)'
                    logger.info(prn_str_pe_val)

            if ep%(args.epochs//args.num_cycles) > (args.epochs//args.num_cycles) - 4 and ep%(args.epochs//args.num_cycles) < (args.epochs//args.num_cycles) - 1:
                torch.save(self.net.state_dict(), os.path.join(args.log_dir, f'full_samples_net_ep{ep}.pth'))
                self.evaluate_full_samples(
                    train_loader, val_loader, test_loader, desc_prefix=f"Full Samples Epoch {ep}"
                )

            if cycle_updated : 
                if val_loader is not None:
                    tic = time.time()
                    losses_val[ep], errors_val[ep], targets_val, logits_val, logits_all_val = \
                        self.evaluate(val_loader)
                    toc = time.time()
                    prn_str = f'(Epoch {ep}) Validation summary: '
                    prn_str += f'loss = {losses_val[ep]:.4f}, prediction error = {errors_val[ep]:.4f} '
                    prn_str += f'(time: {toc-tic:.4f} seconds)'
                    logger.info(prn_str)

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

    def evaluate_full_samples(self, train_loader, val_loader, test_loader, desc_prefix="Full BMA"):
        '''
        Evaluate collected model checkpoints using Bayesian Model Averaging.
        Loads models one by one and accumulates predictions for BMA.
        '''
        args = self.args
        logger = self.logger

        logger.info(f"Starting Bayesian Model Averaging evaluation from: {args.log_dir}")

        model_files = sorted([f for f in os.listdir(args.log_dir) if f.startswith('full_samples_net_ep') and f.endswith('.pth')])

        if not model_files:
            logger.info("No model checkpoints found matching pattern 'full_samples_net_ep*.pth'.")
            return

        logger.info(f"Found {len(model_files)} model checkpoints for BMA")

        # Store BMA results for each dataset
        bma_results = {}
        
        # Evaluate on each dataset
        for dataset_name, data_loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
            if data_loader is None:
                continue
                
            logger.info(f"Performing BMA evaluation on {dataset_name} set...")
            
            # Accumulators for BMA
            all_targets = []
            all_logits_sum = None
            all_logits_individual = []
            total_loss = 0
            total_error = 0
            total_samples = 0
            num_models = 0
            
            # Process each model checkpoint
            for model_idx, model_file in enumerate(model_files):
                model_path = os.path.join(args.log_dir, model_file)
                
                # Create fresh network instance
                eval_net = copy.deepcopy(self.net)
                eval_net = eval_net.to(args.device)
                
                try:
                    state_dict = torch.load(model_path, map_location=args.device)
                    eval_net.load_state_dict(state_dict)
                    eval_net.eval()
                except Exception as e:
                    logger.error(f"Failed to load model {model_file}: {e}")
                    continue
                
                # Evaluate this model on the dataset
                model_targets = []
                model_logits = []
                model_loss = 0
                model_error = 0
                model_samples = 0
                
                with torch.no_grad():
                    with tqdm(data_loader, unit="batch", desc=f"BMA {dataset_name} - Model {model_idx+1}/{len(model_files)}") as tepoch:
                        for x, y in tepoch:
                            x, y = x.to(args.device), y.to(args.device)
                            
                            out = eval_net(x)
                            
                            # Calculate loss and error for this model
                            loss_batch = self.criterion(out, y)
                            pred = out.data.max(dim=1)[1]
                            err_batch = pred.ne(y.data).sum()
                            
                            # Store outputs
                            model_targets.append(y.cpu().numpy())
                            model_logits.append(out.cpu().numpy())
                            
                            # Accumulate statistics
                            model_loss += loss_batch.item() * len(y)
                            model_error += err_batch.item()
                            model_samples += len(y)
                
                # Convert to arrays
                model_targets = np.concatenate(model_targets, axis=0)
                model_logits = np.concatenate(model_logits, axis=0)
                
                # Store targets (only need to do this once)
                if model_idx == 0:
                    all_targets = model_targets
                
                # Accumulate logits for BMA
                if all_logits_sum is None:
                    all_logits_sum = model_logits.copy()
                else:
                    all_logits_sum += model_logits
                
                all_logits_individual.append(model_logits)
                
                # Accumulate statistics
                total_loss += model_loss
                total_error += model_error
                total_samples += model_samples
                num_models += 1
                
                # Log individual model performance
                avg_loss = model_loss / model_samples
                avg_error = model_error / model_samples
                logger.info(f"Model {model_file} on {dataset_name}: loss={avg_loss:.4f}, error={avg_error:.4f}")
                
                # Clean up to save memory
                del eval_net
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            if num_models == 0:
                logger.warning(f"No valid models found for {dataset_name} evaluation")
                continue
                
            # Calculate BMA predictions
            bma_logits = all_logits_sum / num_models
            bma_predictions = np.argmax(bma_logits, axis=1)
            
            # Calculate BMA performance
            bma_loss = np.mean([self.criterion(torch.tensor(bma_logits), torch.tensor(all_targets)).item()])
            bma_error = np.mean(bma_predictions != all_targets)
            
            # Store results
            bma_results[dataset_name] = {
                'loss': bma_loss,
                'error': bma_error,
                'num_models': num_models,
                'targets': all_targets,
                'logits': bma_logits,
                'logits_all': np.stack(all_logits_individual, axis=2),  # Shape: [samples, classes, models]
                'individual_avg_loss': total_loss / (total_samples * num_models),
                'individual_avg_error': total_error / (total_samples * num_models)
            }
            
            logger.info(f"BMA results on {dataset_name}: loss={bma_loss:.4f}, error={bma_error:.4f} (averaged over {num_models} models)")
            logger.info(f"Individual models average on {dataset_name}: loss={bma_results[dataset_name]['individual_avg_loss']:.4f}, error={bma_results[dataset_name]['individual_avg_error']:.4f}")
        
        # Save BMA results
        results_path = os.path.join(args.log_dir, 'bma_evaluation_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(bma_results, f)
        logger.info(f"BMA evaluation results saved to {results_path}")
        
        # Save BMA predictions for test set (for calibration analysis)
        if 'test' in bma_results:
            test_results = bma_results['test']
            fname = self.save_logits(
                test_results['targets'], 
                test_results['logits'], 
                test_results['logits_all'], 
                suffix='test_bma'
            )
            logger.info(f'BMA test predictions saved at {fname}')
        
        logger.info("Finished Bayesian Model Averaging evaluation.")
        return bma_results

    def evaluate_point_estimate(self, data_loader, net_to_evaluate, desc_prefix="Point Estimate"):
        '''
        Evaluate the given network (point estimate) on the data_loader.
        
        Returns:
            loss = averaged CE loss
            err = averaged prediction error
        '''
        args = self.args
        net_to_evaluate.eval() # Ensure evaluation mode

        loss, error, nb_samples = 0, 0, 0
        
        with torch.no_grad(): # Important: disable gradient calculations
            with tqdm(data_loader, unit="batch", desc=f"{desc_prefix} Eval") as tepoch:
                for x, y in tepoch:
                    x, y = x.to(args.device), y.to(args.device)
                    
                    out = net_to_evaluate(x)
                    
                    loss_batch = self.criterion(out, y)
                    pred = out.data.max(dim=1)[1]
                    err_batch = pred.ne(y.data).sum()
                    
                    loss += loss_batch.item() * len(y)
                    error += err_batch.item()
                    nb_samples += len(y)
                    
                    tepoch.set_postfix(loss=loss/nb_samples if nb_samples > 0 else 0, error=error/nb_samples if nb_samples > 0 else 0)
        
        if nb_samples == 0:
            return 0.0, 0.0 # Avoid division by zero if data_loader is empty
            
        return loss/nb_samples, error/nb_samples

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
                    self.Ninflate, self.nd, should_sample=should_sample
                )
                if hasattr(args, 'clip_grad') and args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), args.clip_grad)
                
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
                        
                        # In train_one_epoch method, replace the moment update section around line 300:

                        # Update running moments for this cycle using Welford's algorithm
                        if cycle_number not in self.cycle_theta_mom1:
                            # First sample - initialize
                            self.cycle_theta_mom1[cycle_number] = theta_vec.clone()
                            self.cycle_theta_mom2[cycle_number] = torch.zeros_like(theta_vec)  # This will store sum of squared deviations
                            self.samples_per_cycle[cycle_number] = 1
                        else:
                            # Update using Welford's algorithm
                            n = self.samples_per_cycle.get(cycle_number, 0) + 1
                            delta = theta_vec - self.cycle_theta_mom1[cycle_number]
                            self.cycle_theta_mom1[cycle_number] += delta / n
                            delta2 = theta_vec - self.cycle_theta_mom1[cycle_number]
                            self.cycle_theta_mom2[cycle_number] += delta * delta2
                            self.samples_per_cycle[cycle_number] = n
                    
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
                            # In evaluate method, replace variance calculation:

                            if cycle in self.cycle_theta_mom2:
                                n_samples = self.samples_per_cycle.get(cycle, 0)
                                if n_samples > 1:
                                    # Use Welford's variance: Var = M2 / (n-1)
                                    cycle_variance = self.cycle_theta_mom2[cycle] / (n_samples - 1)
                                else:
                                    # Single sample - use small variance
                                    cycle_variance = torch.ones_like(self.cycle_theta_mom1[cycle]) * 1e-12
                                cycle_variance.clamp_(min=1e-12)
                                nn.utils.vector_to_parameters(cycle_variance, param_vars.parameters())
                                nn.utils.vector_to_parameters(self.cycle_theta_mom1[cycle], param_means.parameters())
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
        fname = os.path.join(self.args.log_dir, f"{self.current_cycle}_ckpt.pt")

        torch.save(
            {   'last_theta': nn.utils.parameters_to_vector(self.net.parameters()),
                'cycle_theta_mom1': self.cycle_theta_mom1,
                'cycle_theta_mom2': self.cycle_theta_mom2,
                'cycle_likelihoods': self.cycle_likelihoods,
                'cycle_states': self.cycle_states,
                'epoch': epoch,
                'current_cycle': self.current_cycle,
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
        self.current_cycle = ckpt.get('current_cycle', 0)
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
            # In evaluate method, replace variance calculation:
            cycle = self.current_cycle
            if cycle in self.cycle_theta_mom2:
                n_samples = self.samples_per_cycle.get(cycle, 0)
                if n_samples > 1:
                    # Use Welford's variance: Var = M2 / (n-1)
                    cycle_variance = self.cycle_theta_mom2[cycle] / (n_samples - 1)
                else:
                    # Single sample - use small variance
                    cycle_variance = torch.ones_like(self.cycle_theta_mom1[cycle]) * 1e-12
                cycle_variance.clamp_(min=1e-12)
        
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
    SGHMC sampler model.

    Actually no parameters involved.
    '''

    def __init__(self, ND, runner, prior_sig=1.0, bias='informative', momentum_decay=0.05):

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
        self.momentum_decay = momentum_decay
        self.runner = runner

    def forward(self, x, y, net, net0, criterion, lrs, Ninflate=1.0, nd=1.0, should_sample=False):
        '''
        Evaluate minibatch SGHMC updates for a given batch.
        Args:
            x, y = batch input, output
            net = workhorse network (its parameters will be filled in)
            net0 = prior mean parameters
            criterion = loss function
            lrs = learning rates in list (adjusted to "eta" in SGHMC)
            momentum_decay = momentum decay parameter (referred to as α in SGHMC papers)
            Ninflate = inflate N by this order of magnitude
            nd = noise discount factor
        Returns:
            loss = NLL loss on the batch
            out = class prediction on the batch
        Effects:
            net has .grad fields filled with SGHMC updates
        '''
        
        bias = self.bias
        N = self.ND * Ninflate 
        if len(lrs) == 1:
            lr_body, lr_head = lrs[0], lrs[0]
        else:
            lr_body, lr_head = lrs[0], lrs[1]
        
        # Initialize momentum if not already done
        if not hasattr(self, 'momentum_buffer'):
            self.momentum_buffer = {}
            for name, param in net.named_parameters():
                self.momentum_buffer[name] = torch.zeros_like(param)

        # self.momentum_decay = momentum_decay
        
        # Forward pass with theta
        out = net(x)

        # Evaluate NLL loss
        loss = criterion(out, y)

        # Gradient d{loss_nll}/d{theta} 
        net.zero_grad()
        loss.backward()
        
        # Compute and set: 
        # v_t = (1-α)v_{t-1} - lr * grad_U(θ) + N(0, 2(α)lr)
        # where grad_U(θ) = -(1/N) * d{logp(th)}/d{th} + d{loss}/d{th}
        with torch.no_grad():
            for (pname, p), p0 in zip(net.named_parameters(), net0.parameters()):
                if p.grad is not None:
                    if net.readout_name not in pname:
                        lr = lr_body
                    else:
                        lr = lr_head
                    
                    # Get momentum for this parameter
                    v = self.momentum_buffer[pname]
                    
                    # Compute gradient term including prior
                    if 'bias' in pname and bias == 'uninformative':
                        grad_U = p.grad + self.prior_sig * p.data # Only data likelihood gradient
                    else:
                        grad_U = p.grad + self.prior_sig * p.data   # Prior + likelihood gradient
                    
                    # Noise term
                    noise_scale = nd * np.sqrt((2 * self.momentum_decay * lr))/ N 
                    noise = noise_scale * torch.randn_like(p)
                    
                    #add noise only if in sampling phase
                    if should_sample :
                        v = v * (1 - self.momentum_decay) - lr * grad_U + noise
                    else:
                        v = v * (1 - self.momentum_decay) - lr * grad_U 
                    
                    # Store updated momentum
                    self.momentum_buffer[pname] = v
                    
                    # Update gradient using momentum
                    p.data.add_(v)
        
        return loss.item(), out.detach()

