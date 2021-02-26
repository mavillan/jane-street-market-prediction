import copy
import numpy as np
from tqdm.notebook import trange
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Monitor:
    def __init__(self, model, optimizer, scheduler, patience, metric_fn, 
                 experiment_name, num_epochs, dataset_sizes, early_stop_on_metric=False,
                 lower_is_better=True, verbose=True):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.patience = patience
        self.metric_fn = metric_fn
        self.dataset_sizes = dataset_sizes
        self.early_stop_on_metric = early_stop_on_metric
        self.lower_is_better = lower_is_better
        self.verbose = verbose
        
        if verbose:
            self.iter_epochs = trange(num_epochs, desc=experiment_name)
        else:
            self.iter_epochs = range(num_epochs)
            
        if lower_is_better:
            self.epoch_loss = {"train": np.inf, "valid": np.inf}
            self.epoch_metric = {"train": np.inf, "valid": np.inf}
            self.best_loss = np.inf
            self.best_metric = np.inf
        else:
            self.epoch_loss = {"train": -np.inf, "valid": -np.inf}
            self.epoch_metric = {"train": -np.inf, "valid": -np.inf}
            self.best_loss = -np.inf
            self.best_metric = -np.inf
            
        self.train_loss = list()
        self.valid_loss = list()
        self.train_metric = list()
        self.valid_metric = list()
            
        self.best_model_state = model.state_dict()
        self.best_optimizer_state = optimizer.state_dict()

        self.epoch_counter = {"train": 0, "valid": 0}
        self.running_loss = 0.0

        self.es_counter = 0
        
        self.epoch_dwr = list()
        self.epoch_preds = list()
        
    def check_if_improved(self, best, actual):
        if self.lower_is_better and (actual < best):
            return True
        elif not self.lower_is_better and (best > actual):
            return True
        else:
            return False

    def reset_epoch(self):
        self.running_loss = 0.0
        self.epoch_dwr = list()
        self.epoch_preds = list()
    
    def step(self, loss, batch_size, dwr_matrix=None, preds=None):
        self.running_loss += loss.item() * batch_size
        if (self.metric_fn is not None) and (dwr_matrix is not None and preds is not None):
            self.epoch_dwr.append(dwr_matrix.numpy())
            self.epoch_preds.append(preds)

    def log_epoch(self, phase):
        self.epoch_loss[phase] = self.running_loss / self.dataset_sizes[phase]

        dwr_matrix = np.concatenate(self.epoch_dwr, axis=0)
        preds = (torch.cat(self.epoch_preds)).detach().cpu().numpy()
        preds = 1./(1. + np.exp(- preds))
        actions = (preds > 0.5).astype(int)
        self.epoch_metric[phase] = self.metric_fn(dwr_matrix[:,0], dwr_matrix[:,1], dwr_matrix[:,2], actions)
        
        if phase == "train":
            self.train_loss.append(self.epoch_loss[phase])
            self.train_metric.append(self.epoch_metric[phase])
        elif phase == "valid":
            self.valid_loss.append(self.epoch_loss[phase])
            self.valid_metric.append(self.epoch_metric[phase])

        postfix_kwargs = {
            "a_train_loss": f"{self.epoch_loss['train']:0.6f}",
            "b_valid_loss": f"{self.epoch_loss['valid']:0.6f}",
            "c_best_loss":  f"{self.best_loss:0.6f}",}
        if self.metric_fn is not None:
            postfix_kwargs["d_train_metric"] = f"{self.epoch_metric['train']:0.6f}"
            postfix_kwargs["e_valid_metric"] = f"{self.epoch_metric['valid']:0.6f}"
            postfix_kwargs["f_best_metric"] =  f"{self.best_metric:0.6f}"
        postfix_kwargs["g_es_counter"] = self.es_counter
        if self.scheduler is not None:
            try:
                postfix_kwargs["h_last_lr"] = f"{self.scheduler.get_last_lr()[0]:0.8f}"
            except:
                pass
            
        if self.verbose: 
            self.iter_epochs.set_postfix(**postfix_kwargs)
        self.epoch_counter[phase] += 1

        early_stop = False
        if phase == "valid":
            
            if not self.early_stop_on_metric:
                improved = self.check_if_improved(self.best_loss, self.epoch_loss["valid"])
            elif self.early_stop_on_metric:
                improved = self.check_if_improved(self.best_metric, self.epoch_metric["valid"])
                
            if improved:
                self.best_loss = copy.deepcopy(self.epoch_loss["valid"])
                self.best_metric = copy.deepcopy(self.epoch_metric["valid"])
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
                self.es_counter = 0
            else:
                self.es_counter += 1
                if self.es_counter >= self.patience:
                    early_stop = True
                    if self.verbose: 
                        self.iter_epochs.close()
        return early_stop
    
    
def train_step(model, train_dataloader, optimizer, monitor, loss_fn,
               scheduler=None, clip_value=None, tabnet=False,
               lambda_sparse=0., dae=False, alpha=0.):
    model.train()
    monitor.reset_epoch()

    for batch in train_dataloader:
        inputs,targets,weights,dwr = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        weights = weights.to(device)
        
        optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(True):
            if tabnet:
                preds,M_loss = model(inputs)
                loss = loss_fn(preds, targets, weights)
                loss -= lambda_sparse*M_loss
            elif dae:
                preds,decoded = model(inputs)
                loss1 = loss_fn(preds, targets, weights)
                loss2 = nn.functional.mse_loss(decoded, inputs)
                loss = (1-alpha)*loss1 + alpha*loss2
            else:
                preds = model(inputs)
                loss = loss_fn(preds, targets, weights)
            loss.backward()
            if clip_value is not None:
                clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        preds = preds[:,0] if len(preds.shape)>1 else preds
        monitor.step(loss, inputs.size(0), dwr, preds)
                
    monitor.log_epoch("train")

    
def valid_step(model, valid_dataloader, optimizer, monitor, loss_fn, 
               tabnet=False, dae=False, alpha=0):
    model.eval()
    monitor.reset_epoch()
    
    for batch in valid_dataloader:
        inputs,targets,weights,dwr= batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        weights = weights.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.set_grad_enabled(False):
            if tabnet:
                preds,_ = model(inputs)
                loss = loss_fn(preds, targets, weights)
            elif dae:
                preds,decoded = model(inputs)
                loss1 = loss_fn(preds, targets, weights)
                loss2 = nn.functional.mse_loss(decoded, inputs)
                loss = (1-alpha)*loss1 + alpha*loss2
            else:
                preds = model(inputs)
                loss = loss_fn(preds, targets, weights)
        preds = preds[:,0] if len(preds.shape)>1 else preds
        monitor.step(loss, inputs.size(0), dwr, preds)
    
    early_stop = monitor.log_epoch("valid")
    return early_stop