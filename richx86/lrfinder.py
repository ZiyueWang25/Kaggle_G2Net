import gc
import matplotlib.pyplot as plt
from torch.optim import AdamW#,Adam, SGD
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.nn import functional as torch_functional

from config import Config
from utils import mixup_data,mixed_criterion,get_device
from model import Model
from dataset import DataRetrieverLRFinder,read_data


# import torch_xla
# import torch_xla.core.xla_model as xm
class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion
        self.device = device
        torch.save(model.state_dict(), f'{Config.model_output_folder}/init_params.pt')

    def range_test(self, loader, end_lr = 10, num_iter = 100, 
                   smooth_f = 0.05, diverge_th = 5):
        lrs = []
        losses = []
        best_loss = float('inf')
        lr_scheduler = ExponentialLR(self.optimizer, end_lr, num_iter)
        for step, batch in enumerate(loader):
            if step == num_iter:
                break
            loss = self._train_batch(batch)
            lrs.append(lr_scheduler.get_last_lr()[0])
            #update lr
            lr_scheduler.step()
            if step > 0:
                loss = smooth_f * loss + (1 - smooth_f) * losses[-1]
            if loss < best_loss:
                best_loss = loss
            losses.append(loss)
            if loss > diverge_th * best_loss:
                print("Stopping early, the loss has diverged")
                break
        #reset model to initial parameters
        self.model.load_state_dict(torch.load(f'{Config.model_output_folder}/init_params.pt'))
        return lrs, losses

    def _train_batch(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        scaler = GradScaler()
        X = batch[0].to(self.device,non_blocking=Config.non_blocking)
        targets = batch[1].to(self.device,non_blocking=Config.non_blocking)
        
        if Config.use_mixup:
            (X_mix, targets_a, targets_b, lam) = mixup_data(
                X, targets, Config.mixup_alpha
            )
            with autocast(enabled=False):
                outputs = self.model(X_mix).squeeze()
                loss = mixed_criterion(self.criterion, outputs, targets_a, targets_b, lam)
        else:
            with autocast(enabled=False):
                outputs = self.model(X).squeeze()
                loss = self.criterion(outputs, targets)
        #loss.backward()
        scaler.scale(loss).backward()
        
        if Config.use_tpu:
            xm.optimizer_step(self.optimizer, barrier=True)  # Note: TPU-specific code! 
        else:
            scaler.step(self.optimizer)
            scaler.update()
#             self.optimizer.step()
        return loss.item()
    
                    
class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)
    def get_lr(self):
        curr_iter = self.last_epoch
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]

def plot_lr_finder(lrs, losses, skip_start = 0, skip_end = 0):
    if skip_end == 0:
        lrs = lrs[skip_start:]
        losses = losses[skip_start:]
    else:
        lrs = lrs[skip_start:-skip_end]
        losses = losses[skip_start:-skip_end]
    fig = plt.figure(figsize = (16,8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(lrs, losses)
    ax.set_xscale('log')
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Loss')
    ax.grid(True, 'both', 'x')
    plt.show()
    
if __name__ == "__main__":
    if Config.use_lr_finder:
        train_df,_ = read_data()
        START_LR = 1e-7
        model = Model()
        device = get_device()
        model.to(device,non_blocking=Config.non_blocking)
        optimizer = AdamW(model.parameters(), lr=START_LR, weight_decay=Config.weight_decay, amsgrad=False)
        criterion = torch_functional.binary_cross_entropy_with_logits
    
        train_data_retriever = DataRetrieverLRFinder(train_df['file_path'], train_df["target"].values)
        train_loader = DataLoader(train_data_retriever,
                                    batch_size=Config.batch_size, 
                                    shuffle=True, 
                                    num_workers=Config.num_workers, pin_memory=True, drop_last=True)
    
    if Config.use_lr_finder:
        try:
            END_LR = 10
            NUM_ITER = 150
            lr_finder = LRFinder(model, optimizer, criterion, device)
            lrs, losses = lr_finder.range_test(train_loader, END_LR, NUM_ITER)
        except RuntimeError as e:
            del model, optimizer, criterion, train_data_retriever, train_loader, lr_finder
            gc.collect()
            torch.cuda.empty_cache() 
            print(e)
        
    if Config.use_lr_finder:
        plot_lr_finder(lrs[:-20], losses[:-20])