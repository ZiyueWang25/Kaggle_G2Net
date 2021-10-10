import time
import json
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from config import Config
import utils 

class Trainer:
    def __init__(
        self, 
        model, 
        device, 
        optimizer, 
        criterion, 
        scheduler,
        valid_labels,
        best_valid_score,
        fold,
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.best_valid_score = best_valid_score
        self.valid_labels = valid_labels
        self.fold = fold

    
    def fit(self, epochs, train_loader, valid_loader, save_path): 
        train_losses = []
        valid_losses = []
#         global N_EPOCH_EXPLICIT  #tbs later
        for n_epoch in range(epochs):
            start_time = time.time()
            print('Epoch: ', n_epoch)
            train_loss, train_preds = self.train_epoch(train_loader)
            valid_loss, valid_preds = self.valid_epoch(valid_loader)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(valid_loss)
            valid_score = utils.get_score(self.valid_labels, valid_preds)

            numbers = valid_score
            filename = Config.model_output_folder+f'score_epoch_{n_epoch}.json'          
            with open(filename, 'w') as file_object: 
                json.dump(numbers, file_object) 
            

            if self.best_valid_score < valid_score:
                self.best_valid_score = valid_score
                self.save_model(n_epoch, save_path+'best_model.pth', train_preds, valid_preds)

            print('train_loss: ',train_loss)
            print('valid_loss: ',valid_loss)
            print('valid_score: ',valid_score)
            print('best_valid_score: ',self.best_valid_score)
            print('time used: ', time.time()-start_time)

                   
            
    def train_epoch(self, train_loader):
        if Config.amp:
            scaler = GradScaler()
        self.model.train()
        losses = []
        train_loss = 0
        # preds = []
        for step, batch in enumerate(train_loader, 1):
            self.optimizer.zero_grad()
            X = batch[0].to(self.device,non_blocking=Config.non_blocking)
            targets = batch[1].to(self.device,non_blocking=Config.non_blocking)
            
            if Config.use_mixup:
                (X_mix, targets_a, targets_b, lam) = utils.mixup_data(
                    X, targets, Config.mixup_alpha
                )
                with autocast(enabled=False):
                    outputs = self.model(X_mix).squeeze()
                    loss = utils.mixed_criterion(self.criterion, outputs, targets_a, targets_b, lam)
            else:
                with autocast(enabled=False):
                    outputs = self.model(X).squeeze()
                    loss = self.criterion(outputs, targets)

                
            if Config.gradient_accumulation_steps > 1:
                loss = loss / Config.gradient_accumulation_steps
            scaler.scale(loss).backward()
          
            if (step) % Config.gradient_accumulation_steps == 0:
                scaler.step(self.optimizer)
                scaler.update()
            

            if (not isinstance(self.scheduler, ReduceLROnPlateau)):
                self.scheduler.step()


            loss2 = loss.detach()

         
            losses.append(loss2)
            train_loss += loss2

            if (step) % Config.print_num_steps == 0:
                train_loss = train_loss.item() #synch once per print_num_steps instead of once per batch
                print(f'[{step}/{len(train_loader)}] ', 
                      'avg loss: ',train_loss/step,
                      'inst loss: ', loss2.item())
 
        return train_loss / step, None

    def valid_epoch(self, valid_loader):
        self.model.eval()      
        valid_loss = []
        preds = []
        for step, batch in enumerate(valid_loader, 1):
            with torch.no_grad():
                X = batch[0].to(self.device,non_blocking=Config.non_blocking)
                targets = batch[1].to(self.device,non_blocking=Config.non_blocking)
                outputs = self.model(X).squeeze()
                loss = self.criterion(outputs, targets)
                if Config.gradient_accumulation_steps > 1:
                    loss = loss / Config.gradient_accumulation_steps
                valid_loss.append(loss.detach().item())
                preds.append(outputs.sigmoid().to('cpu').numpy())

        predictions = np.concatenate(preds)
        return np.mean(valid_loss), predictions

    def save_model(self, n_epoch, save_path, train_preds, valid_preds):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
                'scheduler': self.scheduler.state_dict(),
                'train_preds': train_preds,
                'valid_preds': valid_preds,
            },
            save_path,
        )
    
