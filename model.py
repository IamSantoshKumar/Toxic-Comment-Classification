import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
from callbacks import CallbackRunner
from  utility import AverageMeter
from tqdm import tqdm
from torch.cuda import amp

class Tesseract(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_loader = None
        self.valid_loader = None
        self.optimizer = None
        self.scheduler = None
        self.flag = False
        self.fp16 = False
        self.scaler = None
        self.callbackrunner = None
        self.metrics = {}
        self.metrics["train"] = {}
        self.metrics["valid"] = {}
        self.metrics["test"] = {}
        self._train_state = None
        self._model_state = None

    def forward(self, *args, **kwargs):
        super().forward(*args, **kwargs)
        
    def fetch_optimizer(self, *args, **kwargs):
        return

    def fetch_scheduler(self, *args, **kwargs):
        return

    def update_metrics(self, losses, monitor):
        self.metrics[self._model_state].update(monitor)
        self.metrics[self._model_state]["loss"] = losses.avg

    @property
    def model_state(self):
        return self._model_state

    @model_state.setter
    def model_state(self, value):
        self._model_state = value

    @property
    def train_state(self):
        return self._train_state

    @train_state.setter
    def train_state(self, value):
        self._train_state = value
        if self.callbackrunner is not None:
            self.callbackrunner(value)

    def train_one_epoch(self, dataloader):
        self.train()
        losses = AverageMeter()
        tk0 = tqdm(dataloader, total=len(dataloader))
        self.model_state = 'train'
        for idx, data in enumerate(tk0):
            _, loss, metrics = self.train_one_step(data)
            losses.update(loss.item(), dataloader.batch_size)
            if idx==0:
               metrics_meter = {k: AverageMeter() for k in metrics}
            monitor = {}
            for m_m in metrics_meter:
                metrics_meter[m_m].update(metrics[m_m], dataloader.batch_size)
                monitor[m_m] = metrics_meter[m_m].avg
            tk0.set_postfix(loss=losses.avg, stage='train', **monitor)
        tk0.close()
        self.update_metrics(losses=losses, monitor=monitor)
        return losses.avg
    
    def train_one_step(self, data):
        self.optimizer.zero_grad()
        for k, v in data.items():
            data[k] = v.to('cuda')
        with torch.set_grad_enabled(True):
            if self.fp16:
                with amp.autocast():    
                    logit, loss, accuracy = self(**data)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                logit, loss, accuracy = self(**data)
                loss.backward()
                self.optimizer.step()
        return logit, loss, accuracy
    
    def validate_one_epoch(self, dataloader):
        self.eval()
        losses = AverageMeter()
        tk0 = tqdm(dataloader, total=len(dataloader))
        self.model_state = 'valid'
        for idx, data in enumerate(tk0):
            logit, loss, metrics = self.validate_one_step(data)
            losses.update(loss.item(), dataloader.batch_size)
            if idx==0:
               metrics_meter = {k: AverageMeter() for k in metrics}
            monitor = {}
            for m_m in metrics_meter:
                metrics_meter[m_m].update(metrics[m_m], dataloader.batch_size)
                monitor[m_m] = metrics_meter[m_m].avg
            tk0.set_postfix(loss=losses.avg, stage='valid', **monitor)
        tk0.close()
        self.update_metrics(losses=losses, monitor=monitor)
        return losses.avg
    
    def validate_one_step(self, data):
        for k, v in data.items():
            data[k] = v.to('cuda')
        logit, loss, accuracy = self(**data)
        return logit, loss, accuracy
    
    def predict(self, dataset, batch_size, device):
        self.eval()
        if next(self.parameters()).device!=device:
            self.to(device)
        predictions = []
        dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
        )
        
        tk0 = tqdm(dataloader, total=len(dataloader))
        for idx, data in enumerate(tk0):
            logit = self.predict_one_step(data)
            predicted = logit.cpu().numpy()
            predictions.append(predicted)
        tk0.close()
        return np.concatenate(predictions)
    
    def predict_one_step(self, data):
        for k, v in data.items():
            data[k] = v.to('cuda')
        with torch.no_grad():        
            logit, _, _ = self(**data)
        return logit
    
    
    def fit(
        self, 
        train_dataset,
        valid_dataset,
        train_bs=8,
        valid_bs=8,
        epochs=10,
        callback=None,
        fp16=False,
        device='cpu',
        workers = 1
    ):
        if self.train_loader is None:
            self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_bs,
            shuffle=True,
            num_workers=workers,
            drop_last=True
            )
        if self.valid_loader is None:
            self.valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=valid_bs,
            shuffle=False,
            num_workers=workers,
            drop_last=True
            )

        self.optimizer = self.fetch_optimizer()
        
        if self.scheduler is None:
            self.scheduler = self.fetch_scheduler()
        
        if next(self.parameters()).device!=device:
            self.to(device)
            
        self.callbackrunner = CallbackRunner(callback, self) 
        
        self.fp16 = fp16
        
        if self.fp16:
            self.scaler = amp.GradScaler()

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(self.train_loader)
            valid_loss = self.validate_one_epoch(self.valid_loader)
            
            if self.scheduler is None:
                self.scheduler.step()
            self.train_state='on_epoch_end'
            if self.flag:
                break