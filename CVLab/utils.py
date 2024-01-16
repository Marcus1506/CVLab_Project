import torch
import os
from torchmetrics.image import StructuralSimilarityIndexMeasure
import numpy as np
import random
from tqdm import tqdm

from torch.utils.data import DataLoader
from typing import Callable

from .visual_utils import save_losses

def load_model(model: torch.nn.Module, model_path: str, device: torch.device="cpu") -> torch.nn.Module:
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model

def convert_to_onnx(model: torch.nn.Module, model_path: str) -> None:
    """
    Converts model in training mode to ONNX format to check layout.
    """
    model.eval()
    model = load_model(model, model_path)
    dummy_input = torch.randn(1, 3, 512, 512)
    onnx_name = os.path.splitext(model_path)[0] + '.onnx'
    torch.onnx.export(model, dummy_input, onnx_name,
                    export_params=True, opset_version=17, do_constant_folding=False,
                    input_names=["input"], output_names=["output"],
                    training=torch.onnx.TrainingMode.TRAINING,
                    dynamic_axes={'input' : {0 : 'batch_size'},
                                'output' : {0 : 'batch_size'}}, verbose=False)

class MSE_SSIM(torch.nn.Module):
    """
    Calculates combined loss of MSE and SSIM.
    """
    def __init__(self, value_range: tuple[float, float], alpha: float=1., beta: float=1., device: torch.device="cuda"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        # Alpha and beta may need to be adjusted. SSIM is in range (-1, 1), MSE in (0, 1) if we use max normalization.
        self.mse_loss = torch.nn.MSELoss()
        self.ssim_loss = StructuralSimilarityIndexMeasure(data_range=value_range).to(device)
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 1 - SSIM because 1 represents perfect similarity. So the loss is in the range (0, 3)
        return self.alpha * self.mse_loss(output, target) + self.beta * (1 - self.ssim_loss(output, target))

class Monitoring:
    """
    This class handles monitoring capabalities during training.
    Logging loss visuals, saving model checkpoints and early stopping is implemented in this version.
    """
    def __init__(self, model: torch.nn.Module, model_name: str, model_path: str, losses_path: str,
                 patience: int=4, device: torch.device="cpu"):
        self.model = model
        self.model_name = model_name
        self.model_path = model_path
        self.losses_path = losses_path
        self.device = device
        self.patience = patience

        self.count = patience
        self.min_loss = None

        self.train_losses = []
        self.eval_losses = []

        self.exit_training = False
    
    def save_losses(self) -> None:
        """
        Saves plots of losses to a file.
        """
        save_losses(self.train_losses, self.eval_losses, os.path.join(self.losses_path, self.model_name))

    def save_model(self) -> None:
        """
        Saves losses to a file. Assumes trained_models is already created.
        """
        # Send model to standard device
        self.model.to(self.device)
        model_name = self.model_name + ".pt"
        torch.save(self.model.state_dict(), os.path.join(self.model_path, model_name))

    def checkpoint(self) -> None:
        """
        Checkpoints model state and losses.
        """
        self.model.to(self.device)
        torch.save(self.model.state_dict(), os.path.join(self.model_path, self.model_name + ".pt"))
        self.save_losses()

    def step(self, eval_loss: torch.Tensor, train_loss: torch.Tensor) -> None:
        self.eval_losses.append(eval_loss)
        self.train_losses.append(train_loss)
        
        if eval_loss is None:
            raise ValueError("Eval loss is None!")
        if self.min_loss is None:
            self.min_loss = eval_loss
            return

        if eval_loss < self.min_loss:
            self.min_loss = eval_loss
            self.count = self.patience
            self.checkpoint()
        else:
            self.count -= 1
            if self.count == 0:
                print("\nEarly stopping!")
                self.exit_training = True
        self.save_losses()
    
    def finish_training(self) -> None:
        """
        Method saves losses. Meant for the case of reaching last epoch.
        """
        self.save_losses()

# TODO: Add scheduler
def std_training_loop(
        model: torch.nn.Module, train_data: torch.utils.data.Dataset, val_data: torch.utils.data.Dataset, num_epochs: int,
        model_name: str, optimizer: torch.optim.Optimizer, loss_function: torch.nn.Module, minibatch_size: int=16,
        collate_func: None | Callable[[torch.Tensor], torch.Tensor]=None, show_progress: bool = False, try_cuda: bool=False, early_stopping: bool=True,
        patience: int=3, model_path: str="models", losses_path: str="losses", workers: int=0,
        pin_memory: bool=True, prefetch_factor: int=2, true_random: bool=True
        ) -> None:

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and try_cuda else "cpu")
    print(device)
    model.to(device)
    
    train_monitoring = Monitoring(model, model_name, model_path, losses_path, patience, device)

    if true_random:
        rng = np.random.default_rng()
        seed = rng.integers(0, 2**16 - 1, dtype=int)
    else:
        seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    train_dataloader = DataLoader(train_data, collate_fn=collate_func, batch_size=minibatch_size,
                                  shuffle=True, num_workers=workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    # a little less workers for eval is generally good in most cases
    eval_dataloader = DataLoader(val_data, collate_fn=collate_func, batch_size=minibatch_size,
                                  shuffle=True, num_workers=(workers + 2) // 2, pin_memory=pin_memory, prefetch_factor=prefetch_factor)

    # scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=patience // 2)

    # One Epoch on both datasets just to compare
    model.eval()
    mock_train_batch_losses = []
    mock_eval_batch_losses = []
    with torch.no_grad():
        for train_batch, target_batch in tqdm(train_dataloader, disable=not show_progress, leave=False):
            train_batch = train_batch.to(device, non_blocking=pin_memory)
            target_batch = target_batch.to(device, non_blocking=pin_memory)
            pred = model(train_batch)
            loss = loss_function(pred, target_batch)
            mock_train_batch_losses.append(loss.detach().cpu())
    
    with torch.no_grad():
        for eval_batch, target_batch in tqdm(eval_dataloader, disable=not show_progress, leave=False):
            eval_batch = eval_batch.to(device, non_blocking=pin_memory)
            target_batch = target_batch.to(device, non_blocking=pin_memory)
            pred = model(eval_batch)
            loss = loss_function(pred, target_batch)
            mock_eval_batch_losses.append(loss.detach().cpu())
    
    train_monitoring.step(torch.mean(torch.stack(mock_eval_batch_losses).cpu()), torch.mean(torch.stack(mock_train_batch_losses).cpu()))
    
    for epoch in tqdm(range(num_epochs), disable=not show_progress):
        # set model to training mode
        model.train()
        train_batch_losses = []
        for train_batch, target_batch in tqdm(train_dataloader, disable=not show_progress, leave=False):
            train_batch = train_batch.to(device, non_blocking=pin_memory)
            target_batch = target_batch.to(device, non_blocking=pin_memory)

            model.zero_grad()
            pred = model(train_batch)
            loss = loss_function(pred, target_batch)
            loss.backward()
            optimizer.step()
            train_batch_losses.append(loss.detach().cpu())

        eval_batch_losses = []
        model.eval()
        with torch.no_grad():
            for eval_batch, target_batch in tqdm(eval_dataloader, disable=not show_progress, leave=False):
                eval_batch = eval_batch.to(device, non_blocking=pin_memory)
                target_batch = target_batch.to(device, non_blocking=pin_memory)
                pred = model(eval_batch)
                loss = loss_function(pred, target_batch)
                eval_batch_losses.append(loss.detach().cpu())

        eval_loss = torch.mean(torch.stack(eval_batch_losses).cpu())
        train_loss = torch.mean(torch.stack(train_batch_losses).cpu())

        train_monitoring.step(eval_loss, train_loss)
        # scheduler.step(eval_loss)
        if early_stopping:
            if train_monitoring.exit_training:
                break

    train_monitoring.finish_training()
    return

def train_tuple(
        model: torch.nn.Module, Dataloader: torch.utils.data.DataLoader, loss_function: torch.nn.Module,
        show_progress: bool, device: torch.device, pin_memory: bool, accumulation_steps: int=1,
        optimizer: torch.optim.Optimizer|None=None
        ) -> list[torch.Tensor]:
    minibatch_losses = []
    if optimizer:
        model.train()
        for minibatch_count, (input_batch, target_batch) in tqdm(enumerate(Dataloader), disable=not show_progress, leave=False):
            input_images, input_temps = input_batch
            input_images = input_images.to(device, non_blocking=pin_memory)
            input_temps = input_temps.to(device, non_blocking=pin_memory)
            target_batch = target_batch.to(device, non_blocking=pin_memory)
            pred = model(input_images, input_temps)
            loss = loss_function(pred, target_batch) / accumulation_steps
            loss.backward()
            if minibatch_count + 1 % accumulation_steps == 0:
                optimizer.zero_grad()
                optimizer.step()
            minibatch_losses.append(loss.detach().cpu())
    if optimizer is None:
        model.eval()
        with torch.no_grad():
            for input_batch, target_batch in tqdm(Dataloader, disable=not show_progress, leave=False):
                input_images, input_temps = input_batch
                input_images = input_images.to(device, non_blocking=pin_memory)
                input_temps = input_temps.to(device, non_blocking=pin_memory)
                target_batch = target_batch.to(device, non_blocking=pin_memory)
                pred = model(input_images, input_temps)
                loss = loss_function(pred, target_batch)
                minibatch_losses.append(loss.detach().cpu())
    return minibatch_losses

def std_training_loop_tuple(
        model: torch.nn.Module, train_data: torch.utils.data.Dataset, val_data: torch.utils.data.Dataset, num_epochs: int,
        model_name: str, optimizer: torch.optim.Optimizer, loss_function: torch.nn.Module, minibatch_size: int=16, accumulation_steps: int=1,
        collate_func: None | Callable[[torch.Tensor], torch.Tensor]=None, show_progress: bool = False, try_cuda: bool=False, early_stopping: bool=True,
        patience: int=3, model_path: str="models", losses_path: str="losses", workers: int=0,
        pin_memory: bool=True, prefetch_factor: int=2, true_random: bool=True
        ) -> None:

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and try_cuda else "cpu")
    print(device)
    model.to(device)
    
    train_monitoring = Monitoring(model, model_name, model_path, losses_path, patience, device)

    if true_random:
        rng = np.random.default_rng()
        seed = rng.integers(0, 2**16 - 1, dtype=int)
    else:
        seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    train_dataloader = DataLoader(train_data, collate_fn=collate_func, batch_size=minibatch_size,
                                  shuffle=True, num_workers=workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    # a little less workers for eval is generally good in most cases
    eval_dataloader = DataLoader(val_data, collate_fn=collate_func, batch_size=minibatch_size,
                                  shuffle=True, num_workers=(workers + 2) // 2, pin_memory=pin_memory, prefetch_factor=prefetch_factor)

    # scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=patience // 2)

    # One Epoch on both datasets just to compare
    mock_train_batch_losses = train_tuple(model, train_dataloader, loss_function, show_progress, device, pin_memory, accumulation_steps=accumulation_steps)
    mock_eval_batch_losses = train_tuple(model, eval_dataloader, loss_function, show_progress, device, pin_memory, accumulation_steps=accumulation_steps)
    
    train_monitoring.step(torch.mean(torch.stack(mock_eval_batch_losses)), torch.mean(torch.stack(mock_train_batch_losses)))
    
    for epoch in tqdm(range(num_epochs), disable=not show_progress):
        train_batch_losses = train_tuple(model, train_dataloader, loss_function, show_progress, device, pin_memory, optimizer, accumulation_steps)
        eval_batch_losses = train_tuple(model, eval_dataloader, loss_function, show_progress, device, pin_memory, accumulation_steps=accumulation_steps)

        eval_loss = torch.mean(torch.stack(eval_batch_losses))
        train_loss = torch.mean(torch.stack(train_batch_losses))

        train_monitoring.step(eval_loss, train_loss)
        # scheduler.step(eval_loss)
        if early_stopping:
            if train_monitoring.exit_training:
                break

    train_monitoring.finish_training()
    return