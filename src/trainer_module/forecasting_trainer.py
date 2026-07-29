"""训练与评估流程。

该模块负责封装训练、验证、测试和 early stopping 逻辑，并在 train/val/test 边界处理 memory/reset 等实验细节。"""

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# 训练侧的早停器：根据验证集损失保存最佳权重，并在持续退化时停止训练。
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"Early stopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(
                f"Validation loss improved ({self.val_loss_min:.6f} -> {val_loss:.6f}). Saving checkpoint."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# 在 train/val/test 阶段切换前显式清空 memory，避免状态跨阶段污染。
def _reset_model_memory(model, accelerator=None) -> None:
    target_model = accelerator.unwrap_model(model) if accelerator is not None else model
    if hasattr(target_model, "reset_memory"):
        target_model.reset_memory()


# 统一整理 batch，将标签与模型输入拆开，便于 train/test 共用。
def _prepare_batch(batch: Dict[str, Any], device, accelerator=None) -> Tuple[Dict[str, Any], torch.Tensor]:
    model_inputs: Dict[str, Any] = {}
    labels = batch["y"]
    if accelerator is None:
        labels = labels.to(device)

    for key, value in batch.items():
        if key == "y":
            continue
        if torch.is_tensor(value) and accelerator is None:
            model_inputs[key] = value.to(device)
        else:
            model_inputs[key] = value
    return model_inputs, labels


# 主训练循环：负责优化、验证、早停和最佳 checkpoint 恢复。
def train_imm_llm(
    model,
    train_loader,
    val_loader,
    num_epochs=3,
    lr=1e-4,
    device="cuda",
    patience=3,
    checkpoint_path="checkpoint.pt",
    accelerator=None,
    max_train_batches=None,
    max_val_batches=None,
    epoch_progress_formatter: Optional[Callable[[int, int], str]] = None,
):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6,
    )
    huber_criterion = nn.HuberLoss(delta=1.0)

    if accelerator is not None:
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )
        _print = accelerator.print
        _device = accelerator.device
    else:
        model.to(device)
        _print = print
        _device = device

    early_stopping = EarlyStopping(
        patience=patience,
        verbose=True,
        path=checkpoint_path,
        trace_func=_print,
    )
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "train_huber": [],
        "val_huber": [],
    }

    for epoch in range(num_epochs):
        # 每个 epoch 开始前重置 memory，确保在线状态不会跨 epoch 累积。
        _reset_model_memory(model, accelerator)
        model.train()
        train_loss = 0.0
        train_huber = 0.0
        train_batches_processed = 0

        for batch_idx, batch in enumerate(train_loader):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break

            model_inputs, batch_y = _prepare_batch(batch, _device, accelerator)
            optimizer.zero_grad()
            out = model(labels=batch_y, **model_inputs)
            loss = out["loss"]

            if accelerator is not None:
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), max_norm=5.0)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            train_loss += loss.item()

            with torch.no_grad():
                huber_loss = huber_criterion(out["logits"], batch_y)
                train_huber += huber_loss.item()

            train_batches_processed += 1

        if train_batches_processed == 0:
            raise ValueError("No training batches were processed.")

        scheduler.step()
        train_loss /= train_batches_processed
        train_huber /= train_batches_processed

        # 进入验证阶段前再次清空 memory，避免训练态历史影响验证结果。
        _reset_model_memory(model, accelerator)
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_huber = 0.0
        val_batches_processed = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if max_val_batches is not None and batch_idx >= max_val_batches:
                    break

                model_inputs, batch_y = _prepare_batch(batch, _device, accelerator)
                out = model(labels=batch_y, **model_inputs)
                pred = out["logits"]
                val_loss += out["loss"].item()
                val_mae += torch.abs(pred - batch_y).mean().item()
                val_huber += huber_criterion(pred, batch_y).item()
                val_batches_processed += 1

        if val_batches_processed == 0:
            raise ValueError("No validation batches were processed.")

        val_loss /= val_batches_processed
        val_mae /= val_batches_processed
        val_huber /= val_batches_processed

        if epoch_progress_formatter is not None:
            _print(epoch_progress_formatter(epoch + 1, num_epochs))
        _print(
            f"Epoch: {epoch + 1} | Train Loss: {train_loss:.4f} | "
            f"Val MSE: {val_loss:.4f} | Val MAE: {val_mae:.4f} | "
            f"Train Huber: {train_huber:.4f} | Val Huber: {val_huber:.4f}"
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["train_huber"].append(train_huber)
        history["val_huber"].append(val_huber)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            _print("Early stopping triggered.")
            break

    try:
        load_device = _device if accelerator is None else accelerator.device
        model.load_state_dict(torch.load(checkpoint_path, map_location=load_device))
        _print("Loaded best checkpoint successfully.")
    except Exception as exc:
        _print(f"Warning: failed to load best checkpoint: {exc}. Using current weights.")

    return model, history


# 测试循环只做前向评估，并汇总最终 MSE / MAE 与完整预测结果。
def test_imm_llm(model, test_loader, device="cuda", accelerator=None, max_test_batches=None):
    model.eval()
    test_mse = 0.0
    test_mae = 0.0
    preds = []
    trues = []

    _print = accelerator.print if accelerator is not None else print
    _reset_model_memory(model, accelerator)

    with torch.no_grad():
        test_batches_processed = 0
        for batch_idx, batch in enumerate(test_loader):
            if max_test_batches is not None and batch_idx >= max_test_batches:
                break

            model_inputs, batch_y = _prepare_batch(batch, device, accelerator)
            out = model(labels=batch_y, **model_inputs)
            pred = out["logits"]
            test_mse += out["loss"].item()
            test_mae += torch.abs(pred - batch_y).mean().item()
            preds.append(pred.cpu().numpy())
            trues.append(batch_y.cpu().numpy())
            test_batches_processed += 1

    if test_batches_processed == 0:
        raise ValueError("No test batches were processed.")

    test_mse /= test_batches_processed
    test_mae /= test_batches_processed
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    _print(f"Test results -> MSE: {test_mse:.4f} | MAE: {test_mae:.4f}")
    return test_mse, test_mae, preds, trues





