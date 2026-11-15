"""
run_training.KeepModelOnDeviceDDPStrategy: a DDP fit (1 CPU process, gloo) must not call LightningModule.cpu() at
teardown (the stock DDPStrategy does, which OOM-killed the 4-GPU VLM jobs after training), and the module's own cpu()
must be restored afterwards.
"""
import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, TensorDataset

from navsim.planning.script.run_training import KeepModelOnDeviceDDPStrategy


class _Tiny(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)
        self.cpu_calls = 0

    def cpu(self):
        self.cpu_calls += 1
        return super().cpu()

    def training_step(self, batch, batch_idx):
        return self.layer(batch[0]).pow(2).mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _fit(strategy) -> _Tiny:
    module = _Tiny()
    data = DataLoader(TensorDataset(torch.randn(4, 2)), batch_size=2)
    trainer = pl.Trainer(accelerator="cpu", devices=1, strategy=strategy, max_epochs=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(module, data)
    return module


def test_teardown_keeps_the_model_where_it_is(monkeypatch):
    import pytorch_lightning.strategies.strategy as strategy_module

    moved = []
    original = strategy_module._optimizers_to_device
    monkeypatch.setattr(strategy_module, "_optimizers_to_device",
                        lambda optimizers, device: (moved.append((len(optimizers), str(device))),
                                                    original(optimizers, device))[1])
    stock = _fit(DDPStrategy(process_group_backend="gloo"))
    assert stock.cpu_calls == 1  # the behaviour being removed
    assert (1, "cpu") in moved  # stock teardown moves the optimizer to the CPU
    moved.clear()
    strategy = KeepModelOnDeviceDDPStrategy(process_group_backend="gloo")
    kept = _fit(strategy)
    assert kept.cpu_calls == 0
    assert all(n == 0 for n, device in moved if device == "cpu")  # no optimizer moved to the CPU at teardown
    assert len(strategy.optimizers) == 1  # restored afterwards
    assert "cpu" not in vars(kept)  # the shadowing attribute is gone again
    kept.cpu()
    assert kept.cpu_calls == 1
