import torch
from torch.utils.data import DataLoader
from typing import List, Any, Dict


class ExtendedModule(torch.nn.Module):
    def __init__(self):
        super(ExtendedModule, self).__init__()
        self.global_step = 0
        self.log_this_step = False
        self._hparams = {}
        self.user_data = None
        self._log_queue = []

    def log_dict(self, dictionary):
        self._log_queue.append(dictionary)

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        raise NotImplementedError

    @staticmethod
    def load_hyper_parameters(path):
        return torch.load(path, map_location=torch.device('cpu'))['hyper_parameters']

    @staticmethod
    def load_user_data(path):
        data = torch.load(path, map_location=torch.device('cpu'))
        return data['user_data'] if 'user_data' in data else None  # Backwards compatibility

    @classmethod
    def load_from_checkpoint(cls, path, do_not_load_hyper_parameters=False, **kwargs):
        data = torch.load(path, map_location=torch.device('cpu'))

        if do_not_load_hyper_parameters:
            hp = {}
        else:
            hp = data['hyper_parameters']

        hp.update(**kwargs)

        instance = cls(**hp)
        instance.user_data = data['user_data'] if 'user_data' in data else None  # Backwards compatibility
        instance.load_state_dict(data['state_dict'])
        return instance

    @property
    def device(self):
        return next(self.parameters()).device

    def save_hyperparameters(self, ignore=None):
        from inspect import currentframe
        from copy import deepcopy
        from pytorch_lightning.utilities.parsing import get_init_args

        frame = currentframe().f_back
        init_args = get_init_args(frame)

        if ignore is not None:
            init_args = {k: v for k, v in init_args.items() if k not in ignore}

        self._hparams = deepcopy(init_args)

    def save_checkpoint(self, path):
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as fp:
            torch.save({"hyper_parameters": self._hparams, "state_dict": self.state_dict(),
                        "user_data": self.user_data}, fp)

    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def training_step(self, batch: Any, batch_idx: int) -> dict:
        raise NotImplementedError

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int):
        raise NotImplementedError


def __do_log(log_queue: List[Dict], step: int):
    import wandb

    d = dict()

    for q in log_queue:
        d.update(q)

    wandb.log(d, step=step, commit=True)


def train_extended(module: ExtendedModule, max_epoch=1000, log_interval=1, save_every_k_epochs=100):
    from HeterogeneousGraphRL.Util.Progress import maybe_tqdm
    import wandb
    import os
    from time import time

    device = next(module.parameters()).device
    optimizers = module.configure_optimizers()
    train_dataloader = module.train_dataloader()
    global_step = 0

    total = None

    try:
        total = len(train_dataloader)
    except TypeError:
        pass

    for epoch in range(max_epoch):
        with maybe_tqdm(enumerate(iter(train_dataloader)),
                                     desc="Training epoch {}/{}".format(epoch + 1, max_epoch),
                                     total=total, leave=False) as it:
            timestamps = [time()]
            for batch_idx, batch in it:
                module.global_step = global_step
                module.log_this_step = (global_step % log_interval) == 0
                batch_dev = module.transfer_batch_to_device(batch, device, 0)

                for opt in optimizers:
                    opt.zero_grad()

                result = module.training_step(batch_dev, batch_idx)
                loss = result['loss']
                loss.backward()
                it.set_postfix({"loss": loss.item()})

                for opt in optimizers:
                    opt.step()

                timestamps.append(time())
                timestamps = timestamps[-10:]  # Use at most last 10 samples for window
                throughput = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])

                if module.log_this_step:
                    module.log_dict({"trainer/loss": loss.detach(), "trainer/epoch": epoch,
                                    "trainer/batches_per_sec": throughput, "trainer/global_step": global_step})
                    __do_log(module._log_queue, global_step)
                del module._log_queue[:]
                global_step += 1

        save_path = os.path.join("../tmp", wandb.run.project, wandb.run.id, "best.ckpt")
        module.save_checkpoint(save_path)

        if save_every_k_epochs is not None and epoch % save_every_k_epochs == 0:
            save_path = os.path.join("../tmp", wandb.run.project, wandb.run.id, "epoch-{:04d}.ckpt".format(epoch))
            module.save_checkpoint(save_path)
