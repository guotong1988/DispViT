from typing import Callable, Optional
from abc import ABC
import torch

from .worker_fn import get_rank, get_world_size, get_worker_init_fn


class DataLoader(ABC):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        mode: str,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        drop_last: bool = True,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        persistent_workers: bool = False,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset

        world_size = get_world_size()
        self.batch_size = batch_size // world_size if mode == "train" else 1
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        self.persistent_workers = persistent_workers
        self.seed = seed

        # Create samplers
        if mode == "train":
            self.sampler = torch.utils.data.DistributedSampler(self.dataset, num_replicas=world_size, rank=get_rank())
        elif mode == "val":
            self.sampler = InferenceSampler(len(self.dataset))
        else:
            raise ValueError(f"Unrecognized mode: {mode}")
        
    def get_loader(self, epoch):
        # Set the epoch for the sampler
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
        if hasattr(self.dataset, "epoch"):
            self.dataset.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

        # Create and return the dataloader
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=self.sampler,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            worker_init_fn=get_worker_init_fn(
                seed=self.seed,
                num_workers=self.num_workers,
                epoch=epoch,
                worker_init_fn=self.worker_init_fn,
            ),
            drop_last=self.drop_last
        )


class InferenceSampler(torch.utils.data.Sampler):
    """
    Produces indices for inference across all workers.
    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """

    def __init__(self, size):
        """
        Args:
            size (int): the total number of data on the underlying dataset to sample from
        """
        self._size = size
        assert size > 0
        self._rank = get_rank()
        self._world_size = get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_size = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_size[:rank])
        end = min(sum(shard_size[: rank + 1]), total_size)
        return range(begin, end)
    
    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)