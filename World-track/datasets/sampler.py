# from typing import Iterator
# import torch
from torch.utils.data import RandomSampler

from typing import Iterator, Sized
import torch
from torch.utils.data import Sampler

class RandomPairSampler(RandomSampler):
    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64,
                                     generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.arange(0, n, dtype=torch.long).view(-1, 2)[torch.randperm(n // 2, generator=generator)].view(-1).tolist()
            yield from torch.arange(0, n, dtype=torch.long).view(-1, 2)[torch.randperm(n // 2, generator=generator)].view(-1).tolist()[:self.num_samples % n]


class TemporalSampler(Sampler[int]):
    def __init__(self, data_source: Sized, batch_size: int = 2, accumulate_grad_batches: int = 8) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.accumulate_grad_batches = accumulate_grad_batches

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        m = n - (n % (self.batch_size * self.accumulate_grad_batches))

        idx = torch.arange(m, dtype=torch.long).view(self.batch_size, self.accumulate_grad_batches, -1)
        idx = idx.transpose(0, 1).permute(*torch.arange(idx.ndim - 1, -1, -1)).flatten().tolist()
        idx = idx + list(range(m, n))

        yield from idx