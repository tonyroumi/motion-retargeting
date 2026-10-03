import random

import torch


class ReplayBuffer:
    """ Replay buffer for GAN discriminator training. """

    def __init__(self, pool_size: int = 50, prob: float = 0.5):
        self.pool_size = pool_size
        self.prob = prob
        self.items = []
        self.num_items = 0

    @torch.no_grad()
    def query(self, items: torch.Tensor) -> torch.Tensor:
        """
        Returns a batch where each item is either the current input or a randomly
        replayed item from the pool, replacing it with the current one with
        probability `self.prob` once the pool is full.
        """
        if self.pool_size == 0:
            return items

        output = []

        for img in items:
            img = img.unsqueeze(0)

            # Pool not full: always store and return
            if self.num_items < self.pool_size:
                self.items.append(img.clone())
                self.num_items += 1
                output.append(img)
            else:
                # Pool full: probabilistic swap
                if random.random() < self.prob:
                    idx = random.randint(0, self.pool_size - 1)
                    old = self.items[idx].clone()
                    self.items[idx] = img.clone()
                    output.append(old)
                else:
                    output.append(img)

        return torch.cat(output, dim=0)
