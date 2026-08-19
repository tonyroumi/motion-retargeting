import random
import torch

class ImagePool:
    """ Image replay buffer for GAN discriminator training. """

    def __init__(self, pool_size: int, prob: float = 0.5):
        self.pool_size = pool_size
        self.prob = prob
        self.images = []
        self.num_images = 0

    @torch.no_grad()
    def query(self, images: torch.Tensor) -> torch.Tensor:
        """
        Returns a batch where each image is either the current input or a randomly
        replayed image from the pool, replacing it with the current one with
        probability `self.prob` once the pool is full.
        """
        if self.pool_size == 0:
            return images

        output = []

        for img in images:
            img = img.unsqueeze(0)

            # Pool not full: always store and return
            if self.num_images < self.pool_size:
                self.images.append(img.clone())
                self.num_images += 1
                output.append(img)
            else:
                # Pool full: probabilistic swap
                if random.random() < self.prob:
                    idx = random.randint(0, self.pool_size - 1)
                    old = self.images[idx].clone()
                    self.images[idx] = img.clone()
                    output.append(old)
                else:
                    output.append(img)

        return torch.cat(output, dim=0)
