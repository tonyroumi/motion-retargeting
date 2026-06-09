"""
AMP Observation Buffer
======================
Maintains a per-env sliding window of the last `disc_obs_steps` observations
and writes one completed window into the ring buffer **on every call** to `add`.

Buffer layout
-------------
  _window_buf  : (n_envs, disc_obs_steps, obs_dim)  — sliding window per env
  _buf         : (capacity, disc_obs_steps * obs_dim) — ring buffer of completed windows
  _buf_ptr     :  int                                 — next ring-write slot
  _size        :  int                                 — valid slots in ring (≤ capacity)

Workflow
--------
  On env reset — call seed_from_windows(windows) with shape (n_envs, disc_obs_steps, obs_dim).
    This initialises _window_buf from motion-library frames and immediately flushes
    all n_envs windows into the ring.

  On each env step — call add(obs) with shape (n_envs, obs_dim).
    _window_buf is shifted left by one slot (oldest observation dropped), the new
    observation is written into the last slot, and then all n_envs windows are
    flushed into the ring.  Every step therefore produces exactly n_envs new
    ring entries.

    After step k the per-env window holds:
      [ motion_lib[k], ..., motion_lib[K-1], sim[0], ..., sim[k-1] ]
    (K = disc_obs_steps, motion-lib entries replaced one-by-one by sim states)
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Optional, Union

class CircularObsBuffer:
    """
    Sliding-window ring buffer for AMP discriminator observations.

    Parameters
    ----------
    obs_dim        : dimensionality of a single environment observation
    disc_obs_steps : window length K — number of consecutive observations
                     that form one discriminator input
    n_envs         : number of parallel environments
    capacity       : maximum completed windows kept in the ring
    device         : torch device for all internal tensors
    obs_dtype      : storage dtype (default float32)
    """

    def __init__(
        self,
        obs_dim: int,
        disc_obs_steps: int,
        n_envs: int,
        capacity: int,
        device: Union[str, torch.device] = "cpu",
        obs_dtype: torch.dtype = torch.float32,
    ) -> None:

        self.obs_dim        = obs_dim
        self.disc_obs_steps = disc_obs_steps
        self.n_envs         = n_envs
        self.capacity       = capacity
        self.device         = torch.device(device)
        self.obs_dtype      = obs_dtype

        # Sliding window: always holds the last K observations for every env.
        self._window_buf: torch.Tensor = torch.zeros(
            (n_envs, disc_obs_steps, obs_dim),
            dtype=obs_dtype, device=self.device,
        )

        # Ring buffer of completed (flattened) windows.
        self._buf: torch.Tensor = torch.zeros(
            (capacity, disc_obs_steps * obs_dim),
            dtype=obs_dtype, device=self.device,
        )
        self._buf_ptr: int = 0
        self._size:    int = 0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def seed_from_windows(self, windows: torch.Tensor) -> None:
        """
        Initialise the sliding window from motion-library data and flush to ring.

        Called once per episode start (env.reset).

        Parameters
        ----------
        windows : (n_envs, disc_obs_steps, obs_dim) or (len(env_ids), disc_obs_steps, obs_dim)
                Full K-step window from the motion library, ordered oldest → newest.
                When env_ids is provided, the leading dimension must match len(env_ids).
        env_ids : optional 1-D indices of environments being reset. When None,
                all environments are initialised (original behaviour).
        """
        self._window_buf = windows
        self._flush_all()

    def add(
        self,
        obs:     torch.Tensor,
        env_ids: torch.Tensor = None,
    ) -> None:
        """
        Slide the window left by one, append a new simulator observation, and
        flush to the ring buffer immediately.

        Parameters
        ----------
        obs     : (n_envs, obs_dim) or (len(env_ids), obs_dim)
        env_ids : optional subset of envs to update. When None, all envs are
                  updated (original behaviour).
        """
        self._window_buf = obs
        self._flush_all()
        return obs.reshape(self.n_envs, -1)

    def reset_envs(self, env_indices: Union[torch.Tensor, np.ndarray]) -> None:
        """
        Zero the sliding window for envs that just terminated.

        Call this when you have the new motion-lib seed ready to immediately
        follow with seed_from_windows for those specific envs.
        """
        self._window_buf[env_indices] = 0.0

    def sample(
        self,
        batch_size: int,
        stacked:    bool = False,
        generator:  Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Draw a random batch of completed discriminator windows.

        Parameters
        ----------
        batch_size : number of windows to draw (with replacement)
        stacked    : if False → (batch_size, disc_obs_steps * obs_dim)
                     if True  → (batch_size, disc_obs_steps, obs_dim)
        """
        if self._size == 0:
            raise RuntimeError("Buffer is empty — cannot sample.")

        indices = torch.randint(
            self._size,
            (batch_size,),
            device=self.device,
            generator=generator,
        )
        windows = self._buf[indices]

        if stacked:
            windows = windows.view(batch_size, self.disc_obs_steps, self.obs_dim)

        return windows

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    @property
    def window_shape(self) -> tuple[int, int]:
        return (self.disc_obs_steps, self.obs_dim)

    @property
    def flat_window_dim(self) -> int:
        return self.disc_obs_steps * self.obs_dim

    def _flush_all(self) -> None:
        """Write every env's current window into the ring buffer."""
        flat = self._window_buf.reshape(self.n_envs, -1)   # (n_envs, K*obs_dim)

        # Compute target ring indices, handling wrap-around.
        offsets = torch.arange(self.n_envs, device=self.device)
        indices = (self._buf_ptr + offsets) % self.capacity

        self._buf[indices] = flat
        self._buf_ptr = (self._buf_ptr + self.n_envs) % self.capacity
        self._size = min(self._size + self.n_envs, self.capacity)

    def _flush_envs(self, env_ids: torch.Tensor) -> None:
        """Write a subset of envs' current windows into the ring buffer."""
        n = len(env_ids)
        flat = self._window_buf[env_ids].reshape(n, -1)   # (n, K*obs_dim)

        offsets = torch.arange(n, device=self.device)
        indices = (self._buf_ptr + offsets) % self.capacity

        self._buf[indices] = flat
        self._buf_ptr = (self._buf_ptr + n) % self.capacity
        self._size = min(self._size + n, self.capacity)