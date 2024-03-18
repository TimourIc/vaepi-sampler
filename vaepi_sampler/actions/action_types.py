from typing import Callable, Union

import numpy as np
import torch


class LocalAction:
    """General class that represents a local (additive) discretized action for a single particle,
    Should be written to accept torch tensors with batch size since we will be using this class for the loss function later
    """

    def __init__(
        self,
        T_max: float,
        N_T: int,
        dim: int = 1,
        m: int = 1,
        V: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]] = None,
        x0: float = None,
        xf: float = None,
    ):
        """
        args:
            T_max (float): final value of the time integrand (beta)
            N_T (int): total number of space points in discretization of the path (including first and last) -
                even if the paths that you will be sampling will have length N_T-2 because x0 and xf are fixed, please pass the actual N_T here.
            dim (int): number of space dimensions (for now writing everything with the assumption dim=1)
            mass (float): particle mass in whatever units you are working
            V (function): the potential function, should work on toch tensors (e.g. x**2)
            x0 (float): option to fix initial point
            xf (float): option to fix final point
        """
        self.T_max = T_max
        self.N_T = N_T
        self.dt = T_max / (N_T - 1)
        self.dim = dim
        self.m = m
        self.V = V
        self.x0 = x0
        self.xf = xf

    def kinetic_part(self, x: Union[np.ndarray, torch.Tensor]) -> float:
        """Computes the kinetic energy part of the action

        args:
            x (torch.Tensor): a torch tensor of the discretized path of size (batch_size, N) with N the number of elements in the path

        if x0 and xf are fixed during initalization then N should be N=N_T-2 and you have to only provide the values in between
        if x0 and xf are not fixed during initialization then N=N_T and x is the full path

        Logic behind this sligthly awkward implementation is to conceptually only keep things that will have a torch gradient attached in the path torch.Tensor.
        """

        dx = torch.diff(x, dim=-1)
        S_k = (self.m / 2) * (torch.sum(dx**2, dim=-1)) / self.dt

        if self.x0 is not None and self.xf is not None:
            S_k += (
                (self.m / 2)
                * ((x[:, 0] - self.x0) ** 2 + (x[:, -1] - self.xf) ** 2)
                / self.dt
            )

        return S_k

    def potential_part(
        self,
        x: torch.Tensor,
    ) -> float:
        """Computes the potential energy part of the action

        args:
            x (torch.Tensor): a torch tensor of the discretized path of size (batch_size, N) with N the number of elements in the path

        Same logic as with kinetic_part.
        """

        S_V = torch.sum(self.V(x), dim=-1) * self.dt

        if self.x0 is not None and self.xf is not None:
            S_V += (self.V(self.x0) + self.V(self.xf)) * self.dt

        return S_V

    def __call__(self, x: torch.Tensor):
        total_action = self.kinetic_part(x)

        if self.V is not None:
            total_action += self.potential_part(x)

        return total_action
