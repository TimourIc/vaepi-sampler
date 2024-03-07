from vae_path_generator.actions.action_types import LocalAction
import numpy as np
from typing import Union

class HarmonicAction(LocalAction):


    """The action of a simple 1D SHO"""

    def harmonic_potential(x: Union[float, np.ndarray], m: float ,w: float)-> Union[float, np.ndarray]:

        return (m*w**2/2)*x**2

    def __init__(
        self,
        T_max: float,
        N_T: int,
        dim:int =1,
        m:int=1,
        omega: int=1,
        x0=None,
        xf=None
    ):
        
        """
        UNITS: free to choose whatever units you are working in, if you have removed mass and frequency from the action by rescaling just put m=w=1.
        """
        
        def V(x):
            return HarmonicAction.harmonic_potential(x, m ,omega)
        
        super().__init__(T_max, N_T, dim, m ,V, x0, xf)
    