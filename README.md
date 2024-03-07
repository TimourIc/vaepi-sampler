In this repository a reverse VAE architecture is used to sample Feynman paths and estimate the partition function of simple physical systems.

## VAE-path-integral-sampler

Two recent works have made interesting progress towards sampling physical distributions with the help of variational inference and variational autoencoders. In [Y. Che et al., Phys. Rev. B 105, 214205](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.105.214205) the authors show that it is feasible to learn to sample path distributions of Euclidean path integral propagators using a decoder-only architecture by decoding a latent variable onto a path. In another recent work [G. Cantwell arXiv:2209.10423](https://arxiv.org/abs/2209.10423) the author shows that a reverse-VAE architecture can be used to learn distributions commonly found in physics where the distribution is known up to a normalization constant, providing an alternative to MCMC. Here, we will combine both ideas and use the full encoder-decoder (reverse) VAE architecture to compute path integral propagators and partition functions.


## Setup
Clone repository:

```python
git clone git@github.com:TimourIc/vaepi-sampler
````

Create venv and activate:
 
```python
python -m venv venv
source venv/bin/activate
```

Install your package in editable mode (and use vae_path_generator in scripts like a real package):

```python
pip install -e .
```




