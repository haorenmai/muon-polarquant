# PyTorch Muon with Polar Express Method and 8-bit Quantization
This code is based on the Muon optimizer of [PyTorch](https://github.com/pytorch/pytorch), with additional support for the [Polar Express method](https://arxiv.org/abs/2505.16932) and [8-bit quantization](https://arxiv.org/abs/2509.23106).

```python
from _muon_polarquant import Muon

optimizer = Muon(
    params, 
    lr=1e-3, 
    weight_decay=0.1,
    momentum=0.95,
    nesterov=True,
    polar_express=True,
    pe_l=0.001,
    pe_u=1.0,
    pe_steps=6,
    quant=True       
)

```
