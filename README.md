# ORC-Nextrout

The **ORC-Nextrout** algorithm is designed to recover community structures in networks. We do this by taking inspiration from recent approaches that connect community detection with geometry, using the notion of Ollivier-Ricci curvature (ORC) to detect communities, and combining with a recent Optimal Transport (OT) approach that allows tuning for traffic penalization.


ORC-Nextrout is based on the theory described in this paper:

- Community Detection in networks by Dynamical Optimal Transport Formulation. D. Leite, D. Baptista, A. Ibrhaim, E. Facca and C. D. Bacco ([arXiv](https://arxiv.org/abs/2205.08468)).

Please consider citing our work if you use this code.

## Prerequisites

- [Nextrout](https://github.com/Danielaleite/Nextrout)
- [GraphRicciCurvature](https://github.com/saibalmars/GraphRicciCurvature)


## How to use

You can simply clone this repository:

```
git clone https://github.com/Danielaleite/ORC-Nextrout
```

You can check a step-by-step on how to use it on a real network inside the tutorial [here](https://github.com/Danielaleite/ORC-Nextrout/blob/main/code/tutorial.ipynb).

## Authors

* Daniela Leite, Diego Baptista Theuerkauf 

See also the list of [contributors](https://github.com/Danielaleite/ORC-Nextrout/graphs/contributors) who participated in this project.

## License

MIT License

Copyright (c) 2022 Daniela Leite, Diego Baptista Theuerkauf and Caterina De Bacco

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.