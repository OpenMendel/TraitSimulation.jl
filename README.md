
| **Documentation** | **Build Status** | **Code Coverage**  |
|-------------------|------------------|--------------------|
| [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://OpenMendel.github.io/TraitSimulation.jl/latest) [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://OpenMendel.github.io/TraitSimulation.jl/stable) | [![build Actions Status](https://github.com/OpenMendel/TraitSimulation.jl/workflows/CI/badge.svg)](https://github.com/OpenMendel/TraitSimulation.jl/actions) | [![codecov](https://codecov.io/gh/OpenMendel/TraitSimulation.jl/branch/master/graph/badge.svg?token=YyPqiFpIM1)](https://codecov.io/gh/OpenMendel/TraitSimulation.jl) |

# TraitSimulation.jl
Authors: Sarah Ji, Chris German, Kenneth Lange, Janet Sinsheimer, Jin Zhou, Hua Zhou, Eric Sobel

A convenient tool for simulating phenotypes for unrelateds or families under a variety of supported models.

**(1) GLM Traits: Generalized Linear Models 

**(2) VCM Traits: Variance Component Models

**(3) Improved Multinomial Case/Control Traits: Ordered Multinomial Models:

**(4) GLMM Traits: Generalized Linear Mixed Models 

*Trait Simulation in Julia.*

## Installation
This package requires Julia v0.7 or later, which can be obtained from https://julialang.org/downloads/ or by building Julia from the sources in the https://github.com/JuliaLang/julia repository.

The package has not yet been registered and must be installed using the repository location. For example, by executing the following code:

```julia
using Pkg
pkg"add https://github.com/OpenMendel/TraitSimulation.jl"
```

# Citation
If you use OpenMendel analysis packages in your research, please cite the following reference in the resulting publications:

OPENMENDEL: a cooperative programming project for statistical genetics. Zhou H, Sinsheimer JS, Bates DM, Chu BB, German CA, Ji SS, Keys KL, Kim J, Ko S, Mosher GD, Papp JC, Sobel EM, Zhai J, Zhou JJ, Lange K. Hum Genet. 2019 Mar 26. doi: 10.1007/s00439-019-02001-z. [Epub ahead of print] PMID: 30915546
