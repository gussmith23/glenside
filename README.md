# Glenside

![Build and test](https://github.com/gussmith23/glenside/workflows/Build%20and%20test/badge.svg)
![Check formatting](https://github.com/gussmith23/glenside/workflows/Check%20formatting/badge.svg)

Check out the [web demo](https://gussmith23.github.io/glenside-web-demo)!

Glenside 
  is a pure, low-level 
  tensor program representation 
  which enables tensor program optimization via program rewriting, using rewriting frameworks such as
  the [egg equality saturation library](https://egraphs-good.github.io/).
If you are interested
  in transforming and optimizing
  tensor kernels
  (e.g. fusing kernels,
   exploring data layouts,
   or
   mapping to custom hardware),
  then Glenside is of interest to you!
See the [**web demo**](https://gussmith23.github.io/glenside-web-demo)
  for concrete examples.
See our [**MAPS 2021 paper**](https://arxiv.org/abs/2105.09377)
  to understand why Glenside exists
  and how it can be used.
Finally, see the [docs](https://gussmith23.github.io/glenside/glenside)
  for technical documentation.

## Quickstart

Fastest way to verify that this code actually does something is to build the Docker image and run the tests:

```sh
git clone <this repo>
cd <this repo>
docker build --tag glenside .
docker run -it glenside cargo test
```

...and "soon" I will add interactive web demos and pretty visualizations!

## Dependencies

Glenside optionally depends on TVM and CPLEX.
To disable these optional dependencies,
  use the `--no-default-features`
  flag with `cargo`, e.g.
  `cargo test --no-default-features`.

### CPLEX

Glenside uses the [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio)
  ILP solver.
It isn't actually used
  in the core of Glenside anymore,
  and needs to be removed
  or cordoned off,
  but for now,
  to get Glenside fully working,
  you need CPLEX.
To set up CPLEX,
  follow these steps:

1. **Get access to CPLEX.** Students and academics can do so by making an account through their academic program. Download and install CPLEX on your machine.
2. **Set environment variables.** Set `$CPLEX_LIB` to the location of the newly-installed CPLEX library on your machine. For me on OSX, it resides in `/Applications/CPLEX_Studio1210/cplex/lib/x86-64_osx/static_pic/`.

## Publications

- [Pure Tensor Program Rewriting via Access Patterns (Representation Pearl)](https://arxiv.org/abs/2105.09377) (published at the MAPS symposium at PLDI 2021)
- [Enumerating Hardware-Software Splits with Program Rewriting](https://arxiv.org/abs/2003.00290), a two-page proposal of Glenside which was accepted into the [Second Young Architects Workshop](https://sites.psu.edu/yarch2020/) at ASPLOS 2020.
