# Glenside

![Build and test](https://github.com/gussmith23/glenside/workflows/Build%20and%20test/badge.svg)
![Check formatting](https://github.com/gussmith23/glenside/workflows/Check%20formatting/badge.svg)

[web demo](https://gussmith23.github.io/glenside-web-demo)

Hardware–software partition exploration with e-graphs!

Glenside is a research project which seeks to answer the question: given a deep learning program, can we *automatically* generate an accelerator design and compiler by *simultaneously* optimizing over hardware configuration, memory layout, and software schedule? We first introduce a new representation for tensor programs which includes constructs for representing memory layouts and hardware components. We then utilize *equality graphs* (provided by the [`egg` crate](https://docs.rs/egg/)) to run rewrites over the program and explore the design space of hardware/software designs. Finally, we use the resulting expanded design space to construct an ILP problem. Solutions to the ILP problem encode both hardware configurations and software schedules.

See the [docs](https://gussmith23.github.io/glenside/glenside) for technical details and examples!

## Quickstart

Fastest way to verify that this code actually does something is to build the Docker image and run the tests:

```sh
git clone <this repo>
cd <this repo>
docker build --tag glenside .
docker run -it glenside cargo test
```

...and "soon" I will add interactive web demos and pretty visualizations!

## CPLEX

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

- TODO: Glenside is accepted to MAPS at PLDI 2021! Add link to MAPS paper
- [Enumerating Hardware-Software Splits with Program Rewriting](https://arxiv.org/abs/2003.00290), a two-page proposal of Glenside which was accepted into the [Second Young Architects Workshop](https://sites.psu.edu/yarch2020/) at ASPLOS 2020.
