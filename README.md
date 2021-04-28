# Glenside

![Build and test](https://github.com/gussmith23/glenside/workflows/Build%20and%20test/badge.svg)
![Check formatting](https://github.com/gussmith23/glenside/workflows/Check%20formatting/badge.svg)

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

## Publications

- TODO: Glenside is accepted to MAPS at PLDI 2021! Add link to MAPS paper
- [Enumerating Hardware-Software Splits with Program Rewriting](https://arxiv.org/abs/2003.00290), a two-page proposal of Glenside which was accepted into the [Second Young Architects Workshop](https://sites.psu.edu/yarch2020/) at ASPLOS 2020.