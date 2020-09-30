# glenside

![Build and test](https://github.com/gussmith23/glenside/workflows/Build%20and%20test/badge.svg)
![Check formatting](https://github.com/gussmith23/glenside/workflows/Check%20formatting/badge.svg)

Hardware-software partition exploration with e-graphs

## Quickstart

Fastest way to verify that this code actually does something is to build the Docker image and run the tests:

```sh
git clone <this repo>
cd <this repo>
docker build --tag glenside .
docker run -it glenside cargo test
```

...and "soon" I will add interactive web demos and pretty visualizations!
