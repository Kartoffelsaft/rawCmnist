# MNIST in C

This is an implementation of training a deep neural network to recognize images in the MNIST dataset, using (somewhat) plain C.

# Requirements

1. [STB datastructures library](https://github.com/nothings/stb/blob/master/stb_ds.h)
2. [GNU scientific library](https://www.gnu.org/software/gsl/)
3. [MNIST training images & labels](https://github.com/cvdfoundation/mnist) (testing images are unused as of now)

# Building

```sh
gcc main.c -lgsl-lcblas -ggdb -lm -ffast-math
```
