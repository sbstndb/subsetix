# CUDA Interval Intersection (intervalix, because we are french !)

This is a simple experimental project to find intersection between to srt of intervals in CUDA. 

## What is a set of intervals ? 

Let's define the set of intervals. 

An interval is defined by two values `begin` and `end` as `interval = [begin, end)`. Please not that `begin` is **included** and `end` is **excluded**.
A set of intervals is **just** a collection of intervals like `set= {[b1, e1), [b2, e2), ...}`
In our example, the set is **normalized** and **sorted** by construct. This means that for two different intervals i1 and i2 in a set, there will be `intersection(i1, i2) = 0` and that `i1.begin < i2.begin`.


## Example of intersection

Let `A` and `B` be a set. 

`A = {[0,2), [4, 6), [8, 10)}`

`B = {[1,3), [6, 9), [10, 12)}`

Thus, the intersection set is : 

`I = {[1, 2), [8, 9)}`


## What this project does

- Takes two set of intervals
- Use CUDA kernels to perform  the intersection
- employes a two-pass approach: 
    -   The first one count how many intersections each interval in the set1 has with the set2. This helps know how much space is needed for the result.
    -   The second one is calculate the intersection and write them in the preallocated array.

- To find the intersection areao, we use lower_bounds to the left and to the right of the considered interval. 
- We have a simple demo showing how to use it, and measure the performance (while it is not perfect, it is a WIP)
- We provide a `google-test` to show that is just works. 

## Requirements

-   A Cuda capable GPU
-   CUDA Toolkit with `thrust`
-   `Cmake`
-   `google-test` (if not, you can disable by comenting it in the CMake file)

## Building

1. Clone the repo

2. Create a build directory 
```bash
mkdir build && cd build
```

3. Run `CMake`
!!! Be carefull. As it is a WIP, the current CMake specifies the CUDA architecture to be SM_86. You should change it according to your specific hardware.

After changing the architecture : 
```bash
cmake .. 
make
```

4. Run executables

After building, you can run the example from the build directory : 

```bash
./exemple_main
```


