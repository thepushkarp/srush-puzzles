## GPU Puzzles

This collection of 14 GPU puzzles need you to use NUMBA CUDA to write some basic GPU functions. Here are my solutions to problems from [srush/GPU-Puzzles](https://github.com/srush/GPU-Puzzles/). You can open the puzzles directly in Google Colab [here](https://colab.research.google.com/github/srush/GPU-Puzzles/blob/main/GPU_puzzlers.ipynb).

## Solutions

Here are my solutions to the puzzles.

### Puzzle 1: Map

Implement a "kernel" (GPU function) that adds 10 to each position of vector `a`
and stores it in vector `out`.  You have 1 thread per position.

*Tip: Think of the function `call` as being run 1 time for each thread.
The only difference is that `cuda.threadIdx.x` changes each time.*

```python
def map(out, a) -> None:
    local_i = cuda.threadIdx.x
    out[local_i] = a[local_i] + 10
```

### Puzzle 2 - Zip

Implement a kernel that adds together each position of `a` and `b` and stores it in `out`.
You have 1 thread per position.

```python
def zip(out, a, b) -> None:
    local_i = cuda.threadIdx.x
    out[local_i] = a[local_i] + b[local_i]
```

### Puzzle 3 - Guards

Implement a kernel that adds 10 to each position of `a` and stores it in `out`.
You have more threads than positions.

```python
def guard(out, a, size) -> None:
    local_i = cuda.threadIdx.x
    if local_i < size:
        out[local_i] = a[local_i] + 10
```

### Puzzle 4 - Map 2D

Implement a kernel that adds 10 to each position of `a` and stores it in `out`.
Input `a` is 2D and square. You have more threads than positions.

```python
def map_2d(out, a, size) -> None:
    local_i = cuda.threadIdx.x
    local_j = cuda.threadIdx.y
    if local_i < size and local_j < size:
        out[local_i, local_j] = a[local_i, local_j] + 10
```

### Puzzle 5 - Broadcast

Implement a kernel that adds `a` and `b` and stores it in `out`.
Inputs `a` and `b` are vectors. You have more threads than positions.

```python
def broadcast(out, a, b, size) -> None:
    local_i = cuda.threadIdx.x
    local_j = cuda.threadIdx.y
    if local_i < size and local_j < size:
        out[local_i, local_j] = a[local_i, 0] + b[0, local_j]
```

### Puzzle 6 - Blocks

Implement a kernel that adds 10 to each position of `a` and stores it in `out`.
You have fewer threads per block than the size of `a`.

*Tip: A block is a group of threads. The number of threads per block is limited, but we can
have many different blocks. Variable `cuda.blockIdx` tells us what block we are in.*

```python
def block(out, a, size) -> None:
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < size:
        out[i] = a[i] + 10
```

### Puzzle 7 - Blocks 2D

Implement the same kernel in 2D.  You have fewer threads per block
than the size of `a` in both directions.

```python
def block_2d(out, a, size) -> None:
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # FILL ME IN (roughly 4 lines)
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i < size and j < size:
        out[i, j] = a[i, j] + 10
```

### Puzzle 8 - Shared

Implement a kernel that adds 10 to each position of `a` and stores it in `out`.
You have fewer threads per block than the size of `a`.

**Warning**: Each block can only have a *constant* amount of shared
 memory that threads in that block can read and write to. This needs
 to be a literal python constant not a variable. After writing to
 shared memory you need to call `cuda.syncthreads` to ensure that
 threads do not cross.

```python
TPB = 4
def shared(out, a, size) -> None:
    shared = cuda.shared.array(TPB, numba.float32)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    local_i = cuda.threadIdx.x

    if i < size:
        shared[local_i] = a[i]
        cuda.syncthreads()

        # FILL ME IN (roughly 2 lines)
        out[i] = shared[local_i] + 10
```

### Puzzle 9 - Pooling

Implement a kernel that sums together the last 3 position of `a` and stores it in `out`.
You have 1 thread per position. You only need 1 global read and 1 global write per thread.

```python
TPB = 8
def pool(out, a, size) -> None:
    shared = cuda.shared.array(TPB, numba.float32)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    local_i = cuda.threadIdx.x
    # FILL ME IN (roughly 8 lines)
    if i < size:
        shared[local_i] = a[i]
    else:
        shared[local_i] = 0
    cuda.syncthreads()

    s = shared[local_i]
    if local_i > 0:
        s += shared[local_i - 1]
    if local_i > 1:
        s += shared[local_i - 2]

    out[i] = s if i < size else 0
```

### Puzzle 10 - Dot Product

Implement a kernel that computes the dot-product of `a` and `b` and stores it in `out`.
You have 1 thread per position. You only need 2 global reads and 1 global write per thread.

*Note: For this problem you don't need to worry about number of shared reads. We will
 handle that challenge later.*

```python
TPB = 8
def dot(out, a, b, size) -> None:
    shared = cuda.shared.array(TPB, numba.float32)

    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    local_i = cuda.threadIdx.x
    # FILL ME IN (roughly 9 lines)
    if i < size:
        shared[local_i] = a[i] * b[i]
    else:
        shared[local_i] = 0
    cuda.syncthreads()

    stride = 1
    while stride < cuda.blockDim.x:
        if local_i % (2 * stride) == 0:
            shared[local_i] +=shared[local_i + stride]
        stride *= 2
        cuda.syncthreads()
    if local_i == 0:
        out[local_i] = shared[local_i]
```

### Puzzle 11 - 1D Convolution

Implement a kernel that computes a 1D convolution between `a` and `b` and stores it in `out`.
You need to handle the general case. You only need 2 global reads and 1 global write per thread.

```python
MAX_CONV = 4
TPB = 8
TPB_MAX_CONV = TPB + MAX_CONV
def conv(out, a, b, a_size, b_size) -> None:
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    local_i = cuda.threadIdx.x

    # FILL ME IN (roughly 17 lines)
    shared_a = cuda.shared.array(TPB_MAX_CONV, numba.float64)

    if i < a_size:
        shared_a[local_i] = a[i]
    else:
        shared_a[local_i] = 0

    if local_i < b_size - 1 and i + cuda.blockDim.x < a_size:
        shared_a[local_i + cuda.blockDim.x] = a[i + cuda.blockDim.x]
    if local_i < b_size - 1:
        shared_a[local_i + cuda.blockDim.x] = 0

    cuda.syncthreads()

    if i < a_size:
        result = 0
        for j in range(b_size):
            if local_i + j < TPB_MAX_CONV:
                result += shared_a[local_i + j] * b[j]
        out[i] = result
```

## Puzzle 12 - Prefix Sum

Implement a kernel that computes a sum over `a` and stores it in `out`.
If the size of `a` is greater than the block size, only store the sum of
each block.

We will do this using the [parallel prefix sum](https://en.wikipedia.org/wiki/Prefix_sum) algorithm in shared memory. That is, each step of the algorithm should sum together half the remaining numbers.

```python
TPB = 8
def sum(out, a, size: int) -> None:
    cache = cuda.shared.array(TPB, numba.float32)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    local_i = cuda.threadIdx.x
    blockIdx = cuda.blockIdx.x
    # FILL ME IN (roughly 12 lines)
    if i < size:
        cache[local_i] = a[i]
    else:
        cache[local_i] = 0
    cuda.syncthreads()

    stride = 1
    while stride < cuda.blockDim.x:
        if local_i % (2 * stride) == 0:
            cache[local_i] += cache[local_i + stride]
        stride *= 2
        cuda.syncthreads()
    if local_i == 0:
        out[blockIdx] = cache[local_i]
```

### Puzzle 13 - Axis Sum

Implement a kernel that computes a sum over each column of `a` and stores it in `out`.

```python
TPB = 8
def axis_sum(out, a, size: int) -> None:
    cache = cuda.shared.array(TPB, numba.float32)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    local_i = cuda.threadIdx.x
    batch = cuda.blockIdx.y
    # FILL ME IN (roughly 12 lines)
    if i < size:
        cache[local_i] = a[batch, i]
    else:
        cache[local_i] = 0

    cuda.syncthreads()

    stride = 1
    while stride < size:
        if local_i % (2 * stride) == 0:
            cache[local_i] += cache[local_i + stride]
        stride *= 2
        cuda.syncthreads()

    if local_i == 0:
        out[batch, local_i] = cache[local_i]
```

### Puzzle 14 - Matrix Multiply!

Implement a kernel that multiplies square matrices `a` and `b` and
stores the result in `out`.

*Tip: The most efficient algorithm here will copy a block into
 shared memory before computing each of the individual row-column
 dot products. This is easy to do if the matrix fits in shared
 memory.  Do that case first. Then update your code to compute
 a partial dot-product and iteratively move the part you
 copied into shared memory.* You should be able to do the hard case
 in 6 global reads.

First Case (Matrix fits in memory):

```python
TPB = 3
def mm(out, a, b, size: int) -> None:
    a_shared = cuda.shared.array((TPB, TPB), numba.float32)
    b_shared = cuda.shared.array((TPB, TPB), numba.float32)

    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    local_i = cuda.threadIdx.x
    local_j = cuda.threadIdx.y
    # FILL ME IN (roughly 14 lines)

    if i < size and j < size:
        a_shared[local_i, local_j] = a[i, j]
        b_shared[local_i, local_j] = b[i, j]
    else:
        a_shared[local_i, local_j] = 0.0
        b_shared[local_i, local_j] = 0.0

    cuda.syncthreads()

    if i < size and j < size:
        s = 0.0
        for k in range(size):
            s += a_shared[local_i, k] * b_shared[k, local_j]
        out[i, j] = s
```

Second Case (Matrix does not fits in memory):

```python
TPB = 3
def mm(out, a, b, size: int) -> None:
    a_shared = cuda.shared.array((TPB, TPB), numba.float32)
    b_shared = cuda.shared.array((TPB, TPB), numba.float32)

    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    local_i = cuda.threadIdx.x
    local_j = cuda.threadIdx.y
    # FILL ME IN (roughly 14 lines)

    accumulator = 0.0

    num_tiles = (size + TPB -1) // TPB

    for tileIdx in range(num_tiles):
        k_start = tileIdx * TPB

        a_col = k_start + local_j
        if i < size and a_col < size:
            a_shared[local_i, local_j] = a[i, a_col]
        else:
            a_shared[local_i, local_j] = 0.0

        b_row = k_start + local_i
        if b_row < size and j < size:
            b_shared[local_i, local_j] = b[b_row, j]
        else:
            b_shared[local_i, local_j] = 0

        cuda.syncthreads()

        if i < size and j < size:
            for k in range(min(TPB, size - k_start)):
                accumulator += a_shared[local_i, k] * b_shared[k, local_j]

        cuda.syncthreads()

    if i < size and j < size:
        out[i, j] = accumulator
```
