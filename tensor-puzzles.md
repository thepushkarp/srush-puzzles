## Tensor Puzzles

This collection of 21 tensor puzzles need you to implement some standard NumPy functions from scratch using first principles broadcasting. Here are my solutions to problems from [srush/Tensor-Puzzles](https://github.com/srush/Tensor-Puzzles/). You can open the puzzles directly in Google Colab [here](https://colab.research.google.com/github/srush/Tensor-Puzzles/blob/main/Tensor%20Puzzlers.ipynb).

## Rules

The rules are simple:

1. Can use tensor broadcasting
2. Each puzzle needs to be solved in 1 line (&lt;80 columns) of code.
3. You are allowed @, arithmetic, comparison, shape, any indexing (e.g. `a[:j]`, `a[:, None]`, `a[arange(10)]`), and previous puzzle functions.
4. You are _not allowed_ anything else. No view, sum, take, squeeze, tensor.
5. The following functions are implemented for you:

   - **arange** to replace a for-loop

   ```python
   def arange(i: int):
       "Use this function to replace a for-loop."
       return torch.tensor(range(i))
   ```

   - **where** to replace an if-statement

   ```python
   def where(q, a, b):
       "Use this function to replace an if-statement."
       return (q * a) + (~q) * b
   ```

## Solutions

Here are my solutions to the puzzles.

### Puzzle 1 - ones

Compute [ones](https://numpy.org/doc/stable/reference/generated/numpy.ones.html) - the vector of all ones.

**Intuition**: arange(i) creates a vector of i elements from 0 to i-1. Multiplying this vector by 0 and adding 1 makes it equivalent to ones(i).

**Solution column count**: 28

```python
def ones(i: int) -> TT["i"]:
    return arange(i) * 0 + 1
```

### Puzzle 2 - sum

Compute [sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html) - the sum of a vector.

**Intuition**: dot product of a vector with a vector of ones results in its sum (matmul of two vectors is equivalent to dot product).

**Solution column count**: 40

```python
def sum(a: TT) -> TT[1]:
    return a @ ones(a.shape[0])[:, None]
```

### Puzzle 3 - outer

Compute [outer](https://numpy.org/doc/stable/reference/generated/numpy.outer.html) - the outer product of two vectors.

**Intuition**: Broadcasting one vector vertically and the other horizontally. The matmul is equivalent to the outer product.

**Solution column count**: 34

```python
def outer(a: TT["i"],b: TT["j"]) -> TT["i", "j"]:
    return a[:, None] @ b[None, :]
```

### Puzzle 4 - diag

Compute [diag](https://numpy.org/doc/stable/reference/generated/numpy.diag.html) - the diagonal vector of a square matrix.

**Intuition**: Index the matrix with the same arange for rows and cols to get the i==j positions.

**Solution column count**: 52

```python
def diag(a: TT["i", "i"]) -> TT["i"]:
    return a[arange(a.shape[0]), arange(a.shape[0])]
```

### Puzzle 5 - eye

Compute [eye](https://numpy.org/doc/stable/reference/generated/numpy.eye.html) - the identity matrix.

**Intuition**: Broadcast arange(j) vertically and horizontally and compare row and column indices. The same indices result in 1s and the rest are 0s.

**Solution column count**: 64

```python
def eye(j: int) -> TT["i", "i"]:
    return where(arange(j)[:, None] == arange(j)[None, :], 1, 0)
```

### Puzzle 6 - triu

Compute [triu](https://numpy.org/doc/stable/reference/generated/numpy.triu.html) - the upper triangular matrix.

**Intuition**: Similar to eye but instead of &equals; we use &le; between the row and column indices to include the entries above the diagonal.

**Solution column count**: 64

```python
def triu(j: int) -> TT["j", "j"]:
    return where(arange(j)[:, None] <= arange(j)[None, :], 1, 0)
```

### Puzzle 7 - cumsum

Compute [cumsum](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html) - the cumulative sum.

**Intuition**: Dot product of a vector with the upper triangular matrix makes each position the sum of all previous positions.

**Solution column count**: 45

```python
def cumsum(a: TT["i"]) -> TT["i"]:
    return (a[None, :] @ triu(a.shape[0]))[0]
```

### Puzzle 8 - diff

Compute [diff](https://numpy.org/doc/stable/reference/generated/numpy.diff.html) - the running difference.

**Intuition**: Subtract the vector shifted right by one and preserve the first element preserved.

**Solution column count**: 60

```python
def diff(a: TT["i"], i: int) -> TT["i"]:
    return where(arange(i) == 0, a[0], a - a[arange(i) - 1])
```

### Puzzle 9 - vstack

Compute [vstack](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html) - the matrix of two vectors

**Intuition**: Broadcast arange(2) vertically and compare with 0 to select the first or second vector.

**Solution column count**: 65

```python
def vstack(a: TT["i"], b: TT["i"]) -> TT[2, "i"]:
    return where(arange(2)[:, None] == 0, a[None, :], b[None, :])
```

### Puzzle 10 - roll

Compute [roll](https://numpy.org/doc/stable/reference/generated/numpy.roll.html) - the vector shifted 1 circular position.

**Intuition**: Add 1 to every index modulo length, producing the circularly shifted view.

**Solution column count**: 42

```python
def roll(a: TT["i"], i: int) -> TT["i"]:
    return a[(arange(i) + 1) % a.shape[0]]
```

### Puzzle 11 - flip

Compute [flip](https://numpy.org/doc/stable/reference/generated/numpy.flip.html) - the reversed vector

**Intuition**: Index from the back: i - arange(i) - 1 reverses order.

**Solution column count**: 31

```python
def flip(a: TT["i"], i: int) -> TT["i"]:
    return a[i - arange(i) - 1]
```

### Puzzle 12 - compress

Compute [compress](https://numpy.org/doc/stable/reference/generated/numpy.compress.html) - keep only masked entries (left-aligned).

**Intuition**: Convert boolean mask to running indices with cumsum, then scatter original values into a left-aligned result.

**Solution column count**: 76

```python
def compress(g: TT["i", bool], v: TT["i"], i:int) -> TT["i"]:
    return v @ where(g[:, None], arange(i) == (cumsum(1*g) - 1)[:, None], 0)
```

### Puzzle 13 - pad_to

Compute pad_to - eliminate or add 0s to change size of vector.

**Intuition**: Treat `i ≤ j` as a boolean keep-mask and reuse compress to either clip or 0-pad up to length j.

**Solution column count**: 41

```python
def pad_to(a: TT["i"], i: int, j: int) -> TT["j"]:
    return compress(arange(i) <= j, a, j)
```

### Puzzle 14 - sequence_mask

Compute [sequence_mask](https://www.tensorflow.org/api_docs/python/tf/sequence_mask) - pad out to length per batch.

**Intuition**: Compare column indices to per-row lengths; the "&lt;" mask zeros out positions past each row's length.

**Solution column count**: 72

```python
def sequence_mask(values: TT["i", "j"], length: TT["i", int]) -> TT["i", "j"]:
    return (arange(values.shape[1])[None, :] < length[:, None]) * values
```

### Puzzle 15 - bincount

Compute [bincount](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html) - count number of times an entry was seen.

**Intuition**: For every element create a one-hot over bins and sum 1s across rows, counting occurrences.

**Solution column count**: 75

```python
def bincount(a: TT["i"], j: int) -> TT["j"]:
    return ones(a.shape[0]) @ where(a[:, None] == arange(j)[None, :], 1, 0)
```

### Puzzle 16 - scatter_add

Compute [scatter_add](https://pytorch-scatter.readthedocs.io/en/1.3.0/functions/add.html) - add together values that link to the same location.

**Intuition**: Build a one-hot matrix from link, then multiply values through it so identical destinations sum down the columns.

**Solution column count**: 68

```python
def scatter_add(values: TT["i"], link: TT["i"], j: int) -> TT["j"]:
    values @ where(link[:, None] == arange(j)[None, :], 1, 0)
```

### Puzzle 17 - flatten

Compute [flatten](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html)

**Intuition**: Map each flattened index back to its 2-D coordinates (row = //, col = %) and gather.

**Solution column count**: 45

```python
def flatten(a: TT["i", "j"], i:int, j:int) -> TT["i * j"]:
    return a[arange(i*j)//j, arange(i*j) % j]
```

### Puzzle 18 - linspace

Compute [linspace](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)

**Intuition**: Scale arange(n)/(n-1) between 0-1 and stretch/shift by (j-i) and i to land evenly between the endpoints.

**Solution column count**: 61

```python
def linspace(i: TT[1], j: TT[1], n: int) -> TT["n", float]:
    return i + (j - i) * (arange(n) / where(n > 1, n - 1, 1))
```

### Puzzle 19 - heaviside

Compute [heaviside](https://numpy.org/doc/stable/reference/generated/numpy.heaviside.html)

**Intuition**: For x &ne; 0, x>0 already gives 1 or 0; for x==0 fall back to the provided b.

**Solution column count**: 33

```python
def heaviside(a: TT["i"], b: TT["i"]) -> TT["i"]:
    return where(a == 0, b, a>0)
```

### Puzzle 20 - repeat (1d)

Compute [repeat](https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html)

**Intuition**: Create a column of ones of height d and outer-product with the row vector to duplicate it d times.

**Solution column count**: 40

```python
def repeat(a: TT["i"], d: TT[1]) -> TT["d", "i"]:
    return ones(d)[:, None] @ a[None, :]
```

### Puzzle 21 - bucketize

Compute [bucketize](https://pytorch.org/docs/stable/generated/torch.bucketize.html)

**Intuition**: One-hot test each value against all boundaries (v &ge; b), then sum the 1s along the boundary axis to get its bucket index.

**Solution column count**: 85

```python
def bucketize(v: TT["i"], boundaries: TT["j"]) -> TT["i"]:
    return where(v[:, None] >= boundaries[None, :], 1, 0) @ ones(boundaries.shape[0])
```
