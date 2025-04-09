# simpy — Lightweight NumPy-like Library for Numerical Computations

## 📌 Overview

**simpy** is a simple implementation of a numerical array library inspired by **NumPy**. It provides basic functionality for working with multi-dimensional arrays, including:

- Element-wise operations  
- Broadcasting  
- Indexing and slicing  
- Common array manipulations

> ⚠️ This project is under active development, and the current implementation is not complete.

The goal of this project is to understand how numpy works and all its related methods, functions and properties.

---

## ✨ Features (Current Implementation)

### 🧱 Array Creation

- **Initialization**: Create arrays from nested lists with optional data type enforcement (`int`, `float`, or `bool`).
- **Static Methods**:
  - `simpy.arange(start, stop, step)` — Create a 1D array with evenly spaced values.
  - `simpy.zeros(shape)` — Create an array filled with zeros.
  - `simpy.ones(shape)` — Create an array filled with ones.
  - `simpy.empty(shape)` — Create an uninitialized array (random values).
  - `simpy.eye(N, M=None, k=0)` — Create a 2D identity matrix with ones on the diagonal.

---

### 📐 Array Properties

- `shape` — Dimensions of the array.
- `dtype` — Data type of the array elements.
- `ndim` — Number of dimensions.
- `size` — Total number of elements.

---

### ➕ Operations

- **Element-wise Arithmetic**: Addition (`+`), subtraction (`-`), multiplication (`*`), division (`/`), and comparison operators (`==`, `!=`, `<`, `<=`, `>`, `>=`) with broadcasting support.
- **Indexing and Slicing**: Access elements using integer indices, slices, or tuples.
- **Matrix multiplication**: Using matmul operator (@) you can do multiplication of 2 matrices. 

---

### 🛠️ Utility Methods

- `min(axis=None)` — Compute the minimum value along a specified axis or globally.
- `max(axis=None)` — Compute the maximum value along a specified axis or globally.
- `flatten()` — Flatten the array into a 1D array.
- `fill(value)` — Fill the array with a scalar value.

---
### 📊 Statistical & Mathematical Methods
- `sum()` — Compute the sum of array elements.
- `prod()` — Compute the product of array elements.
- `diagonal()` — Extract the diagonal elements of 2D array.
- `mean(axis=None)` — Compute the arithmetic mean along the specified axis.
- `std(axis=None, ddof=0)` — Compute the standard deviation along the specified axis.
- `cov(rowvar=True, ddof=None)` — Compute the covariance matrix.
- `corrcoef(rowvar=True)` — Compute Pearson correlation coefficients.
- `median(axis=None)` — Compute the median along the specified axis

---

## 🧪 Installation

This project is not yet published as a package. To use it:

```bash
git clone https://github.com/whiteprincewithobsession/self_written_numpy.git
cd self_written_numpy
```

Then, import the `simpy` class in your Python script:

```python
from simpy import simpy

arr = simpy([[1, 2], [3, 4]])
print(arr)
```

---


## 🧾 Usage Examples

### 🔸 Creating Arrays

```python
import simpy

# Create a 2x2 array
arr = simpy([[1, 2], [3, 4]])
print(arr)

# Create a 1D array with evenly spaced values
arr = simpy.arange(0, 1, 0.3)
print(arr)

# Create a 3x3 array filled with zeros
zeros = simpy.zeros([3, 3])
print(zeros)
```

---

### 🔸 Element-wise Operations

```python
a = simpy([[1, 2], [3, 4]])
b = simpy([[5, 6], [7, 8]])

# Addition
print(a + b)

# Multiplication by a scalar
print(a * 2)

# Comparison
print(a < 3)
```

---

### 🔸 Indexing and Slicing

```python
arr = simpy([[1, 2, 3], [4, 5, 6]])

# Access a single element
print(arr[0, 1])  # Output: 2

# Slice a column
print(arr[:, 1])  # Output: [2, 5]
```

---

### 🔸 Utility Methods

```python
arr = simpy([[5, 3], [2, 8]])

# Global minimum
print(arr.min())

# Minimum along axis 0
print(arr.min(axis=0))

# Flatten the array
print(arr.flatten())
```

---

### 🔸 Array Methods

```python
arr = simpy([[1, 2, 3], [4, 5, 6]])

# Sum of all elements
print(arr.sum())  # Output: 21

# Product along axis 1
print(arr.prod(axis=1))  # Output: [6, 120]

# Extract diagonal elements
print(arr.diagonal())  # Output: [1, 5]

# Compute mean
print(arr.mean())  # Output: 3.5
print(arr.mean(axis=0))  # Output: [2.5, 3.5, 4.5]

# Standard deviation
print(arr.std())  # Output: 1.707...

# Matrix multiplication
a = simpy([[1, 2], [3, 4]])
b = simpy([[5, 6], [7, 8]])
print(a @ b)  # Output: [[19, 22], [43, 50]]

# Median
print(arr.median())  # Output: 3.5

# Correlation coefficients
corr = arr.corrcoef()
print(corr)  # Output: correlation matrix
```
---

## 🚧 Project Status

The project is still a work in progress. The following features are planned:

- ✅ **Advanced Indexing**: Boolean masks, fancy indexing
- ✅ **Matrix Operations**: Dot product, transpose, etc.
- ⚙️ **Different operations with arrays**: diagonal, std, mean, median, etc.
- ⚠️ **Error Handling**: Better error messages and robustness

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Submit a pull request with a clear description of your changes

---

## 📬 Contact

For questions or feedback, feel free to reach out:

- **Email**: yarik_02022005@mail.ru  
- **GitHub**: [@whiteprincewithobsession](https://github.com/whiteprincewithobsession)

---

Thank you for your interest in **simpy**! 💙
