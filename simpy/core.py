from typing import List, Tuple, Union, Callable, Any, Optional
from math import prod, ceil
import random


class simpy:
    """A simple numpy.array implementation for numerical computations.
    
    Attributes:
        data: The underlying nested list storing array elements
        shape: The dimensions of the array as a list of integers
        dtype: The data type of array elements ('int' or 'float')
        ndim: The number of array dimensions (axes)
        size: The number of elements in the array (production of shape)
    """
    
    def __init__(self, data: List, dtype: Optional[str] = None, display_decimals: int = 6) -> None:
        """Initialize a new simpy array.
        
        Args:
            data: Nested list containing array elements
            dtype: Optional data type ('int' or 'float') to force conversion
        """
        self.data = self._convert_data(data, dtype)
        self.shape = self._compute_shape(data)
        self.dtype = dtype if dtype else self._infer_dtype(data)
        self.size = prod(self.shape)
        self._display_decimals = display_decimals

    @property
    def ndim(self) -> int:
        """Number of array dimensions (axes).
        
        Examples:
            >>> a = simpy([1, 2, 3])
            >>> a.ndim
            1
        """
        return len(self.shape)

    @property
    def display_decimals(self) -> int:
        """Number of decimal places to show for float values.
        
        Examples:
            >>> arr = simpy([1.23456789], display_decimals=2)
            >>> print(arr)
            simpy(array([1.23]), shape=[1], dtype=float, ndim=1)
        """
        return self._display_decimals


    def _compute_shape(self, data: List) -> List[int]:
        """Calculate the shape dimensions of the array.
        
        Returns:
            List of integers representing array dimensions
        """
        shape = []
        curr = data
        while isinstance(curr, list):
            shape.append(len(curr))
            curr = curr[0] if len(curr) > 0 else None
        return shape
    
    def _convert_data(self, data: List, dtype: Optional[str] = None) -> List:
        """Convert array elements to specified data type.
        
        Args:
            data: Input nested list
            dtype: Target data type ('int' or 'float')
            
        Returns:
            New nested list with converted values
            
        Raises:
            ValueError: If invalid dtype is specified
        """
        if dtype not in [None, "int", "float", "bool"]:
            raise ValueError("Invalid data type specified in dtype field.")
        
        if dtype is None:
            return data
        
        def recursive_convert(item: Union[List, Any]) -> Union[List, int, float, bool]:
            if isinstance(item, list):
                return [recursive_convert(x) for x in item]
            if dtype == "int":
                return int(item)
            elif dtype == "float":
                return float(item)
            elif dtype == "bool":
                return bool(item)
            raise TypeError(f"{item} has incompatible format.")
            
        return recursive_convert(data)
    
    def _convert_value(self, value: Union[int, float]) -> Union[int, float]:
        """Convert value to array's dtype."""
        if self.dtype == "int":
            return int(value)
        return float(value)
        
    def _infer_dtype(self, data: List) -> str:
        """Determine the data type of array elements.
        
        Returns:
            'int' if all elements are integers, 'float' otherwise
            
        Raises:
            ValueError: If unable to determine data type
        """
        def recursive_check(item: Union[List, Any]) -> Union[str, None]:
            if isinstance(item, list):
                return recursive_check(item[0]) if item else None
            if isinstance(item, bool):
                return "bool"
            return "int" if isinstance(item, int) else "float" if isinstance(item, float) else None

        inferred_dtype = recursive_check(data)
        if inferred_dtype is None:
            raise ValueError("Unable to determine data type. Specify dtype explicitly.")
        return inferred_dtype

    def __str__(self) -> str:
        """Return a comprehensive string representation of the array showing all key properties.
        
        The representation includes:
        - The class name (simpy)
        - The actual array data
        - The shape of the array
        - The data type (dtype)
        - The number of dimensions (ndim)
        
        Returns:
            str: Formatted string showing array contents and properties
            
        Example:
            Basic 1D array:
            >>> a = simpy([1, 2, 3])
            >>> str(a)
            'simpy(array([1, 2, 3]), shape=[3], dtype=int, ndim=1)'
        """
        def format_value(value):
            if isinstance(value, float):
                return f"{value:.{self.display_decimals}f}"
            return str(value)

        def format_array(data, indent=0):
            if isinstance(data, list):
                if not data: 
                    return "[]"
                if isinstance(data[0], list):
                    prefix = " " * indent
                    lines = [f"{prefix}[{format_array(sub, indent + 1)}]" for sub in data]
                    return "\n" + "\n".join(lines)
                else:
                    elements = " ".join(format_value(x) for x in data)
                    return f"[{elements}]"
            else: 
                return format_value(data)

        formatted_data = format_array(self.data)
        return f"simpy({formatted_data},\nshape={self.shape}, dtype={self.dtype}, ndim={self.ndim})"
        
    def __repr__(self) -> str:
        """Return the official string representation that could recreate the array.
        
        The output follows Python syntax that could be evaluated to recreate
        an equivalent array. Includes all necessary parameters.
        
        Returns:
            str: Valid Python expression to recreate the array
            
        Example:
            >>> arr = simpy([[1, 2], [3, 4]])
            >>> repr(arr)
            'simpy([[1, 2], [3, 4]], dtype="int")'
        """
        dtype_str = f', dtype="{self.dtype}"' if self.dtype else ''
        return f'simpy({self.data}{dtype_str})'

    def _elementwise_op(self, other: Union[int, float, 'simpy'], op: Callable, compare: bool = False) -> 'simpy':
        """Perform element-wise operation with broadcasting support."""
        if isinstance(other, (int, float)):
            return self._apply_scalar(other, op, compare)
        
        elif isinstance(other, simpy):
            try:
                broadcast_shape = self._broadcast_shapes(self.shape, other.shape)
                
                a_data = self._broadcast_array(self.data, self.shape, broadcast_shape)
                b_data = other._broadcast_array(other.data, other.shape, broadcast_shape)
                
                result = self._apply_broadcasted_arrays(a_data, b_data, op)
                
            except ValueError as e:
                raise ValueError(f"Operands could not be broadcast: {self.shape} vs {other.shape}") from e
            
            return simpy(result, dtype='bool') if compare else simpy(result)
        
        else:
            raise TypeError("Unsupported operand type")

    def _apply_broadcasted_arrays(self, a: List, b: List, op: Callable) -> List:
        """Recursively apply operation to broadcasted arrays."""
        if isinstance(a[0], list) or isinstance(b[0], list):
            return [self._apply_broadcasted_arrays(a_sub, b_sub, op) 
                    for a_sub, b_sub in zip(a, b)]
        return [op(x, y) for x, y in zip(a, b)]

    def _broadcast_shapes(self, shape1: List[int], shape2: List[int]) -> List[int]:
        """Calculate broadcasted shape for two arrays."""
        max_ndim = max(len(shape1), len(shape2))
        padded_shape1 = [1] * (max_ndim - len(shape1)) + shape1
        padded_shape2 = [1] * (max_ndim - len(shape2)) + shape2
        
        result_shape = []
        for s1, s2 in zip(padded_shape1, padded_shape2):
            if s1 != 1 and s2 != 1 and s1 != s2:
                raise ValueError(f"Shapes {shape1} and {shape2} are not broadcastable")
            result_shape.append(max(s1, s2))
            
        return result_shape

    def _broadcast_array(self, data: List, original_shape: List[int], target_shape: List[int]) -> List:
        """Broadcast array to target shape."""
        while len(original_shape) < len(target_shape):
            data = [data]
            original_shape = [1] + original_shape
        
        for axis in range(len(target_shape)):
            if original_shape[axis] != target_shape[axis]:
                if original_shape[axis] != 1:
                    raise ValueError("Cannot broadcast")
                data = data * target_shape[axis]
        
        return data

    def _apply_scalar(self, scalar: Union[int, float], op: Callable, compare: bool = False) -> 'simpy':
        """Apply operation between array and scalar.
        
        Args:
            scalar: Numeric value
            op: Operation function
            
        Returns:
            New simpy array with operation results
        """
        def recursive_apply(data: List) -> List:
            if isinstance(data[0], list):
                return [recursive_apply(sub) for sub in data]
            return [op(x, scalar) for x in data]
        return simpy(recursive_apply(self.data))

    def _apply_array(self, other: 'simpy', op: Callable) -> 'simpy':
        """Apply operation between two arrays element-wise.
        
        Args:
            other: Another simpy array
            op: Operation function
            
        Returns:
            New simpy array with operation results
        """
        def recursive_apply(a: List, b: List) -> List:
            if isinstance(a[0], list):
                return [recursive_apply(sub_a, sub_b) for sub_a, sub_b in zip(a, b)]
            return [op(x, y) for x, y in zip(a, b)]
        return simpy(recursive_apply(self.data, other.data))
    
    def __add__(self, other: Union[int, float, 'simpy']) -> 'simpy':
        """Add another array or scalar to this array.
        
        Args:
            other: Array or scalar value to add
            
        Returns:
            simpy: New array with element-wise sums
            
        Example:
            >>> a = simpy([[1, 2], [3, 4]])
            >>> b = simpy([[5, 6], [7, 8]])
            >>> a + b
            array([[6, 8], [10, 12]])
            
            >>> a + 2
            array([[3, 4], [5, 6]])
        """
        return self._elementwise_op(other, lambda x, y: x + y)

    def __sub__(self, other: Union[int, float, 'simpy']) -> 'simpy':
        """Subtract another array or scalar from this array.
        
        Args:
            other: Array or scalar value to subtract
            
        Returns:
            simpy: New array with element-wise differences
            
        Example:
            >>> a = simpy([[5, 6], [7, 8]])
            >>> b = simpy([[1, 2], [3, 4]])
            >>> a - b
            array([[4, 4], [4, 4]])
            
            >>> a - 1
            array([[4, 5], [6, 7]])
        """
        return self._elementwise_op(other, lambda x, y: x - y)

    def __mul__(self, other: Union[int, float, 'simpy']) -> 'simpy':
        """Multiply this array by another array or scalar.
        
        Args:
            other: Array or scalar value to multiply by
            
        Returns:
            simpy: New array with element-wise products
            
        Example:
            >>> a = simpy([[1, 2], [3, 4]])
            >>> b = simpy([[2, 2], [2, 2]])
            >>> a * b
            array([[2, 4], [6, 8]])
            
            >>> a * 3
            array([[3, 6], [9, 12]])
        """
        return self._elementwise_op(other, lambda x, y: x * y)
    
    def __eq__(self, other: Union[int, float, 'simpy']) -> 'simpy':
        """Element-wise equality comparison.
        
        Args:
            other: Scalar value or simpy array to compare with
            
        Returns:
            simpy: New boolean array where each element is True 
                where elements are equal
                
        Example:
            >>> a = simpy([[1, 2], [3, 4]])
            >>> b = simpy([[1, 5], [3, 4]])
            >>> a == b
            array([[ True, False], 
                [ True,  True]])
                
            >>> a == 3
            array([[False, False], 
                [ True, False]])
        """
        return self._elementwise_op(other, lambda x, y: x == y, compare=True)

    def __ne__(self, other: Union[int, float, 'simpy']) -> 'simpy':
        """Element-wise inequality comparison.
        
        Args:
            other: Scalar value or simpy array to compare with
            
        Returns:
            simpy: New boolean array where each element is True 
                where elements are not equal
                
        Example:
            >>> a = simpy([[1, 2], [3, 4]])
            >>> b = simpy([[1, 5], [3, 4]])
            >>> a != b
            array([[False,  True], 
                [False, False]])
                
            >>> a != 3
            array([[ True,  True], 
                [False,  True]])
        """
        return self._elementwise_op(other, lambda x, y: x != y, compare=True)

    def __lt__(self, other: Union[int, float, 'simpy']) -> 'simpy':
        """Element-wise less than comparison.
        
        Args:
            other: Scalar value or simpy array to compare with
            
        Returns:
            simpy: New boolean array where each element is True 
                where elements are less than compared value
                
        Example:
            >>> a = simpy([[1, 5], [3, 4]])
            >>> b = simpy([[2, 3], [2, 5]])
            >>> a < b
            array([[ True, False], 
                [False,  True]])
                
            >>> a < 3
            array([[ True, False], 
                [False, False]])
        """
        return self._elementwise_op(other, lambda x, y: x < y, compare=True)

    def __le__(self, other: Union[int, float, 'simpy']) -> 'simpy':
        """Element-wise less than or equal comparison.
        
        Args:
            other: Scalar value or simpy array to compare with
            
        Returns:
            simpy: New boolean array where each element is True 
                where elements are less than or equal to compared value
                
        Example:
            >>> a = simpy([[1, 2], [3, 4]])
            >>> b = simpy([[1, 3], [2, 4]])
            >>> a <= b
            array([[ True,  True], 
                [False,  True]])
                
            >>> a <= 3
            array([[ True,  True], 
                [ True, False]])
        """
        return self._elementwise_op(other, lambda x, y: x <= y, compare=True)

    def __gt__(self, other: Union[int, float, 'simpy']) -> 'simpy':
        """Element-wise greater than comparison.
        
        Args:
            other: Scalar value or simpy array to compare with
            
        Returns:
            simpy: New boolean array where each element is True 
                where elements are greater than compared value
                
        Example:
            >>> a = simpy([[2, 3], [4, 1]])
            >>> b = simpy([[1, 2], [3, 4]])
            >>> a > b
            array([[ True,  True], 
                [ True, False]])
                
            >>> a > 2
            array([[False,  True], 
                [ True, False]])
        """
        return self._elementwise_op(other, lambda x, y: x > y, compare=True)

    def __ge__(self, other: Union[int, float, 'simpy']) -> 'simpy':
        """Element-wise greater than or equal comparison.
        
        Args:
            other: Scalar value or simpy array to compare with
            
        Returns:
            simpy: New boolean array where each element is True 
                where elements are greater than or equal to compared value
                
        Example:
            >>> a = simpy([[2, 3], [4, 1]])
            >>> b = simpy([[2, 2], [4, 4]])
            >>> a >= b
            array([[ True,  True], 
                [ True, False]])
                
            >>> a >= 3
            array([[False,  True], 
                [ True, False]])
        """
        return self._elementwise_op(other, lambda x, y: x >= y, compare=True)

    def __truediv__(self, other: Union[int, float, 'simpy']) -> 'simpy':
        """Divide this array by another array or scalar.
        
        Args:
            other: Array or scalar value to divide by
            
        Returns:
            simpy: New array with element-wise quotients
            
        Example:
            >>> a = simpy([[4, 6], [8, 10]])
            >>> b = simpy([[2, 3], [4, 5]])
            >>> a / b
            array([[2.0, 2.0], [2.0, 2.0]])
            
            >>> a / 2
            array([[2.0, 3.0], [4.0, 5.0]])
        """
        return self._elementwise_op(other, lambda x, y: x / y)
        
    def __getitem__(self, indices: Union[int, slice, Tuple]) -> 'simpy':
        """Get subset of array using indexing or slicing.
        
        Args:
            indices: Integer index, slice object, or tuple of indices
            
        Returns:
            simpy: New array with selected elements
            
        Raises:
            IndexError: If index is out of bounds
            TypeError: If invalid index type is used
            
        Examples:
            >>> a = simpy([[1, 2], [3, 4]])
            >>> a[0]
            array([1, 2])
            
            >>> a[0, 1]
            array([2])
            
            >>> a[:, 0]
            array([1, 3])
        """
        if not isinstance(indices, tuple):
            indices = (indices,)
    
        if len(indices) > self.ndim:
            raise IndexError(f"Too many indices for array with {self.ndim} dimensions")
        
        def get_item(data: List, idx: Tuple, dim: int = 0) -> Union[List, float, int]:
            if not idx:
                return data
                
            current_idx = idx[0]
            size = len(data)
            
            if isinstance(current_idx, int):
                if current_idx >= size or current_idx < -size:
                    raise IndexError(
                        f"Index {current_idx} is out of bounds for axis {dim} with size {size}"
                    )
                current_idx = current_idx if current_idx >= 0 else size + current_idx
                result = data[current_idx]
                
                if len(idx) == 1:
                    return result
                return get_item(result, idx[1:], dim + 1)
            
            elif isinstance(current_idx, slice):
                start, stop, step = current_idx.indices(size)
                sliced = data[start:stop:step]
                
                if not idx[1:]:
                    return sliced
                return [get_item(sub, idx[1:], dim + 1) for sub in sliced]
            
            else:
                raise TypeError(f"Invalid index type: {type(current_idx)}")
        
        try:
            result = get_item(self.data, indices)
            
            if not isinstance(result, list):
                return result
                
            if isinstance(result[0], list):
                return simpy(result)
                
            return simpy(result)
            
        except IndexError as e:
            raise IndexError(f"Failed to index array: {str(e)}") from e
        #05.04 FIX error IndexError: Failed to index array: Index 1 is out of bounds for axis 0 with size 1
        #06.04 FIXED
    
    @staticmethod
    def arange(start: Union[int, float], stop: Union[int, float], 
               step: Union[int, float] = 1, dtype: Optional[str] = None) -> 'simpy':
        """Create a 1D array with evenly spaced values within a given interval.
        
        Generates values from start (inclusive) to stop (exclusive),
        incrementing by step.
        
        Args:
            start: Start of the interval (inclusive)
            stop: End of the interval (exclusive)
            step: Spacing between values (default 1)
            dtype: Output data type ('int' or 'float')
            
        Returns:
            simpy: New 1D array
            
        Raises:
            ValueError: If step is zero or invalid direction
            
        Example:        
            >>> simpy.arrange(0, 1, 0.3)
            array([0.0, 0.3, 0.6, 0.9])
        """
        if step == 0:
            raise ValueError("step cannot be 0")
            
        if stop is None:
            stop = start
            start = 0
            
        num_elements = ceil((stop - start) / step)
        
        values = []
        current = start
        for _ in range(num_elements):
            if (step > 0 and current >= stop) or (step < 0 and current <= stop):
                break
            values.append(current)
            current += step
            
        return simpy(values, dtype=dtype)
    
    @staticmethod
    def zeros(shape: List[int], dtype: str = "float") -> 'simpy':
        """Create an array of given shape filled with zeros.
        
        Args:
            shape: Dimensions of the array (e.g., [2, 3] for 2x3 matrix)
            dtype: Data type ('int' or 'float')
            
        Example:
            >>> simpy.zeros([2, 3])
            array([[0, 0, 0], [0, 0, 0]])
        """
        def generate(dims: List[int], value: Union[int, float]):
            if len(dims) == 1:
                return [value] * dims[0]
            return [generate(dims[1:], value) for _ in range(dims[0])]
        
        value = 0 if dtype == "int" else 0.0
        return simpy(generate(shape, value), dtype=dtype)

    @staticmethod
    def ones(shape: List[int], dtype: str = "float") -> 'simpy':
        """Create an array of given shape filled with ones.
        
        Args:
            shape: Dimensions of the array (e.g., [2, 3] for 2x3 matrix)
            dtype: Data type ('int' or 'float')
            
        Example:
            >>> simpy.zeros([3, 3])
            array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        """        
        def generate(dims: List[int], value: Union[int, float]):
            if len(dims) == 1:
                return [value] * dims[0]
            return [generate(dims[1:], value) for _ in range(dims[0])]
        
        value = 1 if dtype == "int" else 1.0
        return simpy(generate(shape, value), dtype=dtype)

    @staticmethod
    def empty(shape: List[int], dtype: str = "float") -> 'simpy':
        """Create an uninitialized array of given shape (values are unpredictable)."""
        def generate(dims: List[int]):
            if len(dims) == 1:
                return [random.uniform(0, 100) for _ in range(dims[0])]
            return [generate(dims[1:]) for _ in range(dims[0])]
        
        return simpy(generate(shape), dtype=dtype)

    def fill(self, value: Union[int, float]) -> None:
        """Fill the array with a scalar value.
        
        Args:
            value: Value to fill the array with
            
        Example:
            >>> arr = simpy.zeros([2, 2])
            >>> arr.fill(5)
            array([[5, 5], [5, 5]])
        """
        def recursive_fill(data: List, fill_value: Union[int, float]):
            if isinstance(data[0], list):
                return [recursive_fill(sub, fill_value) for sub in data]
            return [self._convert_value(fill_value)] * len(data)
        
        self.data = recursive_fill(self.data, value)
        self.dtype = "int" if isinstance(value, int) else "float"

    @staticmethod
    def eye(N: int, M: Optional[int] = None, k: int = 0, dtype: str = 'float') -> 'simpy':
        """Return a 2D array with ones on the diagonal and zeros elsewhere.
        
        Args:
            N: Number of rows in the output matrix
            M: Number of columns (optional, defaults to N)
            k: Index of the diagonal (0 = main diagonal, >0 = upper, <0 = lower)
            dtype: Data type of the resulting array ('int' or 'float')
            
        Returns:
            simpy: 2D array of shape (N, M) with ones on the specified diagonal
            
        Examples:
            Identity matrix (3x3):
            >>> simpy.eye(3)
            array([[1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0]])

            Rectangular matrix with diagonal offset:
            >>> simpy.eye(2, 3, k=1)
            array([[0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0]])

        """
        if M is None:
            M = N

        matrix = []
        for i in range(N):
            row = []
            for j in range(M):
                row.append(1 if j == i + k else 0)
            matrix.append(row)

        return simpy(matrix, dtype=dtype) 

    def min(self, axis: Optional[int] = None) -> Union['simpy', float, int]:
        """Return the minimum of the array along the specified axis.
        
        When axis is None (default), returns the global minimum of all elements.
        When axis is specified, reduces the array's dimensionality by 1, 
        Args:
            axis: Axis along which to operate. None (default) computes global min.
                Valid values: 0 (rows), 1 (columns), etc. up to ndim-1

        Returns:
            If axis is None: scalar minimum value (int/float)
            If axis is specified: simpy array with reduced dimensions

        Raises:
            ValueError: If axis is outside valid range [0, ndim-1]

        Examples:
            Global minimum (axis=None):
            >>> a = simpy([[5, 3], [2, 8]])
            >>> a.min()
            2

            1D array behavior:
            >>> b = simpy([7, 1, 5])
            >>> b.min(axis=0)
            simpy(array([1]), shape=[1], dtype=int, ndim=1)
        """
        if axis is None:
            flat = self.flatten().data
            return min(flat)
        else:
            if axis < 0 or axis >= self.ndim:
                raise ValueError(f"axis {axis} is out of bounds for array with {self.ndim} dimensions")
            reduced_data = self._reduce_along_axis(self.data, axis, min)
            squeezed_data = self._squeeze(reduced_data)
            return simpy(squeezed_data)

    def max(self, axis: Optional[int] = None) -> Union['simpy', float, int]:
        """Return the maximum of the array along the specified axis.
        
        When axis is None (default), returns the global maximum of all elements.
        When axis is specified, reduces the array's dimensionality by 1, 
        Args:
            axis: Axis along which to operate. None (default) computes global max.
                Valid values: 0 (rows), 1 (columns), etc. up to ndim-1

        Returns:
            If axis is None: scalar maximum value (int/float)
            If axis is specified: simpy array with reduced dimensions

        Raises:
            ValueError: If axis is outside valid range [0, ndim-1]

        Examples:
            Global maximum (axis=None):
            >>> a = simpy([[5, 3], [2, 8]])
            >>> a.max()
            8

            1D array behavior:
            >>> b = simpy([7, 1, 5])
            >>> b.max(axis=0)
            simpy(array([7]), shape=[1], dtype=int, ndim=1)
        """
        if axis is None:
            flat = self.flatten().data
            return max(flat)
        else:
            if axis < 0 or axis >= self.ndim:
                raise ValueError(f"axis {axis} is out of bounds for array with {self.ndim} dimensions")
            reduced_data = self._reduce_along_axis(self.data, axis, max)
            squeezed_data = self._squeeze(reduced_data)
            return simpy(squeezed_data)

    def _reduce_along_axis(self, data, axis, reduce_func):
        """Helper function to recursively reduce along the specified axis."""
        if axis == 0:
            if not data:
                return []
            if not isinstance(data[0], list):
                return [reduce_func(data)]
            return [reduce_func([sub[i] for sub in data]) for i in range(len(data[0]))]
        else:
            return [self._reduce_along_axis(sub, axis-1, reduce_func) for sub in data]

    def _squeeze(self, data):
        """Remove dimensions of size 1 from the data structure."""
        if isinstance(data, list):
            if len(data) == 1:
                element = self._squeeze(data[0])
                if isinstance(element, list):
                    return element
                else:
                    return [element]
            else:
                squeezed = [self._squeeze(sub) for sub in data]
                if all(not isinstance(sub, list) for sub in squeezed):
                    return squeezed
                else:
                    return squeezed
        else:
            return data

    def flatten(self) -> 'simpy':
        """Return a 1D simpy array with all elements of the original array."""
        def recursive_flatten(data):
            if isinstance(data, list):
                return [item for sublist in data for item in recursive_flatten(sublist)]
            else:
                return [data]
        flattened_data = recursive_flatten(self.data)
        return simpy(flattened_data)

    def sum(self) -> Union[int, float]:
        """Return the sum of all elements in the array.
    
        Recursively traverses nested lists (arrays of arbitrary dimensions) 
        and sums all numeric values. Non-list elements are treated as numbers.
        
        Returns:
            Sum of all elements in the array. Return type depends on input:
            - int if all elements are integers
            - float if any element is float or mixed types are present
            
        Examples:
            Flat array:
            >>> a = simpy([1, 2, 3])
            >>> a.sum()
            6
            
            Nested arrays:
            >>> b = simpy([[1, 2], [3, [4, 5]]])
            >>> b.sum()
            15
            
            Mixed types:
            >>> c = simpy([1.5, [2, 3]])
            >>> c.sum()
            6.5
        
        Notes:
            - For empty arrays, returns 0
            - Raises TypeError if non-numeric elements are encountered
            - Raises ValueError if array is empty
        """
        if not self.data:
            raise ValueError("Cannot compute sum of empty array")
    
        total = 0
        has_float = False
        
        def recursive_sum(data):
            nonlocal total, has_float
            if isinstance(data, list):
                for item in data:
                    recursive_sum(item)
            else:
                if not isinstance(data, (int, float)):
                    raise TypeError(f"Non-numeric element found: {data}")
                if isinstance(data, float):
                    has_float = True
                total += data
        
        try:
            recursive_sum(self.data)
            return float(total) if has_float else int(total)
        except TypeError as e:
            raise TypeError(f"Array contains non-numeric elements: {e}")
    
    def prod(self) -> Union[int, float]:
        """Return the product of all elements in the array.
    
        Recursively traverses nested lists (arrays of arbitrary dimensions)
        and computes the product of all numeric values. Non-list elements
        are treated as numbers.
        
        Returns:
            Product of all elements in the array. Return type depends on input:
            - int if all elements are integers
            - float if any element is float or mixed types are present
            
        Examples:
            Flat array:
            >>> a = simpy([2, 3, 4])
            >>> a.prod()
            24
            
            Nested arrays:
            >>> b = simpy([[2, 3], [4, [5, 2]]])
            >>> b.prod()
            240
            
            Mixed types:
            >>> c = simpy([2.5, [2, 3]])
            >>> c.prod()
            15.0
            
        Notes:
            - For empty arrays, returns 1 (multiplicative identity)
            - Raises TypeError if non-numeric elements are encountered
            - Raises ValueError if array is empty
            - Special cases:
                - Any zero element will make the product zero
                - Handles negative numbers correctly
        """
        total = 1
        flag = False
        def recursive_check(data):
            nonlocal flag
            if isinstance(data, Union[float, int]):
                flag = True
                return
            else:
                for item in data:
                    recursive_check(item)

        if not flag:
            raise ValueError("Empty array. The sum can not be realized.")

        def recursive_prod(data):
            nonlocal total
            if isinstance(data, list):
                for item in data:
                    recursive_prod(item)
            else:
                total *= data
        recursive_prod(self.data)
        return total
    
    def diagonal(self) -> 'simpy':
        """Return the diagonal elements of a 2D array.
    
        Extracts the elements where the row and column indices are equal.
        Only works for 2D arrays (matrices).
        
        Returns:
            simpy: 1D array containing the diagonal elements
            
        Raises:
            ValueError: If the array is not 2D
            
        Examples:
            >>> a = simpy([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> a.diagonal()
            simpy([1, 5, 9], shape=[3], dtype=int, ndim=1)
            
            >>> b = simpy([[1, 2], [3, 4]])
            >>> b.diagonal()
            simpy([1, 4], shape=[2], dtype=int, ndim=1)
        """
        if self.ndim != 2:
            raise ValueError("diagonal() requires 2D array.")
        
        n_rows = len(self.data)
        n_col = len(self.data[0]) if n_rows > 0 else 0
        min_dim = min(n_rows, n_col)

        diag_el = [self.data[i][i] for i in range(min_dim)]
        return simpy(diag_el, dtype=self.dtype)
    
    def mean(self, axis: Optional[int] = None) -> Union[float, 'simpy']:
        """Compute the arithmetic mean along the specified axis.
    
        Args:
            axis: Axis along which to compute the mean.
                None (default): compute mean of all elements (flattened array)
                0: compute mean along columns (for 2D arrays)
                1: compute mean along rows (for 2D arrays)
                
        Returns:
            If axis is None: scalar mean value (float)
            If axis is specified: simpy array with reduced dimensions
            
        Raises:
            ValueError: If axis is invalid or array is empty
            TypeError: If array contains non-numeric elements
            
        Examples:
            1D array:
            >>> a = simpy([1, 2, 3, 4])
            >>> a.mean()
            2.5
            
            2D array (global mean):
            >>> b = simpy([[1, 2], [3, 4]])
            >>> b.mean()
            2.5
            
            2D array (column means):
            >>> b.mean(axis=0)
            simpy([2.0, 3.0], shape=[2], dtype=float, ndim=1)
            
            2D array (row means):
            >>> b.mean(axis=1)
            simpy([1.5, 3.5], shape=[2], dtype=float, ndim=1)
        """
        if self.size == 0:
            raise ValueError("Can't compute mean of empty array.")
        
        if axis is None:
            total = self.sum()
            return total / self.size

        if axis < 0 or axis >= self.ndim:
            raise ValueError(f"axis {axis} is out of bounds for array with {self.ndim} dimensions.")
        
        if self.ndim == 1:
            if axis == 0:
                return simpy([self.sum() / self.size], dtype='float')
            raise ValueError(f'axis {axis} is invalid for 1D array.')
        
        if axis == 0:
            means = []
            n_rows = self.shape[0]
            for col in range(self.shape[1]):
                col_sum = sum(row[col] for row in self.data)
                means.append(col_sum / n_rows)
            return simpy(means, dtype='float')
        
        elif axis == 1:
            means = []
            n_cols = self.shape[1]
            for row in self.data:
                row_sum = sum(row)
                means.append(row_sum / n_cols)
            return simpy(means, dtype='float')
        raise ValueError(f'axis {axis} is invalid for 2D array')
    
    def std(self, axis: Optional[int] = None, ddof: int = 0) -> Union[float, 'simpy']:
        """Compute the standard deviation along the specified axis.
    
        Args:
            axis: Axis along which to compute the std.
                None (default): compute std of all elements (flattened array)
                0: compute std along columns (for 2D arrays)
                1: compute std along rows (for 2D arrays)
            ddof: Delta degrees of freedom. The divisor used is N - ddof, 
                where N represents the number of elements.
                
        Returns:
            If axis is None: scalar std value (float)
            If axis is specified: simpy array with reduced dimensions
            
        Raises:
            ValueError: If axis is invalid or array is empty
            TypeError: If array contains non-numeric elements
            
        Examples:
            1D array:
            >>> a = simpy([1, 2, 3, 4])
            >>> a.std()
            1.118033988749895
            
            2D array (global std):
            >>> b = simpy([[1, 2], [3, 4]])
            >>> b.std()
            1.118033988749895
            
            2D array (column std):
            >>> b.std(axis=0)
            simpy([1.0, 1.0], shape=[2], dtype=float, ndim=1)
            
            2D array (row std):
            >>> b.std(axis=1)
            simpy([0.5, 0.5], shape=[2], dtype=float, ndim=1)
        """

        if self.size == 0:
            raise ValueError("Can't compute mean of empty array.")
        
        means = self.mean(axis=axis) if axis is not None else self.mean()
        if axis is None:
            square_diff = [(x - means)**2 for x in self.flatten().data]
            sum_sq = sum(square_diff)
            div = self.size - ddof
            return (sum_sq / div)**0.5
        
        if axis < 0 or axis >= self.ndim:
            raise ValueError(f"axis {axis} is out of bounds for array with {self.ndim} dimensions.")
        
        if self.ndim == 1:
            if axis == 0:
                squared_diff = [(x - means.data[0])**2 for x in self.data]
                sum_sq = sum(squared_diff)
                divisor = self.size - ddof
                std_val = (sum_sq / divisor)**0.5
                return simpy([std_val], dtype='float')
            raise ValueError(f'axis {axis} is invalid for 1D array.')
    
        if axis == 0:
            stds = []
            n_rows = self.shape[0]
            for col in range(self.shape[1]):
                squared_diff = [(row[col] - means.data[col])**2 for row in self.data]
                sum_sq = sum(squared_diff)
                divisor = n_rows - ddof
                stds.append((sum_sq / divisor)**0.5)
            return simpy(stds, dtype='float')
        
        elif axis == 1:
            stds = []
            n_cols = self.shape[1]
            for i, row in enumerate(self.data):
                squared_diff = [(x - means.data[i])**2 for x in row]
                sum_sq = sum(squared_diff)
                divisor = n_cols - ddof
                stds.append((sum_sq / divisor)**0.5)
            return simpy(stds, dtype='float')
        
        raise ValueError(f'axis {axis} is invalid for 2D array')
        
    def median(self, axis: Optional[int] = None) -> Union[float, 'simpy']:
        """Compute the median along the specified axis.
    
        Args:
            axis: Axis along which to compute the median.
                None (default): compute median of all elements (flattened array)
                0: compute median along columns (for 2D arrays)
                1: compute median along rows (for 2D arrays)
                
        Returns:
            If axis is None: scalar median value (float)
            If axis is specified: simpy array with reduced dimensions
            
        Raises:
            ValueError: If axis is invalid or array is empty
            
        Examples:
            1D array:
            >>> a = simpy([1, 3, 2, 4])
            >>> a.median()
            2.5
            
            2D array (global median):
            >>> b = simpy([[1, 3], [2, 4]])
            >>> b.median()
            2.5
            
            2D array (column medians):
            >>> b.median(axis=0)
            simpy([1.5, 3.5], shape=[2], dtype=float, ndim=1)
            
            2D array (row medians):
            >>> b.median(axis=1)
            simpy([2.0, 3.0], shape=[2], dtype=float, ndim=1)
        """
        if self.size == 0:
            raise ValueError("Can't compute median of empty array.")

        def compute_median(data):
            sorted_data = sorted(data)
            n = len(sorted_data)
            if n % 2 == 1:
                return float(sorted_data[n//2])
            else:
                return (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2.0
        
        if axis is None:
            flatten_data = self.flatten().data
            return compute_median(flatten_data)
        
        if axis < 0 or axis >= self.ndim:
            raise ValueError(f"axis {axis} is out of bounds for array with {self.ndim} dimensions.")
        
        if self.ndim == 1:
            if axis == 0:
                return simpy([compute_median(self.data)], dtype='float')
            raise ValueError(f'axis {axis} is invalid for 1D array.')
        
        if axis == 0:
            medians = []
            for col in range(self.shape[1]):
                column_data = [row[col] for row in self.data]
                medians.append(compute_median(column_data))
            return simpy(medians, dtype='float')
        
        elif axis == 1:
            medians = []
            for row in self.data:
                medians.append(compute_median(row))
            return simpy(medians, dtype='float')
        
        raise ValueError(f'axis {axis} is ivalid for 2D array')
        
    def cov(self, rowvar: bool = True, ddof: Optional[int] = None) -> 'simpy':
        """Estimate the covariance matrix.
        
        Args:
            rowvar: If True (default), each row represents a variable, with 
                observations in the columns. If False, relationship is transposed.
            ddof: Delta degrees of freedom. The divisor used is N - ddof, where N
                represents the number of observations. If None, defaults to 1 for
                sample covariance (like numpy).
                
        Returns:
            simpy: The covariance matrix
            
        Examples:
            >>> x = simpy([[0, 2], [1, 1], [2, 0]]).T
            >>> x.cov()
            simpy([[ 1., -1.],
                [-1.,  1.]], shape=[2, 2], dtype=float, ndim=2)
        """
        if self.size == 0:
            raise ValueError("Can't compute covariance of empty array.")
        
        if ddof is None:
            ddof = 1
        
        if self.ndim == 1:
            data = [self.data]
        else:
            data = self.data if rowvar else [list(row) for row in zip(*self.data)]
        
        n_vars = len(data)
        n_obs = len(data[0]) if n_vars > 0 else 0
        
        means = [sum(col)/n_obs for col in data]
        
        cov_matrix = []
        for i in range(n_vars):
            row = []
            for j in range(n_vars):
                covariance = sum((data[i][k] - means[i]) * (data[j][k] - means[j]) 
                            for k in range(n_obs)) / (n_obs - ddof)
                row.append(covariance)
            cov_matrix.append(row)
        
        return simpy(cov_matrix, dtype='float')

    def corrcoef(self, rowvar: bool = True) -> 'simpy':
        """Return the Pearson correlation coefficients.
        
        Args:
            rowvar: If True (default), each row represents a variable, with 
                observations in the columns. If False, relationship is transposed.
                
        Returns:
            simpy: The correlation matrix
            
        Examples:
            >>> x = simpy([[0, 1, 2], [2, 1, 0]])
            >>> x.corrcoef()
            simpy([[ 1., -1.],
                [-1.,  1.]], shape=[2, 2], dtype=float, ndim=2)
        """
        if self.size == 0:
            raise ValueError("Can't compute correlation of empty array.")
        
        cov_matrix = self.cov(rowvar=rowvar, ddof=0).data
        
        n = len(cov_matrix)
        corr_matrix = []
        
        for i in range(n):
            row = []
            for j in range(n):
                std_i = cov_matrix[i][i]**0.5
                std_j = cov_matrix[j][j]**0.5
                if std_i == 0 or std_j == 0:
                    correlation = float('nan')
                else:
                    correlation = cov_matrix[i][j] / (std_i * std_j)
                row.append(correlation)
            corr_matrix.append(row)
        
        return simpy(corr_matrix, dtype='float')
    
    def __matmul__(self, other: 'simpy') -> 'simpy':
        """Matrix multiplication using @ operator.
        
        Args:
            other: Another simpy array to multiply with
            
        Returns:
            Result of matrix multiplication as new simpy array
            
        Raises:
            ValueError: If matrix dimensions are incompatible
            TypeError: If other is not a simpy array
            
        Examples:
            Matrix multiplication:
            >>> A = simpy([[1, 2], [3, 4]])
            >>> B = simpy([[5, 6], [7, 8]])
            >>> A @ B
            simpy([[19, 22], [43, 50]], shape=[2, 2], dtype=int, ndim=2)
            
            Matrix-vector multiplication:
            >>> M = simpy([[1, 2], [3, 4]])
            >>> v = simpy([5, 6])
            >>> M @ v
            simpy([17, 39], shape=[2], dtype=int, ndim=1)
        """
        if not isinstance(other, simpy):
            raise TypeError("Matrix multiplication requires simpy arrays")
        
        left = self.data
        right = other.data
        
        left_shape = self.shape if self.ndim == 2 else (1, len(self.data))
        right_shape = other.shape if other.ndim == 2 else (len(other.data), 1)
        
        if left_shape[1] != right_shape[0]:
            raise ValueError(
                f"Shapes {left_shape} and {right_shape} not aligned for matrix multiplication"
            )
        
        result = []
        for i in range(left_shape[0]):
            row = []
            for j in range(right_shape[1]):
                total = 0
                for k in range(left_shape[1]):
                    left_val = left[i][k] if self.ndim == 2 else left[k]
                    right_val = right[k][j] if other.ndim == 2 else right[k]
                    total += left_val * right_val
                row.append(total)
            result.append(row)
        
        if self.ndim == 1 and other.ndim == 1: 
            return simpy(result[0][0])
        elif self.ndim == 1: 
            return simpy(result[0])
        elif other.ndim == 1: 
            return simpy([row[0] for row in result])
        return simpy(result)
    
    def argmin(self, axis: int = None) -> Union[int, 'simpy']:
        """Returns the indices of the minimum values along an axis.
    
        Args:
            axis: Axis along which to operate. If None (default), the index is 
                returned for the flattened array. For 1D arrays, only axis=0 
                is valid. For 2D arrays, axis=0 finds minima across rows, 
                and axis=1 across columns.
        
        Returns:
            - If `axis=None`: Single index (int) of the minimum in the flattened array.
            - If `axis=0` or `axis=1`: `simpy` array of indices along the specified axis.
        
        Raises:
            ValueError: If the array is empty or `axis` is invalid.
            NotImplementedError: If called on arrays with ndim > 2.
        
        Examples:
            >>> a = simpy([3, 1, 4, 1])
            >>> a.argmin()
            1  # Index of the first occurrence of the minimum value (1)

            >>> b = simpy([[3, 1, 4], [2, 0, 5]])
            >>> b.argmin(axis=0)
            simpy([1, 1, 0], dtype=int)  # Indices of minima for each column
            >>> b.argmin(axis=1)
            simpy([1, 1], dtype=int)     # Indices of minima for each row
        """        
        if self.size == 0:
            raise ValueError("Can't find smallest element in empty array.")

        if axis is None:
            res = self.flatten()
            min_el = res.data[0]
            index = 0
            for i, j in enumerate(res.data):
                if res.data[i] < min_el:
                    min_el = res.data[i]
                    index = i
            return index

        if axis < 0 or axis >= self.ndim:
            raise ValueError(f"axis {axis} is out of bounds for array with {self.ndim} dimensions.")
        
        if self.ndim == 1:
            if axis != 0:
                raise ValueError("For 1D array, axis can only be 0 or None.")
            min_el = self.data[0]
            index = 0
            for i, val in enumerate(self.data):
                if val < min_el:
                    min_el = val
                    index = i
            return index

        elif self.ndim == 2:
            if axis == 0:  
                min_indices = []
                for col in range(self.shape[1]):
                    min_val = self.data[0][col]
                    min_row = 0
                    for row in range(1, self.shape[0]):
                        if self.data[row][col] < min_val:
                            min_val = self.data[row][col]
                            min_row = row
                    min_indices.append(min_row)
                return simpy(min_indices, dtype='int')  
            
            elif axis == 1:  
                min_indices = []
                for row in range(self.shape[0]):
                    min_val = self.data[row][0]
                    min_col = 0
                    for col in range(1, self.shape[1]):
                        if self.data[row][col] < min_val:
                            min_val = self.data[row][col]
                            min_col = col
                    min_indices.append(min_col)
                return simpy(min_indices, dtype='int')

            else:
                raise ValueError("For 2D array, axis must be 0, 1, or None.")

        else:
            raise NotImplementedError("argmin() is only implemented for 1D and 2D arrays.")
        
    def argmax(self, axis: int = None) -> Union[int, 'simpy']:
        """Returns the indices of the maximum values along an axis.
    
        Args:
            axis: Axis along which to operate. If None (default), the index is 
                returned for the flattened array. For 1D arrays, only axis=0 
                is valid. For 2D arrays, axis=0 finds maxima across rows, 
                and axis=1 across columns.
        
        Returns:
            - If `axis=None`: Single index (int) of the maximum in the flattened array.
            - If `axis=0` or `axis=1`: `simpy` array of indices along the specified axis.
        
        Raises:
            ValueError: If the array is empty or `axis` is invalid.
            NotImplementedError: If called on arrays with ndim > 2.
        
        Examples:
            >>> a = simpy([3, 1, 4, 1])
            >>> a.argmax()
            2  # Index of the maximum value (4)

            >>> b = simpy([[3, 1, 4], [2, 5, 0]])
            >>> b.argmax(axis=0)
            simpy([0, 1, 0], dtype=int)  # Indices of maxima for each column
            >>> b.argmax(axis=1)
            simpy([2, 1], dtype=int)     # Indices of maxima for each row
        """        
        if self.size == 0:
            raise ValueError("Can't find maximal element in empty array.")

        if axis is None:
            res = self.flatten()
            max_el = res.data[0]
            index = 0
            for i, j in enumerate(res.data):
                if res.data[i] > max_el:
                    max_el = res.data[i]
                    index = i
            return index

        if axis < 0 or axis >= self.ndim:
            raise ValueError(f"axis {axis} is out of bounds for array with {self.ndim} dimensions.")
        
        if self.ndim == 1:
            if axis != 0:
                raise ValueError("For 1D array, axis can only be 0 or None.")
            max_el = self.data[0]
            index = 0
            for i, val in enumerate(self.data):
                if val > max_el:
                    max_el = val
                    index = i
            return index

        elif self.ndim == 2:
            if axis == 0:  
                max_indices = []
                for col in range(self.shape[1]):
                    max_val = self.data[0][col]
                    max_row = 0
                    for row in range(1, self.shape[0]):
                        if self.data[row][col] > max_val:
                            max_val = self.data[row][col]
                            max_row = row
                    max_indices.append(max_row)
                return simpy(max_indices, dtype='int')  
            
            elif axis == 1:  
                max_indices = []
                for row in range(self.shape[0]):
                    max_val = self.data[row][0]
                    max_col = 0
                    for col in range(1, self.shape[1]):
                        if self.data[row][col] > max_val:
                            max_val = self.data[row][col]
                            max_col = col
                    max_indices.append(max_col)
                return simpy(max_indices, dtype='int')

            else:
                raise ValueError("For 2D array, axis must be 0, 1, or None.")

        else:
            raise NotImplementedError("argmax() is only implemented for 1D and 2D arrays.")  

    def setdiff1d(self, other: 'simpy') -> 'simpy':
        """Compute the set difference between two arrays (elements in `self` not in `other`).

        Returns the unique values in `self` that are not present in `other`. 
        The result is sorted in ascending order (unlike Python's set difference).
        Input arrays are flattened before computation.

        Args:
            other: Array to compare against. Must be an instance of `simpy`.

        Returns:
            A new `simpy` array containing the sorted unique values present in `self` 
            but not in `other`. The output is always 1-dimensional.

        Raises:
            TypeError: If `other` is not a `simpy` array.

        Examples:
            >>> a = simpy([1, 2, 3, 2, 4])
            >>> b = simpy([3, 5, 6])
            >>> a.setdiff1d(b)
            simpy([1, 2, 4])  # Unique elements in 'a' not in 'b'

            >>> c = simpy([[1, 2], [3, 4]])
            >>> d = simpy([3, 5])
            >>> c.setdiff1d(d)
            simpy([1, 2, 4])  # Flattened 'c' minus elements from 'd'

        Note:
            1. The operation is not symmetric: `a.setdiff1d(b) != b.setdiff1d(a)`.
        """
        if not isinstance(other, simpy):
            raise TypeError("Passed argument is not an instance of simpy class") 
        
        fst_elements = set(self.flatten().data)
        sec_elements = set(other.flatten().data)
        res = []
        for i in fst_elements:
            if i not in sec_elements:
                res.append(i)
        return res
    
    def transpose(self) -> 'simpy':
        """Transpose the array by reversing its dimensions.
        
        For a 2D array, this swaps rows and columns. For 1D arrays, returns 
        a view of the same array (no change). Higher dimensional arrays 
        reverse the order of axes.
        
        Returns:
            A new `simpy` array with axes reversed. For 2D arrays, this means
            the first axis (rows) becomes the second axis (columns), and vice versa.
            
        Raises:
            ValueError: If the array is empty
            
        Examples:
            1D array (unchanged):
            >>> a = simpy([1, 2, 3])
            >>> a.transpose()
            simpy([1, 2, 3], shape=[3], dtype=int, ndim=1)
            
            2D array:
            >>> b = simpy([[1, 2], [3, 4], [5, 6]])
            >>> b.transpose()
            simpy([[1, 3, 5], 
                [2, 4, 6]], shape=[2, 3], dtype=int, ndim=2)
                
            3D array (axes reversed):
            >>> c = simpy([[[1, 2], [3, 4]], [[[5, 6], [7, 8]]])
            >>> c.transpose()
            simpy([[[1, 5], [3, 7]], 
                [[2, 6], [4, 8]]], shape=[2, 2, 2], dtype=int, ndim=3)
        """
        if self.size == 0:
            raise ValueError("Can't transpose an empty array")
        
        if self.ndim == 1:
            return simpy(self.data.copy())
        
        def recursive_transpose(data, depth=0):
            if depth == self.ndim - 1:
                return data
            transposed = list(zip(*data))
            return [recursive_transpose(list(item), depth+1) for item in transposed]
        
        transposed_data = recursive_transpose(self.data)
        return simpy(transposed_data, dtype=self.dtype)
    
    def minor(self, row: int, col: int) -> 'simpy':
        """Return the minor matrix by removing specified row and column.
        
        Args:
            row: Index of row to remove (0-based)
            col: Index of column to remove (0-based)
            
        Returns:
            simpy: New matrix with specified row and column removed
            
        Raises:
            IndexError: If row or column indices are out of bounds
        """
        if row < 0 or row >= self.shape[0]:
            raise IndexError(f"Row index {row} is out of bounds")
        if col < 0 or col >= self.shape[1]:
            raise IndexError(f"Column index {col} is out of bounds")
        
        minor_data = []
        for i in range(self.shape[0]):
            if i == row:
                continue
            minor_row = []
            for j in range(self.shape[1]):
                if j == col:
                    continue
                minor_row.append(self.data[i][j])
            minor_data.append(minor_row)
        
        return simpy(minor_data, dtype=self.dtype)

    
    def det(self) -> float:
        """Compute the determinant of a square matrix.
        
        Uses recursive Laplace expansion for matrices larger than 2x2.
        For 1x1 matrices, returns the single element.
        For 2x2 matrices, uses the direct formula ad - bc.
        
        Returns:
            The determinant as a float (even for integer matrices)
            
        Raises:
            ValueError: If the matrix is not square or has ndim != 2
            TypeError: If the matrix contains non-numeric elements
            
        Examples:
            >>> a = simpy([[1, 2], [3, 4]])
            >>> a.det()
            -2.0
            
            >>> b = simpy([[5]])
            >>> b.det()
            5.0
            
            >>> c = simpy([[2, 4, 1], [0, 3, -1], [1, 2, 0]])
            >>> c.det()
            5.0
        """
        if self.ndim != 2:
            raise ValueError("Determinant is only defined for 2D matrices")
            
        if self.shape[0] != self.shape[1]:
            raise ValueError("Matrix must be square to compute determinant")
            
        n = self.shape[0]
        
        if n == 1:
            return float(self.data[0][0])
            
        if n == 2:
            return float(self.data[0][0] * self.data[1][1] - 
                        self.data[0][1] * self.data[1][0])
        
        determinant = 0.0
        
        for col in range(n):
            submatrix = []
            for i in range(1, n):
                row = []
                for j in range(n):
                    if j != col:
                        row.append(self.data[i][j])
                submatrix.append(row)
                
            sub_det = simpy(submatrix).det()
            sign = (-1) ** col
            determinant += sign * self.data[0][col] * sub_det
            
        return determinant
    
    def inv(self) -> 'simpy':
        """Compute the multiplicative inverse of a square matrix.

        Returns a new matrix that, when multiplied with the original,
        yields the identity matrix.
        
        Algorithm:
            1. Computes the matrix of cofactors
            2. Transposes to get the adjugate matrix
            3. Multiplies by 1/determinant
        
        Returns:
            simpy: The inverse matrix with dtype='float'
        
        Raises:
            ValueError: If matrix is not square, has ndim != 2, 
                    or is singular (det = 0)
        
        Examples:
            >>> m = simpy([[1, 2], [3, 4]])
            >>> m.inv()
            simpy([[-2.0, 1.0], [1.5, -0.5]])
        
        Notes:
            1. For numerical stability, consider adding a small epsilon 
            when checking determinant == 0
            2. Matrix inversion has O(n³) complexity for dense matrices
        """
        if self.ndim != 2:
            raise ValueError(f"Can't find inversed matrix with ndim: {self.ndim}")

        if self.shape[0] != self.shape[1]:
            raise ValueError(f"The number of rows and columns are different:\n Rows: {self.shape[0]}\n Columns: {self.shape[1]}")
        
        determinant = self.det()

        if determinant == 0:
            raise ValueError(f"Zero Determinant")
        
        if self.shape[0] == 2:
            a, b = self[0, 0], self[0, 1]
            c, d = self[1, 0], self[1, 1]
            inversed_determinant = 1 / determinant
            return simpy([[d * inversed_determinant, -b * inversed_determinant],
                          [-c * inversed_determinant, a * inversed_determinant]], dtype='float')

        cofactors = []
        for r in range(self.shape[0]):
            cofactor_row = []
            for c in range(self.shape[1]):
                minor = self.minor(r, c)
                cofactor = ((-1) ** (r + c)) * minor.det()
                cofactor_row.append(cofactor)
            cofactors.append(cofactor_row)
        
        adjugate = simpy(cofactors).transpose()
        inverse = adjugate * (1 / determinant)
        return inverse
    
    def _compute_eigenvalues(self, max_iter=100, tol=1e-6) -> List[float]:
        """Internal method to compute eigenvalues of a square matrix.
    
        Uses QR algorithm for matrices > 2x2 and direct formula for 2x2 matrices.
        
        Parameters:
            max_iter: Maximum number of QR iterations
            tol: Tolerance for convergence checking
        
        Returns:
            List[float]: Eigenvalues sorted by magnitude (descending)
        
        Raises:
            ValueError: If matrix is not square
            RuntimeError: If QR algorithm fails to converge
        
        Notes:
            1. This is a simplified implementation - consider:
            - Adding complex number support
            - Implementing more robust QR decomposition
            - Adding shift strategies for faster convergence
        """
        if self.ndim != 2 or self.shape[0] != self.shape[1]:
            raise ValueError("Eigenvalues require square matrix")
        
        if self.shape[0] == 2:
            a, b, c, d = self.data[0][0], self.data[0][1], self.data[1][0], self.data[1][1]
            trace = a + d
            det = a*d - b*c
            discr = trace**2 - 4*det
            
            if discr >= 0:
                sqrt_discr = discr**0.5
                return [(trace + sqrt_discr)/2, (trace - sqrt_discr)/2]
            else:
                return [trace/2, trace/2]
        
        A = [row.copy() for row in self.data]
        n = self.shape[0]
        
        for _ in range(max_iter):
            Q = [[0]*n for _ in range(n)]
            R = [[0]*n for _ in range(n)]
            
            for j in range(n):
                v = [A[i][j] for i in range(n)]
                for k in range(j):
                    R[k][j] = sum(Q[i][k] * A[i][j] for i in range(n))
                    v = [v[i] - Q[i][k] * R[k][j] for i in range(n)]
                
                R[j][j] = sum(x**2 for x in v)**0.5
                if R[j][j] < tol:
                    Q = [[1 if i==j else 0 for j in range(n)] for i in range(n)]
                else:
                    Q = [[v[i]/R[j][j] for i in range(n)] for j in range(n)]
            
            A = [[sum(R[i][k] * Q[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
        
        eigenvalues = [A[i][i] for i in range(n)]
        return sorted(eigenvalues, key=lambda x: -abs(x))
        
    def norm(self, ord: Union[str, int] = None) -> Union[float, 'simpy', None]:
        """Compute matrix or vector norm.
    
        Supported norm types:
            - None/'fro': Frobenius norm (sqrt of sum of squares)
            - 1: Maximum column sum (matrices) or max abs (vectors)
            - -1: Minimum column sum (matrices) or min abs (vectors) 
            - 2: Spectral norm (largest singular value)
            - -2: Smallest singular value
            - 'inf': Maximum row sum
            - '-inf': Minimum row sum
            - 0: Number of non-zero elements (vectors only)
            - 'nuc': Nuclear norm (sum of singular values)
        
        Parameters:
            ord: Order of the norm (see above for options)
        
        Returns:
            Computed norm as float. Returns None for unsupported norm types.
        
        Raises:
            ValueError: For invalid norm orders or incompatible dimensions
            NotImplementedError: For norms not implemented for given array shape
        
        Examples:
            >>> v = simpy([3, 4])
            >>> v.norm(2)  # Euclidean norm
            5.0
            
            >>> m = simpy([[1, 2], [3, 4]])
            >>> m.norm('fro')  # Frobenius norm
            5.477...
        
        Notes:
            1. Spectral norm (ord=2) requires square matrices
            2. Nuclear norm may be inaccurate for non-diagonal matrices
        """
        if ord in (None, 'fro'):
            if self.ndim == 1:
                return sum(x**2 for x in self.data)**0.5
            
            elif self.ndim == 2:
                return sum(x**2 for row in self.data for x in row)**0.5
            
            raise NotImplementedError("Norm is only implemented for 1D and 2D arrays")
        
        elif ord in (1, -1):
            if self.ndim == 1:
                module_data = [abs(x) for x in self.data]
                return max(module_data) if ord == 1 else min(module_data)
                
            elif self.ndim == 2:
                col_sums = []
                for col in range(self.shape[1]):
                    col_sum = sum(abs(row[col]) for row in self.data)
                    col_sums.append(col_sum)
                
                return max(col_sums) if ord == 1 else min(col_sums)
                
            raise NotImplementedError("Norm is only implemented for 1D and 2D arrays")
        
        elif ord in ('inf', '-inf'):
            if self.ndim == 1:
                return sum([abs(x) for x in self.data])
            
            elif self.ndim == 2:
                row_sums = []
                for row in range(self.shape[0]):
                    row_sum = sum(abs(row[col] for col in row))
                    row_sums.append(row_sum)
                
                return max(row_sums) if ord == 'inf' else min(row_sums)
            
            raise NotImplementedError("Norm is only implemented for 1D and 2D arrays")

        elif ord == 0:
            if self.ndim == 1:
                return sum(1 for x in self.data if x != 0)
        
            raise ValueError("ord=0 can be used only for 1D arrays")
        
        elif ord in (2, -2):
            if self.ndim == 1:
                return sum(x**2 for x in self.data)**0.5
            
            elif self.ndim == 2:
                if self.shape[0] != self.shape[1]:
                    raise ValueError("Spectral norm requires square matrix")
                eigenvalues = [abs(x) for x in self._compute_eigenvalues()]
        
                return max(eigenvalues) if ord == 2 else min(eigenvalues)
            
            raise NotImplementedError("Spectral norm requires 2D array")

        elif ord == 'nuc':
            if self.ndim == 2:
                return sum(abs(x) for x in self._compute_eigenvalues())**0.5
            raise ValueError("ord='nuc' requires 2D array")

        raise ValueError(f"Invalid norm ord: '{ord}'")