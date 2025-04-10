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
    
    def __init__(self, data: List, dtype: str = None, display_decimals: int = 6) -> None:
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
    
    def _convert_data(self, data: List, dtype: str) -> List:
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
                return f"{value:.1f}"  # Format floats to one decimal place
            return str(value)

        def format_array(data, indent=0):
            if isinstance(data, list):
                if not data:  # Empty list
                    return "[]"
                if isinstance(data[0], list):  # Multi-dimensional array
                    prefix = " " * indent
                    lines = [f"{prefix}[{format_array(sub, indent + 1)}]" for sub in data]
                    return "\n" + "\n".join(lines)
                else:  # 1D array
                    elements = " ".join(format_value(x) for x in data)
                    return f"[{elements}]"
            else:  # Scalar value
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
        
        if len(indices) > len(self.shape):
            raise IndexError(f"Too many indices for array with {len(self.shape)} dimensions")
        
        def get_item(data: List, idx: Tuple, dim: int = 0) -> List:
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
                    return [result] if not isinstance(result, list) else result
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
            return simpy(result if isinstance(result[0], list) else [result])
        except IndexError as e:
            raise IndexError(f"Failed to index array: {str(e)}") from e
        #05.04 FIX error IndexError: Failed to index array: Index 1 is out of bounds for axis 0 with size 1
        #06.04 FIXED
    
    @staticmethod
    def arange(start: Union[int, float], stop: Union[int, float], 
               step: Union[int, float] = 1, dtype: str = None) -> 'simpy':
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
                # Check if all elements are non-list (scalars)
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
        
        # First compute covariance matrix
        cov_matrix = self.cov(rowvar=rowvar, ddof=0).data
        
        n = len(cov_matrix)
        corr_matrix = []
        
        # Convert covariance to correlation
        for i in range(n):
            row = []
            for j in range(n):
                std_i = cov_matrix[i][i]**0.5
                std_j = cov_matrix[j][j]**0.5
                if std_i == 0 or std_j == 0:
                    # Handle case of zero variance
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