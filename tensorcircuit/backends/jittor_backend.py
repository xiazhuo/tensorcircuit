"""
Jittor backend. Not in the tensornetwork package.
"""

# pylint: disable=invalid-name

import logging
import warnings
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import scipy
try:  # old version tn compatiblity
    from tensornetwork.backends import base_backend

    tnbackend = base_backend.BaseBackend

except ImportError:
    from tensornetwork.backends import abstract_backend

    tnbackend = abstract_backend.AbstractBackend

from .abstract_backend import ExtendedBackend

logger = logging.getLogger(__name__)

dtypestr: str
Tensor = Any
pytree = Any
jt: Any
eins: Any

class jittor_optimizer:
    def __init__(self, optimizer: Any) -> None:
        self.optimizer = optimizer
        self.is_init = False

    # def _apply_gradients(self, grads: Tensor, params: Tensor) -> None:
    #     self.optimizer.apply_gradients([(grads, params)])

    def update(self, grads: pytree, params: pytree) -> pytree:
        # flatten grad and param
        params, treedef = JittorBackend.tree_flatten(None, params)
        grads, _ = JittorBackend.tree_flatten(None, grads)
        if self.is_init is False:
            self.optimizer = self.optimizer(params)
            self.is_init = True
        with jt.no_grad():
            for g, p in zip(grads, params):
                p.grad = g
        self.optimizer.step()
        self.optimizer.zero_grad()
        # reorg the param
        params = JittorBackend.tree_unflatten(None, treedef, params)
        return params

class JittorBackend(tnbackend, ExtendedBackend):  # type: ignore
    def __init__(self) -> None:
        super().__init__()
        try:
            import jittor
            from jittor import einops           
        except ImportError:
            raise ImportError(
                "Jittor not installed, please switch to a different "
                "backend or install Jittor."
            )
        global jt
        jt = jittor
        global eins
        eins = einops
        self.name = "jittor"
    
    def convert_to_tensor(self, a: Tensor) -> Tensor:
        # to be completed
        if np.iscomplexobj(a):
            real_part = jt.array(np.real(a))
            imag_part = jt.array(np.imag(a))
            return jt.nn.ComplexNumber(real_part,imag_part)
        else:
            a = jt.array(a)
            return a
        
    def cast(self, a: Tensor, dtype: str) -> Tensor:
        if isinstance(a,jt.nn.ComplexNumber):
            if dtype == 'complex64':
                real = jt.cast(a.real,'float32')
                image = jt.cast(a.imag,'float32')
                return jt.nn.ComplexNumber(real,image)
        return jt.cast(a,dtype)
    
    def sum(
        self: Any,
        a: Tensor,
        axis: Optional[Sequence[int]] = None,
        keepdims: bool = False,
    ) -> Tensor:
        if axis == None:
            axis = ()
        return jt.sum(a, dims=axis, keepdims=keepdims)
    
    def addition(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
        return tensor1 + tensor2
    
    def subtraction(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
        return tensor1 - tensor2
    
    def divide(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
        return tensor1 / tensor2
    
    def conj(self, tensor: Tensor) -> Tensor:
        return tensor.conj()

    def multiply(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
        return tensor1 * tensor2
    
    def sign(self, tensor: Tensor) -> Tensor:
        return jt.nn.sign(tensor)
    
    def norm(self, tensor: Tensor) -> Tensor:
        return jt.norm(tensor)
    
    def shape_tuple(self, tensor: Tensor) -> Tuple[int]:
        return tuple(tensor.shape) 
    
    def shape_concat(self, values: Tensor, axis: int) -> Tensor:
        return np.concatenate(values, axis)
    
    def shape_tensor(self, tensor: Tensor) -> Tensor:
        return jt.array(list(tensor.shape))
    
    def sparse_shape(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
        return self.shape_tuple(tensor)
    
    def shape_prod(self, values: Tensor) -> int:
        return np.prod(np.array(values))
    
    def tensordot(
        self, a: Tensor, b: Tensor, axes: Union[int, Sequence[Sequence[int]]]
    ) -> Tensor:
        # TODO: Implement this function
        raise NotImplementedError("The tensordot operation not found in jittor documentation")
    
    def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
        return jt.out(tensor1, tensor2)
    
    def transpose(self, tensor: Tensor, perm: Optional[Sequence[int]] = None) -> Tensor:
        if isinstance(tensor,jt.nn.ComplexNumber):
            if perm is None:
                perm = tuple(range(tensor.real.ndim - 1, -1, -1))
            return tensor.permute(perm)
        if perm is None:
                perm = tuple(range(tensor.ndim - 1, -1, -1))
        return tensor.permute(perm)
    
    def reshape(self, tensor: Tensor, shape: Tensor) -> Tensor:
        return jt.reshape(tensor, tuple(np.array(shape).astype(int)))
    
    def eye(
        self, N: int, dtype: Optional[str] = None, M: Optional[int] = None
    ) -> Tensor:
        if dtype is None:
            dtype = 'float32'# for test
        if not M:
            M = N
        return jt.init.eye(shape = (N,M), dtype=dtype)

    # def ones(self, shape: Sequence[int], dtype: Optional[str] = None) -> Tensor:
    #     if dtype is None:
    #         dtype = 'float32'# for test
    #         print(dtype)
    #     return jt.ones(shape, dtype=dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> Tensor:
        if dtype is None:
            dtype = 'float32'
        r = jt.ones(shape)
        return self.cast(r, dtype)

    def zeros(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> Tensor:
        if dtype is None:
            dtype = 'float32'
        return jt.zeros(shape, dtype=dtype)

    def copy(self, a: Tensor) -> Tensor:
        return a.copy()

    def expm(self, a: Tensor) -> Tensor:
        return a.exp()

    def abs(self, a: Tensor) -> Tensor:
        return a.abs()

    def sin(self, a: Tensor) -> Tensor:
        return a.sin()

    def cos(self, a: Tensor) -> Tensor:
        return a.cos()

    # acos acosh asin asinh atan atan2 atanh cosh (cos) tan tanh sinh (sin)
    def acos(self, a: Tensor) -> Tensor:
        return a.acos()

    def acosh(self, a: Tensor) -> Tensor:
        return a.acosh()

    def asin(self, a: Tensor) -> Tensor:
        return a.arcsin()

    def asinh(self, a: Tensor) -> Tensor:
        return a.arcsinh()

    def atan(self, a: Tensor) -> Tensor:
        return a.arctan()

    def atan2(self, y: Tensor, x: Tensor) -> Tensor:
        return jt.arctan2(y, x)

    def atanh(self, a: Tensor) -> Tensor:
        return a.arctanh()

    def cosh(self, a: Tensor) -> Tensor:
        return a.cosh()

    def tan(self, a: Tensor) -> Tensor:
        return a.tan()

    def tanh(self, a: Tensor) -> Tensor:
        return a.tanh()

    def sinh(self, a: Tensor) -> Tensor:
        return a.sinh()

    def size(self, a: Tensor) -> Tensor:
        return jt.size(a)
    
    def inv(self, matrix: Tensor) -> Tensor:
        return jt.linalg.inv(matrix)
    
    def eigh(self, matrix: Tensor):
        eigenvalues, eigenvectors = jt.linalg.eigh(matrix)
        sorted_indices = jt.argsort(eigenvalues)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        return eigenvalues, eigenvectors

    def eigvalsh(self, a: Tensor) -> Tensor:
        # to be discussed
        eigenvalues, eigenvectors = jt.linalg.eigh(a)
        return eigenvalues
    
    def det(self: Any, a: Tensor) -> Tensor:
        return jt.linalg.det(a)
    
    def cholesky(self, 
        tensor: Tensor,
        pivot_axis: int = -1,
        non_negative_diagonal: bool = False) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError("parameters lack")
    
    def svd(
        self,
        tensor: Tensor,
        pivot_axis: int = -1,
        max_singular_values: Optional[int] = None,
        max_truncation_error: Optional[float] = None,
        relative: Optional[bool] = False
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError("parameters lack")
    
    def qr(
        self,
        tensor: Tensor,
        pivot_axis: int = -1,
        non_negative_diagonal: bool = False
        ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError("parameters lack")
    def rq(
        self,
        tensor: Tensor,
        pivot_axis: int = -1,
        non_negative_diagonal: bool = False
        ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError("parameters lack")


    def kron(self, a: Tensor, b: Tensor) -> Tensor:
        # TODO: Implement this function
        raise NotImplementedError("The kron operation not found in jittor documentation")

    def dtype(self, a: Tensor) -> str:
        # not clear
        return a.dtype()

    def numpy(self, a: Tensor) -> Tensor:
        # need to check type first?
        return a.numpy()
    
    def onehot(self: Any, a: Tensor, num: int) -> Tensor:
        #dtype must be int
        a = jt.array(a,dtype = 'int32')
        return jt.nn.one_hot(a,num)
    
    def broadcast_left_multiplication(self, tensor1: Tensor,
        tensor2: Tensor) -> Tensor:
        return jt.unsqueeze(tensor1,1) * tensor2
    
    def broadcast_right_multiplication(self, tensor1: Tensor,
        tensor2: Tensor) -> Tensor:
        return tensor1 * jt.unsqueeze(tensor2,0)
    
    def i(self, dtype: Any = None) -> Tensor:
        # TODO: Implement this function
        raise NotImplementedError("complex operation")

    def stack(self, a: Sequence[Tensor], axis: int = 0) -> Tensor:
        return jt.stack(a, axis=axis)

    def concat(self, a: Sequence[Tensor], axis: int = 0) -> Tensor:
        return jt.concat(a, dim=axis)

    def tile(self, a: Tensor, rep: Tensor) -> Tensor:
        # TODO: Implement this function
        # 构建重排模式
        pattern = ' '.join([f'd{i}' for i in range(len(a.shape))])
        repeat_pattern = ' '.join([f'({rep[i].item()} d{i})' for i in range(len(rep))])
        # 构建 axes_lengths 仅包括原始张量的维度
        axes_lengths = {f'd{i}': a.shape[i] for i in range(len(a.shape))}
        return eins.repeat(a, f'{pattern} -> {repeat_pattern}', **axes_lengths)

    def mean(
        self,
        a: Tensor,
        axis: Optional[Sequence[int]] = None,
        keepdims: bool = False,
    ) -> Tensor:
        if axis == None:
            return a.mean(keepdims = keepdims)
        return a.mean(dim=axis, keepdims=keepdims)

    def std(
        self, a: Tensor, axis: Optional[Sequence[int]] = None, keepdims: bool = False
    ) -> Tensor:
        # TODO: Implement this function
        raise NotImplementedError("jittor does not has std operation that supports parameters like axis and keepdims")

    def unique_with_counts(self, a: Tensor, **kws: Any) -> Tuple[Tensor, Tensor]:
        # need test
        output, index, counts = jt.unique(a, return_inverse=True, return_counts=True)
        return output, counts
    


    def min(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        if axis == None:
            axis = ()
        return a.min(axis)

    def max(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        return a.max(axis)

    def argmax(self, a: Tensor, axis: int = 0) -> Tensor:
        return jt.argmax(a, dim=axis)[0]

    def argmin(self, a: Tensor, axis: int = 0) -> Tensor:
        return jt.argmin(a, dim=axis)[0]

    def sigmoid(self, a: Tensor) -> Tensor:
        return a.sigmoid()

    def relu(self, a: Tensor) -> Tensor:
        #maybe implement it directly?
        return jt.nn.relu(a)

    def softmax(self, a: Sequence[Tensor], axis: Optional[int] = None) -> Tensor:
        return jt.nn.softmax(a, dim=axis)
    
    def cumsum(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        if axis == None:
            b = jt.reshape(a,(-1,))
        return jt.cumsum(b, axis)

    def is_tensor(self, a: Any) -> bool:
        return jt.is_var(a)

    def real(self, a: Tensor) -> Tensor:
        return a.real()

    def imag(self, a: Tensor) -> Tensor:
        return a.imag()

    def arange(self, start: int, stop: Optional[int] = None, step: int = 1) -> Tensor:
        return jt.arange(start=start, end=stop, step=step)

    def mod(self, x: Tensor, y: Tensor) -> Tensor:
        return x % y

    def right_shift(self, x: Tensor, y: Tensor) -> Tensor:
        return jt.right_shift(x, y)

    def left_shift(self, x: Tensor, y: Tensor) -> Tensor:
        return jt.left_shift(x, y)

    def solve(self, A: Tensor, b: Tensor,**kws: Any) -> Tensor:  # type: ignore
        return jt.linalg.solve(A,b)

    def searchsorted(self, a: Tensor, v: Tensor, side: str = "left") -> Tensor:
        if side == "left":
            return jt.searchsorted(a,v,right = False)
        return jt.searchsorted(a,v,right = True)

    def set_random_state(
        self, seed: Optional[int] = None, get_only: bool = False
    ) -> Any:
        g = cp.random.default_rng(seed)  # None auto supported
        if get_only is False:
            self.g = g
        return g

    def stateful_randn(
        self,
        g: "cp.random.Generator",
        shape: Union[int, Sequence[int]] = 1,
        mean: float = 0,
        stddev: float = 1,
        dtype: str = "32",
    ) -> Tensor:
        if isinstance(dtype, str):
            dtype = dtype[-2:]
        if isinstance(shape, int):
            shape = (shape,)
        r = g.normal(loc=mean, scale=stddev, size=shape)
        if dtype == "32":
            r = r.astype(np.float32)
        elif dtype == "64":
            r = r.astype(np.float64)
        elif not isinstance(dtype, str):
            r = r.astype(dtype)
        else:
            raise ValueError("unspported `dtype` %s" % dtype)
        return r

    def stateful_randu(
        self,
        g: "cp.random.Generator",
        shape: Union[int, Sequence[int]] = 1,
        low: float = 0,
        high: float = 1,
        dtype: str = "32",
    ) -> Tensor:
        if isinstance(dtype, str):
            dtype = dtype[-2:]
        if isinstance(shape, int):
            shape = (shape,)
        r = g.random(shape) * (high - low) + low
        if dtype == "32":
            r = r.astype(np.float32)
        elif dtype == "64":
            r = r.astype(np.float64)
        elif not isinstance(dtype, str):
            r = r.astype(dtype)
        else:
            raise ValueError("unspported `dtype` %s" % dtype)
        return r

    def stateful_randc(
        self,
        g: "cp.random.Generator",
        a: Union[int, Sequence[int], Tensor],
        shape: Union[int, Sequence[int]],
        p: Optional[Union[Sequence[float], Tensor]] = None,
    ) -> Tensor:
        if isinstance(shape, int):
            shape = (shape,)
        return g.choice(a, size=shape, replace=True, p=p)
    
    def scatter(self, operand: Tensor, indices: Tensor, updates: Tensor) -> Tensor:
        operand_new = jt.copy(operand)
        operand_new[tuple([indices[:, i] for i in range(indices.shape[1])])] = updates
        return operand_new

    def coo_sparse_matrix(
        self, indices: Tensor, values: Tensor, shape: Tensor
    ) -> Tensor:
        return jt.sparse.sparse_array(indices,values,shape)

    def sparse_dense_matmul(
        self,
        sp_a: Tensor,
        b: Tensor,
    ) -> Tensor:
        return jt.sparse.spmm(sp_a,b)

    def to_dense(self, sp_a: Tensor) -> Tensor:
        return sp_a.to_dense()

    def is_sparse(self, a: Tensor) -> bool:
        # to be tested
        return isinstance(a, jt.SparseVar)  

    def cond(
        self,
        pred: bool,
        true_fun: Callable[[], Tensor],
        false_fun: Callable[[], Tensor],
    ) -> Tensor:
        if pred:
            return true_fun()
        return false_fun()

    def switch(self, index: Tensor, branches: Sequence[Callable[[], Tensor]]) -> Tensor:
        return branches[index.numpy()]()

    def device(self, a: Tensor) -> str:
        logger.warning(
            "All jittor.Var objects automatically share usage of CPU or GPU based on a global setting,"
        "so you will get the global setting."
        )
        if jt.flags.use_cuda:
            return "gpu"
        return "cpu"

    def device_move(self, a: Tensor, dev: Any) -> Tensor:
        logger.warning(
            "All jittor.Var objects automatically share usage of CPU or GPU based on a global setting,"
        "so you are changing the global setting."
        )
        if dev == "cpu":
            jt.flags.use_cuda = 0
        if dev == "cpu":
            jt.flags.use_cuda = 1

    def _dev2str(self, dev: Any) -> str:
        raise NotImplementedError("All jittor.Var objects automatically share usage of CPU or GPU based on a global setting, the specific device does not need to be concerned")

    def _str2dev(self, str_: str) -> Any:
        raise NotImplementedError("All jittor.Var objects automatically share usage of CPU or GPU based on a global setting, the specific device does not need to be concerned")

    def stop_gradient(self, a: Tensor) -> Tensor:
        return a.stop_grad()

    def grad(
        self,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Any]:
        return jt.grad(f,argnums,has_aux)

    def value_and_grad(
        self,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Tuple[Any, Any]]:
        def wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, Any]:
            # Define a function that computes both value and gradients
            def wrapped_function(*args: Any, **kwargs: Any) -> Tuple[Any, Any]:
                # Convert args to Jittor tensors if they are not already
                args_jt = [jt.array(arg) if not isinstance(arg, jt.Var) else arg for arg in args]
                
                # Compute the value of f
                value = f(*args_jt, **kwargs)

                # Compute gradients with respect to specified arguments
                if isinstance(argnums, int):
                    argnums = [argnums]
                grads = [jt.grad(value, args_jt[argnum], has_aux) for argnum in argnums]

                # Return value and gradients
                if has_aux:
                    return value, grads
                else:
                    return value, grads[0] if len(grads) == 1 else grads

            return wrapped_function(*args, **kwargs)

        return wrapper

    def jit(
        self,
        f: Callable[..., Any],
        static_argnums: Optional[Union[int, Sequence[int]]] = None,
        jit_compile: Optional[bool] = None,
        **kws: Any,
    ) -> Callable[..., Any]:
        return f

    def vmap(
        self, f: Callable[..., Any], vectorized_argnums: Union[int, Sequence[int]] = 0
    ) -> Any:
        logger.warning(
            "Jittor backend has no intrinsic vmap like interface"
            ", use vectorize instead (plain for loop)"
        )
        if isinstance(vectorized_argnums, int):
            vectorized_argnums = (vectorized_argnums,)

        def wrapper(*args: Any, **kws: Any) -> Tensor:
            results = []
            for barg in zip(*[args[i] for i in vectorized_argnums]):  # type: ignore
                narg = []
                j = 0
                for k in range(len(args)):
                    if k in vectorized_argnums:  # type: ignore
                        narg.append(barg[j])
                        j += 1
                    else:
                        narg.append(args[k])
                results.append(f(*narg, **kws))
            return jt.array(results)

        return wrapper

    def vectorized_value_and_grad(
        self,
        f: Callable[..., Any],
        argnums: Union[int, Sequence[int]] = 0,
        vectorized_argnums: Union[int, Sequence[int]] = 0,
        has_aux: bool = False,
    ) -> Callable[..., Tuple[Any, Any]]:
        raise NotImplementedError("Vectorization not supported")
        if isinstance(vectorized_argnums, int):
            vectorized_argnums = (vectorized_argnums,)

        def wrapper(
            *args: Any, **kws: Any
        ) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, ...]]]:
            jf = self.value_and_grad(f, argnums=argnums, has_aux=has_aux)
            jf = self.vmap(jf, vectorized_argnums=vectorized_argnums)
            vs, gs = jf(*args, **kws)

            if isinstance(argnums, int):
                argnums_list = [argnums]
                gs = [gs]
            else:
                argnums_list = argnums  # type: ignore
                gs = list(gs)
            for i, (j, g) in enumerate(zip(argnums_list, gs)):
                if j not in vectorized_argnums:  # type: ignore
                    gs[i] = self.tree_map(partial(jt.sum, dim=0), g)
            if isinstance(argnums, int):
                gs = gs[0]
            else:
                gs = tuple(gs)

            return vs, gs

        return wrapper

    vvag = vectorized_value_and_grad

    def vjp(
        self,
        f: Callable[..., Any],
        inputs: Union[Tensor, Sequence[Tensor]],
        v: Union[Tensor, Sequence[Tensor]],
    ) -> Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]:
        raise NotImplementedError("Jittor interface not found")

    def jvp(
        self,
        f: Callable[..., Any],
        inputs: Union[Tensor, Sequence[Tensor]],
        v: Union[Tensor, Sequence[Tensor]],
    ) -> Tuple[Union[Tensor, Sequence[Tensor]], Union[Tensor, Sequence[Tensor]]]:
        raise NotImplementedError("Jittor interface not found")
    

    def item(self, tensor):
        return tensor.item()
    
    def randn(self,
            shape: Tuple[int, ...],
            dtype: Optional[Any] = None,
            seed: Optional[int] = None) -> Tensor:
        if seed:
            jt.set_seed(seed)
        dtype = dtype if dtype is not None else 'float32'
        return jt.randn(shape, dtype=dtype)
    
    def random_uniform(self,
                    shape: Tuple[int, ...],
                    boundaries: Optional[Tuple[float, float]] = (0.0, 1.0),
                    dtype: Optional[Any] = None,
                    seed: Optional[int] = None) -> Tensor:
        if seed:
            jt.set_seed(seed)
        dtype = dtype if dtype is not None else 'float32'
        low = 0.0 if boundaries is None else boundaries[0]
        high = 1.0 if boundaries is None else boundaries[1]
        return jt.init.uniform(shape, dtype,low,high)
    
    def index_update(self, tensor: Tensor, mask: Tensor,
                assignee: Tensor) -> Tensor:
        t = jt.array(tensor).clone()
        t[mask] = assignee
        return t
    
    def matmul(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
        if (tensor1.ndim() <= 1) or (tensor2.ndim() <= 1):
            raise ValueError("inputs to `matmul` have to be a tensors of order > 1,")
        return jt.nn.matmul(tensor1, tensor2)
    
    def diagonal(self, tensor: Tensor, offset: int = 0, axis1: int = -2,
                axis2: int = -1) -> Tensor:
        raise NotImplementedError("Jittor interface not found") 
        



    def slice(self, tensor: Tensor, start_indices: Tuple[int, ...],
        slice_sizes: Tuple[int, ...]) -> Tensor:
        if len(start_indices) != len(slice_sizes):
            raise ValueError("Lengths of start_indices and slice_sizes must be identical.")
        obj = tuple(slice(start, start + size)for start, size in zip(start_indices, slice_sizes))
        return tensor[obj]
    


    def diagflat(self, tensor: Tensor, k: int = 0) -> Tensor:
        return jt.diag(jt.flatten(tensor),k)
    
    def trace(self, tensor: Tensor, offset: int = 0, axis1: int = -2,
            axis2: int = -1) -> Tensor:
        if offset != 0:
            errstr = (f"offset = {offset} must be 0 (the default)"
                f"with PyTorch backend.")
            raise NotImplementedError(errstr)
        if axis1 == axis2:
            raise ValueError(f"axis1 = {axis1} cannot equal axis2 = {axis2}")
        N = len(tensor.shape)
        if N > 25:
            raise ValueError(f"Currently only tensors with ndim <= 25 can be traced"f"in the PyTorch backend (yours was {N})")
        if axis1 < 0:
            axis1 = N+axis1
        if axis2 < 0:
            axis2 = N+axis2
        inds = list(map(chr, range(98, 98+N)))
        indsout = [i for n, i in enumerate(inds) if n not in (axis1, axis2)]
        inds[axis1] = 'a'
        inds[axis2] = 'a'
        return jt.linalg.einsum(''.join(inds) + '->' +''.join(indsout), tensor) 
    
    def eps(self, dtype) -> float:
        raise NotImplementedError("Jittor interface not found")
    

    optimizer = jittor_optimizer

