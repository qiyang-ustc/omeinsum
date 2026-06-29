"""Adapter module for integrating omeco tensor contraction optimizer."""

from __future__ import annotations

from typing import Sequence

try:
    import omeco
    OMECO_AVAILABLE = True
except ImportError:
    OMECO_AVAILABLE = False
    omeco = None  # type: ignore

try:
    import opt_einsum.paths
    OPT_EINSUM_AVAILABLE = True
except ImportError:
    OPT_EINSUM_AVAILABLE = False


_PathOptimizerBase = opt_einsum.paths.PathOptimizer if OPT_EINSUM_AVAILABLE else object


def einsum_to_omeco_format(
    equation: str,
    shapes: Sequence[tuple[int, ...]],
) -> tuple[list[list[int]], list[int], dict[int, int], dict[str, int]]:
    """
    Convert einsum equation and tensor shapes to omeco (ixs, out, sizes) format.

    omeco uses integer indices, so we map each unique character to an integer.

    Args:
        equation: Einsum equation like "ibfj,iaep,xabcd,xefgh,jcgq->pdhq"
        shapes: Sequence of tensor shapes

    Returns:
        ixs: List of index lists for each tensor (integer indices)
        out: List of output indices (integer indices)
        sizes: Dictionary mapping integer indices to dimension sizes
        char_to_int: Mapping from character to integer index (for debugging)
    """
    if "->" not in equation:
        raise ValueError("Only explicit einsum equations with '->' are supported.")

    input_part, output_part = equation.split("->")
    input_subscripts = [s.strip() for s in input_part.split(",")]
    output_subscript = output_part.strip()

    if len(input_subscripts) != len(shapes):
        raise ValueError(
            f"Number of subscripts ({len(input_subscripts)}) doesn't match "
            f"number of shapes ({len(shapes)})"
        )

    # Build mapping from character to integer index
    all_chars: set[str] = set()
    for sub in input_subscripts:
        all_chars.update(sub)
    all_chars.update(output_subscript)
    char_to_int = {c: i for i, c in enumerate(sorted(all_chars))}

    # Convert to integer indices
    ixs = [[char_to_int[c] for c in sub] for sub in input_subscripts]
    out = [char_to_int[c] for c in output_subscript]

    # Build size dictionary with integer keys
    sizes: dict[int, int] = {}
    for subscript, shape in zip(input_subscripts, shapes):
        if len(subscript) != len(shape):
            raise ValueError(
                f"Subscript '{subscript}' has {len(subscript)} indices but "
                f"shape has {len(shape)} dimensions"
            )
        for idx_char, dim_size in zip(subscript, shape):
            idx_int = char_to_int[idx_char]
            if idx_int in sizes:
                if sizes[idx_int] != dim_size:
                    raise ValueError(
                        f"Inconsistent size for index '{idx_char}': "
                        f"{sizes[idx_int]} vs {dim_size}"
                    )
            else:
                sizes[idx_int] = dim_size

    return ixs, out, sizes, char_to_int


def omeco_tree_to_path(tree, num_tensors: int) -> list[tuple[int, int]]:
    """
    Convert omeco contraction tree to opt_einsum path format.

    opt_einsum uses a "linear" path format where each step specifies
    indices of operands to contract. After each contraction, the result
    is appended to the end and indices are adjusted.

    Args:
        tree: omeco contraction tree object
        num_tensors: Number of input tensors

    Returns:
        path: List of (i, j) tuples specifying contraction order
    """
    # omeco returns a tree structure. We need to extract the contraction
    # sequence. The tree typically has methods to get the path.
    if hasattr(tree, "flat_path"):
        # If omeco provides a flat_path method
        return tree.flat_path()

    if hasattr(tree, "get_path"):
        return tree.get_path()

    if hasattr(tree, "to_dict"):
        return _omeco_tree_dict_to_path(tree.to_dict())

    # Fallback: try to traverse the tree structure
    # This is a placeholder - actual implementation depends on omeco's tree format
    if hasattr(tree, "__iter__"):
        # If tree is iterable (list of contraction pairs)
        return list(tree)

    # If we can't extract the path, raise an error
    raise TypeError(
        f"Cannot convert omeco tree of type {type(tree)} to path. "
        "Please check omeco version and API."
    )


def _omeco_tree_dict_to_path(tree_dict) -> list[tuple[int, int]]:
    path = []
    current_indices: list[int] = []
    next_idx = 0

    def collect_leaves(node) -> None:
        if "tensor_index" in node:
            current_indices.append(int(node["tensor_index"]))
            return
        for child in node["args"]:
            collect_leaves(child)

    collect_leaves(tree_dict)
    current_indices.sort()
    next_idx = (max(current_indices) + 1) if current_indices else 0

    def process(node) -> int:
        nonlocal next_idx
        if "tensor_index" in node:
            return int(node["tensor_index"])
        args = node["args"]
        if len(args) != 2:
            raise TypeError("omeco tree conversion expects binary contraction nodes")
        left_idx = process(args[0])
        right_idx = process(args[1])
        pos_left = current_indices.index(left_idx)
        pos_right = current_indices.index(right_idx)
        if pos_left > pos_right:
            pos_left, pos_right = pos_right, pos_left
        path.append((pos_left, pos_right))
        new_idx = next_idx
        next_idx += 1
        current_indices.remove(left_idx)
        current_indices.remove(right_idx)
        current_indices.append(new_idx)
        return new_idx

    process(tree_dict)
    return path


class OmecoOptimizer(_PathOptimizerBase):
    """
    Path optimizer that uses omeco for contraction path finding.

    This class wraps omeco's optimization methods and provides a callable
    interface compatible with opt_einsum's path optimizer protocol.

    Args:
        method: Optimization method - "greedy", "treesa", or "treesa_slicer"
        optimize_tc: If True, optimize only time complexity (sc_weight=0).
                     If False (default), uses default balanced optimization.
        **kwargs: Additional arguments passed to the omeco method
    """

    def __init__(self, method: str = "greedy", optimize_tc: bool = False, **kwargs) -> None:
        if not OMECO_AVAILABLE:
            raise ImportError(
                "omeco is not installed. Install it with: pip install omeco"
            )
        self.method = method
        self.optimize_tc = optimize_tc
        self.kwargs = kwargs

    def _get_method_object(self):
        """Get the omeco method object based on method name."""
        method_map = {
            "greedy": omeco.GreedyMethod,
            "treesa": omeco.TreeSA,
            "treesa_slicer": omeco.TreeSASlicer,
        }
        if self.method not in method_map:
            raise ValueError(
                f"Unknown omeco method: {self.method}. "
                f"Available: {list(method_map.keys())}"
            )

        kwargs = self.kwargs.copy()

        # For TreeSA, support optimize_tc mode
        if self.method == "treesa" and self.optimize_tc:
            if "score" not in kwargs:
                # Use tc-only score function (sc_weight=0)
                kwargs["score"] = omeco.ScoreFunction(tc_weight=1.0, sc_weight=0.0)

        return method_map[self.method](**kwargs)

    def __call__(
        self,
        inputs: list[frozenset[str]],
        output: frozenset[str],
        size_dict: dict[str, int],
        memory_limit: int | None = None,
    ) -> list[tuple[int, int]]:
        """
        Find optimal contraction path using omeco.

        Args:
            inputs: List of frozensets, each containing index chars for a tensor
            output: Frozenset of output index chars
            size_dict: Dictionary mapping index chars to dimension sizes
            memory_limit: Optional memory limit (not used by all methods)

        Returns:
            path: List of (i, j) tuples specifying contraction order
        """
        # Build mapping from character to integer index
        all_chars: set[str] = set()
        for inp in inputs:
            all_chars.update(inp)
        all_chars.update(output)
        char_to_int = {c: i for i, c in enumerate(sorted(all_chars))}

        # Convert opt_einsum format (frozensets of chars) to omeco format (lists of ints)
        ixs = [[char_to_int[c] for c in sorted(inp)] for inp in inputs]
        out = [char_to_int[c] for c in sorted(output)]
        sizes = {char_to_int[c]: size_dict[c] for c in size_dict}

        method_obj = self._get_method_object()
        tree = omeco.optimize_code(ixs, out, sizes, method_obj)

        return omeco_tree_to_path(tree, len(inputs))

    def __repr__(self) -> str:
        return f"OmecoOptimizer(method={self.method!r})"


def get_omeco_complexity(
    equation: str,
    shapes: Sequence[tuple[int, ...]],
    method: str = "greedy",
    optimize_tc: bool = False,
    **kwargs,
) -> dict:
    """
    Get contraction complexity metrics using omeco.

    Args:
        equation: Einsum equation
        shapes: Sequence of tensor shapes
        method: Optimization method
        optimize_tc: If True, optimize only time complexity (sc_weight=0)
        **kwargs: Additional arguments for the method

    Returns:
        Dictionary with complexity metrics:
        - tree: The contraction tree
        - path: Contraction path in opt_einsum format
        - tc: Time complexity (log2 of FLOPs)
        - sc: Space complexity (log2 of max intermediate size)
        - complexity_obj: The raw complexity object
    """
    if not OMECO_AVAILABLE:
        raise ImportError("omeco is not installed")

    ixs, out, sizes, char_to_int = einsum_to_omeco_format(equation, shapes)

    method_map = {
        "greedy": omeco.GreedyMethod,
        "treesa": omeco.TreeSA,
        "treesa_slicer": omeco.TreeSASlicer,
    }

    # For TreeSA, support optimize_tc mode
    if method == "treesa" and optimize_tc and "score" not in kwargs:
        kwargs["score"] = omeco.ScoreFunction(tc_weight=1.0, sc_weight=0.0)

    method_obj = method_map[method](**kwargs)

    tree = omeco.optimize_code(ixs, out, sizes, method_obj)
    complexity = omeco.contraction_complexity(tree, ixs, sizes)

    path = omeco_tree_to_path(tree, len(ixs))

    return {
        "tree": tree,
        "path": path,
        "tc": complexity.tc,  # time complexity (log2 of flops)
        "sc": complexity.sc,  # space complexity (log2 of max intermediate)
        "complexity_obj": complexity,
        "char_to_int": char_to_int,
    }


if OPT_EINSUM_AVAILABLE:
    # Register as opt_einsum PathOptimizer subclass if available
    class OmecoPathOptimizer(OmecoOptimizer):
        """OmecoOptimizer as a proper opt_einsum PathOptimizer subclass."""

        def __init__(self, method: str = "greedy", optimize_tc: bool = False, **kwargs) -> None:
            super().__init__(method=method, optimize_tc=optimize_tc, **kwargs)
