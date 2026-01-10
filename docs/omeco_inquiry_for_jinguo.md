# omeco TreeSA 分析报告

## 摘要

我们对 omeco TreeSA 在 Kitaev CTMRG 方程上进行了详细分析，发现了一个有趣的现象。

## 发现

**FLOP 计数 vs 实际运行时间不一致：**

当 d/D < 0.5 时，TreeSA 找到的路径理论上有更多 FLOP，但**实际运行更快**。

## 详细数据

### 理论 FLOP 对比

Kitaev 方程：`iABt,ijkl,xjAp,xkBq,yJap,yKbq,labc->tJKc`

| d | D | d/D | FLOP 比 (omeco/opt) |
|---|---|-----|---------------------|
| 2 | 4 | 0.50 | 1.00x |
| 2 | 6 | 0.33 | 1.56x |
| 2 | 8 | 0.25 | 1.45x |
| 3 | 8 | 0.38 | 1.88x |
| 4 | 8 | 0.50 | 1.00x |

### 实际运行时间对比 (CPU, chi=40, d=2, D=6)

| 方法 | FLOP | 最大中间张量 | 运行时间 |
|------|------|-------------|----------|
| opt_einsum(optimal) | 4.48e+8 | **2.07e+6** | 14.7ms |
| omeco(TreeSA) | 6.97e+8 | **6.91e+5** | 10.9ms |

**关键发现：**
- omeco 有 1.56x 更多 FLOP
- 但 omeco 最大中间张量小 **3 倍**
- omeco 实际运行快 **0.74x**

## 路径对比

```
opt_einsum path: [(0, 1), (0, 5), (0, 4), (0, 3), (0, 2), (0, 1)]
  Largest intermediate: 2,074,000 elements

omeco path: [(4, 6), (0, 2), (0, 1), (2, 3), (0, 2), (0, 1)]
  Largest intermediate: 691,200 elements (3x smaller!)
```

## 分析

TreeSA 似乎在优化**内存效率**而非单纯的 FLOP 数。较小的中间张量带来：
1. 更好的 CPU 缓存利用率
2. 更少的内存分配/释放开销
3. 更快的实际运行时间

## 问题

1. TreeSA 的代价函数是什么？是否有意优化内存占用？

2. 为什么 d/D >= 0.5 时两种方法找到相同路径，而 d/D < 0.5 时不同？

3. 如果 TreeSA 确实在优化内存效率，这对 GPU 场景如何？（GPU 通常更受 FLOP 限制）

## 复现代码

```python
import omeco
import opt_einsum
import numpy as np
import time

equation = 'iABt,ijkl,xjAp,xkBq,yJap,yKbq,labc->tJKc'
chi, d, D = 40, 2, 6

shapes = [(chi,D,D,chi), (chi,D,D,chi), (d,D,D,D), (d,D,D,D),
          (d,D,D,D), (d,D,D,D), (chi,D,D,chi)]

# opt_einsum
arrays = [np.random.randn(*s) for s in shapes]
path_opt, info_opt = opt_einsum.contract_path(equation, *arrays, optimize='optimal')
print(f'opt_einsum FLOP: {float(info_opt.opt_cost)/2:.2e}')
print(f'opt_einsum largest: {info_opt.largest_intermediate:.2e} elements')

# omeco
input_part, output_part = equation.split('->')
subscripts = input_part.split(',')
all_chars = set()
for sub in subscripts:
    all_chars.update(sub)
all_chars.update(output_part)
char_to_int = {c: i for i, c in enumerate(sorted(all_chars))}

ixs = [[char_to_int[c] for c in sub] for sub in subscripts]
out = [char_to_int[c] for c in output_part]
sizes = {char_to_int[c]: s for sub, shape in zip(subscripts, shapes)
         for c, s in zip(sub, shape)}

tree = omeco.optimize_code(ixs, out, sizes, omeco.TreeSA())
comp = omeco.contraction_complexity(tree, ixs, sizes)
print(f'omeco FLOP: {2**comp.tc:.2e}')
print(f'omeco sc (log2 largest): {comp.sc}')
print(f'omeco largest: {2**comp.sc:.2e} elements')
```

## 环境

- omeco 0.2.0
- Python 3.13
- macOS (Apple Silicon)
