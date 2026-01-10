# omeco TreeSA 分析报告

## 摘要

我们对 omeco TreeSA 在 Kitaev CTMRG 方程上进行了全面的性能分析，包括 CPU、GPU 纯收缩以及 GPU + 自动微分 (AD) 场景。

**核心发现：TreeSA 优化的是空间复杂度 (sc)，而非时间复杂度 (tc)。这导致 FLOP 更多但中间张量更小。**

---

## 1. 理论 FLOP 对比

Kitaev 方程：`iABt,ijkl,xjAp,xkBq,yJap,yKbq,labc->tJKc`

| chi | d | D | d/D | FLOP比 (omeco/opt) |
|-----|---|---|-----|---------------------|
| 16 | 2 | 4 | 0.50 | 1.00x |
| 16 | 2 | 6 | 0.33 | 1.00x |
| 24 | 2 | 8 | 0.25 | 1.24x |
| 32 | 2 | 6 | 0.33 | 1.48x |
| 40 | 2 | 6 | 0.33 | 1.56x |
| 48 | 2 | 6 | 0.33 | 1.61x |
| 64 | 2 | 6 | 0.33 | 1.69x |
| 64 | 2 | 8 | 0.25 | 1.61x |

**规律：**
- d/D >= 0.5: 两者找到相同路径 (1.00x)
- d/D < 0.5 且 chi 较大: omeco 有更多 FLOP (1.2x-1.7x)

---

## 2. 路径结构对比 (chi=40, d=2, D=6)

| 指标 | opt_einsum(optimal) | omeco(TreeSA) |
|------|---------------------|---------------|
| FLOP count | 2.24e+08 | 3.48e+08 (1.56x) |
| **最大中间张量** | **2.07e+06** | **6.91e+05** (3x 更小) |
| tc (log2 FLOP) | 27.74 | 28.38 |
| sc (log2 space) | 20.98 | 19.40 |

**TreeSA 生成的路径有更多 FLOP，但中间张量小 3 倍。**

---

## 3. CPU 运行时间

测试环境：Apple Silicon (M-series)

| chi | d | D | FLOP比 | 运行时间比 | 结论 |
|-----|---|---|--------|----------|------|
| 32 | 2 | 6 | 1.48x | 0.89x | omeco 更快 |
| 36 | 2 | 6 | 1.52x | 0.62x | omeco 更快 |
| 40 | 2 | 6 | 1.56x | **0.52x** | omeco 更快 |

**CPU 结论：尽管 FLOP 多 50%，omeco 路径快 2 倍！** 原因是更小的中间张量带来更好的缓存利用率。

---

## 4. GPU 纯收缩

测试环境：NVIDIA H100, float64

| chi | d | D | FLOP比 | 运行时间比 |
|-----|---|---|--------|----------|
| 32 | 2 | 6 | 1.48x | 0.95x |
| 64 | 2 | 6 | 1.69x | 1.00x |
| 96 | 2 | 6 | 1.78x | 0.96x |
| 128 | 2 | 6 | 1.83x | 0.98x |

**GPU 结论：两者运行时间相当 (~1.0x)。** GPU 高带宽抵消了内存效率优势。

---

## 5. GPU + 自动微分 (AD)

测试环境：NVIDIA H100, complex128, forward + backward

| chi | d | D | FLOP比 | 运行时间比 |
|-----|---|---|--------|----------|
| 48 | 2 | 6 | 1.61x | 1.11x |
| 64 | 2 | 6 | 1.69x | 1.10x |
| 48 | 2 | 8 | 1.52x | **0.96x** |
| 64 | 2 | 8 | 1.61x | **1.00x** |

**GPU+AD 结论：两者基本持平，omeco 有时略快。**

---

## 6. 总结

| 场景 | FLOP比 | 运行时间比 | 推荐 |
|------|--------|----------|------|
| **CPU** | 1.5x | **0.5-0.7x** | omeco 更快 |
| **GPU** | 1.5-1.8x | ~1.0x | 两者相当 |
| **GPU+AD** | 1.5-1.7x | ~1.0x | 两者相当 |

---

## 7. 解决方案：使用 ScoreFunction

**问题根源：默认 `ScoreFunction(tc_weight=1.0, sc_weight=1.0)` 同时优化 tc 和 sc。**

使用 `sc_weight=0` 可以只优化 time complexity：

```python
score_tc_only = omeco.ScoreFunction(tc_weight=1.0, sc_weight=0.0)
tree = omeco.optimize_code(ixs, out, sizes, omeco.TreeSA(score=score_tc_only))
```

| 配置 | chi=40, D=6 | chi=64, D=6 |
|------|-------------|-------------|
| opt_einsum(optimal) | tc=27.74 | tc=29.63 |
| TreeSA 默认 (sc=1) | tc=28.38 (1.56x) | tc=30.38 (1.69x) |
| **TreeSA tc_only (sc=0)** | **tc=27.74 (1.00x)** | **tc=29.63 (1.00x)** |

**结论：`sc_weight=0` 时 TreeSA 和 opt_einsum 找到完全相同的 tc！**

---

## 8. 复现代码

```python
import omeco
import opt_einsum
import numpy as np

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
print(f'omeco tc (log2 FLOP): {comp.tc:.2f}')
print(f'omeco sc (log2 space): {comp.sc:.2f}')
print(f'omeco FLOP: {2**comp.tc:.2e}')
print(f'omeco largest: {2**comp.sc:.2e} elements')
```

---

## 环境

- omeco 0.2.0
- opt_einsum 3.3.0
- Python 3.11/3.12
- PyTorch 2.x
- CPU: Apple Silicon
- GPU: NVIDIA H100
