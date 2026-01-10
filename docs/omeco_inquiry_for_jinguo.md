# omeco TreeSA 在 d/D < 0.5 时的问题

## 发现

我们在测试 omeco 时发现一个规律：**当 d/D < 0.5 时，TreeSA 找到的路径比 opt_einsum(optimal) 差 1.6x-2.2x**。

## 测试数据

Kitaev CTMRG 方程：`iABt,ijkl,xjAp,xkBq,yJap,yKbq,labc->tJKc`

| d | D | d/D | omeco/opt_einsum | 状态 |
|---|---|-----|------------------|------|
| 2 | 4 | 0.50 | 1.00x | OK |
| 3 | 4 | 0.75 | 1.00x | OK |
| 4 | 4 | 1.00 | 1.00x | OK |
| 2 | 6 | 0.33 | 1.69x | **FAIL** |
| 3 | 6 | 0.50 | 1.00x | OK |
| 2 | 8 | 0.25 | 1.61x | **FAIL** |
| 3 | 8 | 0.38 | 2.18x | **FAIL** |
| 4 | 8 | 0.50 | 1.00x | OK |

**规律**：d/D >= 0.5 时正常，d/D < 0.5 时异常。

## 实际影响

对于自旋-1/2系统 (d=2)：
- D=4: 正常
- D=5,6,7,8,...: TreeSA 找到次优路径

## 排除的可能性

已测试但无效的调整：
1. ❌ 增加迭代次数 (ntrials=100, niters=200)
2. ❌ 改变温度参数 (各种 betas)
3. ❌ 改变索引排序方式
4. ❌ 使用 `optimize_treesa()` 而非 `optimize_code()`

TreeSA 每次返回完全相同的结果（无随机性），说明它稳定收敛到一个局部最优。

## 复现代码

```python
import omeco
import opt_einsum
import numpy as np

equation = 'iABt,ijkl,xjAp,xkBq,yJap,yKbq,labc->tJKc'
chi, d, D = 64, 2, 8  # d/D = 0.25 < 0.5, 会有问题
shapes = [(chi,D,D,chi), (chi,D,D,chi), (d,D,D,D), (d,D,D,D),
          (d,D,D,D), (d,D,D,D), (chi,D,D,chi)]

# opt_einsum
arrays = [np.empty(s) for s in shapes]
_, info = opt_einsum.contract_path(equation, *arrays, optimize='optimal')
opt_flops = float(info.opt_cost) / 2  # 归一化 MAC 计数

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
omeco_flops = 2 ** comp.tc

print(f'opt_einsum: {opt_flops:.2e}')
print(f'omeco:      {omeco_flops:.2e}')
print(f'ratio:      {omeco_flops/opt_flops:.2f}x')
# 输出: ratio: 1.61x
```

## 问题

这是预期行为还是 bug？如果是预期行为，有什么方法可以改善 d/D < 0.5 情况下的结果？

## 环境

- omeco 0.2.0
- Python 3.13
- macOS (Apple Silicon)
