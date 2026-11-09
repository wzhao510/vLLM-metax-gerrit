# QuaRot W4A8 Online Rotation 参考实现说明

## 1. 背景

QuaRot 会将 Transformer 主干 residual stream 旋转到新的正交坐标系。设原始 hidden state 为：

$$
X \in \mathbb{R}^{T \times N},
$$

旋转后的 residual 为：

$$
X_{\mathrm{rot}} = XQ.
$$

对于普通 Decoder Block，可以通过离线修改相邻 Linear 权重，让 residual stream 一直保持在旋转坐标系中。

但在 Final Norm 位置存在特殊情况。对于带 MTP 的模型，Final Norm 的输出不仅进入 LM Head，也可能继续作为 MTP hidden state：

```text
main decoder output
        ↓
    final norm
     ├── lm_head
     └── MTP hidden / hnorm
```

如果只通过修改 Head 权重补偿 rotation，只能保证 Head logits 路径等价，MTP 接收到的 hidden state 仍然发生变化。

因此，我们的参考实现在 Final Norm 前增加 online inverse rotation，以结束旋转域：

```text
rotated hidden XQ
        ↓
online Q^-1
        ↓
original hidden X
        ↓
original Final Norm
     ├── LM Head
     └── MTP hidden
```

即：

$$
XQ \xrightarrow{\text{online }Q^{-1}} X
\xrightarrow{\text{original Final Norm}}
\text{Head / MTP}.
$$

这样 Final Norm、LM Head 和 MTP 后续路径都可以继续保持原模型语义。

---

## 2. 为什么需要 Standalone Online Rotation

vllm 官方现有 online Hadamard 更适合绑定在某个 Linear 前而且要求变换维度是 2 的幂，例如：

```text
activation（维度是 2 的幂）
   ↓
online Hadamard
   ↓
down_proj
```

本次 QuaRot 的需求不同。

rotation 位于：

```text
residual stream
      ↓
online inverse rotation
      ↓
Final Norm
```

它不属于单个 Linear，因此需要一个可以直接在模型 forward 中调用的 standalone rotation module。

模型侧只需要在需要的位置显式挂载：

```python
if self.prenorm_hadamard_rotation is not None:
    hidden_states = self.prenorm_hadamard_rotation(hidden_states)
```

具体插入位置仍由量化根据模型结构决定。

---

## 3. Online Rotation 与 Composite Hadamard

当前实现提供统一的 online rotation 配置入口，并通过：

```python
rotation_config = get_online_rotation_module_config(config, module_name)
rotation = build_rotation(rotation_config)
```

根据具体 module 的 `type` 构造 rotation。当前 QuaRot W4A8 场景使用：

```text
type = composite_hadamard
```

对应 `CompositeHadamardRotation`。

框架接入后希望能够在统一入口下支持以下两类 rotation：

```text
CompositeHadamardRotation：H_K ⊗ H_P + sign 的结构化快速旋转
DenseOrthogonalRotation：显式保存完整正交矩阵 Q，运行时直接执行 xQ
```

对应的 `type` 可分别使用：

```text
composite_hadamard
dense_orthogonal
```

当前参考 patch 实现和使用的是 `composite_hadamard`。后续扩展其他 `type` 时，不需要修改模型侧的挂载逻辑；具体类名和配置命名可以按照 vLLM 框架规范调整。

目标 rotation 支持：

$$
N = K \times P,\qquad P=2^m,
$$

并实现：

$$
Q =
D\frac{H_K^T \otimes H_P}{\sqrt N}.
$$

其中：

- `N`：rotation dimension；
- `K`：base order；
- `P=N/K`：power-of-two 部分；
- `D`：离线 QuaRot 阶段生成的 random sign；
- `H_K`：与离线转换保持一致的 Hadamard base。

当 `K=1` 时，可以退化为标准 power-of-two Hadamard。

因此除了标准 Hadamard 外，也可以支持：

```text
6144 = 12 × 512
7168 = 28 × 256
14336 = 28 × 512
```

等非 power-of-two 维度。

`sign` 和 `hk` 与离线 QuaRot 使用的具体 rotation 绑定，因此需要作为模型 checkpoint state 保存。

---

## 4. 本次 Patch 修改

当前 patch 主要完成两部分：

```text
1. 基于 online_rotation 配置和 build_rotation 的 standalone rotation 接入
2. Main Final Norm / MTP SharedHead 的 pre-norm rotation
```

其中，当前 Main Final Norm / MTP SharedHead 的 pre-norm rotation 是为 GLM-5.2 和 GLM-5.3 设计的。

当前参考 patch 涉及：

```text
models/deepseek_v2.py
models/deepseek_mtp.py
```

### 4.1 Main Model

在模型初始化时，根据当前 module key 查询是否存在 rotation 配置。

如果存在，则通过统一工厂创建对应 rotation：

```python
rotation_config = get_online_rotation_module_config(
    config,
    rotation_module_name,
)
self.prenorm_hadamard_rotation = build_rotation(rotation_config)
```

否则保持：

```python
self.prenorm_hadamard_rotation = None
```

因此普通模型不会进入新的执行路径。

Final Norm 前的数据流由：

```text
hidden + residual
      ↓
RMSNorm
```

修改为：

```text
hidden + residual
      ↓
online inverse Hadamard
      ↓
RMSNorm
```

这里 rotation 必须作用在最终 residual sum 上。

---

### 4.2 MTP SharedHead

MTP `SharedHead` 同样在自己的 Final Norm 前增加可选 rotation：

```text
MTP rotated hidden
        ↓
online Q^-1
        ↓
shared_head.norm
        ↓
head / next MTP state
```

这样 Main 和 MTP 两条 post-norm 路径都可以恢复原模型语义。

---

## 5. 配置与 Checkpoint

不同挂载位置可能具有不同的：

```text
type
hidden_size
K
inverse
```

因此这些参数应该属于具体 module。

当前参考实现的配置形式：

```json
{
  "online_rotation": {
    "modules": {
      "model.prenorm_hadamard_rotation": {
        "type": "composite_hadamard",
        "hidden_size": 6144,
        "K": 12,
        "inverse": true
      },
      "model.layers.78.shared_head.prenorm_hadamard_rotation": {
        "type": "composite_hadamard",
        "hidden_size": 6144,
        "K": 12,
        "inverse": true
      }
    }
  }
}
```

其中：

$$
P=\frac{\text{hidden\_size}}{K}
$$

可由运行时直接推导。

对应 checkpoint：

```text
model.prenorm_hadamard_rotation.sign
model.prenorm_hadamard_rotation.hk

model.layers.78.shared_head.prenorm_hadamard_rotation.sign
model.layers.78.shared_head.prenorm_hadamard_rotation.hk
```

即：

```text
config module path
        ↕
PyTorch module path
        ↕
checkpoint key prefix
```

保持一致。

不同位置即使使用相同 rotation，也可以分别保存一份 `sign/hk`。这部分数据很小，可以避免额外引入共享 storage 或特殊 load-weight 逻辑。

其中，`sign` 的形状为 `[N]`，`hk` 的形状为 `[K, K]`。对于当前配置 `N = 6144`、`K = 12`：

| 资产 | 元素数 | BF16 / FP16 | FP32 |
| --- | ---: | ---: | ---: |
| `prenorm_hadamard_rotation.sign` | 6,144 | 12 KiB | 24 KiB |
| `prenorm_hadamard_rotation.hk` | 144 | 288 B | 576 B |
| 合计 | 6,288 | 12.28 KiB | 24.56 KiB |

---

## 6. 当前获得的能力

本次修改后，可以支持：

### 6.1 Final Norm 前退出 QuaRot 旋转域

```text
rotated residual
    ↓
online Q^-1
    ↓
original Final Norm
```

从而让 Final Norm 后的 Head / MTP hidden 恢复到原模型坐标。

### 6.2 避免仅为 QuaRot 拆分 Main/MTP Head

如果通过 pre-norm inverse rotation 恢复原坐标，则 Final Norm 和共享 Head 可以继续保留原模型结构。

### 6.3 支持非 power-of-two Hadamard

支持：

$$
N=K\times2^m
$$

而不仅是标准 \(2^m\) Hadamard。

### 6.4 提供可复用的 standalone rotation module

具体模型只需要：

```python
if self.prenorm_hadamard_rotation is not None:
    hidden_states = self.prenorm_hadamard_rotation(hidden_states)
```

即可在需要的位置接入。

具体插入点仍由量化侧提供。

---

## 7. 希望协助评估的内容

当前 vLLM-MetaX 实现可以作为设计和代码参考，不要求最终接入方式与当前 patch 保持一致。主要希望协助评估以下内容。

### 7.1 框架接入方式

评估 standalone online rotation 是否适合接入 vLLM 框架，以及比较合适的模块归属、接口形式和首期适配范围。

### 7.2 性能与 Runtime 兼容

当前 Main Final Norm 路径从：

```text
fused residual add + norm
```

变成：

```text
residual add
    ↓
online Hadamard
    ↓
norm
```

因此需要重点关注：

- online Hadamard kernel latency；
- 显式 residual add 带来的额外读写；
- 端到端 decode / prefill 开销。

如果这部分开销明显，后续可以考虑：

```text
add + Hadamard
```

或者：

```text
add + Hadamard + norm
```

进一步融合。

同时需要评估以下运行方式的兼容性：

```text
torch.compile
CUDA Graph
PP
```

等现有运行路径下均可正常工作。

---

### 7.3 Rotation 类型与扩展

接入后希望通过统一的 `online_rotation` 配置入口支持 `composite_hadamard` 和 `dense_orthogonal`。当前参考实现的配置与加载流程为：

```text
online_rotation.modules
    → module path
    → type + rotation config
    → build_rotation
    → sign/hk checkpoint key
```

后续可以根据框架现有规范进一步确认：

- rotation type 的扩展方式；
- dense orthogonal matrix 的 checkpoint 组织及加载方式；
- `sign/hk` 的 buffer/load-weight 方式；

---


## 8. 总结

本次 patch 主要补充的是 QuaRot 所需要的 standalone online rotation：

$$
\boxed{
XQ
\xrightarrow{\text{online }Q^{-1}}
X
\xrightarrow{\text{Final Norm}}
\text{Head / MTP}
}
$$

当前已经完成：

```text
online_rotation 统一配置入口
get_online_rotation_module_config
build_rotation
CompositeHadamardRotation（composite_hadamard）
Main Final Norm pre-rotation
MTP SharedHead pre-rotation
PP boundary handling
config-driven optional loading
```

后续主要希望 vLLM 组协助评估：

```text
框架接入方式
运行开销及 runtime 兼容性
Composite Hadamard / Dense Orthogonal 的统一扩展方式
```

现有 vLLM-MetaX patch 仅作为参考，不限定最终接入方式必须沿用当前代码结构。我们可以配合补充数值正确性、性能及运行兼容性测试。
