# HSM架构图细节分析与改进建议

## 📊 当前架构图分析

### ✅ **做得好的地方**

1. **整体布局清晰**：三部分结构（输入→采样→循环处理）逻辑清楚
2. **颜色编码合理**：不同模块用不同颜色区分（蓝/橙/绿）
3. **数据流明确**：箭头方向清晰，展示了从输入到输出的完整流程
4. **视觉元素丰富**：使用图标（地球、放大镜、大脑、正弦波等）增强可读性
5. **循环结构突出**：用虚线框明确标注了Recurrent Loop

---

## 🔍 **详细问题分析与改进建议**

### 1. **Glance Network部分**

**当前状态**：
- ✅ 有3个Conv块
- ✅ 有2个CoordAttention模块
- ✅ 有Global Average Pooling
- ✅ 输出g

**需要改进**：
- ❌ **缺少维度标注**：应该明确标注 `g ∈ ℝ²⁵⁶` 或 `d=256`
- ❌ **CoordAttention位置不明确**：论文说是在Conv Block 2和3**之后**，图中应该更明确标注位置
- ⚠️ **Conv块数量**：论文描述是"convolutional encoder with two CoordAttention modules inserted after the second and third convolutional blocks"，这意味着可能有3个或更多Conv块，需要确认

**改进建议**：
```
Glance Network G
├─ Conv Block 1
├─ Conv Block 2
├─ CoordAttention ①  ← 明确标注位置
├─ Conv Block 3
├─ CoordAttention ②  ← 明确标注位置
├─ Global Average Pooling
└─ Output: g ∈ ℝ²⁵⁶  ← 添加维度标注
```

---

### 2. **Hierarchical Sampler部分**

**当前状态**：
- ✅ 展示了Explore (ε=0.60) 和 Exploit (ε=0.30) 两个阶段
- ✅ 有视觉化展示（散点vs聚集）
- ✅ 输入了Saliency Map S

**需要改进**：
- ❌ **缺少关键参数**：
  - 温度参数 `τ = 0.12` 未标注
  - Phase切换点 `α = 0.4` (t < 0.4T) 应该更明确
  - Momentum参数 `μ = 0.45` 未显示
  - Step clamp参数 `s_max = 0.18` 未显示
- ❌ **采样公式不完整**：应该显示完整的采样逻辑
  - `Uniform([0,1]²)` with prob ε
  - `Multinomial(softmax(S/τ))` with prob (1-ε)
- ❌ **Momentum Smoothing未展示**：这是关键组件，应该单独标注
- ❌ **输入来源不完整**：Sampler应该接收 `p_{t-1}` (前一步的预测位置)用于momentum，但图中似乎缺少这个连接

**改进建议**：
```
Hierarchical Saliency-Guided Sampler
├─ Input: Saliency Map S, p_{t-1} (previous fixation)
├─ Phase Decision:
│   ├─ if t < 0.4T: ε = 0.60 (Phase 1: High Exploration)
│   └─ else: ε = 0.30 (Phase 2: Low Exploration)
├─ Sampling:
│   ├─ with prob ε: c_t ~ Uniform([0,1]²)
│   └─ with prob (1-ε): c_t ~ Multinomial(softmax(S/τ)), τ=0.12
├─ Momentum Smoothing: c_t ← (1-μ)c_t + μ(p_{t-1} + δ·s_max)
│   └─ μ=0.45, s_max=0.18
└─ Step Clamp: ||c_t - p_{t-1}|| ≤ s_max
```

---

### 3. **Focus Network部分**

**当前状态**：
- ✅ 有CoordAttention模块
- ✅ 有Conv块
- ✅ 输入了candidate c_t
- ✅ 有momentum反馈（p̂_{t-1}）

**需要改进**：
- ❌ **Patch提取细节缺失**：
  - Patch size: `min(H,W)/4` 未标注
  - Resize to `64×64` 未标注
- ❌ **Conv块数量不明确**：论文说"two convolutional blocks"，图中显示3个绿色块，需要确认
- ❌ **CoordAttention位置**：应该在Conv Block 2之后，需要明确标注
- ❌ **缺少维度标注**：输出 `f_t ∈ ℝ²⁵⁶` 未显示
- ⚠️ **输入来源**：Focus Network的输入应该是 `I` (原图) 和 `c_t`，但图中似乎只显示了patch，应该明确标注是从I中提取的patch

**改进建议**：
```
Focus Network F
├─ Input: Image I, candidate c_t
├─ Extract Patch: size = min(H,W)/4, centered at c_t
├─ Resize: 64×64
├─ Conv Block 1
├─ Conv Block 2
├─ CoordAttention  ← 明确标注位置
├─ Global Average Pooling
└─ Output: f_t ∈ ℝ²⁵⁶  ← 添加维度标注
```

---

### 4. **Feature Fusion部分**

**当前状态**：
- ✅ 展示了三个输入：g, f_t, pos encoding
- ✅ 有加法操作

**需要改进**：
- ❌ **公式不完整**：应该显示完整公式 `h_t = g + f_t + MLP_pos(p_t)`
- ❌ **MLP_pos未明确**：应该标注这是一个MLP，而不是简单的"pos encoding"
- ❌ **输入p_t的来源**：应该是 `c_t` (candidate) 或 `p_{t-1}`，需要明确
- ❌ **缺少维度标注**：输出 `h_t ∈ ℝ²⁵⁶` 未显示

**改进建议**：
```
Feature Fusion
├─ Input:
│   ├─ g ∈ ℝ²⁵⁶ (from Glance Network)
│   ├─ f_t ∈ ℝ²⁵⁶ (from Focus Network)
│   └─ MLP_pos(p_t) ∈ ℝ²⁵⁶ (position encoding)
├─ Operation: h_t = g + f_t + MLP_pos(p_t)
└─ Output: h_t ∈ ℝ²⁵⁶
```

---

### 5. **Mamba Block部分**

**当前状态**：
- ✅ 有Mamba SSM表示
- ✅ 有MLP decoder
- ✅ 有序列处理示意

**需要改进**：
- ❌ **关键参数缺失**：
  - `d_model = 256`
  - `d_state = 256`
  - `d_conv = 4`
  - `expansion factor = 2`
- ❌ **Mamba特性未展示**：
  - Selective state-space mechanism
  - Input-dependent gating
  - Full history attention
  - Learned IOR (Inhibition of Return)
- ❌ **序列表示不清晰**：应该明确显示处理的是序列 `{h_1, h_2, ..., h_T}` → `{o_1, o_2, ..., o_T}`
- ❌ **缺少维度标注**：输入输出维度未标注

**改进建议**：
```
Mamba Sequence Model
├─ Input: Sequence {h_1, h_2, ..., h_T}, each h_t ∈ ℝ²⁵⁶
├─ Mamba Block:
│   ├─ d_model = 256
│   ├─ d_state = 256
│   ├─ d_conv = 4
│   ├─ expansion = 2
│   ├─ Selective State-Space Mechanism
│   ├─ Input-dependent Gating
│   └─ Full History Attention (Learned IOR)
├─ Output: Sequence {o_1, o_2, ..., o_T}, each o_t ∈ ℝ²⁵⁶
└─ MLP_dec: o_t → p̂_t
```

---

### 6. **Position Decoder部分**

**当前状态**：
- ✅ 有MLP和Sigmoid
- ✅ 输出360° p̂_t

**需要改进**：
- ❌ **公式不完整**：应该显示 `p̂_t = σ(MLP_dec(o_t))`
- ❌ **MLP层数未标注**：论文说是"two-layer MLP"
- ❌ **输出维度**：应该标注 `p̂_t ∈ [0,1]²`
- ❌ **最终输出格式**：应该显示完整的scanpath `P̂ = {p̂_1, p̂_2, ..., p̂_T}`

**改进建议**：
```
Position Decoder
├─ Input: o_t ∈ ℝ²⁵⁶ (from Mamba Block)
├─ MLP_dec: 2-layer MLP
├─ Activation: Sigmoid
├─ Formula: p̂_t = σ(MLP_dec(o_t))
└─ Output: p̂_t ∈ [0,1]²
```

---

### 7. **数据流和连接问题**

**需要改进**：
- ❌ **缺少关键反馈连接**：
  - `p̂_t` 应该反馈到下一时刻的Sampler（用于momentum）
  - 当前图中似乎有momentum反馈，但连接关系不够清晰
- ❌ **Teacher Forcing未展示**：训练时使用teacher forcing，但图中未体现
- ❌ **时间步标注**：Recurrent Loop应该更明确标注 `t = 1, 2, ..., T`
- ❌ **全局特征g的复用**：g在整个序列中复用，这个信息应该更突出

**改进建议**：
```
数据流应该明确显示：
1. Glance Network: I → g (计算一次，全局复用)
2. For t = 1 to T:
   a. Hierarchical Sampler: S, p_{t-1} → c_t
   b. Focus Network: I, c_t → f_t
   c. Feature Fusion: g, f_t, MLP_pos(c_t) → h_t
   d. Mamba: {h_1, ..., h_t} → o_t
   e. Decoder: o_t → p̂_t
   f. Feedback: p̂_t → p_{t-1} (for next step)
```

---

### 8. **Training Loss部分**

**当前状态**：
- ✅ 有Soft-DTW和Coverage Loss的视觉表示

**需要改进**：
- ❌ **公式缺失**：应该显示完整的损失函数公式
  - `L_path = SoftDTW_γ(P̂, P*)`, γ = 0.1
  - `L_cov = -(2w_x·(σ_x+r_x)/2 + w_y·(σ_y+r_y)/2)`, w_x=2.0, w_y=1.0
  - `L = L_path + λ·L_cov`, λ = 0.05
- ❌ **参数未标注**：所有超参数应该标注
- ❌ **连接关系**：应该明确显示Loss如何连接到各个模块（虽然Loss不直接影响前向传播，但应该标注训练目标）

**改进建议**：
```
Training Loss
├─ Soft-DTW Loss:
│   └─ L_path = SoftDTW_γ(P̂, P*), γ = 0.1
├─ Coverage Loss:
│   └─ L_cov = -(2w_x·(σ_x+r_x)/2 + w_y·(σ_y+r_y)/2)
│       └─ w_x = 2.0, w_y = 1.0
└─ Total Loss: L = L_path + λ·L_cov, λ = 0.05
```

---

## 🎨 **视觉设计改进建议**

### 1. **数学符号规范化**
- 所有维度应该使用数学符号：`ℝ²⁵⁶`, `[0,1]²`
- 公式应该使用正确的数学排版
- 参数应该用等号明确标注：`ε = 0.60`, `τ = 0.12`

### 2. **颜色编码优化**
- 建议统一颜色方案：
  - Glance Network: 蓝色系
  - Hierarchical Sampler: 绿色系（或保持橙色，但要与Focus区分）
  - Focus Network: 橙色系
  - Feature Fusion: 浅紫色
  - Mamba Block: 深紫色
  - CoordAttention: 红色强调（当前是绿色，建议改）

### 3. **布局优化**
- 当前布局基本合理，但可以考虑：
  - 将Training Loss移到右上角或单独区域，避免与主流程混淆
  - 更明确地展示时间步的迭代关系
  - 可以考虑将Recurrent Loop画得更像一个循环，而不是线性流程

### 4. **标注完整性**
- 所有关键参数都应该标注
- 所有维度信息都应该显示
- 所有公式都应该完整

---

## 📋 **优先级改进清单**

### 🔴 **高优先级（必须修改）**
1. ✅ 添加所有维度标注（ℝ²⁵⁶, [0,1]²等）
2. ✅ 补充Hierarchical Sampler的关键参数（τ, μ, s_max, α）
3. ✅ 明确Momentum Smoothing的公式和位置
4. ✅ 补充Mamba Block的关键参数（d_model, d_state等）
5. ✅ 完善所有公式（h_t, p̂_t, Loss等）

### 🟡 **中优先级（建议修改）**
6. ⚠️ 明确CoordAttention的位置标注
7. ⚠️ 补充Patch提取的细节（size, resize）
8. ⚠️ 优化数据流连接（特别是反馈路径）
9. ⚠️ 统一颜色编码方案

### 🟢 **低优先级（可选优化）**
10. 💡 优化Training Loss的展示位置
11. 💡 增强时间步的视觉表示
12. 💡 添加更多视觉辅助元素（如参数卡片）

---

## 🎯 **总结**

当前架构图在**整体布局**和**视觉设计**方面已经做得很好，但在**技术细节**和**数学标注**方面还有较大改进空间。主要问题是：

1. **参数标注不完整**：很多关键超参数未显示
2. **公式不完整**：很多公式只显示了概念，没有完整的数学表达式
3. **维度信息缺失**：所有中间表示和输出的维度都未标注
4. **细节不够精确**：一些组件的具体实现细节（如patch size, MLP层数等）未明确

建议按照上述改进清单，逐步完善这些细节，使架构图既美观又准确，符合顶级会议的标准。
