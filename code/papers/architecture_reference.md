# MambaGaze 架构图参考

## 整体流程图

```
输入
┌─────────────────────────────────────────────────────────────────┐
│  360° Panoramic Image I (3×H×W)    Saliency Map S (1×H×W)      │
└───────────────┬─────────────────────────────┬───────────────────┘
                │                             │
                ▼                             │
┌───────────────────────────┐                 │
│      Glance Network G     │                 │
│  ┌─────────────────────┐  │                 │
│  │  Conv Block 1       │  │                 │
│  │  Conv Block 2       │  │                 │
│  │  CoordAttention ①   │  │                 │
│  │  Conv Block 3       │  │                 │
│  │  CoordAttention ②   │  │                 │
│  │  GlobalAvgPool      │  │                 │
│  └─────────────────────┘  │                 │
│   global feature g ∈ R^d  │                 │
└───────────────┬───────────┘                 │
                │                             │
                │         ┌───────────────────┘
                │         │
                │         ▼
                │  ┌──────────────────────────────────────────┐
                │  │   Hierarchical Saliency-Guided Sampler   │
                │  │                                          │
                │  │  step t:                                 │
                │  │  ┌─────────────────────────────────┐    │
                │  │  │  if t < 0.4T  →  ε = 0.60       │    │
                │  │  │  else         →  ε = 0.30       │    │
                │  │  └─────────────────────────────────┘    │
                │  │                                          │
                │  │  with prob ε:  c_t ~ Uniform([0,1]²)    │
                │  │  otherwise:    c_t ~ Multinomial(S/τ)   │
                │  │                                          │
                │  │  momentum smoothing (μ=0.45):            │
                │  │  c_t ← (1-μ)c_t + μ(p_{t-1} + δs_max)  │
                │  │  step clamp: ‖c_t - p_{t-1}‖ ≤ s_max   │
                │  └──────────────────┬───────────────────────┘
                │                     │ candidate fixation c_t
                │                     ▼
                │  ┌──────────────────────────────────────────┐
                │  │         Focus Network F                  │
                │  │                                          │
                │  │  extract patch at c_t (size H/4×W/4)    │
                │  │  resize to 64×64                         │
                │  │  Conv Block 1                            │
                │  │  Conv Block 2 → CoordAttention           │
                │  │  GlobalAvgPool                           │
                │  │  local feature f_t ∈ R^d                │
                │  └──────────────────┬───────────────────────┘
                │                     │
                └──────────┬──────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Feature Fusion                                 │
│                                                                  │
│   h_t = g  +  f_t  +  MLP_pos(p_t)                             │
│         ↑      ↑           ↑                                     │
│      global  local    position encoding                          │
│      (reused) (step t)   of current fixation                    │
└──────────────────────────┬───────────────────────────────────────┘
                           │  h_t ∈ R^d
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Mamba Sequence Model                           │
│                                                                  │
│   input:  H = {h_1, h_2, ..., h_T}                             │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  Mamba Block                                             │  │
│   │  d_model=256, d_state=256, d_conv=4, expand=2           │  │
│   │                                                          │  │
│   │  selective SSM: input-dependent gating                  │  │
│   │  → each output attends to full fixation history         │  │
│   │  → learned inhibition of return                         │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│   output: {o_1, o_2, ..., o_T}  ∈ R^d                         │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Position Decoder                              │
│                                                                  │
│   p̂_t = σ( MLP_dec( o_t ) )  ∈ [0,1]²                        │
│                                                                  │
│   predicted scanpath: P̂ = {p̂_1, p̂_2, ..., p̂_T}             │
└──────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Training Loss                                 │
│                                                                  │
│   L_path = SoftDTW_γ(P̂, P*)          γ = 0.1                  │
│   L_cov  = -(2w_x·(σ_x+r_x)/2 + w_y·(σ_y+r_y)/2)             │
│   L      = L_path + λ·L_cov           λ = 0.05                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## CoordAttention 模块细节

```
输入 X ∈ R^(C×H×W)
        │
        ├──── 水平方向 avg pool ──→ X_h ∈ R^(C×H×1)
        │                                │
        └──── 垂直方向 avg pool ──→ X_w ∈ R^(C×1×W)
                                         │
                    concat → [X_h, X_w^T] ∈ R^(C×(H+W)×1)
                                         │
                                    Conv 1×1 + BN + ReLU
                                         │
                              split → [f_h, f_w]
                                    │           │
                               Conv 1×1     Conv 1×1
                               sigmoid      sigmoid
                                    │           │
                               a_h ∈ R^(C×H×1)  a_w ∈ R^(C×1×W)
                                         │
                    输出 = X ⊗ a_h ⊗ a_w  (element-wise multiply)
```

---

## 两阶段采样策略示意

```
fixation step:  1   2   3   4   5   6   7   8   9   10  ...  T
                |←── Phase 1 (40%) ──→|←────── Phase 2 (60%) ──→|
exploration ε:  0.60 0.60 0.60 0.60   0.30 0.30 0.30 0.30 0.30

Phase 1: 高探索率，广泛覆盖全景图显著区域
Phase 2: 低探索率，精细化注视局部显著区域
```

---

## 画图建议（draw.io / PPT）

- 整体用**从左到右**的横向布局，三列：
  1. 左列：输入图像 + Glance Network（蓝色）
  2. 中列：Hierarchical Sampler（绿色）+ Focus Network（橙色）
  3. 右列：Feature Fusion + Mamba Block + Decoder（紫色）
- 用虚线箭头表示 teacher forcing（训练时 GT → 下一步输入）
- 用实线箭头表示推理时的自回归流程
- Mamba Block 内部可以画一个小的 SSM 展开图（可选）
