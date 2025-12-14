# Kiểm tra Dynamic Head - So sánh với Paper

## 1. Công thức chính trong Paper

### Paper (Formula 7):
```
W(F) = π_C(π_S(π_L(F) · F) · F) · F
```

**Thứ tự áp dụng:**
1. **π_L** (Scale-aware) → 2. **π_S** (Spatial-aware) → 3. **π_C** (Task-aware)

---

## 2. So sánh từng Attention Mechanism

### ✅ **Scale-aware Attention (π_L)**

#### Paper (Formula 8):
```
π_L(F) · F = σ(f(1/(S·C) ΣS,C F)) · F
```
- `f(·)`: Linear function (1x1 convolution)
- `σ(x) = max(0, min(1, (x+1)/2))`: **Hard sigmoid function**

#### Code hiện tại:
```python
self.scale_attention = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),      # ✅ Đúng: Global average pooling
    nn.Conv2d(channels, channels // 4, 1),  # ✅ Đúng: 1x1 conv
    nn.ReLU(inplace=True),
    nn.Conv2d(channels // 4, channels, 1),  # ✅ Đúng: 1x1 conv
    nn.Sigmoid()  # ❌ SAI: Dùng Sigmoid thay vì Hard Sigmoid
)
```

**Đánh giá:**
- ✅ Có global average pooling (1/(S·C) ΣS,C F)
- ✅ Có 1x1 convolution (f(·))
- ❌ **SAI**: Dùng `Sigmoid()` thay vì `Hard Sigmoid`

**Cần sửa:**
```python
def hard_sigmoid(x):
    return torch.clamp((x + 1) / 2, 0, 1)
```

---

### ❌ **Spatial-aware Attention (π_S)**

#### Paper (Formula 9):
```
π_S(F) · F = (1/L) Σ(l=1 to L) Σ(k=1 to K) w_l,k F(l; p_k + Δp_k; c) · Δm_k
```

**Đặc điểm:**
- Sparse sampling với K positions
- Deformable convolution với offsets (Δp_k)
- Weight factors (Δm_k)
- Aggregation across levels (1/L Σ)

#### Code hiện tại:
```python
self.spatial_attention = nn.Sequential(
    nn.Conv2d(channels, channels // 4, 1),
    nn.ReLU(inplace=True),
    nn.Conv2d(channels // 4, 1, 1),
    nn.Sigmoid()
)
```

**Đánh giá:**
- ❌ **SAI HOÀN TOÀN**: Code chỉ dùng simple Conv + Sigmoid
- ❌ Không có sparse sampling
- ❌ Không có deformable convolution
- ❌ Không có aggregation across levels
- ❌ Không giống paper

**Cần sửa:**
- Implement sparse sampling với K positions
- Sử dụng deformable convolution hoặc offset learning
- Aggregate across feature pyramid levels

---

### ❌ **Task-aware Attention (π_C)**

#### Paper (Formula 10):
```
π_C(F) · F = max(α¹(F) · F_C + β¹(F), α²(F) · F_C + β²(F))
```

**Đặc điểm:**
- Sử dụng **Dynamic ReLU**
- `[α¹, α², β¹, β²]^T = θ(·)`: Learning control activation threshold
- Channel-wise activation với max operation

#### Code hiện tại:
```python
self.task_attention = nn.ModuleList([
    nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(channels, channels // 4, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels // 4, channels, 1),
        nn.Sigmoid()
    ) for _ in range(num_tasks)
])
```

**Đánh giá:**
- ❌ **SAI**: Dùng Sigmoid thay vì Dynamic ReLU
- ❌ Không có max operation
- ❌ Không có α, β parameters
- ❌ Không giống paper

**Cần sửa:**
- Implement Dynamic ReLU với α, β parameters
- Sử dụng max operation cho channel-wise activation

---

## 3. So sánh tổng thể

| Component | Paper | Code hiện tại | Trạng thái |
|-----------|-------|---------------|------------|
| **Công thức chính** | W(F) = π_C(π_S(π_L(F)·F)·F)·F | ✅ Áp dụng tuần tự | ✅ ĐÚNG |
| **Scale-aware (π_L)** | Hard sigmoid | ❌ Sigmoid | ❌ SAI |
| **Spatial-aware (π_S)** | Sparse sampling + Deformable | ❌ Simple Conv | ❌ SAI |
| **Task-aware (π_C)** | Dynamic ReLU | ❌ Sigmoid | ❌ SAI |
| **General View** | ✅ Có | ✅ Có | ✅ ĐÚNG |
| **Task-specific heads** | ✅ Có | ✅ Có | ✅ ĐÚNG |

---

## 4. Các vấn đề cần sửa

### ❌ **Vấn đề 1: Scale-aware Attention**

**Cần sửa:**
```python
def hard_sigmoid(x):
    """Hard sigmoid: σ(x) = max(0, min(1, (x+1)/2))"""
    return torch.clamp((x + 1) / 2, 0, 1)

class ScaleAwareAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            # Không dùng Sigmoid, dùng hard_sigmoid trong forward
        )
    
    def forward(self, x):
        scale_weight = self.scale_attention(x)
        scale_weight = hard_sigmoid(scale_weight)  # Hard sigmoid
        return x * scale_weight
```

### ❌ **Vấn đề 2: Spatial-aware Attention**

**Cần implement:**
- Sparse sampling với K positions
- Deformable convolution hoặc offset learning
- Aggregation across feature pyramid levels

**Code mẫu:**
```python
class SpatialAwareAttention(nn.Module):
    def __init__(self, channels, K=9):
        super().__init__()
        self.K = K
        # Offset learning
        self.offset_conv = nn.Conv2d(channels, 2 * K, 3, padding=1)
        # Weight learning
        self.weight_conv = nn.Conv2d(channels, K, 3, padding=1)
        # ... (phức tạp, cần implement đầy đủ)
```

### ❌ **Vấn đề 3: Task-aware Attention**

**Cần implement Dynamic ReLU:**
```python
class DynamicReLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Learn α¹, α², β¹, β²
        self.theta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels * 4, 1)  # [α¹, α², β¹, β²]
        )
    
    def forward(self, x):
        # Get α, β parameters
        params = self.theta(x)  # [B, 4*C, 1, 1]
        alpha1, alpha2, beta1, beta2 = params.chunk(4, dim=1)
        
        # Dynamic ReLU: max(α¹·F_C + β¹, α²·F_C + β²)
        out1 = alpha1 * x + beta1
        out2 = alpha2 * x + beta2
        return torch.max(out1, out2)

class TaskAwareAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dynamic_relu = DynamicReLU(channels)
    
    def forward(self, x):
        return self.dynamic_relu(x)
```

---

## 5. Kết luận

### ❌ **Dynamic Head hiện tại KHÔNG giống paper:**

1. **Scale-aware**: Dùng Sigmoid thay vì Hard Sigmoid
2. **Spatial-aware**: Hoàn toàn khác - không có sparse sampling, deformable conv
3. **Task-aware**: Dùng Sigmoid thay vì Dynamic ReLU

### ✅ **Những gì đúng:**

1. Thứ tự áp dụng attention (π_L → π_S → π_C)
2. Có General View
3. Có task-specific heads

### 📝 **Khuyến nghị:**

1. **Ưu tiên cao**: Sửa Scale-aware (dễ - chỉ cần thay Sigmoid)
2. **Ưu tiên trung bình**: Sửa Task-aware (cần implement Dynamic ReLU)
3. **Ưu tiên thấp**: Sửa Spatial-aware (phức tạp nhất - cần sparse sampling)

---

## 6. Tóm tắt

| Câu hỏi | Trả lời |
|---------|---------|
| **Dynamic Head đã giống paper chưa?** | ❌ **CHƯA** - 3/3 attention mechanisms đều khác |
| **Vấn đề nghiêm trọng nhất?** | ❌ Spatial-aware - hoàn toàn khác paper |
| **Có thể sửa được không?** | ✅ Có - nhưng cần implement lại 3 attention mechanisms |

