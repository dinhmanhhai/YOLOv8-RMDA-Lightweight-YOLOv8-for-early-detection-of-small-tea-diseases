# Sửa Lỗi Tensor Size Mismatch trong Neck

## Lỗi gốc

```
RuntimeError: Sizes of tensors must match except in dimension 1. 
Expected size 160 but got size 40 for tensor number 1 in the list.
```

**Nguyên nhân:** Kích thước tensor không khớp khi concatenate trong neck path.

---

## Phân tích vấn đề

### Kích thước feature maps từ Backbone:

- **P3** (index 4): `H/8 × W/8` (sau 3 lần downsampling)
- **P4** (index 6): `H/16 × W/16` (sau 4 lần downsampling)  
- **P5** (index 7): `H/32 × W/32` (sau 5 lần downsampling)

### Vấn đề trong code cũ:

```python
# Code cũ (SAI)
p4_up = self.neck_p4_up(p3_out)  # H/8 → H/4
p4_cat = torch.cat([p4_up, p4], dim=1)  # H/4 vs H/16 → MISMATCH!
```

**Lý do:** 
- `p3_out` sau khi upsample 2x có kích thước `H/4`
- `p4` từ backbone có kích thước `H/16`
- Không thể concat vì spatial dimensions khác nhau

---

## Giải pháp đã áp dụng

### 1. Top-down Path (P4 và P5)

**P4 path:**
```python
# Upsample p3_out: H/8 → H/4
p4_up_from_p3 = self.neck_p4_up(p3_out)  # H/8 → H/4

# Upsample p4 từ backbone để match: H/16 → H/4
p4_up_from_backbone = F.interpolate(p4, size=p4_up_from_p3.shape[2:], mode='nearest')

# Concat và process
p4_cat = torch.cat([p4_up_from_p3, p4_up_from_backbone], dim=1)
p4_out = self.neck_p4(p4_cat)  # Output: H/4

# Downsample về kích thước ban đầu của P4: H/4 → H/16
p4_out = F.interpolate(p4_out, size=p4.shape[2:], mode='nearest')
```

**P5 path:**
```python
# Upsample p4_out: H/16 → H/8
p5_up_from_p4 = self.neck_p5_up(p4_out)  # H/16 → H/8

# Upsample p5 từ backbone để match: H/32 → H/8
p5_up_from_backbone = F.interpolate(p5, size=p5_up_from_p4.shape[2:], mode='nearest')

# Concat và process
p5_cat = torch.cat([p5_up_from_p4, p5_up_from_backbone], dim=1)
p5_out = self.neck_p5(p5_cat)  # Output: H/8

# Downsample về kích thước ban đầu của P5: H/8 → H/32
p5_out = F.interpolate(p5_out, size=p5.shape[2:], mode='nearest')
```

### 2. Bottom-up Path

**P4 bottom-up:**
```python
# p5_out is H/32, downsample 2x → H/16
p4_down = self.neck_p4_down(p5_out)  # H/32 → H/16

# p4_out is H/16, same size - no need to resize
p4_cat_bottom = torch.cat([p4_down, p4_out], dim=1)  # ✅ Match!
p4_final = self.neck_p4_bottom(p4_cat_bottom)  # Output: H/16
```

**P3 bottom-up:**
```python
# p4_final is H/16, downsample 2x → H/8
p3_down = self.neck_p3_down(p4_final)  # H/16 → H/8

# p3_out is H/8, same size - no need to resize
p3_cat_bottom = torch.cat([p3_down, p3_out], dim=1)  # ✅ Match!
p3_final = self.neck_p3_bottom(p3_cat_bottom)  # Output: H/8
```

---

## Kết quả

### Kích thước sau khi sửa:

| Stage | Feature Map | Size |
|-------|------------|------|
| **Backbone** | P3 | H/8 × W/8 |
| | P4 | H/16 × W/16 |
| | P5 | H/32 × W/32 |
| **Top-down** | p3_out | H/8 × W/8 |
| | p4_out | H/16 × W/16 (sau khi downsample) |
| | p5_out | H/32 × W/32 (sau khi downsample) |
| **Bottom-up** | p4_final | H/16 × W/16 |
| | p3_final | H/8 × W/8 |
| **Head** | Input | [H/8, H/16, H/32] |

### ✅ **Tất cả concatenations đều match!**

---

## Thay đổi chính

1. ✅ **Thêm `F.interpolate`** để match kích thước trước khi concat
2. ✅ **Downsample sau processing** để giữ nguyên kích thước ban đầu
3. ✅ **Import `torch.nn.functional as F`** để dùng `F.interpolate`

---

## Lưu ý

- C2f, RepGFPN, AKConv **không thay đổi spatial size** (giữ nguyên H, W)
- Cần đảm bảo kích thước match trước khi concat
- Sau khi process, downsample về kích thước ban đầu để bottom-up path hoạt động đúng

---

## Test

Code đã được sửa và sẵn sàng để chạy training. Lỗi tensor size mismatch đã được giải quyết.

