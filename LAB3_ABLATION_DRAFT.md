## Phần 4: Ablation Study (2.9)

### Mục tiêu
Để hiểu rõ ảnh hưởng của các hyperparameters, chạy 2 experiments bổ sung:
- **Run 1:** max_len=128, hidden_dim=64, dropout=0.2 (nhỏ hơn)
- **Run 2:** max_len=256, hidden_dim=64, dropout=0.2 (giữ max_len)

Mục đích so sánh ảnh hưởng của `max_len` và `hidden_dim` so với baseline.

---

### Kết quả So sánh

| Configuration | max_len | hidden_dim | dropout | Val F1 | Test F1 | Truncation | Epochs | Ghi chú |
|---|---|---|---|---|---|---|---|---|
| **Baseline** | 256 | 128 | 0.3 | 0.550 | 0.542 | 34.75% | 5 | Early stop epoch 5 |
| **Run 1** | 128 | 64 | 0.2 | 0.5473 | 0.5387 | **83.45%** ⚠️ | 6 | Giảm max_len + hidden |
| **Run 2** | 256 | 64 | 0.2 | [Running...] | [Running...] | [To compute] | [TBD] | Giảm hidden_dim |

---

### Phân tích Kết quả

#### 1. Ảnh hưởng của max_len

**So sánh:**
- **Run 1** (max_len=128): Val F1 = **0.5473**, Truncation = **83.45%**
- **Run 2** (max_len=256): Val F1 = [Waiting...], Truncation = [Waiting...]

**Nhận xét từ Run 1:**
- Giảm max_len từ 256 → 128 **làm tăng truncation rate lên 83.45%** (từ 34.75%)
- Điều này có nghĩa **83.45% review bị cắt ngắn** → mất quá nhiều thông tin
- Val F1 vẫn giữ ở mức 0.5473 (gần bằng baseline 0.550), nhưng **test F1 giảm từ 0.542 → 0.5387**
- **Kết luận:** Giảm max_len quá nhiều không tốt vì mất thông tin từ cuối review

#### 2. Ảnh hưởng của hidden_dim

**So sánh:**
- **Baseline** (hidden_dim=128): Val F1 = 0.550
- **Run 1 + Run 2** (hidden_dim=64): Val F1 ≈ 0.547 (Run 1)

**Nhận xét:**
- Giảm hidden_dim từ 128 → 64 **không ảnh hưởng nhiều đến kết quả** (0.550 → 0.547 = -0.3%)
- Nhưng giảm hidden_dim **làm giảm số parameters gầnđấy 50%** → mô hình nhỏ hơn, nhanh hơn
- Trade-off: Hiệu suất gần như nhau, nhưng mô hình nhẹ hơn

**Kết luận từ Run 1:** Có thể sử dụng hidden_dim=64 thay vì 128 mà không mất nhiều hiệu suất

#### 3. Ảnh hưởng của dropout

**So sánh:**
- **Baseline** (dropout=0.3): Val F1 = 0.550
- **Run 1** (dropout=0.2): Val F1 = 0.5473

**Nhận xét:**
- Giảm dropout từ 0.3 → 0.2 làm **Val F1 giảm nhẹ từ 0.550 → 0.5473** (-0.27%)
- Nhưng mô hình vẫn không overfitting (Run 1 train đủ 6 epochs, Baseline dừng ở epoch 5)
- dropout=0.2 yếu hơn dropout=0.3 trong việc chống overfitting

---

### Kết luận Tổng quả

**Run 1 (max_len=128, hidden_dim=64, dropout=0.2):**
- ✅ Val F1: 0.5473 (tương đương baseline)
- ❌ Test F1: 0.5387 (giảm hơn baseline 0.542)
- ⚠️ **Truncation rate: 83.45% - quá cao**, mất quá nhiều thông tin
- **Đánh giá:** Không nên dùng max_len=128

**Run 2 (max_len=256, hidden_dim=64, dropout=0.2):**
- [Đang chạy...]
- **Kỳ vọng:** Sẽ tốt hơn Run 1 vì giữ lại thông tin (max_len=256)

**Khuyến nghị:**
1. **Nên giữ max_len=256** - Cắt cân bằng giữa thông tin (34.75% truncation) và tính toán
2. **Có thể giảm hidden_dim về 64** - Không ảnh hưởng nhiều, mà mô hình nhẹ hơn
3. **Nên tăng dropout lên 0.3** - Tốt hơn 0.2 trong việc chống overfitting
4. **Để cải thiện hơn nữa:** Chuyển sang **LSTM thay vì RNN** - Để xử lý vanishing gradient tốt hơn

---

### Phụ lục: Chi tiết Epochs

**Run 1 - Epoch History (max_len=128, hidden_dim=64):**
```
Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Train F1 | Val F1
------|-----------|----------|-----------|---------|----------|----------
1     | 0.703     | 0.691    | 51.08%    | 52.50%  | 0.511    | 0.520
2     | 0.685     | 0.691    | 55.33%    | 52.88%  | 0.553    | 0.527
3     | 0.678     | 0.693    | 57.67%    | 52.28%  | 0.576    | 0.456
4     | 0.693     | 0.691    | 52.06%    | 52.88%  | 0.519    | 0.528
5     | 0.689     | 0.691    | 53.02%    | 53.20%  | 0.530    | 0.531
6     | 0.682     | **0.679** | 56.45%    | **54.90%** | 0.564 | **0.547**
```

**Nhận xét:** Val loss nhất quán ở ~0.69 (không giảm rõ), nhưng Val acc cải thiện ở epoch 6. Mô hình đã train đủ 6 epochs, không early stop.
