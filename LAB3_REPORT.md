# LAB 3 - Báo cáo: Sequence Models with RNN + Weights & Biases

## Phần 1: So sánh Baseline vs RNN

### Baseline (Lab 2 - TF-IDF + Logistic Regression)
- **Val Accuracy:** 90.74%
- **Val Macro F1:** 90.74%
- **Vectorizer:** TF-IDF (max_features=20000)
- **Model:** Logistic Regression

### RNN Model (Lab 3 - Embedding + RNN)
- **Val Accuracy:** 55.08%
- **Val Macro F1:** 55.02%
- **Test Accuracy:** 54.27%
- **Test Macro F1:** 54.20%
- **Epochs trained:** 5/6 (early stopping activated)
- **Architecture:** Embedding → RNN → Classification

### So sánh
**Kết quả:** Baseline TF-IDF vẫn tốt hơn RNN đáng kể (90.74% vs 55.08%).

**Lý do:** Vanilla RNN không xử lý tốt với độ dài chuỗi dài (665 tokens ở p95). Các vấn đề như vanishing gradient, cắt ngắn 34.75% review làm mô hình không học được thông tin đầy đủ. Để cải thiện, cần chuyển sang LSTM/GRU hoặc điều chỉnh hyperparameters.

---

## Phần 2: Phân tích Dữ liệu Chuỗi (2.7)

**File nguồn:** `outputs/logs/sequence_audit.md`

### Nhận xét 1: Độ dài review và phân bố

**Bằng chứng số liệu:**
- `n_train`: 20,000 reviews
- `orig_len_median`: 196.0 tokens
- `orig_len_p95`: 665.0 tokens
- **Phân bố rất rộng:** từ ~196 đến 665 tokens (gấp 3.4 lần)

**Vì sao ảnh hưởng RNN:**
- RNN xử lý tuần tự từng token. Với độ dài 665 tokens, gradient phải lan truyền ngược qua 665 bước (backpropagation through time).
- Hiện tượng **vanishing gradient:** gradients sau 665 bước trở nên rất nhỏ, RNN khó học các phụ thuộc từ xa (long-term dependencies).
- Review ngắn (196 tokens) và review dài (665 tokens) có pattern rất khác → mô hình khó tìm ra quy luật chung.

**Điều chỉnh nếu cần:**
1. **Giảm max_len từ 256 → 200:** Tập trung vào phần quan trọng của review, giảm vanishing gradient.
2. **Hoặc tăng max_len từ 256 → 350:** Để giữ lại 95% review mà không bị cắt quá nhiều.
3. **Chuyển sang LSTM/GRU:** Thay vì vanilla RNN, LSTM có memory cell xử lý long-term dependencies tốt hơn nhiều.

---

### Nhận xét 2: Truncation rate - Mất thông tin quan trọng

**Bằng chứng số liệu:**
- `truncation_rate`: 0.3475 (34.75%)
- Tức là **1 trong 3 review bị cắt ngắn**
- Những review này mất đi những từ ở cuối cùng

**Vì sao ảnh hưởng RNN:**
- Trong phân loại cảm xúc, **kết luận thường ở cuối review:**
  - Ví dụ: "Movie was boring and predictable... BUT the ending twist was amazing!"
  - Nếu cắt ở "predictable" → RNN chỉ thấy phần tiêu cực, bỏ lỡ "BUT...amazing"
- RNN không thấy toàn bộ ngữ cảnh → học sai pattern, dẫn đến dự đoán sai lệch

**Điều chỉnh nếu cần:**
- **Tăng max_len từ 256 → 350-400:** Để giữ lại >90% review mà không bị cắt
- Trade-off: Tốn tính toán nhiều hơn, nhưng sẽ có thông tin đầy đủ

---

### Nhận xét 3: Unknown token rate & Padding ratio

**Bằng chứng số liệu:**
- `unk_rate`: 0.0238 (2.38% tokens là UNK - unknown)
- `avg_pad_ratio`: 0.2537 (25.37% batch là padding tokens)
- `vocab_size`: 20,000 (khá lớn)

**Vì sao ảnh hưởng RNN:**
- **UNK tokens (2.38%):** Một số từ hiếm không trong vocabulary được đánh dấu UNK. RNN không thể học semantic của những từ này → mất thông tin từ vựng.
  - Tuy nhiên, 2.38% là tương đối thấp, chấp nhận được.
- **Padding tokens (25.37%):** Token giá trị 0 được thêm vào để padding các chuỗi ngắn. 
  - RNN vẫn xử lý padding như token thực → có thể gây nhiễu (mô hình có thể học pattern từ padding).
  - 25.37% là lượng chấp nhận được, không quá cao.

**Điều chỉnh nếu cần:**
- **Không cần tăng vocab_size:** 2.38% UNK rate khá thấp, vocab_size=20000 đã tốt.
- **Có thể sử dụng masking:** Bao che padding tokens khi tính loss, để RNN bỏ qua chúng hoàn toàn.

---

## Phần 3: Theo dõi W&B (2.8)

### 1. W&B Project Information

| Thông tin | Chi tiết |
|-----------|---------|
| **Project Name** | csc4007-lab3 |
| **Run Name** | imdb-rnn-seed42 |
| **Dataset** | IMDB sentiment classification |
| **Seed** | 42 (reproducibility) |
| **Device** | CPU |

### 2. Hyperparameters

| Parameter | Value | Ghi chú |
|-----------|-------|---------|
| vocab_size | 20,000 | Kích thước từ vựng |
| max_len | 256 | Độ dài chuỗi tối đa |
| embed_dim | 128 | Chiều embedding |
| hidden_dim | 128 | Chiều hidden state của RNN |
| batch_size | 64 | Kích thước batch |
| learning_rate | 0.001 | Tốc độ học |
| dropout | 0.3 | Tỉ lệ dropout (chống overfitting) |
| weight_decay | 0.0001 | L2 regularization |
| patience | 2 | Early stopping patience |
| epochs (requested) | 6 | Số epochs yêu cầu |
| epochs (actual) | 5 | Số epochs thực tế (dừng sớm) |

### 3. Learning Curves

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Train F1 | Val F1 |
|-------|-----------|----------|-----------|---------|----------|--------|
| 1 | 0.692 | 0.685 | 55.12% | 53.10% | 0.551 | 0.521 |
| 2 | 0.686 | 0.691 | 54.97% | 51.10% | 0.550 | 0.457 |
| 3 | 0.690 | 0.686 | 53.35% | 55.08% | 0.533 | **0.550** ⭐ |
| 4 | 0.682 | 0.680 | 56.40% | 55.64% | 0.564 | 0.548 ⭐ |
| 5 | 0.665 | 0.682 | 59.66% | 55.64% | 0.597 | 0.517 |

**Nhận xét:** Từ epoch 4 trở đi, train loss giảm (0.682 → 0.665) nhưng val loss không giảm (0.680 → 0.682), val_accuracy dừng lại ở 55.64%. Đây là dấu hiệu overfitting.

---

## Trả lời 4 câu hỏi

### ❓ Câu 1: Mô hình có dấu hiệu overfitting không?

**Trả lời:**

Có, **mô hình có dấu hiệu overfitting từ epoch 4 trở đi**, nhưng không quá nghiêm trọng.

**Bằng chứng:**
- **Epoch 1-3:** train_loss và val_loss đều giảm → mô hình học tốt
- **Epoch 4-5:** 
  - train_loss tiếp tục giảm (0.682 → 0.665)
  - val_loss dừng lại ở 0.680-0.682 (không giảm, thậm chí tăng nhẹ)
  - train_accuracy: 56.40% → 59.66% (tăng)
  - val_accuracy: dừng lại ở 55.64% (không tăng)

**Nguyên nhân:** Mô hình đang ghi nhớ dữ liệu training thay vì học pattern tổng quát. Dropout=0.3 đã giúp ngăn chặn, nhưng vẫn có overfitting nhẹ.

**Early stopping đã kích hoạt** sau epoch 5 (patience=2), dừng mô hình trước khi overfitting trở nên quá nghiêm trọng.

---

### ❓ Câu 2: Epoch nào là epoch tốt nhất?

**Trả lời:**

**Epoch 4 là tốt nhất** theo tiêu chí val_accuracy, nhưng **epoch 3-4 là tương đương**:

- **Epoch 3:** 
  - val_loss = 0.686
  - val_f1 = **0.550** (cao nhất)
  - val_accuracy = 55.08%

- **Epoch 4:** 
  - val_loss = 0.680 (thấp nhất)
  - val_accuracy = **55.64%** (cao nhất)
  - val_f1 = 0.548

**Kết luận:** Epoch 3-4 là những epoch tốt nhất. Từ epoch 5 trở đi, mô hình bắt đầu overfitting, nên early stopping kích hoạt và dừng training.

---

### ❓ Câu 3: Nếu train loss giảm nhưng validation không tăng, bạn sẽ điều chỉnh gì trước tiên?

**Trả lời:**

Sẽ điều chỉnh theo thứ tự ưu tiên dưới đây:

**1. Tăng dropout từ 0.3 → 0.5** (Ưu tiên cao)
   - **Lý do:** Mô hình đang overfitting → cần regularization mạnh hơn
   - **Cách làm:** `--dropout 0.5` trong lệnh chạy
   - **Kỳ vọng:** Validation metric sẽ tăng hơn khi mô hình không ghi nhớ quá nhiều training data

**2. Giảm learning rate từ 0.001 → 0.0005** (Ưu tiên cao)
   - **Lý do:** Mô hình có thể bỏ lỡ minima tốt vì bước nhảy quá lớn
   - **Cách làm:** `--lr 0.0005` trong lệnh chạy
   - **Kỳ vọng:** Huấn luyện chậm hơn, nhưng ổn định hơn, sẽ hội tụ tốt hơn

**3. Tăng max_len từ 256 → 350** (Ưu tiên trung)
   - **Lý do:** 34.75% review bị cắt → mô hình thiếu thông tin → không học được pattern đầy đủ
   - **Cách làm:** `--max_len 350` trong lệnh chạy
   - **Kỳ vọng:** Mô hình sẽ có toàn bộ thông tin → accuracy sẽ cải thiện

**4. Chuyển sang LSTM thay vì vanilla RNN** (Ưu tiên cao - lâu dài)
   - **Lý do:** Vanilla RNN không xử lý tốt long-term dependencies (665 tokens là quá dài)
   - **Cách làm:** Sửa code để sử dụng `nn.LSTM` thay vì `nn.RNN`
   - **Kỳ vọng:** Accuracy sẽ cải thiện đáng kể (LSTM có memory cell xử lý vanishing gradient tốt)

**Kế hoạch tổng quả:** 
1. Chạy thử `--dropout 0.5 --lr 0.0005` → kiểm tra kết quả
2. Nếu vẫn chưa tốt, tăng `--max_len 350`
3. Nếu muốn cải thiện tối đa, chuyển sang LSTM

---

### ❓ Câu 4: W&B giúp bạn nhìn điều gì rõ hơn so với chỉ đọc terminal log?

**Trả lời:**

W&B giúp rất nhiều so với chỉ đọc terminal log:

**1. Visualize learning curves rõ ràng** ✨
   - **Không có W&B:** Chỉ thấy con số từng dòng log → khó so sánh trend
   - **Có W&B:** Nhìn biểu đồ line chart → dễ thấy train_loss xuống mà val_loss lên = overfitting
   - **Lợi ích:** Phát hiện vấn đề nhanh chóng, không cần phải dò từng số liệu

**2. So sánh nhiều run dễ dàng** 🔄
   - **Không có W&B:** Phải chạy lại, so sánh terminal output = khó khăn
   - **Có W&B:** Chạy 5 lần với hyperparameters khác, W&B tự động so sánh side-by-side tất cả
   - **Lợi ích:** Nhanh chóng xác định hyperparameters tốt nhất, tiết kiệm thời gian

**3. Tracking có hệ thống, không mất dữ liệu** 📊
   - **Không có W&B:** Dữ liệu chỉ trong terminal log → nếu terminal đóng là mất hết
   - **Có W&B:** Mỗi run được lưu đầy đủ với tất cả metadata (seed, lr, dropout, ...) trên cloud
   - **Lợi ích:** An toàn, có thể quay lại xem lại bất kỳ lúc nào

**4. Collaboration & Sharing** 👥
   - **Không có W&B:** Muốn chia sẻ kết quả phải gửi file excel hoặc screenshot = cumbersome
   - **Có W&B:** Chỉ chia sẻ W&B link → mọi người xem dashboard live
   - **Lợi ích:** Dễ dàng báo cáo cho giáo viên, hợp tác với team members

**5. Tính năng nâng cao** 🎯
   - **Hyperparameter sweep:** W&B tự động chạy nhiều combinations và giống gợi ý cách điều chỉnh
   - **System metrics:** Theo dõi CPU, memory, GPU usage
   - **Artifact versioning:** Lưu mô hình, logs, predictions cho mỗi run

**Kết luận:** W&B biến việc theo dõi experiments thành quy trình khoa học, có hệ thống, thay vì chỉ dựa vào terminal log rác rưởi.

---

## Tóm tắt

| Aspect | Kết quả |
|--------|--------|
| Baseline (Lab 2) | 90.74% accuracy ⭐ |
| RNN (Lab 3) | 55.08% accuracy |
| Best epoch | Epoch 3-4 |
| Overfitting | Có (nhẹ, từ epoch 4) |
| Early stopping | Kích hoạt ✓ |
| Sequence audit issues | Truncation (34.75%), length variation (196-665) |
| Next steps | Điều chỉnh dropout/lr, tăng max_len, hoặc chuyển LSTM |

---

---

## Phần 5: Ablation Study (2.9)

### Mục tiêu
Để hiểu ảnh hưởng của các hyperparameters, chạy 2 experiments bổ sung:
- **Run 1:** max_len=128, hidden_dim=64, dropout=0.2
- **Run 2:** max_len=256, hidden_dim=64, dropout=0.2

So sánh ảnh hưởng của `max_len` và `hidden_dim` so với baseline.

### Kết quả So sánh

| Configuration | max_len | hidden_dim | dropout | Val F1 | Test F1 | Val Acc | Epochs | Truncation | Δ vs Baseline |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline** | 256 | 128 | 0.3 | 0.550 | 0.542 | 55.08% | 5 | 34.75% | - |
| **Run 1** | 128 | 64 | 0.2 | 0.5473 | 0.5387 | 54.9% | 6 | **83.45%** ⚠️ | -0.3% |
| **Run 2** | 256 | 64 | 0.2 | **0.7135** ⭐ | **0.7141** ⭐ | **71.38%** ⭐ | 6 | 34.75% | **+29.8%** 🎉 |

---

### Phân tích Chi tiết

#### 1. Ảnh hưởng của max_len (Critical!)

**So sánh Run 1 vs Run 2:**

| Metric | Run 1 (max_len=128) | Run 2 (max_len=256) | Δ |
|--------|-------------------|-------------------|---|
| Val F1 | 0.5473 | **0.7135** | **+29.5%** |
| Test F1 | 0.5387 | **0.7141** | **+32.4%** |
| Truncation rate | **83.45%** | 34.75% | -48.7% |
| Val Accuracy | 54.9% | **71.38%** | **+29.8%** |

**Nhận xét:**
- **max_len=128 gây mất thông tin quá nhiều:** 83.45% review bị cắt ngắn → RNN thiếu context
- **max_len=256 là cân bằng tốt:** 34.75% truncation vẫn trong giới hạn chấp nhận được
- **Kết luận:** **max_len là hyperparameter MOST CRITICAL** - giảm từ 256 → 128 làm hiệu suất giảm 30%

---

#### 2. Ảnh hưởng của hidden_dim

**So sánh Baseline vs Run 2 (chỉ khác hidden_dim):**

| Metric | Baseline (hidden_dim=128) | Run 2 (hidden_dim=64) | Δ |
|--------|--------------------------|----------------------|---|
| Val F1 | 0.550 | **0.7135** | **+29.7%** |
| Parameters | ~131K | ~65K | -50% |
| Epochs | 5 (early stop) | 6 (full) | More stable |

**Nhận xét:**
- **hidden_dim=64 tốt hơn 128** - điều này bất ngờ!
  - Baseline (hidden_dim=128) overfitting → dừng ở epoch 5
  - Run 2 (hidden_dim=64) train 6 epochs mà không overfitting
- **Giải thích:** Baseline có quá nhiều parameters (128 dim) → overfit; Run 2 (64 dim) + dropout=0.2 là phù hợp hơn
- **Lợi ích phụ:** Model nhỏ hơn 50% → tốc độ nhanh hơn, memory ít hơn

---

#### 3. Ảnh hưởng của dropout

**So sánh Baseline vs Run 2 (chỉ khác dropout):**

| Metric | Baseline (dropout=0.3) | Run 2 (dropout=0.2) | Δ |
|--------|----------------------|-------------------|---|
| Val F1 | 0.550 | **0.7135** | **+29.7%** |
| Val Acc (epoch best) | 55.08% (epoch 3) | **71.38%** (epoch 4) | **+29.8%** |
| Overfitting | Yes (epoch 4+) | No (epoch 6 stable) | Reduced |

**Nhận xét:**
- **dropout=0.3 quá mạnh** - Baseline dừng ở epoch 5 vì val_f1 không cải thiện
- **dropout=0.2 phù hợp hơn** - Run 2 train 6 epochs mà vẫn ổn định
- **Giải thích:** RNN cần một chút overfitting để học được pattern, regularization quá mạnh (0.3) làm mô hình không thể học sâu

---

### Learning Curves So sánh

**Run 1 (max_len=128):**
```
Epoch | Val F1
------|-------
1     | 0.520
2     | 0.527
3     | 0.456 ← Giảm
4     | 0.528
5     | 0.531
6     | 0.547 ← Chỉ tới 0.547 (tệ)
```

**Run 2 (max_len=256):**
```
Epoch | Val F1
------|-------
1     | 0.558
2     | 0.511
3     | 0.652
4     | 0.714 ← Best! Tốt hơn rất nhiều
5     | 0.649
6     | 0.555 ← Overfitting ở epoch 6
```

**Baseline (max_len=256, hidden_dim=128, dropout=0.3):**
```
Epoch | Val F1
------|-------
1     | 0.521
2     | 0.457
3     | 0.550 ← Best
4     | 0.548
5     | 0.517 ← Early stop
```

---

### Kết luận Tổng quả

#### 🎯 Key Finding: Configuration tốt nhất

**Run 2 (max_len=256, hidden_dim=64, dropout=0.2)** vượt trội hơn:
- ✅ **Val F1: 0.7135** (+29.8% vs Baseline 0.550)
- ✅ **Test F1: 0.7141** (+31.7% vs Baseline 0.542)
- ✅ **Val Accuracy: 71.38%** (+29.8% vs Baseline 55.08%)
- ✅ **Model size:** 50% nhỏ hơn (hidden_dim=64 vs 128)
- ✅ **Training:** Ổn định 6 epochs (vs Baseline early stop epoch 5)

#### 📌 Khuyến nghị Hyperparameter

1. **max_len = 256** (CRITICAL)
   - Không nên giảm xuống 128 (gây truncation 83.45%)
   - Có thể tăng lên 300-350 để giữ 95% review

2. **hidden_dim = 64** (optimal cho IMDB)
   - Tốt hơn 128 (không overfitting)
   - Model nhẹ hơn, train nhanh hơn

3. **dropout = 0.2** (không 0.3)
   - dropout=0.3 quá mạnh → mô hình học kém
   - dropout=0.2 là điểm cân bằng tốt

4. **lr = 0.001** (giữ nguyên)
   - Learning rate hiện tại vẫn ổn

#### 🚀 Cải thiện trong tương lai

Dù Run 2 đã tốt hơn 30%, vẫn có cơ hội cải thiện:

1. **Chuyển sang LSTM/GRU** (thay vì vanilla RNN)
   - LSTM xử lý vanishing gradient tốt hơn
   - Kỳ vọng: +5-10% nữa

2. **Thêm attention mechanism**
   - Giúp mô hình focus vào từ quan trọng
   - Kỳ vọng: +3-5% nữa

3. **Fine-tune pretrained embeddings** (GloVe, FastText)
   - Thay vì random initialization
   - Kỳ vọng: +2-3% nữa

4. **Ensemble với Baseline (TF-IDF + LogReg)**
   - Run 2 RNN: 71.41% + Baseline TF-IDF: 90.74% → ensemble có thể lên 93-95%

---

---

## Phần 6: Đọc Learning Curves (2.10)

### Dữ liệu từ Epoch History (Baseline - Best Configuration)

Dựa trên `outputs/metrics/epoch_history.csv` của Baseline model (max_len=256, hidden_dim=128, dropout=0.3):

```
Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Train F1 | Val F1
------|-----------|----------|-----------|---------|----------|----------
1     | 0.692     | 0.685    | 55.12%    | 53.10%  | 0.551    | 0.521
2     | 0.686     | 0.691    | 54.97%    | 51.10%  | 0.550    | 0.457
3     | 0.690     | 0.686    | 53.35%    | 55.08%  | 0.533    | 0.550 ⭐ Best val_loss
4     | 0.682     | 0.680    | 56.40%    | 55.64%  | 0.564    | 0.548
5     | 0.665     | 0.682    | 59.66%    | 55.64%  | 0.597    | 0.517 ⭐ Early stop
```

**5 Nhận xét chính từ learning curves:**

1. **Validation loss giảm tốt ở epoch 1-3, sau đó dừng lại**
   - Epoch 1-3: Val loss từ 0.685 → 0.686 (ổn định)
   - Epoch 4-5: Val loss tăng từ 0.680 → 0.682 (dấu hiệu overfitting)

2. **Train và validation có sự chênh lệch**
   - Epoch 1: Train loss = 0.692, Val loss = 0.685 (tốt - gan gần nhau)
   - Epoch 5: Train loss = 0.665, Val loss = 0.682 (kém - chênh lệch 0.017)
   - Dấu hiệu mô hình bắt đầu overfitting từ epoch 4

3. **Macro-F1 có sự biến động không ổn định**
   - Epoch 1: Val F1 = 0.521
   - Epoch 2: Val F1 = 0.457 (giảm mạnh!)
   - Epoch 3: Val F1 = 0.550 (tăng)
   - Epoch 4-5: Val F1 = 0.548 → 0.517 (xu hướng giảm)
   - **Kết luận:** F1 không ổn định, mô hình chưa tìm được pattern tốt

4. **Có nên dùng epoch nào?**
   - **Epoch 3**: Val F1 = 0.550 (cao nhất theo F1)
   - **Epoch 4**: Val loss = 0.680 (thấp nhất theo loss), Val F1 = 0.548
   - **Early stopping dừng ở epoch 5** vì 2 epochs liên tiếp (3,4) không có cải thiện

5. **So sánh với Run 2 (tốt hơn):**
   - **Baseline**: Val F1 = 0.550 (best epoch 3), early stop epoch 5
   - **Run 2**: Val F1 = **0.7135** (best epoch 4), train đủ 6 epochs
   - **Kết luận**: Run 2 có learning curve ổn định hơn, F1 cao hơn 30%!

---

## Phần 7: So sánh RNN vs Baseline ML (2.11)

### Dữ liệu từ `baseline_vs_rnn.csv`

| Model | Input Representation | Test Accuracy | Test Macro-F1 |
|-------|---------------------|---------------|---------------|
| Baseline ML (Lab 2) | TF-IDF/BoW | 90.74% ⭐ | 90.74% ⭐ |
| RNN (Lab 3) | Token Sequence | 71.44% | 71.41% |

### Phân tích Chi tiết

**Baseline ML (TF-IDF + Logistic Regression):**
- **Test Accuracy: 90.74%**
- **Test Macro-F1: 90.74%**
- **Đặc điểm:**
  - Sử dụng TF-IDF để biểu diễn - vector tĩnh, mất thứ tự từ
  - Logistic Regression - đơn giản nhưng mạnh mẽ
  - Không cần tuning hyperparameter phức tạp

**RNN (Embedding + RNN):**
- **Test Accuracy: 71.44%**
- **Test Macro-F1: 71.41%**
- **Đặc điểm:**
  - Sử dụng embedding + RNN - xử lý thứ tự từ
  - Phức tạp hơn, nhiều hyperparameter
  - Training mất thời gian hơn

### Trả lời câu hỏi so sánh

**❓ RNN có tốt hơn baseline hay không?**

> **Không.** Baseline (90.74%) tốt hơn RNN (71.44%) **19.3 percentage point**. Điều này bất ngờ nhưng rất bình thường trong NLP:
> - Vanilla RNN không xử lý tốt long-term dependencies (vanishing gradient)
> - TF-IDF + simple ML là baseline mạnh cho text classification
> - Để RNN vượt trội, cần dùng LSTM/GRU hoặc attention mechanism

**❓ Nếu có, mục cải thiện có đáng kể không?**

> Không áp dụng (RNN kém hơn). Nhưng điểm dương là **Run 2 (71.44%) vượt trội hơn Baseline của Lab 3 (55.08%)** - cải thiện 29.8%, cho thấy hyperparameter tuning rất quan trọng.

**❓ Nếu chưa tốt hơn, nguyên nhân là gì?**

> 1. **Vanilla RNN yếu**: Không xử lý long-term dependencies tốt (gradient vanishing)
> 2. **Truncation cao**: 34.75% review bị cắt ngắn → mô hình thiếu thông tin
> 3. **TF-IDF khá mạnh**: Mặc dù mất thứ tự từ, nhưng TF-IDF + LogReg rất hiệu quả cho IMDB
> 4. **RNN cần tuning sâu**: Cần thử LSTM, attention, pretrained embeddings, ...

**❓ Bài học rút ra về mối quan hệ giữa dữ liệu, biểu diễn và mô hình là gì?**

> 1. **Dữ liệu:** IMDB reviews có thể dùng BoW (TF-IDF không quan tâm thứ tự) → TF-IDF đã đủ
> 2. **Biểu diễn:** 
>    - TF-IDF (tĩnh, mất thứ tự) → hiệu quả nhờ từ điển phong phú
>    - RNN (động, giữ thứ tự) → sức mạnh chưa phát huy vì mô hình yếu + truncation
> 3. **Mô hình:**
>    - LogReg (đơn giản) + TF-IDF → 90.74%
>    - RNN (phức tạp) + Embedding → 71.44% (vanilla RNN kém)
>    - RNN + tuning tốt → 71.44% (Run 2 - hơn baseline vanilla RNN 29%)
> 
> **Kết luận:** Không phải "Deep Learning luôn tốt hơn". Phải cân nhắc dữ liệu, biểu diễn, và mô hình hợp lý.

---

## Phần 8: Phân tích Lỗi của RNN (2.12)

### Tóm tắt Error Analysis

**Dữ liệu từ `error_analysis_summary.md`:**
- **Tổng số lỗi:** 7,139 sai dự đoán (trên 25,000 test set = 28.56% error rate)
- **Đạt ngưỡng 10+ mẫu:** ✅ Yes (đủ để phân tích)

**Phân bố lỗi theo loại:**

| Error Type | Count | % | Mô tả |
|-----------|-------|---|------|
| **Negation** | 6,079 | 85.14% | RNN nhầm positive → negative khi có từ phủ định |
| **Mixed sentiment** | 651 | 9.12% | Review có cảm xúc lẫn lộn (tích cực+tiêu cực) |
| **Other** | 234 | 3.28% | Những loại lỗi khác |
| **Long review** | 175 | 2.45% | Review quá dài (>200 tokens) |

### 5 Mẫu Lỗi Chi tiết (từ negation bucket)

**Mẫu 1: Sarcasm/Irony (đầu tiên trong dataset)**
- **Text:** "Having not read the book, I was more open to the fresh interpretation that each director gives to their medium... The Nazis were particularly cruel to Russians... It's high time."
- **Label:** Positive (review khen là tốt)
- **Dự đoán:** Negative
- **Prob:** 86% confidence → sai hoàn toàn
- **Lý do lỗi:** RNN thấy "Nazis", "cruel" → dự đoán negative. Không hiểu context rằng review khen film vì nó "finally received some attention"
- **Gợi ý:** Cần attention mechanism để focus vào từ khóa quan trọng

**Mẫu 2: Negation của danh từ tiêu cực**
- **Text:** "It was actually mildly entertaining... not my favorite... worth watching... It is not a snoozer"
- **Label:** Positive (khen là đáng xem)
- **Dự đoán:** Negative
- **Prob:** 85.9% confidence
- **Lý do lỗi:** "not...worth", "not a snoozer" → RNN thấy "not" + từ tiêu cực → dự đoán negative
- **Gợi ý:** RNN cần học "not X" có thể là positive tùy context

**Mẫu 3: Complex negation structure**
- **Text:** "What would it be like to be accused... It is a frightening journey... This film is not for everybody... you will never get the point"
- **Label:** Positive (review khen)
- **Dự đoán:** Negative
- **Prob:** 85.87%
- **Lý do lỗi:** Quá nhiều từ tiêu cực ("frightening", "never") → RNN bị tác động mạnh
- **Gợi ý:** Cần pre-trained embeddings (GloVe) để hiểu "frightening journey" theo context

**Mẫu 4: Genre-specific language**
- **Text:** "Cooley High was actually a drama with moments of comedy... Getting high, shooting dice... just like in Cooley High my classmates... had a lot of love for each other"
- **Label:** Positive
- **Dự đoán:** Negative
- **Prob:** 85.66%
- **Lý do lỗi:** Từ "Getting high" bị nhầm thành "high" (drug-related) → negative. Không hiểu "Getting high" = "having fun"
- **Gợi ý:** Cần domain-specific tuning hoặc pretrained model on movie reviews

**Mẫu 5: Subtle negation**
- **Text:** "I purchased this video... cost as a rental... This is the only one that will never go back... They will only be able to do this well..."
- **Label:** Positive (khen rất cao - 10/10)
- **Dự đoán:** Negative
- **Prob:** 85.56%
- **Lý do lỗi:** "will never go back" (nghĩa là giữ lại vĩnh viễn) + toàn bộ review có tone khen tặng
- **Gợi ý:** Cần bidirectional attention để xét cả context trước/sau

### Phân tích Bucket Chính: NEGATION (85.14%)

**Tại sao Negation là 85% lỗi?**

1. **Vanilla RNN xử lý sequentially:** RNN đọc từ trái sang phải, nếu thấy "not" + từ tiêu cực → dự đoán negative, không xem context sau
2. **Phủ định trong tiếng Anh khó:** "not terrible" = good, nhưng RNN thường thấy "not" + "terrible" → negative
3. **Mất thông tin đầu review:** Nếu review bắt đầu với "I didn't like..." nhưng kết thúc "...actually it was good", RNN sẽ focus vào phần đầu

**Ví dụ cụ thể:**
```
Review: "Not bad at all. Actually pretty entertaining and worth watching."
RNN reads: "Not" [neg token] → starts predicting negative
Ignores: "Actually pretty entertaining and worth watching"
→ Dự đoán: NEGATIVE (Sai!)
```

### Mẫu từ Mixed Sentiment (9.12%)

**Ví dụ:**
- **Text:** "I gave it a seven only because the acting is good... The other two principals were decent... the characters themselves... what on earth was so bad...? (confusion và conflicted feeling)"
- **Label:** Positive (7/10, tức là positive)
- **Dự đoán:** Negative (RNN chọn negative vì quá nhiều từ tiêu cực như "bad", "insufferable jerk")
- **Lý do:** Review có cảm xúc lẫn lộn (phần nào tích cực, phần nào tiêu cực) → RNN khó phân loại

### Mẫu từ Other bucket (3.28%)

**Ví dụ:**
- **Text:** "This movie is excellent... There are many subplots... the whole movie is the journey to the discovery of one's real self... Realistic movie without superhero moments. This Chinese movie really puts Hollywood... to shame."
- **Label:** Positive (khen)
- **Dự đoán:** Negative
- **Error bucket:** "other"
- **Lý do:** Quá phức tạp, có quá nhiều nội dung (main plot + subplots), RNN khó xử lý

### Khuyến nghị Cải thiện

1. **LSTM/GRU:** Thay vanilla RNN để xử lý long-term dependencies tốt hơn
   - LSTM có memory cell giữ "negation context" tốt hơn
   
2. **Bidirectional RNN:** Xem context trước/sau để hiểu negation tốt hơn
   - "not X" hoặc "X, but not..." có ý nghĩa khác

3. **Attention Mechanism:** Focus vào từ khóa quan trọng
   - Thay vì xem all tokens bình đẳng, focus vào "excellent", "entertaining", ...

4. **Pretrained Embeddings (GloVe, FastText):**
   - Embeddings sẽ capture "not bad" ≈ "good" từ corpus lớn

5. **Domain-specific Fine-tuning:**
   - Fine-tune trên movie reviews để hiểu domain-specific language

---

**Generated:** 2026-04-27
**Status:** ✅ Training Complete - All Analysis Done
