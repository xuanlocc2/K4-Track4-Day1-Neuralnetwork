# Báo cáo Lab Day 1 — Nguyễn Văn Xuân Lộc — 2A202602870

> Mọi con số trỏ về `exp_id` trong `experiments.xlsx` / `results/<exp_id>.json`. Các so sánh trong mục 3 là **1 seed (seed 1)** của thí nghiệm so với **trung bình 3 seed của baseline**; ngưỡng nhiễu 2σ = 0,0141 (val macro-F1).

## 1. Thiết lập

- Môi trường: Google Colab, GPU Tesla T4, PyTorch 2.11.0+cu130.
- Dữ liệu: Forest CoverType; `train` 464 809 / `eval` 116 203. Validation: 20% của train (phân tầng, seed 42) → 371 847 train / 92 962 val. 10 cột số được chuẩn hoá bằng thống kê của train.
- Model: `M-base` 54→256→128→7 (47 879 tham số), ReLU, logits không softmax. Baseline: cross-entropy, SGD+momentum 0,9, **lr 0,2**, batch 512, 20 epoch, He init, không dropout, FP32. lr chọn bằng sweep 10 epoch trên val (lr 0,003→0,5; best val loss thấp nhất ở 0,2; val macro-F1 của 0,2 = 0,825 và 0,3 = 0,828 không phân biệt được).
- Mốc tham chiếu: accuracy "đoán lớp đa số" trên val = **0,4876**.
- Các chủ đề đã thử: ☑ loss ☑ optimizer ☑ hyper-parameter ☑ dropout ☑ clipping ☑ mixed precision ☑ init

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả |
|---|---|
| Số tham số / shape logits | 47 879 / (B, 7) |
| Loss bước 0 (so với ln 7 = 1,946) | 2,378 (model Part 1); 1,90–2,27 tuỳ seed trong baseline |
| Quá khớp 20 mẫu: loss cuối | 0,00094 (acc 1,00; Adam lr 1e-3, 300 bước) |
| Mọi tham số có gradient khác 0 | ☑ có (6/6 tham số, grad norm 0,39–2,36) |
| Baseline, số seed đã chạy | 3 (`base-s1..s3`) |
| Baseline: val acc (TB ± σ) | 0,9125 ± 0,0035 |
| Baseline: val macro-F1 (TB ± σ) | 0,8624 ± 0,0071 |

**Ngưỡng nhiễu dùng trong báo cáo:** 2σ = **0,0141** (val macro-F1); với accuracy 2σ = 0,0069.

Loss bước 0 cao hơn ln 7 vì He init cho logit có độ lệch chuẩn ≈ 1 (không phải 0): khi logit không đều, mạng "tự tin sai" ở nhiều mẫu và cross-entropy kỳ vọng vượt ln C. Đo trực tiếp: loss bước 0 thay đổi theo seed (1,90 / 1,98 / 2,27). Quá khớp 20 mẫu về ≈ 0 chứng tỏ model, loss và vòng cập nhật đúng.

**Baseline:** val acc 0,9125 so với mốc đoán đa số 0,4876. Best epoch 19–20/20 và val loss còn giảm nhẹ ở cuối: mô hình **chưa hội tụ hẳn** và chưa quá khớp (val − train loss ≈ 0,025). Grad norm phẳng 0,39–0,44.

## 3. Kết quả theo chủ đề

### 3.1 Hàm mất mát — CE vs MSE
- Dự đoán: Mình đoán MSE trên one-hot sẽ kém cross-entropy: gradient nhỏ hơn nên học chậm hơn, và các lớp hiếm sẽ bị bỏ qua nhiều hơn. Giá trị loss của hai cách khác thang đo nên mình sẽ chỉ so accuracy và macro-F1.
- Kết quả: `loss-mse` val macro-F1 **0,7816** vs baseline 0,8624 (−0,081 ≫ 2σ); val acc 0,8819 vs 0,9125. Ảnh: `figures/compare_loss.png`.
- Giải thích: MSE được tính trên logit so với one-hot, trung bình trên N×7 phần tử, nên gradient theo logit là 2(z−y)/7 — nhỏ hơn và tuyến tính theo sai số. `grad_norm` chỉ 0,06–0,08 so với 0,40 của CE, tức là hiệu quả giống lr nhỏ hơn nhiều và học chậm (best epoch 20, vẫn đang giảm). CE có gradient p−y đi qua softmax, nên mẫu bị đoán sai nặng nhận gradient lớn; các lớp hiếm (lớp 3, 4) vì thế cũng tụt F1 nhiều hơn. Giá trị loss (0,027) **không so được** với CE (0,229) vì khác thang đo; chỉ so accuracy/macro-F1.

### 3.2 Bộ tối ưu hoá
- Dự đoán: Mình đoán Adam sẽ học nhanh hơn SGD ở những epoch đầu vì nó tự chỉnh bước cho từng tham số. SGD thuần thì cần lr lớn hơn Adam rất nhiều (momentum 0.9 làm lr hiệu dụng của baseline khoảng 10 lần lr, nên SGD thuần phải thử lr cỡ 0.5–2). AdamW với weight decay nhỏ chắc gần giống Adam. Mình nghĩ nếu mỗi cái được chỉnh lr cho tốt thì kết quả chênh nhau không nhiều.
- Mỗi bộ ở lr tốt nhất trong các lr đã thử (val macro-F1; baseline 0,8624):

| Bộ tối ưu | exp_id | lr | val macro-F1 | best epoch |
|---|---|---|---|---|
| SGD+momentum (baseline) | `base-s1..s3` | 0,2 | 0,8624 (TB 3 seed) | 19–20 |
| SGD thuần | `opt-sgd-lr0.5` | 0,5 | 0,8311 | 19 |
| Adam | `opt-adam-lr3e-3` | 3e-3 | **0,8702** | 19 |
| AdamW (wd 0,01) | `opt-adamw-lr1e-3` | 1e-3 | 0,8403 | 18 |

- Độ nhạy lr: SGD thuần lr 0,1 / 0,5 / 2,0 → 0,755 / 0,831 / 0,822 (`opt-sgd-lr0.1`, `opt-sgd-lr0.5`, `opt-sgd-lr2.0`); Adam lr 3e-4 / 1e-3 / 3e-3 → 0,788 / 0,845 / 0,870. SGD lr 2,0 dao động mạnh giữa các epoch (val loss nhảy 0,28→0,45). Adam lr 3e-4 quá chậm (val loss còn giảm ở epoch 20). Ảnh chồng: `figures/compare_optimizer.png` (gồm SGD lr 0,5 / 2,0, Adam 3 mức lr, AdamW và `base-s1`; **không** có SGD lr 0,1, đường cong riêng của nó ở `figures/opt-sgd-lr0.1.png`).
- Giải thích: SGD+momentum 0,9 có lr hiệu dụng ≈ lr/(1−0,9) = 10×lr, nên baseline lr 0,2 tương đương SGD thuần ≈ 2,0; SGD thuần cần lr 0,5–2 để so sánh công bằng. Adam chuẩn hoá bước theo căn bậc hai của moment bậc hai nên ít phụ thuộc thang gradient của từng tham số, nhưng vẫn cần lr đủ lớn trong 20 epoch. AdamW lr 1e-3, wd 0,01: phân rã mỗi bước chỉ lr·wd = 1e-5, nên khác Adam 1e-3 (−0,004) nằm trong nhiễu.
- **Kết luận:** Adam lr 3e-3 hơn baseline +0,0078 < 2σ (0,0141) → **không phân biệt được** với SGD+momentum khi cả hai được chỉnh lr. Kết luận "Adam thua" nếu dùng lr mặc định 1e-3 (−0,018, vượt nhiễu) chỉ là do lr chưa chỉnh.

### 3.3 Hyper-parameter
- Dự đoán: Mình đoán baseline chưa học xong trong 20 epoch (best epoch gần cuối), nên tăng lên 40 epoch sẽ tốt hơn. Mạng rộng hơn (512-256) cũng nên tốt hơn một chút. Batch lớn (2048) chạy nhanh hơn mỗi epoch nhưng ít bước cập nhật hơn nên có thể kém hơn, nhân lr lên 4 lần có thể cứu được. Batch nhỏ (128) chậm hơn mà chưa chắc tốt hơn.

| exp_id | thay đổi | val macro-F1 | Δ vs TB baseline | vượt 2σ? | s/epoch |
|---|---|---|---|---|---|
| `hp-ep40` | 40 epoch | 0,8770 | +0,0146 | ≈ có (sát ngưỡng) | 1,31 |
| `hp-wide` | 512-256 | 0,8772 | +0,0148 | ≈ có (sát ngưỡng) | 1,34 |
| `hp-deep` | 256-128-64 | 0,8572 | −0,0053 | không | 1,48 |
| `hp-b128` | batch 128 | 0,8236 | −0,0388 | có | **5,23** |
| `hp-b2048` | batch 2048 | 0,8320 | −0,0305 | có | **0,35** |
| `hp-b2048-lr0.8` | batch 2048, lr ×4 | 0,8263 | −0,0361 | có | 0,34 |

Ảnh: `figures/compare_hparam.png`.
- Giải thích: số bước cập nhật mỗi epoch = N/batch (≈ 726 với batch 512, ≈ 2 900 với 128, ≈ 182 với 2048). `hp-ep40` cho thêm bước nên val loss giảm tiếp (best epoch 37, 0,196); khoảng cách val − train loss tăng 0,026→0,038, là dấu hiệu bắt đầu quá khớp nhẹ. Batch 2048 nhanh gấp ≈ 4 mỗi epoch nhưng chỉ có 1/4 số bước; tăng lr ×4 theo quy tắc tuyến tính không cứu được (0,8263), vì lr 0,8 với momentum 0,9 đã quá lớn so với vùng ổn định (xem 3.5). Batch 128 chậm 4× mà không tốt hơn (best epoch 15, gradient nhiễu hơn ở lr 0,2). `hp-wide` tốt hơn gần bằng `hp-ep40` mà chi phí thời gian/epoch gần như không đổi (GPU chưa kín tải ở mạng nhỏ), bộ nhớ 192 MB vs 173 MB.
- Lưu ý: chỉ 1 seed, Δ của `hp-ep40` và `hp-wide` (+0,0146 / +0,0148) chỉ **sát** ngưỡng 2σ = 0,0141, nên kết luận "tốt hơn" còn yếu.

### 3.4 Dropout
- Dự đoán: Mình đoán dropout sẽ không giúp, vì baseline có train loss ≈ val loss (gap chỉ khoảng 0.025) nên chưa overfit. Thêm dropout chỉ làm mạng khó học hơn, nên macro-F1 sẽ giảm, và giảm nhiều hơn khi dropout 0.3.
- Kết quả: `drop-0.1` val macro-F1 0,8426 (−0,020, vượt nhiễu); `drop-0.3` 0,7921 (−0,070). Khoảng cách val − train loss: 0,026 (baseline) → 0,012 → 0,007. Ảnh: `figures/compare_dropout.png`.
- Giải thích: dropout thu hẹp khoảng cách đúng như kỳ vọng, nhưng baseline **không quá khớp** (train loss 0,20 vs val 0,23, best epoch cuối) mà đang **chưa khớp đủ**; thêm nhiễu chỉ làm train loss tăng (0,203→0,230→0,304) và val tệ hơn. Dropout chỉ nên dùng khi val loss bắt đầu tăng trong khi train loss vẫn giảm.

### 3.5 Gradient clipping
- Dự đoán: Mình đoán ở lr bình thường clipping gần như không đổi kết quả, vì grad norm của baseline khá phẳng. Ở lr rất cao (1.0, 2.0) không clip thì huấn luyện sẽ hỏng hoặc rất dao động, còn có clip thì ổn định hơn nhưng chưa chắc bằng baseline.
- Ngưỡng c = 0,29 (≈ 0,7 × grad norm trung bình của baseline 0,41).
- Lr bình thường: `clip-0.29` có kích hoạt (grad norm trước clip trung bình 0,55–0,59 > c ở mọi epoch) nhưng val macro-F1 0,8551 (−0,0074 < 2σ): không đổi đáng kể, chỉ giảm nhẹ tốc độ vì bước bị co.
- Lr cao (`figures/compare_clipping.png`):

| exp_id | lr | clip | val macro-F1 | val acc |
|---|---|---|---|---|
| `clip-hi1.0-noclip` | 1,0 | không | 0,7775 | 0,8618 |
| `clip-hi1.0-clip` | 1,0 | 0,29 | **0,8244** | 0,8870 |
| `clip-hi2.0-noclip` | 2,0 | không | 0,0936 | 0,4876 |
| `clip-hi2.0-clip` | 2,0 | 0,29 | 0,5994 | 0,7790 |

- Giải thích: ở lr 2,0 không clip, huấn luyện sụp về dự đoán lớp đa số (val acc 0,4876 = mốc đoán đa số, loss kẹt 1,21 và grad norm tụt còn 0,06–0,18). Mô hình vẫn không NaN nhưng mạng bị đẩy vào vùng không học được. Clipping giới hạn độ dài bước nên cứu được phần lớn (F1 0,094→0,599; ở lr 1,0: 0,778→0,824, vượt nhiễu), nhưng không khôi phục hẳn mức của baseline vì lr vẫn quá lớn để hội tụ tốt. Hạn chế: `grad_norm` ghi lại là trung bình mỗi epoch trước clip, nên không thấy được từng đợt đột biến.

### 3.6 Mixed precision

| exp_id | precision | s/epoch | peak mem (MB) | val macro-F1 | val acc |
|---|---|---|---|---|---|
| `base-s1` | FP32 | 1,32 | 173 | 0,8549 | 0,9086 |
| `amp-fp16` | FP16 + GradScaler | 1,79 | 178 | 0,8463 | 0,9089 |
| `amp-bf16` | BF16 | 1,59 | 178 | 0,8570 | 0,9139 |

- Dự đoán: Mình đoán mạng này quá nhỏ (48k tham số) nên mixed precision sẽ không nhanh hơn, có thể còn chậm hơn vì tốn thêm bước đổi kiểu số. Độ chính xác chắc không đổi nhiều. FP16 cần GradScaler còn BF16 thì không.
- Giải thích: **không nhanh hơn, thậm chí chậm hơn** (+36% FP16, +20% BF16) và không tiết kiệm bộ nhớ. Mạng chỉ 48k tham số và toàn bộ dữ liệu nằm trên GPU, nên thời gian bị chi phối bởi khởi chạy kernel và chi phí autocast/chuyển kiểu, không phải bởi tính toán ma trận mà Tensor Core tăng tốc. Bộ nhớ cực đại bị chi phối bởi dữ liệu trên GPU. Độ chính xác không đổi đáng kể (chênh trong nhiễu). FP16 cần GradScaler vì phạm vi mũ nhỏ (gradient dễ underflow); BF16 có cùng phạm vi mũ với FP32 nên thường không cần.

### 3.7 Khởi tạo tham số
- Dự đoán: Mình đoán zeros sẽ hỏng hoàn toàn vì mọi nơ-ron giống hệt nhau. Normal std 0.01 làm kích hoạt nhỏ dần qua từng lớp nên học chậm hoặc kém. Xavier và He chắc ngang nhau vì mạng chỉ có 3 lớp.

std kích hoạt sau từng Linear ở bước 0 (eval, 2 048 mẫu val):

| init | std L1 / L2 / L3 | val macro-F1 | `exp_id` |
|---|---|---|---|
| zeros | 0 / 0 / 0 | 0,0936 | `init-zeros` |
| normal (0,01) | 0,034 / 0,0040 / 0,00029 | 0,8547 | `init-normal` |
| xavier | 0,27 / 0,23 / 0,20 | 0,8653 | `init-xavier` |
| default (PyTorch) | 0,29 / 0,12 / 0,07 | 0,8550 | `init-default` |
| he (baseline) | 0,65 / 0,68 / 0,62 | 0,8549 (s1) | `base-s1` |

Ảnh: `figures/compare_init.png`.
- `zeros`: mọi nơ-ron trong một lớp giống hệt nhau (đối xứng) và đầu ra ReLU = 0 nên gradient của trọng số các lớp dưới bằng 0; chỉ bias lớp cuối học, mạng chỉ biết tần suất lớp: val acc 0,4876, loss kẹt 1,205 (= entropy của phân phối lớp), grad norm ≈ 0,03.
- `normal` 0,01: độ lệch chuẩn kích hoạt co ≈ 10× mỗi lớp (0,034→0,004→0,0003), logit gần 0 nên loss bước 0 = 1,946 = ln 7; với 3 lớp và lr hiệu dụng lớn mạng vẫn thoát được (0,8547), nhưng mạng sâu hơn sẽ bị tắt.
- He vs Xavier: Xavier giả định kích hoạt tuyến tính (phương sai 2/(fan_in+fan_out)); ReLU triệt một nửa phương sai nên He dùng 2/fan_in để giữ std ≈ hằng qua lớp. Ở mạng 3 lớp chênh lệch chỉ là 0,65 vs 0,2 ở std, và val macro-F1 của he / xavier / default đều trong nhiễu; khác biệt chỉ quan trọng khi mạng sâu hơn.

## 4. Đánh giá cuối trên tập eval

Số lấy từ `eval_result.json` / `scripts/evaluate.py`.

| Cấu hình | Seed nộp | val macro-F1 | **eval macro-F1** | eval accuracy |
|---|---|---|---|---|
| Baseline (`base-s1`) | 1 | 0,8549 | 0,8576 | 0,9084 |
| Cấu hình cuối cùng (`hp-wide`) | 1 | 0,8772 | **0,8799** | 0,9217 |

- Cấu hình cuối: `hp-wide` = baseline với hidden (512, 256) (161 287 tham số), chọn vì val macro-F1 cao nhất trong các lần chạy (0,8772), chỉ hơn `hp-ep40` 0,0002 (không phân biệt được; mình chọn theo quy tắc cố định để không "chọn tay"). **Winner's curse:** chọn lần chạy tốt nhất trong ~30 lần có thiên lệch lạc quan.
- Cải thiện trên eval: +0,0223 macro-F1 (0,8576→0,8799), +0,0133 acc. Trên val cải thiện tương tự (+0,0223 so với `base-s1`, +0,0148 so với trung bình 3 seed). Mức +0,0223 so với ngưỡng 2σ = 0,0141 là vượt nhiễu, nhưng cả hai bên chỉ có 1 seed trên eval nên không có σ eval.
- Val và eval sát nhau (lệch ≈ +0,003 ở cả hai cấu hình) → việc chọn bằng val không bị lệch rõ trên eval.

### 4.1 Phân tích lỗi theo lớp (cấu hình cuối, eval)

| Lớp | Tên | support | precision | recall | F1 |
|---|---|---|---|---|---|
| 0 | Spruce/Fir | 42 368 | 0,9239 | 0,9142 | 0,9190 |
| 1 | Lodgepole Pine | 56 661 | 0,9315 | 0,9365 | 0,9340 |
| 2 | Ponderosa Pine | 7 151 | 0,8902 | 0,9457 | 0,9171 |
| 3 | Cottonwood/Willow | 549 | 0,8926 | 0,7869 | 0,8364 |
| 4 | Aspen | 1 899 | 0,7747 | 0,8004 | **0,7874** |
| 5 | Douglas-fir | 3 473 | 0,8844 | 0,7910 | 0,8351 |
| 6 | Krummholz | 4 102 | 0,9239 | 0,9376 | 0,9307 |

- Lớp khó nhất là **lớp 4 Aspen** (F1 = 0,787, precision 0,775): 16% mẫu Aspen bị đoán thành Lodgepole Pine (304/1 899) và 357 mẫu Lodgepole bị đoán thành Aspen. Hai lỗi lớn nhất tuyệt đối là Spruce/Fir ↔ Lodgepole Pine (3 292 và 2 937 mẫu), do hai lớp chiếm 85% dữ liệu và đặc trưng địa hình/độ cao rất gần nhau.
- Lớp 3 (Cottonwood/Willow) chỉ 549 mẫu (0,47%), recall 0,787: 17% bị nhầm sang Ponderosa Pine (93/549); Douglas-fir bị nhầm với Ponderosa Pine (536 mẫu, 15%). Nguyên nhân: lớp ít mẫu, đặc trưng chồng lấn lớp gần.
- Cách cải thiện sẽ thử: train lâu hơn kết hợp wide (ep40 đã tốt), class-weighted loss hoặc focal loss cho lớp hiếm, và các đặc trưng tương tác (khoảng cách tới nước × độ cao).

## 5. Trả lời các câu hỏi dẫn dắt

1. **Optimizer nào thắng khi chỉnh lr công bằng?** Adam lr 3e-3 (0,8702) vs SGD+momentum lr 0,2 (0,8624): chênh +0,0078 < 2σ (0,0141), nên **không có người thắng rõ**. Khi lr không chỉnh (Adam 1e-3: 0,8446, AdamW: 0,8403) Adam có vẻ thua vượt nhiễu; kết luận đảo chiều chỉ vì lr.
2. **Dropout giúp không khi chưa quá khớp?** Không: dropout 0,1 / 0,3 giảm val macro-F1 0,020 / 0,070 (vượt nhiễu) vì mạng đang chưa khớp đủ; chỉ nên dùng khi val loss tăng trong khi train loss giảm.
3. **Clipping giải quyết gì?** Chặn bước cập nhật quá lớn khi lr cao. Bằng chứng: ở lr 2,0 không clip sụp về đoán đa số (acc 0,4876, F1 0,094), có clip đạt F1 0,599; ở lr 1,0 clip nâng F1 0,778→0,824. Ở lr bình thường clip không đổi kết quả (−0,0074, trong nhiễu).
4. **Mixed precision nhanh hơn?** Không: FP16 1,79 s/epoch, BF16 1,59 s so với FP32 1,32 s; mạng 48k tham số và dữ liệu nằm sẵn trên GPU nên bị chi phối bởi chi phí khởi chạy và chuyển kiểu, không phải phép nhân ma trận.
5. **Vì sao zeros hỏng? He vs Xavier?** Đối xứng + ReLU(0)=0 làm gradient các lớp dưới = 0 (xem 3.7). He dùng phương sai 2/fan_in bù cho ReLU triệt nửa phương sai, Xavier dùng 2/(fan_in+fan_out) cho kích hoạt tuyến tính; quan trọng khi mạng sâu vì std kích hoạt co/giãn theo luỹ thừa của số lớp.
6. **Loss không giảm sau 2 000 bước, 3 phép kiểm tra đầu tiên:** (a) **quá khớp một lô nhỏ** (20 mẫu về 0,001 trong Part 1) để tách lỗi code/mô hình khỏi lỗi tối ưu; (b) **đo loss bước 0 so với ln C, std kích hoạt và grad norm từng lớp** (init zeros cho std 0 và loss kẹt 1,205; normal cho std co 10× mỗi lớp); (c) **quét lr** (lr 2,0 kẹt ở 1,21 và grad norm tụt; lr 0,003 chỉ đạt 0,623 sau 10 epoch) và theo dõi grad norm để biết cần clip hay giảm lr.

## 6. Hạn chế và điều bất ngờ

- Bất ngờ: AMP **chậm hơn** FP32; `init-normal` (std 0,01) vẫn học tốt dù kích hoạt co ≈ 10×/lớp; MSE tụt rất mạnh (−0,081) mà một phần là do gradient nhỏ chứ không chỉ do "loss kém".
- Hạn chế: mọi thí nghiệm Part 3 chỉ 1 seed, trong khi σ của baseline = 0,0071 (ngưỡng 0,0141) là ước lượng từ 3 seed nên bản thân cũng nhiễu; `hp-wide` / `hp-ep40` chỉ sát ngưỡng; chọn cấu hình cuối từ ~30 lần chạy có winner's curse; cùng 20 epoch nhưng khác batch nghĩa là khác số bước; baseline chưa hội tụ nên mọi so sánh "20 epoch" thiên về cấu hình học nhanh; `grad_norm` là trung bình theo epoch.
- Nếu có thêm thời gian: chạy `hp-wide` + 40 epoch (kết hợp) với 3 seed, chạy lại các nhóm tốt nhất với 3 seed, thử Adam 3e-3 + wide, và class-weighted loss cho lớp 3, 4.

## 7. Phụ lục

- File nộp: `REPORT.md`, `experiments.xlsx`, `predictions_eval.csv`, `eval_result.json`, `figures/` (mỗi `exp_id` một ảnh + `compare_*.png`), `results/` (JSON mỗi lần chạy + `part1_overfit20.png`), `code/` (`lab.ipynb`, `data.py`, `model.py`, `optimizer.py`, `train.py`, `plots.py`, `results_table.py`).
- Thời gian chạy: ≈ 1,3–1,8 s/epoch trên T4 (batch 512; 5,2 s với batch 128), tổng ≈ 20–30 phút cho ~30 lần chạy.
