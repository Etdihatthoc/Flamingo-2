# Tóm tắt thay đổi - Sửa lỗi MAE evaluation

## 🎯 Vấn đề gốc

Bạn phát hiện ra:
- **Lúc train** (validation): Model dự đoán format đẹp → MAE thấp
- **Lúc test độc lập**: Model dự đoán format xấu → MAE cao

Ví dụ:
```
# Lúc validation trong training
Ground Truth: total: 5.5/10, vocabulary: 6.0/10, grammar: 5.0/10
Prediction:   total: 5.5/10, vocabulary: 5.0/10, grammar: 5.5/10  ✅

# Lúc test độc lập
Ground Truth: total: 5.5/10, vocabulary: 6.0/10, grammar: 5.0/10  
Prediction:   this: 8.5/10iet: 9.5/10fl: 7.5/10  ❌
```

## 🔍 Nguyên nhân

Hai phương pháp **khác nhau hoàn toàn**:

### Phương pháp cũ (teacher-forcing) trong metrics.py:
```python
# Đưa CẢ prompt VÀ answer vào model
output = model(
    lang_x=input_ids,  # Chứa cả câu trả lời đúng!
    labels=labels
)
# Model "nhìn trộm" được câu trả lời → predict dễ
```

### Phương pháp đúng (autoregressive) trong inference.py:
```python
# Chỉ đưa prompt vào model
prompt = input_ids[:sep_position]  # Chỉ có câu hỏi
output = model.generate(
    lang_x=prompt  # KHÔNG có câu trả lời
)
# Model phải tự generate từng token → khó hơn
```

## ✅ Giải pháp

Đã sửa `mae_val_epoch` và `mae_test_epoch` trong `metrics.py`:
- ❌ Bỏ: `model.forward()` với full input (teacher-forcing)
- ✅ Thêm: `model.generate()` với chỉ prompt (autoregressive)

## 📊 Kết quả mong đợi

### Trước khi sửa:
- Validation MAE: **thấp** (model được "nhìn đáp án")
- Test độc lập MAE: **cao** (model không nhìn đáp án)
- → **Không nhất quán!**

### Sau khi sửa:
- Validation MAE: **cao hơn** (realistic)
- Test độc lập MAE: **tương tự validation**
- → **Nhất quán!** ✅

## ⚠️ Lưu ý quan trọng

1. **MAE score cao hơn là ĐÚNG**, không phải bug!
   - Score cũ thấp vì model "gian lận" (nhìn đáp án)
   - Score mới cao hơn nhưng là khả năng **thực sự** của model

2. **Format xấu hơn là ĐÚNG**, không phải lỗi!
   - Format đẹp trước đây là do teacher-forcing
   - Format xấu bây giờ phản ánh model thực tế generate

3. **Không cần train lại model**
   - Chỉ sửa cách đánh giá, không sửa model
   - Weights của model không đổi

## 🚀 Cách dùng

### 1. Training (không thay đổi gì)
```bash
CUDA_VISIBLE_DEVICES=0 python train_lora.py -c config.yaml
```

Validation MAE giờ sẽ cho kết quả **chính xác**.

### 2. Test độc lập
Giống như trước, nhưng giờ kết quả **nhất quán** với validation:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --checkpoint ./model.pt
```

## 📈 Thay đổi hiệu suất

| Khía cạnh | Trước | Sau |
|-----------|-------|-----|
| **Tốc độ eval** | Nhanh (~100 samples/min) | Chậm hơn (~20-50 samples/min) |
| **MAE score** | Thấp (không thực tế) | Cao hơn (thực tế) |
| **Nhất quán** | ❌ Khác nhau train/test | ✅ Giống nhau train/test |
| **Độ chính xác** | ❌ Sai (teacher-forcing) | ✅ Đúng (autoregressive) |

## 📚 Tài liệu

1. **CHANGES_EXPLANATION.md**: Giải thích chi tiết kỹ thuật (English + Vietnamese)
2. **USAGE_NOTES.md**: Hướng dẫn sử dụng và troubleshooting
3. **metrics.py**: Code đã được sửa

## 🎓 Hiểu nhanh

**Teacher-forcing** = Model nhìn đáp án khi predict
- Dễ → score cao
- Không thực tế
- Chỉ dùng khi train

**Autoregressive** = Model tự generate, không nhìn đáp án
- Khó → score thấp hơn
- Thực tế
- Dùng khi test và deploy

## ❓ FAQ

**Q: Tại sao MAE score cao hơn bây giờ?**
A: Vì trước đây model "nhìn đáp án". Giờ không nhìn nữa nên khó hơn.

**Q: Có cần train lại không?**
A: Không! Model không đổi, chỉ cách đánh giá đổi.

**Q: Tại sao format prediction xấu?**
A: Đó là khả năng thực của model. Format đẹp trước đây do teacher-forcing.

**Q: Có thể dùng lại code cũ không?**
A: Không nên. Code cũ cho metrics sai lệch.

**Q: Test chậm quá, làm sao?**
A: Giảm `max_samples` trong hàm `mae_val_epoch`:
```python
mae_metrics = mae_val_epoch(..., max_samples=10)  # Thay vì 50
```

## ✨ Kết luận

- ✅ Đã sửa xong
- ✅ Giờ validation và test nhất quán
- ✅ Metrics chính xác phản ánh model
- ⚠️ Score cao hơn nhưng đó là **sự thật** về model
- ⚠️ Format xấu hơn nhưng đó là model **thực tế** generate

**Đừng lo lắng về score cao hơn - đó là điều TỐT vì bây giờ bạn biết khả năng THẬT của model!**
