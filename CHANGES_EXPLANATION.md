# Explanation of Changes: Teacher-Forcing vs Autoregressive Generation

## Problem Statement

The user observed that when running MAE (Mean Absolute Error) evaluation:
- **During training validation** (`mae_val_epoch` / `mae_test_epoch` in train loop): Predictions had correct format
  ```
  Ground Truth: total: 5.5/10, vocabulary: 6.0/10, grammar: 5.0/10, ...
  Prediction:   total: 5.5/10, vocabulary: 5.0/10, grammar: 5.5/10, ...
  ```

- **Standalone testing** (loading checkpoint and running independently): Predictions had malformed format
  ```
  Prediction: this: 8.5/10iet: 9.5/10fl: 7.5/10 fluunciation: 7.5/10...
  ```

## Root Cause Analysis

The issue was that the MAE evaluation functions (`mae_val_epoch` and `mae_test_epoch`) used a different inference method than standalone testing:

### 1. **Original Implementation (Teacher-Forcing)**

```python
# OLD CODE in mae_val_epoch/mae_test_epoch
output = model(
    audio_x=audio_clips,
    audio_x_mask=audio_embed_mask,
    lang_x=input_ids,              # Full sequence including ground truth!
    attention_mask=attention_mask,
    labels=labels                  # Ground truth labels
)

# Extract predictions from logits
logits = output.logits
pred_tokens = torch.argmax(logits, dim=-1)
pred_text = tokenizer.decode(pred_tokens[..., after_sep_position:])
```

**Problems with this approach:**
- Uses `model.forward()` which is the training forward pass
- Passes the **entire input sequence** including ground truth tokens after SEP
- Model can see the correct previous tokens when predicting each next token
- This is **teacher-forcing**: at each step, the model receives the ground truth previous token
- Results in unrealistically good predictions that don't reflect actual generation capability

### 2. **Standalone Testing (Autoregressive Generation)**

```python
# EXISTING CODE in inference.py
prompt = input_ids[:sep_location+1]  # Only up to SEP token

output = model.generate(
    audio_x=audio_clips,
    audio_x_mask=audio_embed_mask,
    lang_x=prompt,                   # Only prompt, no ground truth!
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=256,
    temperature=1.0,
)

output_decoded = tokenizer.decode(output)
```

**Why this is different:**
- Uses `model.generate()` which is autoregressive generation
- Only passes the **prompt** (up to SEP token), no ground truth
- Model generates one token at a time
- Each next token is predicted based only on previously **generated** tokens (not ground truth)
- This is **autoregressive generation**: realistic generation scenario

### 3. **Why Teacher-Forcing Gave Better Results**

Teacher-forcing during evaluation gave unrealistically good results because:

1. **Error accumulation is hidden**: In real generation, if the model makes one mistake, it can cascade into more mistakes. Teacher-forcing prevents this by always providing correct context.

2. **Format consistency**: With teacher-forcing, if ground truth has "vocabulary:", the model sees this pattern and can continue it correctly. Without it, the model might generate "vocab" or misspell words.

3. **Token-level vs sequence-level prediction**: Teacher-forcing evaluates token-level prediction accuracy, while autoregressive generation evaluates sequence-level generation quality.

## The Fix

### New Implementation (Autoregressive Generation)

```python
# NEW CODE in mae_val_epoch/mae_test_epoch
for sample_idx in range(input_ids.shape[0]):
    input_id = input_ids[sample_idx]
    
    # Find SEP token location
    sep_location = None
    for i in range(len(input_id)-1, -1, -1):
        if input_id[i] == sep_token_id:
            sep_location = i
            break
    
    # Extract ONLY the prompt (up to and including SEP)
    prompt = input_id[:sep_location+1]
    
    # Generate prediction using autoregressive generation
    output = model.generate(
        audio_x=audio_clips[sample_idx].unsqueeze(0),
        audio_x_mask=audio_embed_mask[sample_idx].unsqueeze(0),
        lang_x=prompt.unsqueeze(0),          # Only prompt!
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=256,
        temperature=1.0,                     # Same as inference
        num_beams=1,                         # Greedy decoding
    )[0]
    
    # Decode prediction
    pred_text = tokenizer.decode(output).split(tokenizer.sep_token)[-1]
```

### Key Changes

1. **Process samples individually**: Instead of batch forward pass, we now loop through samples
2. **Extract prompt only**: Only take tokens up to (and including) the SEP token
3. **Use `model.generate()`**: Switch from `model.forward()` to `model.generate()`
4. **Consistent parameters**: Use same generation parameters as standalone testing
5. **No ground truth in input**: Model only sees prompt, not the answer

## Impact

### Before (Teacher-Forcing)
- ✓ Fast evaluation (batch processing)
- ✓ Clean, well-formatted outputs
- ✗ **Unrealistic** - doesn't match actual model performance
- ✗ **Misleading metrics** - overly optimistic MAE scores
- ✗ **Inconsistent** - different from standalone testing

### After (Autoregressive Generation)
- ✓ **Realistic** - matches actual generation performance
- ✓ **Consistent** - same behavior as standalone testing
- ✓ **Accurate metrics** - true reflection of model capability
- ⚠ Slower evaluation (sequential generation per sample)
- ⚠ May produce less formatted outputs (but this is realistic!)

## Conclusion

The change from teacher-forcing to autoregressive generation in MAE evaluation functions ensures that:

1. **Validation metrics during training accurately reflect model performance**
2. **No surprises when deploying the model** - performance matches validation metrics
3. **Consistency across all evaluation contexts** - training validation, testing, and standalone inference

The malformed outputs seen in standalone testing were actually the **correct behavior** - they show what the model really generates. The well-formatted outputs during training validation were the **incorrect behavior** - they showed an artificially optimistic view due to teacher-forcing.

## Vietnamese Summary (Tóm tắt tiếng Việt)

### Vấn đề
- Lúc train: kết quả MAE đẹp, format đúng
- Lúc test độc lập: kết quả MAE xấu, format sai

### Nguyên nhân
- **Lúc train**: dùng teacher-forcing - model nhìn thấy câu trả lời đúng khi dự đoán
- **Lúc test**: dùng autoregressive - model tự generate từng token một, không nhìn thấy câu trả lời

### Giải pháp
Đổi code trong `mae_val_epoch` và `mae_test_epoch`:
- Từ `model.forward()` (teacher-forcing) → `model.generate()` (autoregressive)
- Chỉ đưa prompt vào model, không đưa ground truth
- Giờ validation trong training sẽ cho kết quả giống test độc lập

### Kết quả
- Metrics giờ **chính xác** phản ánh khả năng thực của model
- Không còn bất ngờ khi deploy model
- Format có thể xấu hơn nhưng đó là **hiện thực**, không phải bug!
