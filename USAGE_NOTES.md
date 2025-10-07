# Usage Notes for Updated MAE Evaluation

## What Changed?

The `mae_val_epoch` and `mae_test_epoch` functions in `audio_flamingo_2/train/metrics.py` now use **autoregressive generation** instead of **teacher-forcing**.

## How to Use

### 1. Training with Validation (No changes needed)

Your existing training code will work as before:

```bash
CUDA_VISIBLE_DEVICES=0 python train_lora.py -c /path/to/config.yaml
```

The validation will now show **realistic** MAE scores that match standalone testing.

### 2. Standalone Testing

Create a test script (or use existing `inference.py` pattern):

```python
from metrics import mae_test_epoch
from train_utils import get_autocast, get_cast_dtype

# Load model and config...

mae_metrics, sample_logs = mae_test_epoch(
    model=model,  # Your loaded model
    data_config=data_config,
    clap_config=clap_config,
    tokenizer=tokenizer,
    batch_size=1,  # Can be larger, but generation is sequential per sample
    autocast=get_autocast(args.precision, cache_enabled=True),
    cast_dtype=get_cast_dtype(args.precision),
    device_id=0,
    max_samples=100  # Number of samples to evaluate
)

print(f"MAE Metrics: {mae_metrics}")
```

### 3. Expected Behavior Changes

#### Before (Teacher-Forcing)
```python
# During validation in training
Ground Truth: total: 5.5/10
vocabulary: 6.0/10
grammar: 5.0/10

Prediction: total: 5.5/10
vocabulary: 5.0/10
grammar: 5.5/10
# Well formatted, low MAE

# During standalone test
Prediction: this: 8.5/10iet: 9.5/10fl: 7.5/10
# Malformed, high MAE
```

#### After (Autoregressive Generation)
```python
# During validation in training
Ground Truth: total: 5.5/10
vocabulary: 6.0/10
grammar: 5.0/10

Prediction: total: 5.0/10
vocab: 5.5/10
grammar: 5.0/10
# May have variations, realistic MAE

# During standalone test
Prediction: total: 5.0/10
vocab: 5.5/10
grammar: 5.0/10
# Similar to validation, consistent MAE
```

## Performance Considerations

### Speed
- **Before**: Fast batch processing (~100 samples/min)
- **After**: Slower sequential generation (~20-50 samples/min depending on GPU)

**Why slower?** Generation must be done sequentially, one sample at a time, because each token depends on previous generated tokens.

### Memory
- **Before**: Higher memory usage (stores full forward pass)
- **After**: Lower memory usage (only stores generation state)

### Accuracy of Metrics
- **Before**: Unrealistically low MAE (teacher-forcing advantage)
- **After**: Realistic MAE that matches deployment performance

## Tuning Generation Parameters

You can adjust generation parameters in the code if needed:

```python
# In metrics.py, mae_val_epoch and mae_test_epoch functions
output = model.generate(
    audio_x=audio_clips[sample_idx].unsqueeze(0),
    audio_x_mask=audio_embed_mask[sample_idx].unsqueeze(0),
    lang_x=prompt.unsqueeze(0),
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=256,      # Adjust if needed
    temperature=1.0,         # Lower = more deterministic (try 0.7-1.0)
    num_beams=1,            # Increase for beam search (1-5)
    do_sample=False,        # Set to True for sampling
)
```

### Parameter Effects:
- **temperature**: 
  - `0.0-0.7`: More deterministic, consistent outputs
  - `0.8-1.0`: Balanced
  - `>1.0`: More random, diverse outputs
  
- **num_beams**:
  - `1`: Greedy decoding (fastest, used by default)
  - `3-5`: Beam search (slower, potentially better quality)
  
- **max_new_tokens**:
  - Adjust based on expected output length
  - Current: 256 tokens

## Troubleshooting

### Issue: Evaluation is too slow
**Solution 1**: Reduce `max_samples` parameter
```python
mae_metrics, _ = mae_val_epoch(..., max_samples=10)  # Faster evaluation
```

**Solution 2**: Reduce `num_beams` (if you changed it)
```python
num_beams=1  # Fastest, greedy decoding
```

### Issue: Outputs are poorly formatted
**This is expected!** The model's true generation quality is now being measured. Options:

1. **Accept it**: This is the model's real performance
2. **Improve model**: Continue training or adjust hyperparameters
3. **Adjust generation**: Try `temperature=0.8` for more consistency

### Issue: MAE scores are higher than before
**This is correct!** The old scores were artificially low due to teacher-forcing. New scores reflect true model performance.

### Issue: Results still don't match standalone testing
Check these:
1. Are you using the same checkpoint?
2. Are generation parameters identical?
3. Is the model in `.eval()` mode?
4. Are you using the same random seed?

## FAQ

**Q: Should I retrain my model?**
A: No, the model hasn't changed. Only the evaluation method is now correct.

**Q: Why are my validation MAE scores higher now?**
A: The old scores were unrealistic due to teacher-forcing. New scores are accurate.

**Q: Can I switch back to the old method?**
A: Not recommended. The old method gives misleading metrics. If needed, check git history.

**Q: Will this affect my trained model's weights?**
A: No, this only changes evaluation. Training and model weights are unchanged.

**Q: Should I use temperature=1.0 or something else?**
A: Use 1.0 for evaluation to match training conditions. Lower values (0.7-0.9) may give slightly better formatted outputs but less diverse.

## Summary

✅ **Do**: Trust the new metrics - they're accurate
✅ **Do**: Expect slower evaluation speed
✅ **Do**: Use these metrics to compare model versions
✅ **Do**: Report these metrics in papers/reports

❌ **Don't**: Compare new metrics with old metrics (they measure different things)
❌ **Don't**: Expect identical outputs to teacher-forcing evaluation
❌ **Don't**: Be alarmed by "worse" formatting - it's realistic

---

For more details, see `CHANGES_EXPLANATION.md`.
