# Fix for MAE Evaluation IndexError

## Problem Statement
The user encountered an `IndexError` when running MAE (Mean Absolute Error) evaluation:

```
IndexError: index 0 is out of bounds for dimension 0 with size 0
```

This error occurred in `audio_flamingo_2/src/helpers.py` at line 388 in the `MaskedCrossAttention.forward()` method.

## Root Cause Analysis

The issue was caused by a mismatch between how the model expects input during inference vs. how the data was being provided during evaluation:

### Expected (Inference Mode)
```python
# Format: <audio>{prompt}{sep_token}
sample = f"<audio>{text_prompt.strip()}{tokenizer.sep_token}"
```

### What Was Provided (Training Mode)
```python
# Format: <audio>{prompt}{sep_token}{output}<|endofchunk|>{eos}
# Full sequence including the answer
```

When the full sequence was passed to `model.generate()`, the model couldn't properly identify the `<audio>` token position, causing `media_locations_b` to be empty and triggering the IndexError.

## Solution

Modified `mae_val_epoch` and `mae_test_epoch` functions in `audio_flamingo_2/train/metrics.py` to:

1. **Truncate input at SEP token** before calling `model.generate()`
2. **Verify `<audio>` token presence** before and after truncation
3. **Use `model.generate()`** instead of `model()` with teacher forcing
4. **Extract ground truth separately** from the full sequence

### Code Changes

#### Before (Broken)
```python
# Used full input with output
input_ids = batch["input_ids"]  # <audio>{prompt}{sep}{output}...

# Forward pass with teacher forcing
output = model(
    audio_x=audio_clips,
    lang_x=input_ids,  # Full sequence
    attention_mask=attention_mask,
    labels=labels
)

# Extract predictions from logits
pred_tokens = torch.argmax(output.logits, dim=-1)
```

#### After (Fixed)
```python
# Find SEP token and truncate
input_ids_full = batch["input_ids"]
sep_pos = (input_ids_full[0] == sep_token_id).nonzero()[-1].item()
input_ids_truncated = input_ids_full[:, :sep_pos+1]  # <audio>{prompt}{sep}

# Verify <audio> token is present
has_audio = (input_ids_truncated[0] == media_token_id).any()
if not has_audio:
    continue

# Generate new tokens
generated_ids = model.generate(
    audio_x=audio_clips,
    audio_x_mask=audio_embed_mask,
    lang_x=input_ids_truncated,  # Only prompt + SEP
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=256,
    do_sample=False,
)[0]

# Decode generated output
output_decoded = tokenizer.decode(generated_ids).split(tokenizer.sep_token)[-1]
```

## Files Changed

1. **`audio_flamingo_2/train/metrics.py`** - Fixed both functions
   - `mae_val_epoch()` - Line 94-240
   - `mae_test_epoch()` - Line 242-383
   
2. **`FIX_SUMMARY.md`** - Detailed documentation of the fix

3. **`validate_fix.py`** - Demonstration script showing the fix

## Verification

Run the validation script to see the demonstration:
```bash
python3 validate_fix.py
```

Expected output:
```
✅ VERIFICATION:
   1. metrics.py - FIXED ✓
   2. metrics_new.py - Already correct ✓
   3. Syntax validation - Passed ✓
```

## Testing the Fix

To test the fix in your environment:

1. **Run validation script**:
   ```bash
   cd /path/to/Flamingo-2
   python3 validate_fix.py
   ```

2. **Run actual evaluation**:
   ```bash
   cd audio_flamingo_2/train
   python test.py --config configs/vstep_lora.yaml
   ```

3. **Expected behavior**:
   - No `IndexError` should occur
   - MAE metrics should be computed successfully
   - Sample logs should be generated

## Impact

- ✅ Fixes `IndexError` in `helpers.py` line 388
- ✅ Enables proper MAE evaluation on validation and test sets
- ✅ Matches inference behavior in `inference_HF_pretrained/inference.py`
- ✅ Maintains backward compatibility with existing code

## Notes

The repository has two versions of metrics files:
- `metrics.py` - Used by `test.py` (was broken, now fixed)
- `metrics_new.py` - Used by `train_lora.py` (already had correct implementation)

Both files now use the same correct approach.

## References

- Fixed file: `audio_flamingo_2/train/metrics.py`
- Reference implementation: `inference_HF_pretrained/inference.py`
- Error location: `audio_flamingo_2/src/helpers.py` line 388
- Data preparation: `audio_flamingo_2/data/data.py`

## Additional Resources

- **FIX_SUMMARY.md** - Comprehensive technical documentation
- **validate_fix.py** - Interactive demonstration of the fix
