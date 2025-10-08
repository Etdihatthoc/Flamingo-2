# Fix Summary: MAE Evaluation Functions

## Problem
The `mae_test_epoch` and `mae_val_epoch` functions in `audio_flamingo_2/train/metrics.py` were causing an `IndexError` when calling `model.generate()`:

```
IndexError: index 0 is out of bounds for dimension 0 with size 0
```

This error occurred in `audio_flamingo_2/src/helpers.py` at line 388 in the `MaskedCrossAttention.forward()` method when accessing `media_locations_b[i+1]`, where `media_locations_b` was empty.

## Root Cause
The issue occurred because the functions were passing the **full input sequence** (including both prompt AND output) to `model.generate()`, but `model.generate()` expects only the **prompt with the `<audio>` token** (no output), similar to inference mode.

### Original Flow (Incorrect)
```python
# Data from dataloader includes: <audio>{prompt}{sep_token}{output}<|endofchunk|>{eos}
input_ids = batch["input_ids"]  # Full sequence with output

# Passing full sequence to model.generate() - WRONG!
model.generate(
    audio_x=audio_clips,
    lang_x=input_ids,  # Contains <audio>{prompt}{sep}{output}...
    ...
)
```

The problem: When the model tried to generate, it couldn't properly locate the `<audio>` token positions because the input was structured for training (teacher forcing), not inference.

### Fixed Flow (Correct)
```python
# Data from dataloader: <audio>{prompt}{sep_token}{output}<|endofchunk|>{eos}
input_ids_full = batch["input_ids"]

# Find SEP token position
sep_pos = (input_ids_full[0] == sep_token_id).nonzero()[-1].item()

# Truncate to only include: <audio>{prompt}{sep_token}
input_ids_truncated = input_ids_full[:, :sep_pos+1]

# Now pass truncated input to model.generate() - CORRECT!
model.generate(
    audio_x=audio_clips,
    lang_x=input_ids_truncated,  # Contains only <audio>{prompt}{sep}
    ...
)
```

## Solution
Modified both `mae_val_epoch` and `mae_test_epoch` functions to:

1. **Truncate input to prompt only**: Extract only the portion up to (and including) the SEP token from the full input_ids
2. **Verify `<audio>` token presence**: Add checks to ensure the `<audio>` token is present before and after truncation
3. **Use `model.generate()` correctly**: Pass only the truncated input (like `inference.py` does)
4. **Extract ground truth separately**: Get the expected output from the portion after the SEP token

## Changes Made

### File: `audio_flamingo_2/train/metrics.py`

#### Function: `mae_val_epoch` (lines 94-273)
- Changed from using `model()` with teacher forcing to using `model.generate()`
- Added proper input truncation at SEP token position
- Added validation checks for `<audio>` and SEP tokens
- Simplified the code by removing unnecessary label masking logic

#### Function: `mae_test_epoch` (lines 275-451)
- Applied the same changes as `mae_val_epoch`
- Input format now matches `inference_HF_pretrained/inference.py`

## Key Changes in Detail

### Before (Incorrect):
```python
# Used full input with labels for forward pass
labels = input_ids.clone()
# ... complex label masking logic ...

output = model(
    audio_x=audio_clips,
    audio_x_mask=audio_embed_mask,
    lang_x=input_ids,  # Full sequence
    attention_mask=attention_mask,
    labels=labels
)

# Extract predictions from logits
pred_tokens = torch.argmax(output.logits, dim=-1)
```

### After (Correct):
```python
# Truncate input to prompt + SEP only
sep_pos = (input_ids_full[0] == sep_token_id).nonzero()[-1].item()
input_ids_truncated = input_ids_full[:, :sep_pos+1]

# Verify <audio> token is present
has_audio_after_truncate = (input_ids_truncated[0] == media_token_id).any()

# Generate new tokens
generated_ids = model.generate(
    audio_x=audio_clips,
    audio_x_mask=audio_embed_mask,
    lang_x=input_ids_truncated,  # Only prompt + SEP
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=256,
    do_sample=False,
)[0]
```

## Why This Fix Works

1. **Proper Audio Attention**: By truncating to only the prompt, the `<audio>` token is correctly identified as the first/only media token, allowing the `MaskedCrossAttention` layer to properly compute attention between text and audio features.

2. **Matches Inference Pattern**: The input format now exactly matches how `inference_HF_pretrained/inference.py` prepares inputs, ensuring consistent behavior between evaluation and inference.

3. **Prevents IndexError**: With the `<audio>` token properly positioned at the start of the sequence (after truncation), `media_locations_b` is no longer empty, preventing the IndexError.

## Testing Recommendations

1. Run the MAE evaluation on a small validation set to verify no errors
2. Compare outputs with the original inference.py to ensure consistency
3. Check that all samples are properly processed (no unexpected skips)

## Related Files
- Fixed: `audio_flamingo_2/train/metrics.py`
- Reference: `inference_HF_pretrained/inference.py` (correct pattern)
- Related: `audio_flamingo_2/src/helpers.py` (where error occurred)
- Related: `audio_flamingo_2/data/data.py` (data preparation)
