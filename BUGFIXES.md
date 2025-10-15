# Bug Fixes Documentation

This document tracks bugs found and fixed in the Granny dev branch.

## Bug #1: FileDirValue validates before creating directory

**Issue:** FileDirValue.setValue() validates that a directory exists before creating it, causing failure when setting output directory paths.

**Location:** `Granny/Models/Values/FileDirValue.py:28-31`

**Problem Code:**
```python
self.value = value
if not self.validate():
    raise ValueError("Not a directory. Please specify a directory.")
os.makedirs(self.value, exist_ok=True)
```

**Root Cause:** The validation happens before directory creation, but validate() checks if directory exists.

**Fix:** Create directory first, then validate.

**Impact:** Prevents segmentation analysis from running - crashes during initialization.

**Status:** FIXED

**Fix Applied:**
```python
# Create directory first, then validate
os.makedirs(self.value, exist_ok=True)
if not self.validate():
    raise ValueError("Not a directory. Please specify a directory.")
```

---

## Bug #2: FileNameValue circular validation logic

**Issue:** FileNameValue.setValue() has flawed logic that tries to validate self.value before it's set to the new value.

**Location:** `Granny/Models/Values/FileNameValue.py:23`

**Problem Code:**
```python
self.value = value if self.validate() else None
```

**Root Cause:** validate() checks self.value (old value) instead of the new value parameter.

**Fix:** Set value first, allow model names that aren't files yet.

**Impact:** Can cause TypeError when model names are used instead of file paths.

**Status:** FIXED

**Fix Applied:**
```python
def setValue(self, value: str):
    self.value = value
    self.is_set = True

def validate(self) -> bool:
    return self.value is not None and (os.path.isfile(self.value) or bool(self.value.strip()))
```

---

## Testing Strategy

1. **Bug #1 Test:** Run segmentation with output directory creation
2. **Bug #2 Test:** Use model names vs file paths for segmentation
3. **Integration Test:** Full segmentation pipeline with apple tray demo image

Expected result: `granny -i cli --analysis segmentation --model pome_fruit-v1_0 --input demo/granny_smith_images/apple_tray` should work without manual directory creation.

## Test Results

**All tests passed!**

**Integration Test Result:**
```bash
$ granny -i cli --analysis segmentation --model pome_fruit-v1_0 --input demo/granny_smith_images/apple_tray

	model                    : (user) pome_fruit-v1_0
	input                    : (user) demo/granny_smith_images/apple_tray

0: 704x1024 18 fruits, 25.6ms
Speed: 5.8ms preprocess, 25.6ms inference, 90.8ms postprocess per image at shape (1, 3, 704, 1024)
```

- Automatic directory creation works
- Model name resolution works  
- Segmentation detects 18 fruits successfully
- No manual directory creation required