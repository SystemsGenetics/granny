# Parameter Adjustments

This guide covers the configurable parameters available in Granny for fine-tuning analysis results.

## Segmentation Parameters

### Confidence Threshold

The confidence threshold parameter allows you to control the quality and quantity of fruit detections during segmentation analysis.

**Parameter:** `--confidence_threshold`

**Usage:**

```bash
granny -i cli --analysis segmentation --confidence_threshold 0.8 --input /path/to/images
```

**Details:**

- **Default Value:** 0.25
- **Range:** 0.0 - 1.0
- **Type:** Float
- **Required:** No (uses default if not specified)

**How it works:**

The confidence threshold filters YOLO model predictions based on how confident the model is about each detection. Only detections with confidence scores above the threshold are kept for processing.

**Recommended Values:**

| Confidence | Use Case | Expected Results |
|------------|----------|------------------|
| 0.1 - 0.3 | Maximum detection coverage | More fruits detected, may include false positives |
| 0.25 | Default balanced setting | Good balance of accuracy and coverage |
| 0.5 - 0.7 | Higher quality detections | Fewer fruits but higher accuracy |
| 0.8 - 0.95 | Only high-confidence fruits | Most reliable detections, may miss some valid fruits |
| 0.95+ | Extremely strict filtering | Only the most obvious fruit detections |

**Example Scenarios:**

**High-Quality Analysis (Fewer, More Accurate Results):**

```bash
granny -i cli --analysis segmentation --confidence_threshold 0.85 --input ./fruit_images/
```

**Maximum Coverage (More Detections, Some False Positives):**

```bash
granny -i cli --analysis segmentation --confidence_threshold 0.15 --input ./fruit_images/
```

**Testing Different Thresholds:**

You can run the same image with different confidence thresholds to find the optimal setting for your specific images:

```bash
# Conservative approach
granny -i cli --analysis segmentation --confidence_threshold 0.8 --input ./test_images/

# Default approach  
granny -i cli --analysis segmentation --confidence_threshold 0.25 --input ./test_images/

# Liberal approach
granny -i cli --analysis segmentation --confidence_threshold 0.1 --input ./test_images/
```

Compare the number of detected fruits and manually verify the results to determine the best threshold for your use case.

**Note:** This parameter only affects the segmentation analysis. Subsequent analysis steps (starch, color, etc.) will use whatever fruits were detected during segmentation.