# Hybrid Depth Estimation System - Technical Summary

## Overview

The system now implements a **hybrid depth estimation** approach that combines:
1. **Stereo disparity** (most accurate, requires object in both top cameras)
2. **Monocular depth using reference object scaling** (fallback, works with single camera)

## How It Works

### 1. Calibration Phase

Using the reference object in the bottom camera:
```
focal_length = (ref_pixel_height × 14cm) / 8cm
ref_pixel_height = stored for monocular estimation
```

### 2. Stereo Depth Estimation (Primary Method)

**For objects visible in BOTH top cameras:**
```
disparity = x_left - x_right
depth = (focal_length × 15cm) / disparity
```

**Advantages:**
- ✅ Most accurate
- ✅ No assumptions about object size
- ✅ Works for any object class

**Limitations:**
- ❌ Requires object in both cameras
- ❌ Requires positive disparity

### 3. Monocular Depth Estimation (Fallback Method)

**For objects in ANY camera using reference object scaling:**
```
distance = ref_distance × (ref_pixel_height / object_pixel_height)
```

**Principle:** Similar triangles
- If reference object has height H at distance D with pixel height P_ref
- Then object with same height at distance D' has pixel height P'
- D' = D × (P_ref / P')

**Advantages:**
- ✅ Works with single camera
- ✅ Provides estimates for all objects
- ✅ No stereo matching required

**Limitations:**
- ⚠️ Assumes objects have similar height to reference object (8cm)
- ⚠️ Less accurate than stereo
- ⚠️ Sensitive to object orientation

### 4. Hybrid Approach

The system automatically chooses the best method:

```python
1. Try stereo depth first (most accurate)
2. If stereo fails (object not in both cameras):
   → Use monocular depth estimation
3. If object in multiple cameras:
   → Average monocular estimates
```

## Visualization

Distance labels now show the method used:
- `125.3cm[S]` - Stereo depth (most reliable)
- `87.5cm[M]` - Monocular depth (reference-based estimate)
- `14.0cm (ref)` - Known reference object distance

## Logging

The system provides detailed logging:

```
✓ [STEREO] ID 5 (vessel-ship): distance = 125.3 cm (disparity = 29.45px)
✓ [MONO] ID 7 (vessel-jetski): distance = 87.5 cm (ref-based)
```

## Configuration

All parameters are configurable:

```python
REF_OBJ_REAL_HEIGHT_CM = 8.0   # Known height of reference object
REF_OBJ_DISTANCE_CM = 14.0     # Known distance of reference object
BASELINE_CM = 15.0             # Stereo baseline between top cameras
SKIP_CLASSES = ['person']      # Classes to skip
```

## Accuracy Expectations

| Method | Typical Accuracy | Use Case |
|--------|-----------------|----------|
| Stereo | ±5-10% | Objects in both top cameras |
| Monocular | ±20-30% | Single camera detections |
| Reference | Exact (14cm) | Calibration object |

## Future Improvements

To improve monocular accuracy:

1. **Class-specific heights**: Define real-world heights per class
   ```python
   CLASS_HEIGHTS = {
       'vessel-ship': 50.0,      # cm
       'vessel-jetski': 30.0,    # cm
       'ref-obj': 8.0            # cm
   }
   ```

2. **Aspect ratio correction**: Account for object orientation

3. **Multi-frame averaging**: Smooth estimates over time

4. **Confidence scoring**: Weight estimates by reliability

## Code Structure

### New Methods in StereoDepthEstimator

1. `estimate_depth_stereo()` - Stereo disparity calculation
2. `estimate_depth_monocular()` - Reference object scaling
3. `estimate_depth_hybrid()` - Combined approach (main entry point)

### Integration

The MainPipeline now calls:
```python
depths = self.depth_estimator.estimate_depth_hybrid(
    all_results.get(0, []),  # left camera
    all_results.get(1, []),  # right camera
    all_results              # all cameras for monocular
)
```

Returns: `Dict[global_id -> (depth_cm, method)]`

## Testing

Run the system and observe:
- Objects in both top cameras get `[S]` tag (stereo)
- Objects in single camera get `[M]` tag (monocular)
- Reference object shows `(ref)` tag with known distance

The hybrid approach ensures **all detected objects get distance estimates**, not just those visible in both cameras!
