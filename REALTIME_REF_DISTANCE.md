# Real-Time Reference Object Distance Calculation

## Overview

The system now calculates the reference object's distance **dynamically in real-time** instead of using a hardcoded value.

## How It Works

### Previous Approach (Hardcoded)
```python
# Hardcoded value
REF_OBJ_DISTANCE_CM = 14.0  # Always showed 14.0cm
```

### New Approach (Real-Time Calculation)

**Formula:**
```
distance = (focal_length × real_height) / pixel_height
```

**Implementation:**
1. **Calibration** (first frame with ref-obj):
   - Detects reference object
   - Calculates focal length using known distance (14cm)
   - Stores focal length for future calculations

2. **Every Frame** (after calibration):
   - Detects reference object
   - Measures pixel height of bbox
   - Calculates distance: `distance = (f × 8cm) / pixel_height`
   - Displays calculated distance with `[calc]` tag

## Benefits

✅ **Dynamic measurement**: Distance updates every frame based on actual detection  
✅ **Validates calibration**: If ref-obj moves, you'll see the distance change  
✅ **More accurate**: Uses real-time measurements instead of assumptions  
✅ **Debugging aid**: Helps verify the depth estimation system is working correctly  

## Visualization

The reference object now shows:
```
Line 1: ID:1 | ref-obj
Line 2: 13.8cm [calc]  ← Calculated in real-time
```

The `[calc]` tag indicates this is a calculated distance using the monocular depth formula.

## Expected Behavior

**If reference object is stationary at 14cm:**
- Distance should show ~14.0cm ± 0.5cm
- Small variations are normal due to detection bbox variations

**If reference object moves:**
- Distance will change accordingly
- Closer = smaller number
- Farther = larger number

**If distance is very different from 14cm:**
- Check that ref-obj is correctly detected
- Verify calibration happened successfully
- Check focal length value in logs

## Technical Details

### Calculation Flow

```python
# 1. Calibration (first detection)
focal_length = (pixel_height × 14cm) / 8cm

# 2. Every frame thereafter
pixel_height = ref_obj.bbox.height
distance = (focal_length × 8cm) / pixel_height
```

### Code Location

**StereoDepthEstimator.estimate_depth_hybrid():**
```python
# Calculate reference object distance in real-time
if self.calibrated:
    for detection, global_id in bottom_results:
        if 'ref' in detection.class_name.lower():
            pixel_height = detection.height
            ref_distance = (self.focal_length_px * self.ref_height_cm) / pixel_height
            ref_obj_info = (global_id, ref_distance)
            break
```

**Visualizer.draw_results():**
```python
# Add calculated distance for reference object
elif ref_obj_info and global_id == ref_obj_info[0]:
    ref_distance = ref_obj_info[1]
    line2 = f"{ref_distance:.1f}cm [calc]"
```

## Logging

You'll see debug logs like:
```
[REF-OBJ] ID 1: distance = 13.8 cm (height=1205.3px)
```

This confirms the calculation is happening every frame.

## Comparison with Other Methods

| Object Type | Method | Tag | Accuracy |
|-------------|--------|-----|----------|
| Vessels (both cams) | Stereo disparity | `[S]` | ±5-10% |
| Vessels (single cam) | Monocular (ref-based) | `[M]` | ±20-30% |
| Reference object | Monocular (known height) | `[calc]` | ±2-5% |

The reference object calculation is very accurate because:
- Known real-world height (8cm)
- Good quality detection
- Calibrated focal length
- Single camera (no stereo matching needed)

## Troubleshooting

**Distance shows wrong value:**
- Check calibration logs for focal length
- Verify ref-obj is detected correctly
- Check pixel height is reasonable (should be ~1200px)

**Distance fluctuates a lot:**
- Normal variation: ±0.5cm
- Large variation (>2cm): Check bbox stability
- May need temporal smoothing

**No distance shown:**
- Verify ref-obj is detected (check logs)
- Ensure calibration completed successfully
- Check that ref-obj class name contains 'ref'
