# Hardcoded Focal Length Configuration

## Change Summary

The system now uses a **hardcoded focal length of 3060px** for all cameras instead of calculating it from the reference object.

## Configuration

```python
# Camera parameters
FOCAL_LENGTH_PX = 3060.0  # Hardcoded focal length for all cameras
```

## What Changed

### Before (Calculated)
```python
# Calibration calculated focal length
focal_length = (pixel_height × 14cm) / 8cm
```

### After (Hardcoded)
```python
# Direct assignment
focal_length = 3060.0  # Fixed value
```

## Benefits

✅ **Consistent**: Same focal length used for all cameras  
✅ **Faster**: No calibration calculation needed  
✅ **Predictable**: Known camera parameters  
✅ **Simpler**: Less code complexity  

## How It Works Now

1. **Initialization**: Focal length set to 3060px immediately
2. **Reference object detection**: Only used to set `ref_pixel_height` for monocular estimation
3. **Depth calculation**: Uses hardcoded 3060px for all calculations

## Depth Formulas

### Stereo Depth
```
depth = (3060px × 15cm) / disparity
```

### Monocular Depth  
```
distance = (3060px × 8cm) / pixel_height
```

### Reference Object Distance
```
distance = (3060px × 8cm) / pixel_height
```

## Log Output

You'll see:
```
Depth estimator initialized (baseline=15.0cm, focal_length=3060.0px)
Reference object detected: height=1205.7px (focal_length=3060.0px)
```

## To Change Focal Length

Simply update the configuration:
```python
FOCAL_LENGTH_PX = 3060.0  # Change this value
```

All depth calculations will automatically use the new value.
