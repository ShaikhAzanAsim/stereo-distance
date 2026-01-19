# Global Re-ID and Visualization Improvements

## Changes Made

### 1. Improved Global Re-Identification

**Goal**: Ensure the same ship gets the same ID across all 3 cameras

#### Matching Weight Adjustments
```python
# Before
COSINE_WEIGHT = 0.7  # Appearance similarity
IOU_WEIGHT = 0.2     # Spatial overlap
CLASS_WEIGHT = 0.1   # Class consistency
MATCH_THRESHOLD = 0.85

# After
COSINE_WEIGHT = 0.8  # Increased - appearance is key for cross-camera matching
IOU_WEIGHT = 0.1     # Reduced - objects appear in different positions across cameras
CLASS_WEIGHT = 0.1   # Same
MATCH_THRESHOLD = 0.75  # Lowered - more lenient for cross-camera matching
```

**Rationale:**
- **Appearance (80%)**: Ships look similar across cameras, so appearance matching is most reliable
- **Spatial (10%)**: Objects appear in different positions in different camera views
- **Lower threshold (0.75)**: Allows more lenient matching across cameras while still maintaining accuracy

#### Track Persistence
```python
# Before
MAX_TRACK_AGE = 30  # frames
EMBEDDING_HISTORY_SIZE = 10

# After
MAX_TRACK_AGE = 60  # Doubled - maintain IDs longer
EMBEDDING_HISTORY_SIZE = 15  # Increased - more stable embeddings
```

**Benefits:**
- IDs persist longer even if object temporarily leaves frame
- More embedding history = more stable appearance representation
- Better cross-camera matching consistency

#### Cross-Camera Consistency Bonus
```python
# New feature
if len(track.camera_ids) > 1:
    score += 0.05  # Small bonus for multi-camera tracks
```

**How it works:**
- Tracks that already appear in multiple cameras get a small matching bonus
- Helps maintain the same ID when a ship appears in a new camera
- Encourages ID consistency across all 3 views

### 2. Improved Visualization

**Problem**: Distance labels were getting cut off in tall videos

**Solution**: Two-line label display

```
Before (single line):
ID:5 | vessel-ship | 125.3cm[S]  ← Gets cut off

After (two lines):
ID:5 | vessel-ship
125.3cm [S]  ← Always visible
```

#### Label Format
- **Line 1**: `ID:{global_id} | {class_name}`
- **Line 2**: `{distance}cm [{method}]`
  - `[S]` = Stereo depth (most accurate)
  - `[M]` = Monocular depth (reference-based)
  - `[ref]` = Known reference object distance

#### Visual Improvements
- Each line has its own colored background
- Better readability with proper spacing
- Distance always visible, even in tall video frames

## Expected Results

### Better Global ID Consistency
✅ Same ship should maintain same ID across all 3 cameras  
✅ Ships tracked longer (60 frames vs 30 frames)  
✅ More stable embeddings (15 history vs 10)  
✅ Cross-camera bonus helps maintain consistency  

### Better Visualization
✅ Distance labels always visible (two-line format)  
✅ Clear method indication [S] or [M]  
✅ No label cutoff in tall videos  
✅ Easier to read at a glance  

## Testing

Run the system and observe:

1. **Global ID Consistency**:
   - Watch a ship appear in multiple cameras
   - Verify it has the same ID in all views
   - Check logs for successful cross-camera matches

2. **Visualization**:
   - Verify labels are on two lines
   - Confirm distance is always visible
   - Check method tags [S] or [M]

## Configuration

If you need to tune matching further:

```python
# Make matching more strict (fewer cross-camera matches)
MATCH_THRESHOLD = 0.80

# Make matching more lenient (more cross-camera matches)
MATCH_THRESHOLD = 0.70

# Prioritize appearance even more
COSINE_WEIGHT = 0.85
IOU_WEIGHT = 0.05

# Keep tracks even longer
MAX_TRACK_AGE = 90
```

## Troubleshooting

**If ships still get different IDs across cameras:**
- Lower `MATCH_THRESHOLD` to 0.70
- Increase `COSINE_WEIGHT` to 0.85
- Check that OSNet embeddings are being extracted properly

**If too many false matches:**
- Raise `MATCH_THRESHOLD` to 0.80
- Increase `CLASS_WEIGHT` to 0.15

**If labels still cut off:**
- Reduce `DISPLAY_SCALE` to show larger frames
- Adjust `font_scale` in visualization code
