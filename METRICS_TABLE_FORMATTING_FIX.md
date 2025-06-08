# Metrics Table Formatting Fix Summary

## Issue
The `metrics_summary_table.png` in the performance folder had inconsistent formatting compared to the more polished `sector_metrics_summary_table.png` in the sectors folder.

## Key Differences Identified

### Before (Performance Table):
- Gray header (#f2f2f2)
- No alternating row colors (all white)
- Smaller row height (scale 1.3)
- Title row included
- Different column width distribution

### After (Matching Sector Table):
- Blue header (#6788d8) with white text
- Alternating row colors (#f0f0f0 and white)
- Taller rows for better readability (scale 2.5)
- No title (cleaner look)
- Model column gets 55% width, metrics share 45%

## Changes Applied

### 1. Color Scheme
```python
# Header color
header_color = '#6788d8'  # Blue header matching sector table

# Alternating row colors
for i in range(len(table_data)):
    if i % 2 == 0:
        row_colors = ['#f0f0f0'] * len(table_data.columns)  # Light gray
    else:
        row_colors = ['white'] * len(table_data.columns)
```

### 2. Column Widths
```python
# Model column gets 55% of width (same as sector table)
model_width = 0.55
other_width = 0.45 / (total_cols - 1)
```

### 3. Table Styling
```python
# Font size and row height matching sector table
table.set_fontsize(10)
table.scale(1.0, 2.5)  # Much taller rows

# White bold text on blue header
cell.set_text_props(weight='bold', color='white')

# Black borders with proper line width
cell.set_edgecolor('black')
cell.set_linewidth(0.5)
```

### 4. Layout
- Removed title row for cleaner appearance
- Used `plt.subplots_adjust()` instead of `tight_layout()`
- Left-aligned Model column text

## Result
The performance metrics table now has:
- Professional blue header with white text
- Alternating gray/white rows for easy reading
- Properly sized rows with good spacing
- Clear black borders
- Consistent formatting with sector tables

## Pipeline Integration
The formatting is automatically applied when running:
```bash
python main.py --visualize
python main.py --all
```

The updated code is in `src/visualization/plots/metrics.py` in the `MetricsTable` class.