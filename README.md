# Event Analyzer

Event Analyzer is a PyQt6 desktop app for interactively exploring large CSV time-series files, placing time dividers, selecting regions, applying thresholds, and exporting exceedance summaries.

## File Tree

```text
event_analyzer/
  analysis/
    exceedance.py          # Contiguous exceedance-event detection
    statistics.py          # Region summary statistics
  controllers/
    divider_manager.py     # Draggable vertical dividers
    plot_controller.py     # Dataset to plot coordination
    region_selector.py     # Region selection and highlight
    session.py             # JSON session save/load
    threshold_manager.py   # Draggable horizontal threshold
  data/
    csv_loader.py          # Polars lazy CSV inspection/loading
    downsample.py          # Min/max visual downsampling
    models.py              # Data classes
    time_utils.py          # Numeric and datetime time parsing
  plotting/
    exceedance_chart.py    # Matplotlib bar chart and SVG export
    time_axis_item.py      # Time-aware PyQtGraph axis
    time_series_plot.py    # Interactive PyQtGraph plot widget
  ui/
    main_window.py         # Main PyQt6 application window
    time_slider.py         # X-axis aligned slider
scripts/
  generate_sample_csv.py
tests/
  test_exceedance.py
```

## Technology Choices

The live time-series plot uses PyQtGraph because it handles pan, zoom, grids, linked axes, and large interactive line plots much faster than Matplotlib. The exceedance bar chart uses Matplotlib because its SVG output and publication-style labels are stronger for static charts.

CSV handling uses Polars. The app reads a small preview immediately, inspects schema separately, then lazily scans and collects only the selected time, target, and auxiliary columns. This is more memory-efficient than loading the entire CSV width, while still keeping selected columns in memory for fast interaction and repeated threshold analysis. Selected columns are stored as sorted NumPy arrays; the plot keeps separate downsampled display buffers so full-resolution arrays remain available for analysis and export.

The current implementation fully supports wide CSV files where each case is a separate target column. The model layer is kept separate from the UI so long-format support can be added later by transforming `case_id`, `parameter_name`, and `value` columns into the same `TimeSeriesDataset` structure.

## Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

For editable development:

```bash
pip install -e .
```

## Generate Sample Data

```bash
python scripts/generate_sample_csv.py --out sample_timeseries.csv --rows 5000
```

The sample file contains numeric time, datetime time, three target cases, two auxiliary parameters, and a non-numeric note column.

## Run

```bash
python -m time_series_threshold_app
```

Alternative local entry points:

```bash
python -m event_analyzer
python main.py
```

After `pip install -e .`:

```bash
event-analyzer
time-series-threshold-app
```

## Basic Workflow

1. Open a CSV file.
2. Choose the time column.
3. Check one or more target columns and optional auxiliary columns.
4. Click `Plot Selected`.
5. Move the slider to trace values.
6. Right-click the plot or slider to add dividers.
7. Left-click a plotted region to select it.
8. Right-click the plot or use the threshold control to add a threshold.
9. Export SVG plots, event CSV, or selected-region CSV.

## Features

- Wide-format CSV target selection.
- Future-ready model structure for long-format reshaping.
- Numeric and datetime time axes.
- Numeric validation with user-facing warnings.
- Preview table and column filtering.
- Interactive PyQtGraph plot with pan, zoom, grid, legend, tracer, and auxiliary overlays.
- Multiple auxiliary y-axes through linked PyQtGraph view boxes.
- Draggable, editable, sorted vertical dividers.
- Region selection before, between, and after dividers.
- Draggable threshold line and exceedance highlighting.
- Exceedance event detection with nonuniform time spacing and linear threshold-crossing interpolation.
- Matplotlib exceedance duration bar chart.
- SVG export for the main plot and bar chart.
- CSV export for event summaries and selected region data.
- JSON session save/load.
- Project/session save and load including selected columns, auxiliary axes, dividers, threshold, region, colors, visibility, and theme.
- Recent files menu and dark/light theme toggle.
- Target case visibility toggles and color customization.
- Unit-label extraction from column names such as `pressure [Pa]` and `temperature (C)`.
- Selected-region statistics table with min, max, mean, median, standard deviation, time above threshold, and event counts.
- Keyboard shortcuts for opening CSVs, saving SVGs, adding dividers/thresholds, resetting the view, and exporting summaries.

## Large CSV Performance Notes

- The loader uses staged background work with progress callbacks and cooperative cancellation. Preview rows are loaded first so the UI can show columns and a preview quickly.
- Selected data is held in memory as one sorted `float64` time array plus one `float64` NumPy array per selected target or auxiliary column. This avoids repeatedly reading the CSV for tracing, plotting, and threshold analysis.
- PyQtGraph receives display-only arrays generated by bucketed min/max downsampling. The reducer preserves first/last samples, local minima and maxima, selected region boundaries, and a capped set of points around threshold crossings.
- Threshold analysis uses the full-resolution selected target arrays, not the downsampled display arrays. The detector sorts the time axis once and reuses that order across cases.
- `DataManager(max_loaded_cells=...)` can enforce a configurable cell budget before selected columns are loaded. The default leaves this unset because acceptable memory use depends on the workstation and dataset.
- Cancellation is cooperative. It is checked between loader stages and between analyzed cases; a single in-progress Polars collect call may need to finish before cancellation takes effect.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

For a quick syntax/import quality check:

```bash
python -m compileall event_analyzer tests
```

The pytest suite covers CSV loading with numeric and datetime time columns, missing values, invalid column selection, target/auxiliary selection, divider sorting/editing, region selection, threshold event detection, multiple events per case, nonuniform time spacing, and export wrapper behavior.

Some tests are skipped automatically when optional runtime packages such as Polars, PyQt6, or Matplotlib are not installed. GUI testing is intentionally minimal: the automated suite focuses on non-GUI logic, because drag interactions, native file dialogs, and exact rendered SVG appearance are brittle in headless CI. Those behaviors are exercised through controller/service unit tests and should still receive manual smoke testing before releases.

To generate sample data for manual or exploratory testing:

```bash
python scripts/generate_sample_csv.py --out sample_timeseries.csv --rows 5000
```
