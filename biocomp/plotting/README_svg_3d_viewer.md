# Interactive SVG 3D Plot Viewer

This feature allows 3D plots from the biocompiler to be saved as SVG files with embedded metadata and viewed interactively in a web browser.

## Features

- Save 3D plots as SVG with each z-slice tagged with metadata
- Interactive HTML viewer with:
  - "Show All Slices" checkbox to display all slices at once
  - Slider control to select individual slices when checkbox is unchecked
  - Display of z-values and slice information
  - Embedded plot metadata viewer

## Usage

### 1. Creating a 3D Plot with SVG Export

When creating a 3D plot, simply specify `.svg` as the output format:

```python
from biocomp.plotutils import FigureSpec, SimpleLayout
from biocomp.plotting.plotting_3d import smooth_3d

# Create figure spec with SVG output
fig_spec = FigureSpec(
    title="My 3D Plot",
    output_dir="/path/to/output",
    output_file="my_3d_plot.svg",  # Use .svg extension
    layout=SimpleLayout(axes_size=(8, 8)),
    metadata={"description": "Interactive 3D plot"}
)

# Create and save the plot
figax = fig_spec.make_figure()
smooth_3d(
    X=X, Y=Y,
    input_names=["X", "Y", "Z"],
    output_name="Output",
    rescaler=rescaler,
    ax=[figax.ax],
    zslices=[np.array([0.2, 0.5, 0.8])],  # Your z-slices
    # ... other parameters
)
fig_spec.finalize(figax)
```

### 2. Viewing the Interactive SVG

1. Open `biocomp/biocomp/plotting/svg_viewer.html` in your web browser
2. Load your SVG file using the file selector or drag-and-drop
3. Use the controls:
   - **Show All Slices** checkbox: Toggle between showing all slices or individual selection
   - **Slider**: When checkbox is unchecked, select which z-slice to display
   - The viewer shows the current z-value and slice information

### 3. Implementation Details

The system works by:

1. **Tagging slices during plot creation**: Each 2D slice in the 3D plot is assigned a unique GID (group ID) encoding its slice index and z-value
2. **Post-processing the SVG**: The `FigureSpec._postprocess_svg()` method adds custom data attributes to tagged elements
3. **Interactive viewing**: The HTML viewer uses JavaScript to parse these attributes and control visibility

Each slice group element gets these attributes:
- `id`: `biocomp_3dslice_{index}_z{value}`
- `data-biocomp-type`: `3dslice`
- `data-z-value`: The actual z-value
- `data-slice-index`: The slice index
- `class`: `biocomp-3d-slice` (for CSS targeting)

### 4. Example

See `biocomp-devbook/demo_3d_svg_workflow.py` for a complete working example.

## Technical Notes

- The viewer uses CSS classes and JavaScript to control slice visibility
- All child elements of a slice (including images, axes, labels) are shown/hidden together
- The SVG post-processing removes problematic XML namespace prefixes for browser compatibility
- Metadata is embedded in the SVG and displayed in the viewer