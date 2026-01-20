# Gompertz Simulation Comparison Tool

This document describes the new academic-style comparison visualization tool for the Gompertz tree-diameter simulator.

## Files Created

1. **gompertz_sim.py** - Standalone Python script for running the comparison
2. **gompertz_sim.ipynb** - Jupyter notebook with the same functionality
3. **output/gompertz_comparison_academic.png** - High-resolution PNG output
4. **output/gompertz_comparison_academic.pdf** - Vector PDF output for publications

## New Academic Visualization Style

The new visualization matches the reference image style with the following features:

- **White background** with black axes
- **Serif fonts** resembling Computer Modern/LaTeX style
- **Inward-facing tick marks** on all four sides of the plot
- **Clean colored lines** without markers (cyan for Xkminus2, orange for Xkminus1)
- **Boxed legend** with rectangular border in the top-right corner
- **Dashed reference line** showing the asymptotic diameter D

## Usage

### Standalone Python Script

```bash
python gompertz_sim.py
```

This will:
1. Run both Gompertz simulation versions with default parameters
2. Generate a comparison plot
3. Save outputs to the `output/` directory

### Jupyter Notebook

Open `gompertz_sim.ipynb` in Jupyter or JupyterLab and run all cells.

## Default Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| r | 0.3 | Growth rate parameter |
| D | 100.0 | Asymptotic diameter (cm) |
| λ | 0.7 | Noise rate parameter |
| years | 50 | Number of years to simulate |
| X₀ | 15.0 | Initial diameter |
| X₁ | 17.0 | Second year diameter (Xkminus2 only) |

## Output Files

Both PNG (300 DPI) and PDF formats are generated:
- `output/gompertz_comparison_academic.png`
- `output/gompertz_comparison_academic.pdf`

## Comparison of Models

The comparison plot shows two stochastic Gompertz trajectories:

1. **Cyan line ($X^{(2)}_k$)**: Version with $X_{k-2}$ term - uses the average of the two previous years
2. **Orange line ($X^{(1)}_k$)**: Version with $X_{k-1}$ term only - uses only the previous year
3. **Gray dashed line**: Asymptotic diameter $D = 100$ cm

The visualization highlights how the two different model formulations produce different growth trajectories over time.
