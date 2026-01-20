#!/usr/bin/env python3
"""
Gompertz Tree-Diameter Simulation Comparison

This script implements and compares two versions of the stochastic Gompertz model
for simulating tree diameter growth over time, with a publication-quality academic
visualization style matching LaTeX/PGFPlots appearance.

Models:
    Version 1 (X_{k-2}): X_k = D^(1-exp(-r)) * [(X_{k-1}+X_{k-2})/2]^(exp(-r)) * ε_{k-1}
    Version 2 (X_{k-1}): X_k = D^(1-exp(-r)) * X_{k-1}^(exp(-r)) * ε_{k-1}

Features:
    - Academic LaTeX-style visualization
    - Inward-facing tick marks on all four sides
    - Clean colored lines without markers
    - Boxed legend in top-right corner
    - Outputs both PNG and PDF formats
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def configure_academic_style():
    """Configure matplotlib for academic LaTeX-style plotting."""
    plt.rcParams.update({
        # Font settings for LaTeX-like appearance
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        
        # Figure settings
        'figure.figsize': (8, 5.5),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        
        # Line settings
        'lines.linewidth': 2.0,
        'lines.markersize': 0,  # No markers for academic style
        
        # Grid settings
        'grid.alpha': 0.3,
        
        # Mathtext settings
        'mathtext.fontset': 'stix',
    })


def simulate_gompertz_two_step(r, D, lam, n_years, X0=15.0, X1=17.0):
    """
    Simulate the Gompertz process with two-step dependency.
    
    Parameters
    ----------
    r : float
        Growth rate parameter.
    D : float
        Asymptotic diameter (maximum achievable diameter).
    lam : float
        Rate parameter for the exponential noise distribution.
    n_years : int
        Number of years to simulate.
    X0 : float, optional
        Initial diameter at year 0 (default: 15.0 cm).
    X1 : float, optional
        Diameter at year 1 (default: 17.0 cm).
    
    Returns
    -------
    np.ndarray
        Array of simulated diameters X_k for k = 0, 1, ..., n_years.
    """
    # Initialize the diameter array
    X = np.zeros(n_years + 1)
    X[0] = X0
    X[1] = X1
    
    # Pre-compute constant terms for efficiency
    exponent = np.exp(-r)
    constant_term = D ** (1 - exponent)
    
    # Generate all exponential noise variables at once
    epsilon = np.random.exponential(scale=1.0 / lam, size=n_years)
    
    # Simulate the process for k >= 2
    for k in range(2, n_years + 1):
        avg_previous = (X[k - 1] + X[k - 2]) / 2.0
        X[k] = constant_term * (avg_previous ** exponent) * epsilon[k - 1]
    
    return X


def simulate_gompertz_one_step(r, D, lam, n_years, X0=15.0):
    """
    Simulate the Gompertz process with one-step dependency.
    
    Parameters
    ----------
    r : float
        Growth rate parameter.
    D : float
        Asymptotic diameter (maximum achievable diameter).
    lam : float
        Rate parameter for the exponential noise distribution.
    n_years : int
        Number of years to simulate.
    X0 : float, optional
        Initial diameter at year 0 (default: 15.0 cm).
    
    Returns
    -------
    np.ndarray
        Array of simulated diameters X_k for k = 0, 1, ..., n_years.
    """
    # Initialize the diameter array
    X = np.zeros(n_years + 1)
    X[0] = X0
    
    # Pre-compute constant terms for efficiency
    exponent = np.exp(-r)
    constant_term = D ** (1 - exponent)
    
    # Generate all exponential noise variables at once
    epsilon = np.random.exponential(scale=1.0 / lam, size=n_years)
    
    # Simulate the process for k >= 1
    for k in range(1, n_years + 1):
        X[k] = constant_term * (X[k - 1] ** exponent) * epsilon[k - 1]
    
    return X


def create_academic_comparison_plot(X_two_step, X_one_step, params, output_dir="output"):
    """
    Create a publication-quality comparison plot with academic LaTeX style.
    
    Features:
        - Inward-facing tick marks on all four sides
        - LaTeX-style serif fonts
        - Clean colored lines without markers
        - Boxed legend in top-right corner
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    # Generate year indices
    years = np.arange(len(X_two_step))
    
    # Plot trajectories with academic style (no markers)
    line1, = ax.plot(years, X_two_step, color='#17becf', linewidth=2.0,
                     label=r'$X^{(2)}_k$ (with $X_{k-2}$ term)')
    line2, = ax.plot(years, X_one_step, color='#ff7f0e', linewidth=2.0,
                     label=r'$X^{(1)}_k$ (with $X_{k-1}$ term only)')
    
    # Add reference line for asymptotic diameter D
    ax.axhline(y=params['D'], color='#7f7f7f', linestyle='--', 
               linewidth=1.5, alpha=0.8,
               label=r'$D = ' + f'{params["D"]:.0f}' + r'$ cm')
    
    # Set inward-facing tick marks on all four sides
    ax.tick_params(axis='both', which='major', direction='in', 
                   length=6, width=1.0)
    ax.tick_params(axis='both', which='minor', direction='in',
                   length=3, width=0.8)
    
    # Set axis labels with LaTeX-style formatting
    ax.set_xlabel(r'$k$ (Year)', fontsize=13)
    ax.set_ylabel(r'$X_k$ (cm)', fontsize=13)
    
    # Set title
    ax.set_title(r'Gompertz Stochastic Process Comparison' + '\n' +
                 r'$r = ' + f'{params["r"]}' + r'$, $\lambda = ' + 
                 f'{params["lambda"]}' + r'$',
                 fontsize=14, pad=10)
    
    # Configure legend with rectangular box (sharp corners)
    legend = ax.legend(loc='upper right', frameon=True, 
                       fancybox=False, edgecolor='black',
                       framealpha=1.0, borderpad=0.8,
                       labelspacing=0.5)
    
    # Set legend box properties for sharp rectangular appearance
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_edgecolor('black')
    
    # Add light grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set axis limits with some padding
    ax.set_xlim(-1, len(years) + 1)
    y_min = min(X_two_step.min(), X_one_step.min()) * 0.95
    y_max = max(X_two_step.max(), X_one_step.max()) * 1.05
    ax.set_ylim(max(0, y_min), y_max)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save as PNG
    png_filename = os.path.join(output_dir, "gompertz_comparison_academic.png")
    plt.savefig(png_filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"PNG saved: {png_filename}")
    
    # Save as PDF
    pdf_filename = os.path.join(output_dir, "gompertz_comparison_academic.pdf")
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"PDF saved: {pdf_filename}")
    
    return fig, ax


def print_summary(X_two_step, X_one_step, params):
    """Print simulation summary to console."""
    print("=" * 60)
    print("GOMPERTZ SIMULATION SUMMARY")
    print("=" * 60)
    
    print("\nSimulation Parameters:")
    print(f"  Growth rate (r):         {params['r']}")
    print(f"  Asymptotic diameter (D): {params['D']} cm")
    print(f"  Noise rate (λ):          {params['lambda']}")
    print(f"  Number of years:         {params['years']}")
    
    print("\nInitial Conditions:")
    print(f"  X₀: {params['X0']} cm (both versions)")
    print(f"  X₁: {params['X1']} cm (Xkminus2 version only)")
    
    print("\nFinal Diameter Values:")
    print(f"  Xkminus2 version: {X_two_step[-1]:.4f} cm")
    print(f"  Xkminus1 version: {X_one_step[-1]:.4f} cm")
    print(f"  Asymptote D:      {params['D']} cm")
    
    print("\nOutput Files:")
    print(f"  PNG: output/gompertz_comparison_academic.png")
    print(f"  PDF: output/gompertz_comparison_academic.pdf")
    
    print("=" * 60)


def main():
    """Main function to run the Gompertz simulation comparison."""
    # Configure academic style
    configure_academic_style()
    
    # Ensure output directory exists
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define default simulation parameters
    params = {
        'r': 0.3,       # Growth rate parameter
        'D': 100.0,     # Asymptotic diameter (cm)
        'lambda': 0.7,  # Rate parameter for exponential noise
        'years': 50,    # Number of years to simulate
        'X0': 15.0,     # Initial diameter
        'X1': 17.0      # Second year diameter (for Xkminus2 version)
    }
    
    print("Running simulations...")
    print(f"Parameters: r = {params['r']}, D = {params['D']}, λ = {params['lambda']}, years = {params['years']}")
    
    # Run simulations
    X_two_step = simulate_gompertz_two_step(
        r=params['r'],
        D=params['D'],
        lam=params['lambda'],
        n_years=params['years'],
        X0=params['X0'],
        X1=params['X1']
    )
    
    X_one_step = simulate_gompertz_one_step(
        r=params['r'],
        D=params['D'],
        lam=params['lambda'],
        n_years=params['years'],
        X0=params['X0']
    )
    
    print("Simulations complete.")
    
    # Create and save the comparison plot
    create_academic_comparison_plot(X_two_step, X_one_step, params, output_dir)
    
    # Print summary
    print_summary(X_two_step, X_one_step, params)


if __name__ == "__main__":
    main()
