#!/usr/bin/env python3
"""
MAD4037 - Stochastic Process TP
Gompertz Tree-Diameter Simulator (Version with X_{k-2} term)

This script simulates the stochastic Gompertz model for tree diameter growth
using the recurrence relation that depends on the average of the two previous
values.

Model:
    X_k = D^(1-exp(-r)) * [(X_{k-1}+X_{k-2})/2]^(exp(-r)) * ε_{k-1}, k ≥ 2
    ε_k ~ Exponential(λ) i.i.d.

Initial conditions: (X0, X1) = (15, 17) cm (fixed)
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import sys

# Ensure the output directory exists
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def simulate_gompertz_two_step(
    r: float, D: float, lam: float, n_years: int, X0: float = 15.0, X1: float = 17.0
) -> np.ndarray:
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

    # Generate all exponential noise variables at once for efficiency
    # np.random.exponential(scale) uses mean = scale, so we use scale = 1/lambda
    epsilon = np.random.exponential(scale=1.0 / lam, size=n_years)

    # Simulate the process for k ≥ 2
    for k in range(2, n_years + 1):
        # Average of the two previous values
        avg_previous = (X[k - 1] + X[k - 2]) / 2.0

        # Apply the Gompertz recurrence relation
        X[k] = constant_term * (avg_previous**exponent) * epsilon[k - 1]

    return X


def plot_diameter_trajectory(
    X: np.ndarray, params: Dict[str, Any], version: str
) -> None:
    """
    Create and display the diameter trajectory plot.

    Parameters
    ----------
    X : np.ndarray
        Array of diameter values.
    params : dict
        Dictionary containing r, D, lambda parameters.
    version : str
        Version identifier for the plot title and filename.
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate year indices
    years = np.arange(len(X))

    # Plot the trajectory
    ax.plot(years, X, "b-o", markersize=4, linewidth=1.5, label="Diameter trajectory")

    # Add reference line for asymptotic diameter D
    ax.axhline(
        y=params["D"],
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f'Asymptote D = {params["D"]} cm',
    )

    # Customize the plot
    ax.set_xlabel("Year k", fontsize=12)
    ax.set_ylabel("Diameter X_k (cm)", fontsize=12)
    ax.set_title(
        f"Gompertz Stochastic Process - {version}\n"
        f'r = {params["r"]}, D = {params["D"]}, λ = {params["lambda"]}',
        fontsize=14,
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Generate filename with parameters
    filename = os.path.join(output_dir,f"gompertz_run_{version}_r{params['r']}_l{params['lambda']}.png")

    # Save the figure
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Figure saved as: {filename}")

    # Display the plot
    plt.show()


def print_parameter_summary(params: Dict[str, Any]) -> None:
    """
    Pretty-print the parameter block to the console.

    Parameters
    ----------
    params : dict
        Dictionary containing simulation parameters.
    """
    print("\n" + "=" * 50)
    print("GOMPERTZ SIMULATION PARAMETERS")
    print("=" * 50)
    for key, value in params.items():
        if isinstance(value, float):
            print(f"  {key:12s}: {value:.4f}")
        else:
            print(f"  {key:12s}: {value}")
    print("=" * 50 + "\n")


def print_last_diameters(X: np.ndarray, n_last: int = 5) -> None:
    """
    Print the last N diameter values to the console.

    Parameters
    ----------
    X : np.ndarray
        Array of diameter values.
    n_last : int, optional
        Number of last values to display (default: 5).
    """
    print(f"Last {n_last} diameter values:")
    print("-" * 30)
    start_idx = max(0, len(X) - n_last)
    for i in range(start_idx, len(X)):
        print(f"  X_{i:3d} = {X[i]:.4f} cm")
    print("-" * 30 + "\n")


def main():
    """
    Main function to run the Gompertz simulation with two-step dependency.

    Handles command-line argument parsing, runs the simulation, and displays results.
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Simulate Gompertz stochastic process with X_{k-2} term.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--r", type=float, default=0.1, help="Growth rate parameter (r > 0)"
    )
    parser.add_argument(
        "--D", type=float, default=100.0, help="Asymptotic diameter D (cm)"
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=1.0,
        dest="lam",
        help="Rate parameter for exponential noise (λ > 0)",
    )
    parser.add_argument(
        "--years", type=int, default=50, help="Number of years to simulate"
    )

    args = parser.parse_args()

    # Validate parameters
    if args.r <= 0:
        print("Error: Growth rate r must be positive.")
        sys.exit(1)
    if args.D <= 0:
        print("Error: Asymptotic diameter D must be positive.")
        sys.exit(1)
    if args.lam <= 0:
        print("Error: Lambda must be positive.")
        sys.exit(1)
    if args.years < 2:
        print("Error: Number of years must be at least 2.")
        sys.exit(1)

    # Create parameter dictionary for display
    params = {
        "r": args.r,
        "D": args.D,
        "lambda": args.lam,
        "years": args.years,
        "X0": 15.0,
        "X1": 17.0,
    }

    # Print parameter summary
    print_parameter_summary(params)

    # Run the simulation
    print("Running simulation...")
    X = simulate_gompertz_two_step(
        r=args.r,
        D=args.D,
        lam=args.lam,
        n_years=args.years,
        X0=15.0,  # Fixed initial condition
        X1=17.0,  # Fixed initial condition
    )
    print("Simulation complete.\n")

    # Print last diameters
    print_last_diameters(X)

    # Create and display the plot
    plot_diameter_trajectory(X, params, version="Xkminus2")


if __name__ == "__main__":
    main()
