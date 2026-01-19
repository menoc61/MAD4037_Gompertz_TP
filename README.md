

# MAD4037_Gompertz_TP Project Structure

```
MAD4037_Gompertz_TP/
├── gompertz_with_Xkminus2.py
├── gompertz_with_Xkminus1_only.py
├── requirements.txt
├── README.md
└── assignment_submission_email.txt
```


```markdown
# MAD4037 – TP Processus stochastique
## Gompertz tree-diameter simulator

This project implements a stochastic version of the Gompertz model for simulating tree diameter growth over time. The model incorporates random environmental fluctuations through exponentially distributed noise.

---

## Mathematical Model

### Version 1: With X_{k-2} term

For this version, the diameter at year k depends on the average of the two previous years:

$$X_k = D^{(1-e^{-r})} \cdot \left(\frac{X_{k-1} + X_{k-2}}{2}\right)^{(e^{-r})} \cdot \varepsilon_{k-1}, \quad k \geq 2$$

where:
- $X_0 = 15$ cm (fixed)
- $X_1 = 17$ cm (fixed)
- $\varepsilon_k \sim \text{Exponential}(\lambda)$ i.i.d.

### Version 2: With X_{k-1} term only

For this simplified version, the diameter at year k depends only on the previous year:

$$X_k = D^{(1-e^{-r})} \cdot X_{k-1}^{(e^{-r})} \cdot \varepsilon_{k-1}, \quad k \geq 1$$

where:
- $X_0 = 15$ cm (fixed)
- $\varepsilon_k \sim \text{Exponential}(\lambda)$ i.i.d.

### Parameter Description

| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| Growth rate | $r$ | Controls the speed of convergence toward the asymptote |
| Asymptotic diameter | $D$ | Theoretical maximum diameter the tree can reach |
| Noise rate | $\lambda$ | Controls the intensity of random environmental fluctuations |

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/menoc61/MAD4037_Gompertz_TP.git
cd MAD4037_Gompertz_TP
pip install -r requirements.txt
```

---

## Usage

### Version with X_{k-2} term

```bash
python gompertz_with_Xkminus2.py --r 0.3 --D 100 --lambda 0.7 --years 50
```

### Version with X_{k-1} term only

```bash
python gompertz_with_Xkminus1_only.py --r 0.3 --D 100 --lambda 0.7 --years 50
```

### Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--r` | 0.1 | Growth rate parameter (must be positive) |
| `--D` | 100.0 | Asymptotic diameter in cm (must be positive) |
| `--lambda` | 1.0 | Rate parameter for exponential noise (must be positive) |
| `--years` | 50 | Number of years to simulate |

---

## Version Differences

### Version 1 (gompertz_with_Xkminus2.py)

This version implements the full second-order model where each year's diameter depends on the average of the two previous years. This creates a smoothing effect and typically produces more stable trajectories. The initial conditions are fixed at $X_0 = 15$ cm and $X_1 = 17$ cm.

### Version 2 (gompertz_with_Xkminus1_only.py)

This version implements a simplified first-order model where each year's diameter depends only on the previous year. This model is more sensitive to recent fluctuations and typically shows more variability in the trajectory. The initial condition is fixed at $X_0 = 15$ cm.

---

## Tuning Parameters

### Effect of r (Growth Rate)

- **Small r (e.g., 0.1)**: Slow convergence toward the asymptotic diameter. The tree grows gradually over many years.
- **Medium r (e.g., 0.3)**: Moderate convergence speed. This is often the most realistic setting for biological growth models.
- **Large r (e.g., 0.98)**: Very rapid convergence. The tree quickly approaches the asymptotic diameter D.

### Effect of λ (Noise Rate)

- **Small λ (e.g., 0.5)**: Low noise intensity. The trajectory is smooth with small random fluctuations around the deterministic path.
- **Medium λ (e.g., 1.0)**: Moderate noise intensity. Noticeable but manageable random variations.
- **Large λ (e.g., 2.0+)**: High noise intensity. Large fluctuations can push the diameter significantly above or below the expected trajectory.

### Effect of D (Asymptotic Diameter)

- **D** sets the ceiling for the diameter. All trajectories, regardless of noise, will tend toward this value on average.
- Larger D values require more years to see convergence.

---

## Output

Both scripts produce:

1. **Console output**:
   - Parameter summary table
   - Last 5 diameter values

2. **Graphical output**:
   - A line plot showing diameter vs. year
   - Reference line at the asymptotic diameter D
   - Saved as `gompertz_run_<version>_r<r>_l<lambda>.png`

---

## Observation

### Influence of r on Trajectory Smoothness and Convergence

The growth rate parameter r has a profound influence on the behavior of the Gompertz stochastic process. When r is small (close to 0.1), the trajectory exhibits a gradual approach to the asymptotic diameter D, with the curve showing a characteristic concave-down shape typical of biological growth. The smoothing effect of the exponential decay term e^{-r} means that each step contributes only a small proportional change, resulting in trajectories that appear smooth and continuous even in discrete time.

As r increases toward 0.98, the convergence becomes dramatically faster. In the limit, the model approaches an exponential decay toward the asymptote, and the trajectory can reach values very close to D within just a few iterations. This rapid convergence can be visually striking, but it also means that the stochastic component has less time to influence the trajectory before the process stabilizes near D.

### Influence of λ on Oscillation and Variability

The noise parameter λ controls the intensity of random environmental fluctuations through the exponential distribution ε ~ Exp(λ). When λ is small (around 0.5), the mean of the exponential noise is larger (mean = 1/λ), which tends to push diameters upward on average. This creates trajectories that often overshoot the deterministic path and show persistent oscillations around the expected value.

When λ is large (2.0 or higher), the exponential noise has a smaller mean and reduced variance, resulting in trajectories that more closely follow the deterministic Gompertz curve. However, even with high λ, the multiplicative nature of the noise ensures that some stochastic variation remains, preventing the trajectory from becoming perfectly deterministic.

The interaction between r and λ is particularly important: low r combined with low λ produces trajectories that are smooth but may show systematic drift away from D due to the noise mean, while high r combined with high λ produces rapid convergence with minimal noise around the asymptotic level. These observations suggest that realistic tree growth simulations require careful calibration of both parameters based on actual growth data.

---

## Repository

GitHub: [https://github.com/menoc61/MAD4037_Gompertz_TP](https://github.com/menoc61/MAD4037_Gompertz_TP)

---

## License

This project is for educational purposes as part of the MAD4037 stochastic processes course.
```

## 5. assignment_submission_email.txt

```
Objet: devoir TP MAD4037 – Momeni Gilles Christian

Bonjour Professeur,

Veuillez trouver ci-joint le travail pratique demandé (simulateur du processus de Gompertz) ainsi qu'un lien vers le depot GitHub :  
https://github.com/menoc61/MAD4037_Gompertz_TP

Bien cordialement,  
Momeni Gilles Christian
```