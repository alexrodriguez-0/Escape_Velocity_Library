# Escape_Velocity_Library

Tools to infer **galaxy cluster masses** from spectroscopic radius–velocity phase space via the **escape-velocity edge**, with a calibrated suppression factor $Z_v$ that corrects for sparse sampling. The workflow follows the methodology in **Rodriguez et al. (2025)** and includes an end-to-end example using **HeCS / HeCS-SZ** spectroscopy (Rines et al. 2013; Rines et al. 2016; >40,000 galaxies).

> **Please cite:** Rodriguez et al. (2025), *Concordance of Weak Lensing and Escape Velocity Cluster Masses*, arXiv:2507.20938.

---

## Repository layout

```
Escape_Velocity_Library/
├── AGAMA_Zv_calibration/              # Precomputed Z_v calibration (grid in redshift, mass, sampling)
│   └── Zv_fits_z_0.01_M200_14.0.pkl   # Example filename pattern (see “Z_v calibration”)
├── Example/
│   ├── run_example.py                  # Minimal end-to-end example (configure paths, run mass inference)
│   └── Rines_galaxy_data.txt           # Example input: RAh RAm RAs DEd DEm DEs redshift
└── Function/
    └── Libraries/
        ├── escape_analysis_functions.py   # Phase-space building, interlopers, edge, MCMC wrapper
        └── escape_theory_functions.py     # v_esc model, cosmology helpers, c(M) mapping, r_eq
```

---

## Z_v calibration (AGAMA_Zv_calibration)

The calibration quantifies the **down-sampling suppression** of the measured escape-velocity edge relative to the true escape profile. It was generated with **AGAMA** and stored as pickled fit files with names like:
```
Zv_fits_z_{z:.2f}_M200_{M:.1f}.pkl
```

**Coverage of the grid** used by this library:
- **Redshift:** $0.0 \le z \le 0.7$
- **Mass:** $14.0 \le \log_{10}(M_{200}/M_\odot) \le 15.6$
- **Sampling:** $50 \le N \le 1200$ member galaxies between $0.2$ and $1.0\,r_{200}$

The $Z_v$ distribution in each bin is modeled as a **skew-normal**, whose parameters are smooth (approximately linear) functions of $\log_{10} N$ with weak dependence on $z$ and $M_{200}$. The analysis draws $Z_v$ for each bin via inverse-CDF sampling consistent with the grid.

> You do **not** need AGAMA installed to run inference—only these precomputed calibration files are required.

---

## Installation

This is a pure-Python library.

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy pandas astropy emcee matplotlib
```

---

## Quick start (example)

1. Edit `Example/run_example.py` to set:
   - `path_to_Zv_calibration = '/path/to/Escape_Velocity_Library/AGAMA_Zv_calibration'`
   - `path_to_galaxy_data = '/path/to/Escape_Velocity_Library/Example/'`
   - `cluster_positional_data = (cl_ra_deg, cl_dec_deg, cl_z)`
   - Initial mass guess/prior widths: `M200`, `M200_err_up`, `M200_err_down` (log-space dex widths)
   - Cosmology: `cosmo_name='FlatLambdaCDM'`, `cosmo_params=[Omega_m, h]` (e.g., `[0.3, 0.7]`)

2. **Input format** for `Rines_galaxy_data.txt` (one galaxy per line):
   ```
   RAh  RAm  RAs   DEd  DEm  DEs   redshift
   ```
   RA and Dec are **sexagesimal**; `DEd` carries the sign. Redshift is dimensionless $z$.

3. Run:
   ```bash
   python Example/run_example.py
   ```

**Outputs**
- Phase-space plot $v_{\rm los}$ vs $r_\perp/r_{200}$ with: interlopers removed, measured **down-sampled** edge, **suppressed** theoretical escape profile, and uncertainty bands.
- Posterior summary for $M_{200}$ (median and 68% credible interval).

---

## Programmatic API (typical)

```python
from escape_analysis_functions import main

results = main(
    path_to_Zv_calibration,         # str: folder with AGAMA_Zv_calibration tables
    galaxy_positional_data,         # ndarray (N,7): [RAh,RAm,RAs,DEd,DEm,DEs,redshift]
    cluster_positional_data,        # tuple: (cl_ra_deg, cl_dec_deg, cl_z)
    M200,                           # float: initial guess for M200 [Msun]
    M200_err_up,                    # float: +dex prior half-width on log10(M200)
    M200_err_down,                  # float: −dex prior half-width on log10(M200)
    cosmo_params,                   # list: e.g., [Omega_m, h] for FlatLambdaCDM
    cosmo_name,                     # str: e.g., 'FlatLambdaCDM'
    nwalkers=250, nsteps=1000       # MCMC controls
)
```

**Key internals** (see `Function/Libraries/`):
- `ClusterDataHandler.calculate_projected_quantities(...)` → $r_\perp$ (Mpc), $v_{\rm los}$ (km/s)
- `ClusterDataHandler.shiftgapper(data, gap_prev, nbin_val, gap_val, coremin)` → interloper removal with optional **core exclusion** radius
- `get_edge(bins, ...)` → edge measurement (max $|v_{\rm los}|$ per radial bin) with **outer-bin monotonic** enforcement
- `EscapeVelocityModeling.sample_Zv(N, z, log10M, rbin)` → draw $Z_v$ for given sampling $N$, redshift $z$, mass, and radial bin
- Theory layer: `v_esc_dehnen`, `dehnen_nfwM200_errors`, `D_A`, `H_z_function`, `q_z_function`, `rho_crit_z`

---

## Modeling assumptions (precise)

### Escape profile in an accelerating background
The observable escape surface is tied to the **effective** potential:
$$v_{\rm esc}^2(r) = -2\big[\Psi(r)-\Psi(r_{\rm eq})\big] - q(z)H^2(z)\big(r^2-r_{\rm eq}^2\big)$$
where $\Psi(r)$ is the matter-only potential, $q(z)$ the **deceleration parameter**, $H(z)$ the Hubble parameter, and $r_{\rm eq}$ the radius where the inward gravitational acceleration balances the outward cosmological term.

### Mass model and c(M)
We represent $\Psi(r)$ with a **Dehnen** profile. When inference is parameterized by $M_{200}$, we map to Dehnen parameters through an **NFW** proxy with a redshift-dependent concentration–mass relation (configurable; a standard choice like Dutton & Macciò 2014 is supported) and $\rho_{\rm crit}(z)$.

### Down-sampled edge and suppression $Z_v$
The measured edge (max $|v_{\rm los}|$ per radial bin) from a finite sample is **suppressed** relative to the true escape profile. We model this with a multiplicative factor $Z_v>1$ whose distribution depends on:
- **Sampling** $N$: number of *member* galaxies between $0.2$ and $1.0\,r_{200}$ **after** interloper removal and centering,
- **Redshift** $z$ and **mass** $M_{200}$ (weak dependence),
- **Radial bin**.

The $Z_v$ distribution per bin is a **skew-normal** with parameters expressed as smooth functions of $\log_{10} N$. We draw $Z_v$ for each bin when comparing data to theory.

### Binning, monotonicity, and interlopers
- **Radial bins:** **5** uniform bins from **0.2** to **1.0** $r_{200}$.
- **Monotonicity:** enforce a **monotonic decrease** of the edge in the **outer three** bins.
- **Interlopers:** **shifting-gapper** with typical settings **20 galaxies/bin** and **600 km/s** initial gap; a **core exclusion** (e.g., `coremin ≈ 0.44 r_{200}` for 5 bins) prevents over-flagging at small radii.
- **Velocity cut:** $|v_{\rm los}| < 4500$ km/s during construction of the phase space.

### Errors and likelihood
- Spectroscopic errors $\sigma_{cz}\sim 30\,\mathrm{km\,s^{-1}}$ are added in quadrature to edge uncertainties.
- Likelihood: Gaussian over the 5 radial bins comparing the **down-sampled** edge to the **suppressed** theoretical curve.
- Inference: `emcee` affine-invariant ensemble sampler on $\log_{10} M_{200}$; report the posterior median and 68% credible interval.

### Centers and systemic velocity
The cluster sky center and systemic redshift are iteratively refined (default ~10 iterations) using galaxies in $0.2\le r_\perp/r_{200}\le 1.0$ to symmetrize the phase space. Outliers at $|v_{\rm los}|>4500\,\mathrm{km\,s^{-1}}$ are removed a priori.

### $M_{200}$ Starting Estimate
The model implicitly assumes an initial guess for the cluster mass to bin the phase-space data and measure the sampling. The weak covariance of this starting estimate is elaborated upon in Rodriguez et al. 2025, where it is found to not be of significant concern.

---

## Reproducibility tips

- Keep **5 bins** on $0.2 \rightarrow 1.0\,r_{200}$ to remain within the $Z_v$ calibration.
- Measure **$N$** for the $Z_v$ call **after** interloper removal and within $0.2\le r/r_{200}\le 1.0$.
- Use a consistent **cosmology** in both reduction and theory (`cosmo_name`, `cosmo_params`).
- If you change shifting-gapper hyper-parameters, verify edge stability (few-percent variations expected within reasonable ranges).

---

## Data acknowledgement

The example uses spectroscopic members from **HeCS** and **HeCS-SZ** (Rines et al. 2013; Rines et al. 2016).

---

## Citation

If you use this library or its results, please cite the arxiv link, or ApJ Submission (under review):

- **Rodriguez et al. (2025)** — *Concordance of Weak Lensing and Escape Velocity Cluster Masses*, **arXiv:2507.20938**.


---

## Contact

- Alexander Rodriguez — <alexcrod@umich.edu
