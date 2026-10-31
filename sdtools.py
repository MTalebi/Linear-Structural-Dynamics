"""sdtools: the reusable engine for the Linear Structural Dynamics book.

One numerical engine, every load. The equation of motion of a single-degree-of-
freedom (SDOF) oscillator

        m u''(t) + c u'(t) + k u(t) = p(t)

is recast as a first-order state-space system  z' = A z + B p(t)  and solved
numerically. Once you have `sdof_response(...)`, every load (harmonic, pulse,
earthquake, ...) is just a different p(t) handed to the same solver.

This module is developed pedagogically in Chapter 3 and imported by every
chapter; it is documented for reuse in Appendix B. Consistent SI units
(kN, m, t, s): 1 kN = 1 t * m/s^2.
"""
import numpy as np
import matplotlib as mpl
from scipy.linalg import expm, solve


# --------------------------------------------------------------------------- #
#  Plot style for DATA plots (axes ON, Nature look). Schematics use figstyle.  #
# --------------------------------------------------------------------------- #
PLT = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00", "#000000"]
INK = "#16191C"


def plot_style(scale=1.0):
    """Apply publication-quality rcParams for result/data plots (idempotent)."""
    b = 12.5 * scale
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica",
                            "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "cm", "mathtext.default": "it",
        "font.size": b, "axes.titlesize": b + 0.5, "axes.labelsize": b,
        "xtick.labelsize": b - 1.5, "ytick.labelsize": b - 1.5,
        "legend.fontsize": b - 1.5, "legend.frameon": False,
        "axes.linewidth": 1.3, "axes.edgecolor": "#16191C",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#E3E8EC", "grid.linewidth": 1.0,
        "axes.axisbelow": True, "lines.linewidth": 2.2,
        "axes.prop_cycle": mpl.cycler(color=PLT),
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


# --------------------------------------------------------------------------- #
#  Natural properties                                                          #
# --------------------------------------------------------------------------- #
def natural(m, c, k):
    """Return the natural dynamic properties of an SDOF oscillator as a dict:
    wn (rad/s), fn (Hz), Tn (s), zeta (-), wd (rad/s), c_cr (critical damping)."""
    wn = np.sqrt(k / m)
    c_cr = 2.0 * m * wn
    zeta = c / c_cr
    wd = wn * np.sqrt(max(1.0 - zeta**2, 0.0))
    return dict(wn=wn, fn=wn / (2 * np.pi), Tn=2 * np.pi / wn,
                zeta=zeta, wd=wd, c_cr=c_cr)


# --------------------------------------------------------------------------- #
#  State-space form and the exact (matrix-exponential) solver                  #
# --------------------------------------------------------------------------- #
def state_space(m, c, k):
    """Continuous-time state-space matrices for z = [u, u']^T:
        z' = A z + B p(t).
    """
    A = np.array([[0.0, 1.0],
                  [-k / m, -c / m]])
    B = np.array([[0.0],
                  [1.0 / m]])
    return A, B


def sdof_response(m, c, k, p, dt, u0=0.0, v0=0.0, method="exp"):
    """Solve m u'' + c u' + k u = p(t) for the relative response.

    Parameters
    ----------
    p : array of applied force sampled at t = 0, dt, 2dt, ...  (length N).
        For base excitation pass p = -m * ag (ag = ground acceleration).
    method : "exp" (exact matrix-exponential, zero-order hold on p) or
             "newmark" (average-acceleration Newmark-beta).

    Returns
    -------
    t, u, v, a : time and relative displacement/velocity/acceleration arrays.
    """
    p = np.asarray(p, float)
    if method == "newmark":
        return newmark(m, c, k, p, dt, u0=u0, v0=v0)
    N = p.size
    A, B = state_space(m, c, k)
    Ad = expm(A * dt)
    Bd = solve(A, (Ad - np.eye(2)) @ B)        # A^{-1}(Ad - I) B, exact for ZOH
    z = np.array([u0, v0])
    u = np.empty(N); vel = np.empty(N)
    u[0], vel[0] = u0, v0
    for i in range(N - 1):
        z = Ad @ z + (Bd[:, 0]) * p[i]
        u[i + 1], vel[i + 1] = z
    a = (p - c * vel - k * u) / m
    t = np.arange(N) * dt
    return t, u, vel, a


def newmark(m, c, k, p, dt, u0=0.0, v0=0.0, gamma=0.5, beta=0.25):
    """Newmark-beta time integration (default = average acceleration, which is
    unconditionally stable). Returns t, u, v, a (relative)."""
    p = np.asarray(p, float)
    N = p.size
    u = np.empty(N); v = np.empty(N); a = np.empty(N)
    u[0], v[0] = u0, v0
    a[0] = (p[0] - c * v0 - k * u0) / m
    a1 = m / (beta * dt**2) + c * gamma / (beta * dt)
    a2 = m / (beta * dt) + c * (gamma / beta - 1.0)
    a3 = m * (1.0 / (2 * beta) - 1.0) + c * dt * (gamma / (2 * beta) - 1.0)
    k_hat = k + a1
    for i in range(N - 1):
        p_hat = p[i + 1] + a1 * u[i] + a2 * v[i] + a3 * a[i]
        u[i + 1] = p_hat / k_hat
        v[i + 1] = (gamma / (beta * dt)) * (u[i + 1] - u[i]) \
            + (1.0 - gamma / beta) * v[i] + dt * (1.0 - gamma / (2 * beta)) * a[i]
        a[i + 1] = (1.0 / (beta * dt**2)) * (u[i + 1] - u[i]) \
            - (1.0 / (beta * dt)) * v[i] - (1.0 / (2 * beta) - 1.0) * a[i]
    t = np.arange(N) * dt
    return t, u, v, a


def central_difference(m, c, k, p, dt, u0=0.0, v0=0.0):
    """Explicit central-difference integration (conditionally stable:
    dt < Tn/pi). Shown in Ch.3 for comparison. Returns t, u, v, a."""
    p = np.asarray(p, float)
    N = p.size
    u = np.empty(N); u[0] = u0
    a0 = (p[0] - c * v0 - k * u0) / m
    u_m1 = u0 - dt * v0 + 0.5 * dt**2 * a0
    k_hat = m / dt**2 + c / (2 * dt)
    A_ = m / dt**2 - c / (2 * dt)
    B_ = k - 2 * m / dt**2
    u_prev = u_m1
    for i in range(N - 1):
        rhs = p[i] - A_ * u_prev - B_ * u[i]
        u[i + 1] = rhs / k_hat
        u_prev = u[i]
    v = np.gradient(u, dt)
    a = (p - c * v - k * u) / m
    t = np.arange(N) * dt
    return t, u, v, a


# --------------------------------------------------------------------------- #
#  Response spectra                                                            #
# --------------------------------------------------------------------------- #
def response_spectrum(ag, dt, periods, zeta=0.05):
    """Elastic response spectra of a ground-acceleration record `ag` (sampled at
    `dt`). For each natural period in `periods`, solve a unit-mass SDOF and take
    the peak relative deformation. Returns (Sd, Sv, Sa) with the pseudo-velocity
    Sv = wn*Sd and pseudo-acceleration Sa = wn^2*Sd."""
    ag = np.asarray(ag, float)
    periods = np.asarray(periods, float)
    m = 1.0
    Sd = np.empty_like(periods)
    for j, T in enumerate(periods):
        wn = 2 * np.pi / T
        k = m * wn**2
        c = 2 * zeta * m * wn
        _, u, _, _ = sdof_response(m, c, k, -m * ag, dt, method="exp")
        Sd[j] = np.max(np.abs(u))
    wn = 2 * np.pi / periods
    return Sd, wn * Sd, wn**2 * Sd


# --------------------------------------------------------------------------- #
#  Ground motions                                                             #
# --------------------------------------------------------------------------- #
def demo_record(dt=0.02, duration=30.0, pga=0.30 * 9.81, seed=7):
    """A reproducible synthetic strong-motion record (Kanai-Tajimi filtered
    noise with a Saragoni-Hart intensity envelope), scaled to a target peak
    ground acceleration. Returns (t, ag) with ag in m/s^2. Real records load the
    same way via load_record(); swap them in for Chapter 6 if available."""
    rng = np.random.default_rng(seed)
    n = int(round(duration / dt))
    t = np.arange(n) * dt
    white = rng.standard_normal(n)
    # Kanai-Tajimi ground filter (2nd order): wg, zg
    wg, zg = 2 * np.pi * 2.5, 0.6
    a_f = np.zeros(n)
    x, xd = 0.0, 0.0
    for i in range(n - 1):
        xdd = white[i] - 2 * zg * wg * xd - wg**2 * x
        xd += xdd * dt
        x += xd * dt
        a_f[i + 1] = 2 * zg * wg * xd + wg**2 * x
    env = (t / 3.0)**2 * np.exp(-1.2 * (t / 3.0))      # Saragoni-Hart envelope
    ag = a_f * env
    ag *= pga / np.max(np.abs(ag))
    return t, ag


def load_record(path, dt, scale=1.0, skiprows=0):
    """Load a ground-acceleration record from a one-column text file (values at
    uniform spacing `dt`). Returns (t, ag*scale)."""
    ag = np.loadtxt(path, skiprows=skiprows).ravel() * scale
    return np.arange(ag.size) * dt, ag
