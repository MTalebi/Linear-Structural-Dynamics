"""osdof: small, tested OpenSeesPy time-history helpers for SDOF and shear-building models.

Kept separate from `sdtools` (which is the pure-NumPy engine) so Chapter 3 can
build the numerical engine without OpenSees in the way. The later chapters use
this for an independent cross-check against the NumPy solver, and Appendix A
documents the pattern line by line.

The model is one mass node on a zeroLength spring to a fixed node, with Rayleigh
mass-proportional damping tuned to the target c (C = alpha_M * M with
alpha_M = c/m gives exactly c for an SDOF), integrated with Newmark's average-
acceleration rule. A force history is applied as a Path time series; for base
excitation pass the effective force p = -m * ag and read the relative response.
"""
import numpy as np
import openseespy.opensees as ops


def opensees_sdof(m, c, k, p, dt, u0=0.0, v0=0.0):
    """Linear SDOF time-history in OpenSeesPy.

    Parameters
    ----------
    p : applied-force array sampled at 0, dt, 2dt, ... (length N).
        For base excitation pass p = -m * ag to get the relative response.
    u0, v0 : initial displacement and velocity (for free vibration).

    Returns
    -------
    t, u, v, a : time and relative displacement/velocity/acceleration arrays.
    """
    p = np.asarray(p, float)
    N = p.size
    ops.wipe()
    ops.model('basic', '-ndm', 1, '-ndf', 1)
    ops.node(1, 0.0); ops.fix(1, 1)
    ops.node(2, 0.0); ops.mass(2, m)
    ops.uniaxialMaterial('Elastic', 1, k)
    ops.element('zeroLength', 1, 1, 2, '-mat', 1, '-dir', 1)
    ops.rayleigh(c / m, 0.0, 0.0, 0.0)               # C = (c/m) M  ->  c

    ops.timeSeries('Path', 1, '-dt', dt, '-values', *map(float, p))
    ops.pattern('Plain', 1, 1)
    ops.load(2, 1.0)

    ops.constraints('Plain'); ops.numberer('Plain'); ops.system('BandGeneral')
    ops.test('NormDispIncr', 1e-10, 10); ops.algorithm('Linear')
    ops.integrator('Newmark', 0.5, 0.25); ops.analysis('Transient')

    if u0 != 0.0 or v0 != 0.0:
        ops.setNodeDisp(2, 1, u0, '-commit')
        ops.setNodeVel(2, 1, v0, '-commit')
        # Commit the dynamically consistent initial acceleration a0 = (p0 - c v0 - k u0)/m
        # so Newmark's first step starts from equilibrium. Without this OpenSees begins from
        # a0 = 0, injecting a spurious O(dt) start-up transient that swamps the true O(dt^2)
        # period-elongation error of the average-acceleration rule.
        ops.setNodeAccel(2, 1, (float(p[0]) - c * v0 - k * u0) / m, '-commit')

    u = np.empty(N); v = np.empty(N); a = np.empty(N)
    u[0], v[0], a[0] = ops.nodeDisp(2, 1), ops.nodeVel(2, 1), ops.nodeAccel(2, 1)
    for i in range(N - 1):
        ops.analyze(1, dt)
        u[i + 1] = ops.nodeDisp(2, 1)
        v[i + 1] = ops.nodeVel(2, 1)
        a[i + 1] = ops.nodeAccel(2, 1)
    return np.arange(N) * dt, u, v, a


def opensees_sdof_ground(m, c, k, ag, dt):
    """Linear SDOF under base excitation, the *easy* OpenSees way.

    Instead of forming the effective force -m*ag by hand and applying it as a
    load (as `opensees_sdof` would), we hand the ground acceleration straight to
    a ``UniformExcitation`` pattern. OpenSees then drives every mass with the
    record and reports the **relative** response directly, with no -m*ag bookkeeping.

    Parameters
    ----------
    ag : ground-acceleration array sampled at 0, dt, 2dt, ... (m/s^2).

    Returns
    -------
    t, u : time and relative displacement of the mass (m).
    """
    ag = np.asarray(ag, float)
    N = ag.size
    ops.wipe()
    ops.model('basic', '-ndm', 1, '-ndf', 1)
    ops.node(1, 0.0); ops.fix(1, 1)
    ops.node(2, 0.0); ops.mass(2, m)
    ops.uniaxialMaterial('Elastic', 1, k)
    ops.element('zeroLength', 1, 1, 2, '-mat', 1, '-dir', 1)
    ops.rayleigh(c / m, 0.0, 0.0, 0.0)               # C = (c/m) M  ->  c

    ops.timeSeries('Path', 1, '-dt', dt, '-values', *map(float, ag))
    ops.pattern('UniformExcitation', 1, 1, '-accel', 1)   # base shaking on DOF 1

    ops.constraints('Plain'); ops.numberer('Plain'); ops.system('BandGeneral')
    ops.test('NormDispIncr', 1e-10, 10); ops.algorithm('Linear')
    ops.integrator('Newmark', 0.5, 0.25); ops.analysis('Transient')

    u = np.zeros(N)
    for i in range(N - 1):
        ops.analyze(1, dt)
        u[i + 1] = ops.nodeDisp(2, 1)
    return np.arange(N) * dt, u


def _build_shear_building(masses, ks):
    """Assemble an n-storey shear building in OpenSees: n mass nodes joined to a
    fixed base by Elastic zero-length storey springs. Node 0 is the base; node i
    (1..n) is floor i. Shared by the eigen and response-history helpers below."""
    masses = np.asarray(masses, float)
    ks = np.asarray(ks, float)
    n = masses.size
    ops.wipe()
    ops.logFile('opensees.log', '-noEcho')           # keep the solver banner quiet
    ops.model('basic', '-ndm', 1, '-ndf', 1)
    ops.node(0, 0.0); ops.fix(0, 1)                  # fixed base
    for i in range(n):
        ops.node(i + 1, 0.0)
        ops.mass(i + 1, float(masses[i]))
    for i in range(n):                               # storey springs in a chain
        ops.uniaxialMaterial('Elastic', i + 1, float(ks[i]))
        ops.element('zeroLength', i + 1, i, i + 1, '-mat', i + 1, '-dir', 1)
    return n


def opensees_mdof_eigen(masses, ks):
    """Natural frequencies and mode shapes of a shear building from OpenSees.

    Builds the chain and calls ``ops.eigen`` (the dense ``-fullGenLapack`` solver,
    right for a small system where we want every mode). Returns the circular
    frequencies and the mode-shape columns, sign-fixed so the roof entry is
    positive.

    Returns
    -------
    wn : natural circular frequencies (rad/s), ascending.
    Phi : (n, n) array whose columns are the mode shapes.
    """
    n = _build_shear_building(masses, ks)
    lam = ops.eigen('-fullGenLapack', n)             # eigenvalues = wn^2
    wn = np.sqrt(np.array(lam))
    Phi = np.array([[ops.nodeEigenvector(i + 1, r + 1, 1)
                     for r in range(n)] for i in range(n)])
    for r in range(n):                               # cosmetic sign fix
        if Phi[-1, r] < 0:
            Phi[:, r] *= -1
    return wn, Phi


def opensees_mdof_ground(masses, ks, zeta, ag, dt):
    """Linear MDOF shear building under base excitation, integrated in OpenSees.

    Uses ``UniformExcitation`` for the ground motion and ``modalDamping`` for a
    clean ``zeta`` in every mode (the matrix analogue of a single damping ratio).
    The modal damping force is state-dependent, so we drive the step to
    convergence with ``algorithm('Newton')``. The result then reproduces a NumPy
    classical-damping Newmark integration to about 1e-9 (Chapter 8).

    Returns
    -------
    t, U : time and the (N, n) array of floor displacements relative to the ground (m).
    """
    ag = np.asarray(ag, float)
    N = ag.size
    n = _build_shear_building(masses, ks)
    ops.eigen('-fullGenLapack', n)                   # modes first ...
    ops.modalDamping(zeta)                           # ... so modal damping can use them

    ops.timeSeries('Path', 1, '-dt', dt, '-values', *map(float, ag))
    ops.pattern('UniformExcitation', 1, 1, '-accel', 1)

    ops.constraints('Transformation'); ops.numberer('Plain'); ops.system('BandGeneral')
    ops.test('NormDispIncr', 1e-10, 50); ops.algorithm('Newton')
    ops.integrator('Newmark', 0.5, 0.25); ops.analysis('Transient')

    U = np.zeros((N, n))
    for i in range(N - 1):
        ops.analyze(1, dt)
        for j in range(n):
            U[i + 1, j] = ops.nodeDisp(j + 1, 1)
    return np.arange(N) * dt, U
