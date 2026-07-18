"""LDP mechanism wrappers and registries."""
import numpy as np
import torch
from scipy.stats import norm as _snorm, beta as _beta_dist, truncnorm as _truncnorm
from scipy.special import betainc as _betainc
from scipy.optimize import brentq as _brentq
from scipy.linalg import hadamard as _hadamard

PERCENTILE = 90.0
LAMBDA_N   = 1000.0
DELTA      = 1e-5


def _project_l1(z: np.ndarray, rho: float) -> np.ndarray:
    """Clip each row of z to L1 ball of radius ρ via scalar scaling."""
    norms = np.abs(z).sum(axis=1, keepdims=True)
    scale = np.minimum(1.0, rho / np.maximum(norms, 1e-10))
    return z * scale


# ── Gaussian noise calibration ────────────────────────────────────────────────

def _agm_sigma(eps: float, delta: float, sensitivity: float) -> float:
    """Optimal σ for (ε,δ)-LDP Gaussian mechanism (Balle & Wang 2018).

    Finds the smallest σ satisfying the exact DP condition:
        δ = Φ(Δ/(2σ) − εσ/Δ) − e^ε · Φ(−Δ/(2σ) − εσ/Δ)
    which is tighter than the classical σ = Δ·√(2 ln(1.25/δ)) / ε.
    """
    def _delta_of(sigma):
        v = sensitivity / (2.0 * sigma)
        return (_snorm.cdf(v - eps * sigma / sensitivity)
                - np.exp(eps) * _snorm.cdf(-v - eps * sigma / sensitivity))

    sigma_hi = sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / eps
    while _delta_of(sigma_hi) > delta:
        sigma_hi *= 2.0
    return _brentq(lambda s: _delta_of(s) - delta, 1e-10, sigma_hi,
                   xtol=1e-8, rtol=1e-8)


# ── PrivUnit2 shared helpers ──────────────────────────────────────────────────

def _sphere_from_t(u: np.ndarray, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """z = t·u + √(1-t²)·w,  w ⊥ u uniform on S^{d-2}. u:(n,d), t:(n,)."""
    G  = rng.standard_normal(u.shape)
    G -= (G * u).sum(axis=1, keepdims=True) * u
    G /= np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-10)
    return t[:, None] * u + np.sqrt(np.maximum(1.0 - t[:, None]**2, 0.0)) * G


def _pu2_sample_t(gamma: float, n: int, from_cap: bool, a: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Rejection-sample t from Beta(a,a) marginal restricted to cap or complement."""
    g   = (1 + gamma) / 2
    out = np.empty(n)
    rem = np.arange(n)
    while len(rem):
        m   = max(len(rem) * 4, 32)
        s   = _beta_dist.rvs(a, a, size=m, random_state=rng)
        acc = s[s >= g] if from_cap else s[s < g]
        k   = min(len(acc), len(rem))
        out[rem[:k]] = 2.0 * acc[:k] - 1.0
        rem = rem[k:]
    return out


def _pu2_find_gamma(eps: float, a: float) -> float:
    """Find optimal γ* for PrivUnit2 via self-consistency A(γ*,ε) = γ*."""
    def _bias(gamma):
        if gamma >= 1:
            return 0.0
        g        = (1 + gamma) / 2
        P        = 1.0 - float(_betainc(a, a, g))
        es_above = 0.5 * (1.0 - float(_betainc(a + 1, a, g)))
        M        = 2.0 * es_above - P
        Z        = np.exp(eps) * P + (1.0 - P)
        return 0.0 if Z < 1e-15 else (np.exp(eps) - 1.0) * M / Z

    return float(_brentq(lambda g: _bias(g) - g, -1 + 1e-8, 1 - 1e-8))


def _pu2_cap_prob(gamma: float, a: float) -> float:
    if gamma <= -1: return 1.0
    if gamma >=  1: return 0.0
    return 1.0 - float(_betainc(a, a, (1 + gamma) / 2))


# ── PA shared setup ──────────────────────────────────────────────────────────

def _pa_setup(W, features, bounds, lambda_n, Z_pub_norm):
    """Shared SVD + SV-weighted λ allocation for PA (Pre/Post-processing Adaptive) variants.

    Returns (d, r, sv, U, sqrt_l, inv_sqrt_l, mu).
    """
    d      = len(features)
    scv    = np.array([bounds[f][1] - bounds[f][0] for f in features])
    W_norm = np.asarray(W, dtype=float) * scv
    W2     = W_norm.reshape(1, -1) if W_norm.ndim == 1 else W_norm

    _, sv_all, Vt = np.linalg.svd(W2, full_matrices=True)
    r  = int(np.sum(sv_all > 1e-10 * sv_all[0]))
    U  = Vt.T
    sv = sv_all[:r]
    mu = Z_pub_norm.mean(axis=0)

    if np.isinf(lambda_n):
        C_row           = float(d)
        sqrt_l_null     = np.zeros(d - r)
        inv_sqrt_l_null = np.zeros(d - r)
    else:
        C_row = d - (d - r) / np.sqrt(lambda_n)
        assert C_row > 0, f"lambda_n={lambda_n} too small for d={d}, r={r}"
        sqrt_l_null     = np.full(d - r, np.sqrt(lambda_n))
        inv_sqrt_l_null = 1.0 / sqrt_l_null

    sv23           = sv ** (2.0 / 3.0)
    inv_sqrt_l_row = C_row * sv23 / sv23.sum()
    sqrt_l_row     = 1.0 / inv_sqrt_l_row

    sqrt_l     = np.concatenate([sqrt_l_row,     sqrt_l_null])
    inv_sqrt_l = np.concatenate([inv_sqrt_l_row, inv_sqrt_l_null])

    return d, r, sv, U, sqrt_l, inv_sqrt_l, mu


# ── Simple mechanism classes ──────────────────────────────────────────────────

class NoNoise:
    """No privacy baseline (ε=∞). Returns z unchanged."""
    def encode(self, z: np.ndarray, eps: float,
               rng: np.random.Generator = None) -> np.ndarray:
        return z.copy()

    def decode(self, z_enc: np.ndarray) -> np.ndarray:
        return z_enc


class L1ClipLaplace:
    """Laplace mechanism with L1 norm clipping in raw latent space.

    Clips z to L1 ball of radius ρ (90th percentile of Z_tr L1 norms),
    then adds Laplace noise with scale 2ρ/ε (L1 sensitivity = 2ρ).
    """
    def __init__(self, rho: float):
        self._rho = rho

    def encode(self, z: np.ndarray, eps: float,
               rng: np.random.Generator = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        z_clip = _project_l1(z, self._rho)
        return z_clip + rng.laplace(0, 2 * self._rho / eps, z_clip.shape)

    def decode(self, z_enc: np.ndarray) -> np.ndarray:
        return z_enc


class GaussianMech:
    """Gaussian mechanism: L2 clip + AGM noise, no preprocessing.

    ρ = 90th percentile of Z_tr L2 norms. L2 sensitivity = 2ρ.
    σ calibrated via AGM (Balle & Wang 2018) for (ε,δ)-LDP.
    """
    def __init__(self, rho: float, delta: float):
        self._rho   = rho
        self._delta = delta
        self._cache: dict[float, float] = {}

    def encode(self, z: np.ndarray, eps: float,
               rng: np.random.Generator = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        if eps not in self._cache:
            self._cache[eps] = _agm_sigma(eps, self._delta, 2 * self._rho)
        nrm    = np.linalg.norm(z, axis=1, keepdims=True)
        z_clip = z * np.minimum(1.0, self._rho / np.maximum(nrm, 1e-10))
        return z_clip + rng.normal(0, self._cache[eps], z_clip.shape)

    def decode(self, z_enc: np.ndarray) -> np.ndarray:
        return z_enc


# ── PA mechanisms ────────────────────────────────────────────────────────────

def make_laplace_pa(W, features, bounds, Z_pub_norm,
                lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """Laplace+PA: SV-weighted λ allocation + L1 clip + Laplace noise.

    Row-space budget allocated by: 1/√λ_i ∝ s_i^{2/3} (Lagrange optimum for
    min Σ s_i²·λ_i s.t. Σ 1/√λ_i = C_row).
    """
    d, r, sv, U, sqrt_l, inv_sqrt_l, mu = _pa_setup(W, features, bounds, lambda_n, Z_pub_norm)

    lambdas_row = (1.0 / inv_sqrt_l[:r]) ** 2
    print(f"  Laplace+PA: r={r}, λ_i(row) min={lambdas_row.min():.4f} "
          f"max={lambdas_row.max():.4f}  "
          f"(uniform λ_r would be "
          f"{(r / d) ** 2 if np.isinf(lambda_n) else 1.0 / ((d - (d - r) / np.sqrt(lambda_n)) / r) ** 2:.4f})")

    z_pub = (Z_pub_norm - mu) @ U * inv_sqrt_l
    rho   = float(np.percentile(np.abs(z_pub).sum(axis=1), percentile))

    class _LaplacePA:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if rng is None:
                rng = np.random.default_rng()
            z_sc = (z - mu) @ U * inv_sqrt_l
            z_cl = _project_l1(z_sc, rho)
            return z_cl + rng.laplace(0, 2.0 * rho / epsilon, z_cl.shape)

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return (z_enc * sqrt_l) @ U.T + mu

    return _LaplacePA()


def make_agm_pa(W, features, bounds, Z_pub_norm,
                      lambda_n=LAMBDA_N, percentile=PERCENTILE, delta=DELTA):
    """AGM+PA: SV-weighted λ + L2 clipping + AGM noise.

    Same SV-weighted λ allocation as Laplace+PA.
    L2 sensitivity after clipping: Δ₂ = 2ρ.
    Noise: AGM (Balle & Wang 2018), σ = _agm_sigma(ε, δ, 2ρ) → (ε,δ)-LDP.
    """
    d, r, sv, U, sqrt_l, inv_sqrt_l, mu = _pa_setup(W, features, bounds, lambda_n, Z_pub_norm)

    z_pub = (Z_pub_norm - mu) @ U * inv_sqrt_l
    rho   = float(np.percentile(np.linalg.norm(z_pub, axis=1), percentile))
    sens  = 2.0 * rho
    _cache: dict[float, float] = {}

    class _AGMPA:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if rng is None:
                rng = np.random.default_rng()
            if epsilon not in _cache:
                _cache[epsilon] = _agm_sigma(epsilon, delta, sens)
            z_sc = (z - mu) @ U * inv_sqrt_l
            nrm  = np.linalg.norm(z_sc, axis=1, keepdims=True)
            z_cl = z_sc * np.minimum(1.0, rho / np.maximum(nrm, 1e-10))
            return z_cl + rng.normal(0, _cache[epsilon], z_cl.shape)

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return (z_enc * sqrt_l) @ U.T + mu

    return _AGMPA()


def make_privunit2_opt_pa(W, features, bounds, Z_pub_norm,
                          lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """PrivUnit2(Opt)+PA: SV-weighted anisotropic PA transform + spherical step-function.

    Combines PA's row-space identification with PrivUnit2's optimal step-function.
    ALL d dimensions participate (including null space), so z_null is protected.

    Encoding:
        z_sc  = (z - μ) @ U ⊙ inv_sqrt_λ          (anisotropic scaling)
        u     = x / ‖x‖                              (normalize to S^{d-1})
        z̃  ~ PrivUnit2(u, ε) on S^{d-1}
        out   = (ρ/A) · z̃
    Decoding:
        z_dec = (out ⊙ sqrt_λ) @ U.T + μ

    Pure ε-LDP: inherited from PrivUnit2 on S^{d-1}.
    """
    if np.isinf(lambda_n):
        raise ValueError("lambda_n must be finite: null space needs non-zero weight for privacy.")

    d, r, sv, U, sqrt_l, inv_sqrt_l, mu = _pa_setup(W, features, bounds, lambda_n, Z_pub_norm)

    z_pub = (Z_pub_norm - mu) @ U * inv_sqrt_l
    rho   = float(np.percentile(np.linalg.norm(z_pub, axis=1), percentile))
    a     = (d - 1) / 2.0
    _cache: dict = {}

    class _PrivUnit2PA:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if rng is None:
                rng = np.random.default_rng()
            if epsilon not in _cache:
                gamma = _pu2_find_gamma(epsilon, a)
                P     = _pu2_cap_prob(gamma, a)
                p_cap = np.exp(epsilon) * P / (np.exp(epsilon) * P + (1.0 - P))
                _cache[epsilon] = (gamma, p_cap)
                print(f"  PrivUnit2(Opt)+PA: ε={epsilon}, γ*={gamma:.4f}, p_cap={p_cap:.4f}, A=γ*={gamma:.4f}")
            gamma, p_cap = _cache[epsilon]

            z_sc  = (z - mu) @ U * inv_sqrt_l
            u_vec = z_sc / np.maximum(np.linalg.norm(z_sc, axis=1, keepdims=True), 1e-10)
            n     = z.shape[0]
            use   = rng.random(n) < p_cap
            out   = np.empty_like(u_vec)

            for from_cap, mask in [(True, use), (False, ~use)]:
                idx = np.where(mask)[0]
                if not len(idx): continue
                out[idx] = _sphere_from_t(u_vec[idx],
                                          _pu2_sample_t(gamma, len(idx), from_cap, a, rng),
                                          rng)
            return (rho / gamma) * out

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return (z_enc * sqrt_l) @ U.T + mu

    return _PrivUnit2PA()


def make_cw_laplace_pa(W, features, bounds, Z_pub_norm,
                   lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """CW(Laplace)+PA: SV-weighted PA transform + coordinate-wise i.n.i.d. Laplace.

    Combines SV-weighted anisotropic scaling with per-coordinate budget allocation
    (λ_i^{1/3} split per [25TIFS]) applied in the transformed space.
    """
    d, r, sv, U, sqrt_l, inv_sqrt_l, mu = _pa_setup(W, features, bounds, lambda_n, Z_pub_norm)

    z_pub   = (Z_pub_norm - mu) @ U * inv_sqrt_l
    rho     = np.percentile(np.abs(z_pub), percentile, axis=0)
    lambdas = 2.0 * rho
    norm_23 = float(np.sum(lambdas ** (2.0 / 3.0)))
    beta    = lambdas ** (1.0 / 3.0) * norm_23
    active  = beta > 0

    class _CWLaplacePA:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if rng is None:
                rng = np.random.default_rng()
            z_sc = (z - mu) @ U * inv_sqrt_l
            noise = np.zeros_like(z_sc)
            if active.any():
                noise[:, active] = rng.laplace(
                    0, beta[active] / epsilon, (z_sc.shape[0], int(active.sum())))
            return z_sc + noise

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return (z_enc * sqrt_l) @ U.T + mu

    return _CWLaplacePA()


def make_cw_agm_pa(W, features, bounds, Z_pub_norm,
                             lambda_n=LAMBDA_N, percentile=PERCENTILE, delta=DELTA):
    """CW(AGM)+PA: SV-weighted PA transform + coordinate-wise i.n.i.d. Gaussian.

    Same transform as CW(Laplace)+PA; noise calibrated via AGM (Balle & Wang 2018)
    with budget split minimizing Σ σ_i² per [25TIFS]. Satisfies (ε,δ)-LDP.
    """
    d, r, sv, U, sqrt_l, inv_sqrt_l, mu = _pa_setup(W, features, bounds, lambda_n, Z_pub_norm)

    z_pub   = (Z_pub_norm - mu) @ U * inv_sqrt_l
    rho     = np.percentile(np.abs(z_pub), percentile, axis=0)
    lambdas = 2.0 * rho
    scale_i = np.sqrt(lambdas * lambdas.sum())
    active  = scale_i > 0
    _cache: dict[float, float] = {}

    class _CWGaussianPA:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if rng is None:
                rng = np.random.default_rng()
            if epsilon not in _cache:
                _cache[epsilon] = _agm_sigma(epsilon, delta, 1.0)
            z_sc = (z - mu) @ U * inv_sqrt_l
            noise = np.zeros_like(z_sc)
            if active.any():
                noise[:, active] = rng.normal(
                    0, _cache[epsilon] * scale_i[active],
                    (z_sc.shape[0], int(active.sum())))
            return z_sc + noise

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return (z_enc * sqrt_l) @ U.T + mu

    return _CWGaussianPA()


def make_privunitg_mc_pa(W, features, bounds, Z_pub_norm,
                           lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """PrivUnitG(MC)+PA: SV-weighted PA transform + PrivUnitG step-function."""
    return make_pa_warp(
        lambda z: make_privunitg_mc(z, percentile),
        W, features, bounds, Z_pub_norm, lambda_n, percentile,
    )


def make_privunitg_paper_pa(W, features, bounds, Z_pub_norm,
                                 lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """PrivUnitG(Paper)+PA: SV-weighted PA transform + paper-exact PrivUnitG."""
    return make_pa_warp(
        lambda z: make_privunitg_paper(z, percentile),
        W, features, bounds, Z_pub_norm, lambda_n, percentile,
    )


def make_pa_warp(inner_factory, W, features, bounds, Z_pub_norm,
                     lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """PA preprocessing wrapper for any mechanism.

    Applies the PA transform (SVD rotation + SV-weighted anisotropic scaling)
    as a plug-in preprocessing, then delegates encode/decode to an inner mechanism
    built on the transformed public data.

    inner_factory(z_pub: np.ndarray) -> mechanism with .encode / .decode
    """
    d, r, sv, U, sqrt_l, inv_sqrt_l, mu = _pa_setup(W, features, bounds, lambda_n, Z_pub_norm)
    z_pub = (Z_pub_norm - mu) @ U * inv_sqrt_l
    inner = inner_factory(z_pub)

    class _Wrapped:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            return inner.encode((z - mu) @ U * inv_sqrt_l, epsilon, rng)

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return (inner.decode(z_enc) * sqrt_l) @ U.T + mu

    return _Wrapped()


# ── Ablation mechanisms ───────────────────────────────────────────────────────

def make_l1clip_laplace(Z_pub_norm: np.ndarray, percentile: float = PERCENTILE):
    """Laplace mechanism with L1 clipping at given percentile. Pure ε-LDP."""
    rho = float(np.percentile(np.abs(Z_pub_norm).sum(axis=1), percentile))
    return L1ClipLaplace(rho)


def make_l1clip_privunit2(Z_pub_norm: np.ndarray, percentile: float = PERCENTILE):
    """PrivUnit2 with L1 clipping only (no PA transform). Pure ε-LDP."""
    rho_l1 = float(np.percentile(np.abs(Z_pub_norm).sum(axis=1), percentile))
    inner  = make_privunit2_opt(Z_pub_norm, percentile)

    class _L1ClipPrivUnit2:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            return inner.encode(_project_l1(z, rho_l1), epsilon, rng)

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return z_enc

    return _L1ClipPrivUnit2()


def make_l1clip_privunitg(Z_pub_norm: np.ndarray, percentile: float = PERCENTILE):
    """PrivUnitG with L1 clipping only (no PA transform). Pure ε-LDP."""
    rho_l1 = float(np.percentile(np.abs(Z_pub_norm).sum(axis=1), percentile))
    inner  = make_privunitg_mc(Z_pub_norm, percentile)

    class _L1ClipPrivUnitG:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            return inner.encode(_project_l1(z, rho_l1), epsilon, rng)

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return z_enc

    return _L1ClipPrivUnitG()


# ── NoPostProc ablations (PA Pre-processing, identity Post-processing) ────────

def make_laplace_pa_no_post_proc(W, features, bounds, Z_pub_norm,
                      lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """Laplace+PA without inverse transform in Post-processing (ablation)."""
    d, r, sv, U, sqrt_l, inv_sqrt_l, mu = _pa_setup(W, features, bounds, lambda_n, Z_pub_norm)
    z_pub = (Z_pub_norm - mu) @ U * inv_sqrt_l
    rho   = float(np.percentile(np.abs(z_pub).sum(axis=1), percentile))

    class _NoDecode:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if rng is None:
                rng = np.random.default_rng()
            z_sc = (z - mu) @ U * inv_sqrt_l
            z_cl = _project_l1(z_sc, rho)
            return z_cl + rng.laplace(0, 2.0 * rho / epsilon, z_cl.shape)

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return z_enc

    return _NoDecode()


def make_privunit2_opt_pa_no_post_proc(W, features, bounds, Z_pub_norm,
                                lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """PrivUnit2+PA without inverse transform in Post-processing (ablation)."""
    d, r, sv, U, sqrt_l, inv_sqrt_l, mu = _pa_setup(W, features, bounds, lambda_n, Z_pub_norm)
    z_pub  = (Z_pub_norm - mu) @ U * inv_sqrt_l
    rho_l1 = float(np.percentile(np.abs(z_pub).sum(axis=1), percentile))
    inner  = make_privunit2_opt(z_pub, percentile)

    class _NoDecode:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            x = (z - mu) @ U * inv_sqrt_l
            return inner.encode(_project_l1(x, rho_l1), epsilon, rng)

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return z_enc

    return _NoDecode()


def make_privunitg_mc_no_post_proc(W, features, bounds, Z_pub_norm,
                                lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """PrivUnitG+PA without inverse transform in Post-processing (ablation)."""
    d, r, sv, U, sqrt_l, inv_sqrt_l, mu = _pa_setup(W, features, bounds, lambda_n, Z_pub_norm)
    z_pub  = (Z_pub_norm - mu) @ U * inv_sqrt_l
    rho_l1 = float(np.percentile(np.abs(z_pub).sum(axis=1), percentile))
    inner  = make_privunitg_mc(z_pub, percentile)

    class _NoDecode:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            x = (z - mu) @ U * inv_sqrt_l
            return inner.encode(_project_l1(x, rho_l1), epsilon, rng)

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return z_enc

    return _NoDecode()


# ── NoReshaping ablations (SVD rotation only, no anisotropic scaling) ─────────

def make_laplace_no_reshaping(W, features, bounds, Z_pub_norm,
                         lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """Laplace+PA rotation only, no anisotropic scaling (ablation)."""
    d, r, sv, U, sqrt_l, inv_sqrt_l, mu = _pa_setup(W, features, bounds, lambda_n, Z_pub_norm)
    z_pub = (Z_pub_norm - mu) @ U
    rho   = float(np.percentile(np.abs(z_pub).sum(axis=1), percentile))

    class _NoReshaping:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if rng is None:
                rng = np.random.default_rng()
            x_rot = (z - mu) @ U
            z_cl  = _project_l1(x_rot, rho)
            return z_cl + rng.laplace(0, 2.0 * rho / epsilon, z_cl.shape)

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return z_enc @ U.T + mu

    return _NoReshaping()


def make_privunit2_opt_pa_no_reshaping(W, features, bounds, Z_pub_norm,
                                   lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """PrivUnit2+PA rotation only, no anisotropic scaling (ablation)."""
    d, r, sv, U, sqrt_l, inv_sqrt_l, mu = _pa_setup(W, features, bounds, lambda_n, Z_pub_norm)
    z_pub = (Z_pub_norm - mu) @ U
    inner = make_privunit2_opt(z_pub, percentile)

    class _NoReshaping:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            return inner.encode((z - mu) @ U, epsilon, rng)

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return inner.decode(z_enc) @ U.T + mu

    return _NoReshaping()


def make_privunitg_mc_pa_no_reshaping(W, features, bounds, Z_pub_norm,
                                   lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """PrivUnitG+PA rotation only, no anisotropic scaling (ablation)."""
    d, r, sv, U, sqrt_l, inv_sqrt_l, mu = _pa_setup(W, features, bounds, lambda_n, Z_pub_norm)
    z_pub = (Z_pub_norm - mu) @ U
    inner = make_privunitg_mc(z_pub, percentile)

    class _NoReshaping:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            return inner.encode((z - mu) @ U, epsilon, rng)

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return inner.decode(z_enc) @ U.T + mu

    return _NoReshaping()


# ── NoPreProc ablations (raw-space encode, PA Post-processing) ────────────────

def make_l1clip_laplace_pa_no_pre_proc(W, features, bounds, Z_pub_norm,
                                  lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """L1Clip+Laplace in raw space, decode applies PA inverse transform (ablation)."""
    d, r, sv, U, sqrt_l, inv_sqrt_l, mu = _pa_setup(W, features, bounds, lambda_n, Z_pub_norm)
    rho = float(np.percentile(np.abs(Z_pub_norm).sum(axis=1), percentile))

    class _NoPreProc:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if rng is None:
                rng = np.random.default_rng()
            z_cl = _project_l1(z, rho)
            return z_cl + rng.laplace(0, 2.0 * rho / epsilon, z_cl.shape)

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return (z_enc * sqrt_l) @ U.T + mu

    return _NoPreProc()


def make_l1clip_privunit2_pa_no_pre_proc(W, features, bounds, Z_pub_norm,
                                    lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """L1Clip+PrivUnit2 in raw space, decode applies PA inverse transform (ablation)."""
    d, r, sv, U, sqrt_l, inv_sqrt_l, mu = _pa_setup(W, features, bounds, lambda_n, Z_pub_norm)
    inner = make_l1clip_privunit2(Z_pub_norm, percentile)

    class _NoPreProc:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            return inner.encode(z, epsilon, rng)

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return (z_enc * sqrt_l) @ U.T + mu

    return _NoPreProc()


def make_l1clip_privunitg_pa_no_pre_proc(W, features, bounds, Z_pub_norm,
                                    lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """L1Clip+PrivUnitG in raw space, decode applies PA inverse transform (ablation)."""
    d, r, sv, U, sqrt_l, inv_sqrt_l, mu = _pa_setup(W, features, bounds, lambda_n, Z_pub_norm)
    inner = make_l1clip_privunitg(Z_pub_norm, percentile)

    class _NoPreProc:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            return inner.encode(z, epsilon, rng)

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return (z_enc * sqrt_l) @ U.T + mu

    return _NoPreProc()


# ── L1 clipping geometry before spherical mechanisms ─────────────────────────

def make_pa_l1clip_privunit2(W, features, bounds, Z_pub_norm,
                             lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """PA transform + L1 clip in transformed space + PrivUnit2."""
    d, r, sv, U, sqrt_l, inv_sqrt_l, mu = _pa_setup(W, features, bounds, lambda_n, Z_pub_norm)
    z_pub  = (Z_pub_norm - mu) @ U * inv_sqrt_l
    rho_l1 = float(np.percentile(np.abs(z_pub).sum(axis=1), percentile))
    inner  = make_privunit2_opt(z_pub, percentile)

    class _L1Clip:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            x = (z - mu) @ U * inv_sqrt_l
            return inner.encode(_project_l1(x, rho_l1), epsilon, rng)

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return (inner.decode(z_enc) * sqrt_l) @ U.T + mu

    return _L1Clip()


def make_pa_l1clip_privunitg(W, features, bounds, Z_pub_norm,
                             lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """PA transform + L1 clip in transformed space + PrivUnitG."""
    d, r, sv, U, sqrt_l, inv_sqrt_l, mu = _pa_setup(W, features, bounds, lambda_n, Z_pub_norm)
    z_pub  = (Z_pub_norm - mu) @ U * inv_sqrt_l
    rho_l1 = float(np.percentile(np.abs(z_pub).sum(axis=1), percentile))
    inner  = make_privunitg_mc(z_pub, percentile)

    class _L1Clip:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            x = (z - mu) @ U * inv_sqrt_l
            return inner.encode(_project_l1(x, rho_l1), epsilon, rng)

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return (inner.decode(z_enc) * sqrt_l) @ U.T + mu

    return _L1Clip()


# ── Gaussian preprocessing mechanisms ────────────────────────────────────────

def make_plan_pub(Z_pub_norm: np.ndarray, p: int = 2,
              percentile: float = PERCENTILE, delta: float = DELTA):
    """PLAN: Variance-Aware Private Mean Estimation [24PET].

    Known-variance scenario: σ̂²_i = var(Z_tr[:,i]), μ̃ = mean(Z_tr).
    Scaling: z_s = (z - μ̃) · σ̂^{-1/(p+2)}, clip to L2 ball of radius C.
    Gaussian noise: N(0, σ²I), σ = 2C·sqrt(2 log(1.25/δ)) / ε  → (ε,δ)-LDP.
    """
    mu     = Z_pub_norm.mean(axis=0)
    var    = np.maximum(Z_pub_norm.var(axis=0), 1e-8)
    scale  = var ** (1.0 / (p + 2))
    inv_sc = 1.0 / scale

    Z_pub_s        = (Z_pub_norm - mu) * inv_sc
    C            = float(np.percentile(np.linalg.norm(Z_pub_s, axis=1), percentile))
    gauss_factor = 2.0 * C * np.sqrt(2.0 * np.log(1.25 / delta))

    class _PLAN:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if rng is None:
                rng = np.random.default_rng()
            z_s  = (z - mu) * inv_sc
            nrm  = np.linalg.norm(z_s, axis=1, keepdims=True)
            z_cl = z_s * np.minimum(1.0, C / np.maximum(nrm, 1e-10))
            return z_cl + rng.normal(0, gauss_factor / epsilon, z_cl.shape)

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return z_enc * scale + mu

    return _PLAN()


# ── PLAN(Paper) helpers ────────────────────────────────────────────────────────

def _priv_quantile_1d_zcdp(values: np.ndarray, lo_init: float, hi_init: float,
                            T: int, q: float, sigma: float,
                            rng: np.random.Generator) -> float:
    """1D private quantile via binary search with Gaussian noise (zCDP).

    T iterations of binary search over [lo_init, hi_init].
    Each iteration: compare noisy count to q·n, halve the interval.
    sigma = per-query noise std on raw count (L2 sensitivity = 1).
    """
    lo, hi = lo_init, hi_init
    n = len(values)
    for _ in range(T):
        mid   = (lo + hi) / 2.0
        count = float(np.sum(values <= mid))
        if count + rng.normal(0.0, sigma) >= q * n:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def make_plan_paper(Z_pub_norm: np.ndarray, p: int = 2,
                    T: int = 20, delta: float = DELTA):
    """PLAN(Paper): Algorithm 1 from Aumüller et al. [24PET], exact implementation.

    Differences from make_plan_pub (public-data approximation):
      • μ̃: PrivQuantile (binary-search + Gaussian noise) on private z  — not public mean.
      • C:  PrivQuantile on private ‖y‖₂ norms                         — not public percentile.
      • Privacy: ρ-zCDP, converted from (ε,δ)-DP via Lemma 2.3
                 ε = ρ + 2√(ρ log 1/δ)  ⟹  √ρ = −√(log 1/δ) + √(log 1/δ + ε).
      • Budget split: ρ₁ = ρ₂ = ρ₃ = ρ/3  (Section 7.2).
      • Noise:  η ~ N(0, 2C²/ρ₃ · I)  [zCDP, sensitivity = 2C for clipped sum].
      • σ̂²: from public data (known-variance scenario, Section 3).

    encode() receives all n private samples at once (shape n×d).
    Returns y_clip + η/n so that mean(encode) = mean(y_clip) + η/n,
    matching the paper's estimator μ̃ + (mean(y_clip) + η/n) · Σ̂^{1/(p+2)}.
    """
    n_pub, d = Z_pub_norm.shape

    # σ̂² from public data (known-variance scenario)
    var    = np.maximum(Z_pub_norm.var(axis=0), 1e-8)
    scale  = var ** (1.0 / (p + 2))   # Σ̂^{1/(p+2)} diagonal
    inv_sc = 1.0 / scale               # Σ̂^{-1/(p+2)} diagonal

    # Coordinate bound M for μ̃ PrivQuantile search range [−M, M]
    M = float(np.percentile(np.abs(Z_pub_norm), 99.0))
    M = max(M, 1e-6)

    # Upper bound for ‖y‖₂ for C PrivQuantile search range [0, M_norm]
    Z_pub_y = (Z_pub_norm - Z_pub_norm.mean(axis=0)) * inv_sc
    M_norm  = float(np.percentile(np.linalg.norm(Z_pub_y, axis=1), 99.9)) * 2.0
    M_norm  = max(M_norm, 1e-6)

    _cache: dict[float, tuple] = {}

    class _PLANPaper:
        def __init__(self):
            self._mu_tilde = np.zeros(d)

        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if rng is None:
                rng = np.random.default_rng()
            n = z.shape[0]

            # (ε,δ) → ρ-zCDP: Lemma 2.3
            log_inv_d = np.log(1.0 / delta)
            rho       = (-np.sqrt(log_inv_d) + np.sqrt(log_inv_d + epsilon)) ** 2

            if epsilon not in _cache:
                rho1 = rho2 = rho3 = rho / 3.0
                # σ for μ̃ PrivQuantile: d dims × T iters each ⟹ ρ/(d·T) per query
                sigma_mu = np.sqrt(d * T / (2.0 * rho1)) if rho1 > 0 else 1e9
                # σ for C  PrivQuantile: 1D × T iters ⟹ ρ/T per query
                sigma_c  = np.sqrt(T / (2.0 * rho2))     if rho2 > 0 else 1e9
                _cache[epsilon] = (rho1, rho2, rho3, sigma_mu, sigma_c)
                print(f"  PLAN-Paper: ε={epsilon}, ρ={rho:.5f}, "
                      f"σ_μ={sigma_mu:.2f}, σ_C={sigma_c:.2f}")

            rho1, rho2, rho3, sigma_mu, sigma_c = _cache[epsilon]

            # Step 4: private coordinate-wise median μ̃ (Algorithm 1, line 4)
            mu_tilde = np.array([
                _priv_quantile_1d_zcdp(z[:, i], -M, M, T, 0.5, sigma_mu, rng)
                for i in range(d)
            ])
            self._mu_tilde = mu_tilde

            # Step 5: y^(j) = (x^(j) − μ̃) ⊙ Σ̂^{-1/(p+2)}
            y = (z - mu_tilde) * inv_sc

            # Step 6: private clipping radius C at quantile (n − √n)/n
            k      = np.sqrt(n)
            q_clip = float(np.clip((n - k) / n, 0.0, 1.0))
            norms  = np.linalg.norm(y, axis=1)
            C = _priv_quantile_1d_zcdp(norms, 0.0, M_norm, T, q_clip, sigma_c, rng)
            C = max(C, 1e-8)

            # Step 7: η ~ N(0, 2C²/ρ₃ · I), sensitivity = 2C, ρ₃-zCDP
            sigma_noise = C * np.sqrt(2.0 / rho3) if rho3 > 0 else 1e9
            eta = rng.normal(0.0, sigma_noise, (d,))

            # Clip y to L2 ball of radius C
            nrm    = np.linalg.norm(y, axis=1, keepdims=True)
            y_clip = y * np.minimum(1.0, C / np.maximum(nrm, 1e-10))

            # Return y_clip + η/n: averaging gives mean(y_clip) + η/n ✓
            return y_clip + eta / n

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return z_enc * scale + self._mu_tilde

    return _PLANPaper()


# ── Inst-Opt ────────────────────────────────────────────────────────────────

def make_inst_opt(Z_pub_norm: np.ndarray, delta: float = DELTA):
    """Inst-Opt: instance-optimal LDP mean estimation (Huang et al., NeurIPS 2021).

    Steps (Section 3.3 + Section 5 LDP adaptation):
      1. Random Hadamard rotation: x̂ = (1/√d) H D x,  D = diag(±1) shared randomness.
      2. Per-dim median shift: c̃ = median(X̂_pub, axis=0),  x̃ = x̂ − c̃.
      3. Optimal L2 clip radius C*(ε): minimises bias-variance tradeoff on public norms.
         ∂E/∂C = 0  ⟹  fraction{‖x̃‖>C} = σ_unit(ε) · √(d/n),
         where σ_unit(ε) = AGM(ε, δ, sensitivity=2).
      4. AGM noise: z_clip + N(0, σ²I),  σ = AGM(ε, δ, 2C*).
    Decode: un-shift then un-rotate (both linear, commute with averaging).
    (ε,δ)-LDP.  Requires d to be a power of 2.
    """
    n, d = Z_pub_norm.shape
    assert (d & (d - 1)) == 0, f"Inst-Opt requires d = power of 2, got {d}"

    # Shared public randomness: fixed diagonal sign flip
    d_vec = np.random.default_rng(0).choice(np.array([-1.0, 1.0]), size=d)
    H = _hadamard(d).astype(float)  # H @ H = d · I, H symmetric

    # Rotate public data: Z_hat[i] = (1/√d) · H @ (d_vec ⊙ x[i])
    # Batch form: Z_hat = (X * d_vec) @ H / √d  (H symmetric ⟹ H = H.T)
    Z_hat   = (Z_pub_norm * d_vec) @ H / np.sqrt(d)
    c_tilde = np.median(Z_hat, axis=0)          # per-dim shift center
    norms   = np.linalg.norm(Z_hat - c_tilde, axis=1)

    _cache: dict[float, tuple[float, float]] = {}

    class _ShiftedCM:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if rng is None:
                rng = np.random.default_rng()
            if epsilon not in _cache:
                # σ_unit = AGM(ε, δ, 2) so that σ(C) = C · σ_unit
                sigma_unit = _agm_sigma(epsilon, delta, 2.0)
                frac_above = float(np.clip(sigma_unit * np.sqrt(d / n), 0.0, 1.0))
                q          = float(np.clip(1.0 - frac_above, 0.0, 1.0))
                C_opt      = max(float(np.quantile(norms, q)), 1e-8)
                sigma      = _agm_sigma(epsilon, delta, 2.0 * C_opt)
                _cache[epsilon] = (C_opt, sigma)
                print(f"  ShiftedCM: ε={epsilon}, C={C_opt:.4f} "
                      f"(q={100*q:.1f}th pct), σ={sigma:.4f}")
            C_opt, sigma = _cache[epsilon]

            z_hat  = (z * d_vec) @ H / np.sqrt(d)          # rotate
            z_til  = z_hat - c_tilde                        # shift
            nrm    = np.linalg.norm(z_til, axis=1, keepdims=True)
            z_clip = z_til * np.minimum(1.0, C_opt / np.maximum(nrm, 1e-10))
            return z_clip + rng.normal(0, sigma, z_clip.shape)

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            # un-shift then un-rotate: z = (1/√d) · (x̂ @ H) ⊙ d_vec
            return (z_enc + c_tilde) @ H * d_vec / np.sqrt(d)

    return _ShiftedCM()


# ── Spherical mechanisms ──────────────────────────────────────────────────────

def make_privunit2_opt(Z_pub_norm: np.ndarray, percentile: float = PERCENTILE):
    """PrivUnit2: optimal step-function mechanism on S^{d-1}. Pure ε-LDP.

    q(z|v) ∝ e^ε · 1[⟨z,v⟩ ≥ γ*] + 1[⟨z,v⟩ < γ*]
    where γ* satisfies the self-consistency condition A(γ*,ε) = γ*.
    Decode scale ρ/γ* gives unbiased estimate of ρ·(z/‖z‖).
    """
    rho = float(np.percentile(np.linalg.norm(Z_pub_norm, axis=1), percentile))
    d   = Z_pub_norm.shape[1]
    a   = (d - 1) / 2.0
    _cache: dict = {}

    class _PrivUnit2:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if rng is None:
                rng = np.random.default_rng()
            if epsilon not in _cache:
                gamma = _pu2_find_gamma(epsilon, a)
                P     = _pu2_cap_prob(gamma, a)
                p_cap = np.exp(epsilon) * P / (np.exp(epsilon) * P + (1.0 - P))
                _cache[epsilon] = (gamma, p_cap)
                print(f"  PrivUnit2(Opt): ε={epsilon}, γ*={gamma:.4f}, "
                      f"p_cap={p_cap:.4f}, A=γ*={gamma:.4f}")
            gamma, p_cap = _cache[epsilon]

            n   = z.shape[0]
            u   = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-10)
            use = rng.random(n) < p_cap
            out = np.empty_like(z)

            for from_cap, mask in [(True, use), (False, ~use)]:
                idx = np.where(mask)[0]
                if not len(idx): continue
                out[idx] = _sphere_from_t(u[idx],
                                          _pu2_sample_t(gamma, len(idx), from_cap, a, rng),
                                          rng)
            return (rho / gamma) * out

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return z_enc

    return _PrivUnit2()


def make_privunitg_mc(Z_pub_norm: np.ndarray, percentile: float = PERCENTILE):
    """PrivUnitG (Asi et al., ICML 2022): Gaussian step-function mechanism. Pure ε-LDP.

    Step function on ⟨g,v⟩ ~ N(0,1) in ambient Gaussian space:
        q(g→z|v) ∝ e^ε · 1[⟨g,v⟩≥γ*] + 1[⟨g,v⟩<γ*],  z = g/‖g‖.
    Optimal γ* maximizes A(γ,ε) = E_q[⟨z,v⟩], found via MC grid search.
    """

    rho = float(np.percentile(np.linalg.norm(Z_pub_norm, axis=1), percentile))
    d   = Z_pub_norm.shape[1]

    def _find_params(eps: float, n_mc: int = 500_000):
        g   = np.random.randn(n_mc, d)
        t   = g[:, 0]
        cos = t / np.linalg.norm(g, axis=1)

        def _A(gamma):
            mask = t >= gamma
            num  = np.exp(eps) * cos[mask].sum() + cos[~mask].sum()
            den  = np.exp(eps) * mask.sum()      + (~mask).sum()
            return float(num / den) if den > 0 else 0.0

        gammas = np.linspace(-3.0, 3.0, 600)
        best   = int(np.argmax([_A(gm) for gm in gammas]))
        lo, hi = gammas[max(0, best - 5)], gammas[min(len(gammas) - 1, best + 5)]
        gammas2 = np.linspace(lo, hi, 300)
        gamma   = float(gammas2[np.argmax([_A(gm) for gm in gammas2])])
        A       = _A(gamma)
        Z       = float(np.exp(eps) * _snorm.sf(gamma) + _snorm.cdf(gamma))
        p_cap   = float(np.exp(eps) * _snorm.sf(gamma) / Z)
        return gamma, p_cap, A

    def _sample_g_batch(u: np.ndarray, gamma: float, from_cap: bool,
                        rng: np.random.Generator) -> np.ndarray:
        n = u.shape[0]
        t = (_truncnorm.rvs(gamma,    np.inf, size=n, random_state=rng) if from_cap else
             _truncnorm.rvs(-np.inf, gamma,   size=n, random_state=rng))
        G  = rng.standard_normal((n, d))
        G -= (G * u).sum(axis=1, keepdims=True) * u
        g  = t[:, None] * u + G
        return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-10)

    _cache: dict = {}

    class _PrivUnitG:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if rng is None:
                rng = np.random.default_rng()
            if epsilon not in _cache:
                gamma, p_cap, A = _find_params(epsilon)
                _cache[epsilon] = (gamma, p_cap, A)
                print(f"  PrivUnitG(MC): ε={epsilon}, γ*={gamma:.4f}, "
                      f"p_cap={p_cap:.4f}, A={A:.4f}")
            gamma, p_cap, A = _cache[epsilon]

            n   = z.shape[0]
            u   = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-10)
            use = rng.random(n) < p_cap
            out = np.empty_like(z)

            for from_cap, mask in [(True, use), (False, ~use)]:
                idx = np.where(mask)[0]
                if not len(idx): continue
                out[idx] = _sample_g_batch(u[idx], gamma, from_cap, rng)
            return (rho / A) * out

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return z_enc

    return _PrivUnitG()


def make_privunitg_paper(Z_pub_norm: np.ndarray, percentile: float = PERCENTILE):
    """PrivUnitG — paper-exact (Asi et al., ICML 2022).

    Matches the paper algorithm exactly: outputs g directly (no sphere
    normalization). Optimal γ* found analytically via dA/dγ=0:

        γ·Z(γ) = (eᵉ−1)·φ(γ),  Z = eᵉ·P(t≥γ) + P(t<γ),  t ~ N(0,1)

    Unlike make_privunitg_mc, no Monte Carlo estimation is used.
    """
    rho = float(np.percentile(np.linalg.norm(Z_pub_norm, axis=1), percentile))
    d   = Z_pub_norm.shape[1]

    def _A(gamma: float, eps: float) -> float:
        phi = float(_snorm.pdf(gamma))
        P   = float(_snorm.sf(gamma))
        Z   = np.exp(eps) * P + (1.0 - P)
        return (np.exp(eps) - 1.0) * phi / Z if Z > 1e-15 else 0.0

    def _find_params(eps: float):
        # dA/dγ = 0  ↔  γ·Z(γ) − (eᵉ−1)·φ(γ) = 0
        def _deriv(gamma):
            phi = float(_snorm.pdf(gamma))
            P   = float(_snorm.sf(gamma))
            Z   = np.exp(eps) * P + (1.0 - P)
            return gamma * Z - (np.exp(eps) - 1.0) * phi

        gamma = float(_brentq(_deriv, -10.0, 10.0, xtol=1e-10))
        A     = _A(gamma, eps)
        P     = float(_snorm.sf(gamma))
        Z     = np.exp(eps) * P + (1.0 - P)
        p_cap = np.exp(eps) * P / Z
        return gamma, p_cap, A

    def _sample_g_batch(u: np.ndarray, gamma: float, from_cap: bool,
                        rng: np.random.Generator) -> np.ndarray:
        n = u.shape[0]
        t = (_truncnorm.rvs(gamma,    np.inf, size=n, random_state=rng) if from_cap else
             _truncnorm.rvs(-np.inf, gamma,   size=n, random_state=rng))
        G  = rng.standard_normal((n, d))
        G -= (G * u).sum(axis=1, keepdims=True) * u
        return t[:, None] * u + G  # raw Gaussian output — no sphere normalization

    _cache: dict = {}

    class _PrivUnitGAsi22:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if rng is None:
                rng = np.random.default_rng()
            if epsilon not in _cache:
                gamma, p_cap, A = _find_params(epsilon)
                _cache[epsilon] = (gamma, p_cap, A)
                print(f"  PrivUnitG(Paper): ε={epsilon}, γ*={gamma:.4f}, "
                      f"p_cap={p_cap:.4f}, A={A:.4f}")
            gamma, p_cap, A = _cache[epsilon]

            n   = z.shape[0]
            u   = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-10)
            use = rng.random(n) < p_cap
            out = np.empty_like(z)

            for from_cap, mask in [(True, use), (False, ~use)]:
                idx = np.where(mask)[0]
                if not len(idx): continue
                out[idx] = _sample_g_batch(u[idx], gamma, from_cap, rng)
            return (rho / A) * out

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return z_enc

    return _PrivUnitGAsi22()


# ── Jacobian row-space computation ────────────────────────────────────────────

def compute_jacobian_row_space(model: torch.nn.Module, Z_pub: torch.Tensor,
                               n_samples: int = 500,
                               sv_gap_ratio: float = 1e-5) -> np.ndarray:
    """Aggregate Jacobians and extract a clean row-space basis matrix.

    Stacks per-sample Jacobians into B ∈ ℝ^{(n_samples*K) × D}, then uses SVD
    with a gap-based rank threshold to separate row space from numerical null space.

    Returns W_eff ∈ ℝ^{r × D}  (rank-r).
    """
    model.eval()
    device = next(model.parameters()).device
    idx    = torch.randperm(len(Z_pub))[:n_samples]
    Z_sub  = Z_pub[idx].to(device)
    K      = model(Z_sub[:1]).shape[1]

    rows = []
    for i in range(len(Z_sub)):
        z   = Z_sub[i:i+1].clone().float().requires_grad_(True)
        out = model(z)
        for k in range(K):
            grad = torch.autograd.grad(out[0, k], z,
                                       retain_graph=(k < K - 1))[0]
            rows.append(grad.detach().cpu().numpy().flatten())

    B = np.array(rows, dtype=np.float64)
    _, s, Vt = np.linalg.svd(B, full_matrices=False)
    r = int(np.sum(s > sv_gap_ratio * s[0]))
    print(f"  Jacobian: B{B.shape}, effective rank={r} "
          f"(sv gap at {s[r-1]:.2f} → {s[r]:.4f}, ratio={s[r]/s[0]:.2e})")
    return np.diag(s[:r]) @ Vt[:r]


# ── Cheng et al. (ICML 2022) ─────────────────────────────────────────────────

def _cheng_water_fill(eigvals: np.ndarray, alpha: float) -> tuple[np.ndarray, int]:
    """Water-filling for optimal σ²_i (Prop 4.6). Returns (sigma2[:Zp], Zp)."""
    sqrt_lam = np.sqrt(eigvals)
    for Zp in range(len(eigvals), 0, -1):
        S = sqrt_lam[:Zp].sum()
        if S < 1e-12:
            continue
        sigma2 = sqrt_lam[:Zp] / S * (1.0 + Zp * alpha) - alpha
        if sigma2[-1] >= 0.0:
            return np.maximum(sigma2, 0.0), Zp
    return np.array([1.0]), 1


def _cheng_setup(W, features, bounds, Z_pub_norm, percentile):
    """Cholesky whitening setup for make_task_aware.

    Returns (d, mu, L, L_inv, r, P) where:
      mu:    public data mean
      L:     Cholesky factor of Cov(X_pub)
      L_inv: inverse of L
      r:     `percentile`-th percentile of ||h||_2 over public data
      P:     task matrix in whitened space (W_norm @ L)
    """
    d      = len(features)
    mu     = Z_pub_norm.mean(axis=0)
    Z_c    = Z_pub_norm - mu
    Cov    = (Z_c.T @ Z_c) / len(Z_pub_norm) + 1e-8 * np.eye(d)
    L      = np.linalg.cholesky(Cov)
    L_inv  = np.linalg.inv(L)
    r      = float(np.percentile(np.linalg.norm(Z_c @ L_inv.T, axis=1), percentile))
    scale  = np.array([bounds[features[j]][1] - bounds[features[j]][0] for j in range(d)])
    W_norm = np.asarray(W, dtype=float) * scale
    W2     = W_norm.reshape(1, -1) if W_norm.ndim == 1 else W_norm
    P      = W2 @ L
    return d, mu, L, L_inv, r, P


def make_task_aware(W, features, bounds, Z_pub_norm, percentile=PERCENTILE):
    """Cheng et al. (2022, ICML) task-aware LDP mechanism for linear MSE settings.

    Encoder-decoder framework (Section 4.1):
      1. Whiten x via Cholesky of Cov(Z_pub): h = (x - μ) @ L^{-T}.
      2. Rotate h into the task-relevant basis Q (eigenvectors of P^T P,
         P = W_norm @ L).
      3. Scale each dimension by σ_i from the water-filling solution (Prop. 4.6):
         σ²_i = (√λ_i / ΣZ'√λ_j)(1 + Z'·8r²/ε²) − 8r²/ε²  for i ≤ Z', 0 o.w.
      4. Add Laplace noise with sensitivity 2r (scale = 2r/ε) per encoded dim.
      5. Decode: x̂ = L · D · φ̃ + μ  where D = E^T(EE^T + 8r²/ε² · I)^{-1}.
    """
    d, mu, L, L_inv, r, P = _cheng_setup(W, features, bounds, Z_pub_norm, percentile)

    PtP                = P.T @ P
    eigvals_raw, Q_raw = np.linalg.eigh(PtP)
    idx     = np.argsort(eigvals_raw)[::-1]
    eigvals = np.maximum(eigvals_raw[idx], 0.0)
    Q       = Q_raw[:, idx]

    _cache:    dict[float, tuple] = {}
    _last_eps: list               = [None]

    class _Cheng:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if epsilon not in _cache:
                alpha      = 8.0 * r ** 2 / epsilon ** 2
                sigma2, Zp = _cheng_water_fill(eigvals, alpha)
                E          = np.sqrt(sigma2)[:, None] * Q[:, :Zp].T  # (Zp, d)
                D          = E.T @ np.linalg.inv(E @ E.T + alpha * np.eye(Zp))
                _cache[epsilon] = (D, Zp, E)
            _last_eps[0] = epsilon
            D, Zp, E     = _cache[epsilon]

            if rng is None:
                rng = np.random.default_rng()
            h     = (z - mu) @ L_inv.T
            norms = np.linalg.norm(h, axis=1, keepdims=True)
            h     = h * np.minimum(1.0, r / np.maximum(norms, 1e-12))
            phi   = h @ E.T
            phi_noisy     = phi + rng.laplace(0.0, 2.0 * r / epsilon, phi.shape)
            z_out         = np.zeros((len(z), d))
            z_out[:, :Zp] = phi_noisy
            return z_out

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            if _last_eps[0] is None:
                raise RuntimeError("decode called before encode")
            D, Zp, _ = _cache[_last_eps[0]]
            return z_enc[:, :Zp] @ D.T @ L.T + mu

    return _Cheng()


# ── Mechanism registry ────────────────────────────────────────────────────────

def build_mechs(dim: int, W: np.ndarray, Z_np: np.ndarray) -> dict:
    """Build LDP mechanism dict."""
    rho_l1   = float(np.percentile(np.abs(Z_np).sum(axis=1), PERCENTILE))
    rho_l2   = float(np.percentile(np.linalg.norm(Z_np, axis=1), PERCENTILE))

    features = [str(j) for j in range(dim)]
    bounds   = {str(j): (0.0, 1.0) for j in range(dim)}
    anr_kw   = dict(W=W, features=features,
                    bounds=bounds,
                    Z_pub_norm=Z_np, lambda_n=LAMBDA_N, percentile=PERCENTILE)

    mechs = {
        "NoNoise":          NoNoise(),
        "Laplace(L1)":      L1ClipLaplace(rho_l1),
        "Laplace+PA":   make_laplace_pa(**anr_kw),
        "AGM":              GaussianMech(rho_l2, DELTA),
        "AGM+PA":   make_agm_pa(**anr_kw, delta=DELTA),
        "PrivUnit2(Opt)":        make_privunit2_opt(Z_np),
        "PrivUnit2(Opt)+PA": make_privunit2_opt_pa(**anr_kw),
        "PrivUnitG(MC)":        make_privunitg_mc(Z_np),
        "PrivUnitG(Paper)":       make_privunitg_paper(Z_np),
        "PrivUnitG(MC)+PA":      make_privunitg_mc_pa(**anr_kw),
        "PrivUnitG(Paper)+PA": make_privunitg_paper_pa(**anr_kw),
        "CW(Laplace)+PA":   make_cw_laplace_pa(**anr_kw),
        "CW(AGM)+PA":   make_cw_agm_pa(**anr_kw, delta=DELTA),
        "PLAN(Pub)":         make_plan_pub(Z_np, delta=DELTA),
        "PLAN(Paper)":       make_plan_paper(Z_np, delta=DELTA),
        "Task-Aware":    make_task_aware(W=W, features=features, bounds=bounds, Z_pub_norm=Z_np),
    }
    # Inst-Opt requires d = power of 2 (Hadamard transform)
    if (dim & (dim - 1)) == 0:
        mechs["Inst-Opt"] = make_inst_opt(Z_np, delta=DELTA)
    return mechs


def build_mechs_ablation(dim: int, W: np.ndarray, Z_np: np.ndarray) -> dict:
    """Ablation-study LDP mechanism dict.

      • NoPostProc:   full PA Pre-processing, identity Post-processing
      • NoReshaping:  SVD rotation only, no anisotropic scaling
      • NoPreProc:    raw-space encode, full PA Post-processing
      • L1 clip:      L1 clipping geometry before spherical mechanisms
    """
    features = [str(j) for j in range(dim)]
    bounds   = {str(j): (0.0, 1.0) for j in range(dim)}
    anr_kw   = dict(W=W, features=features,
                    bounds=bounds,
                    Z_pub_norm=Z_np, lambda_n=LAMBDA_N, percentile=PERCENTILE)

    rho_l1 = float(np.percentile(np.abs(Z_np).sum(axis=1), PERCENTILE))

    return {
        # ── Full PA (baseline upper) ──
        "Laplace+PA":                   make_laplace_pa(**anr_kw),
        "PrivUnit2(Opt)+PA":            make_privunit2_opt_pa(**anr_kw),
        "PrivUnitG(MC)+PA":             make_privunitg_mc_pa(**anr_kw),

        # ── L1 clipping geometry ──
        "Laplace(L1)":                  L1ClipLaplace(rho_l1),
        "PrivUnit2(L1,Opt)":            make_l1clip_privunit2(Z_np, PERCENTILE),
        "PrivUnitG(L1,MC)":             make_l1clip_privunitg(Z_np, PERCENTILE),

        # ── NoPreProc ablations ──
        "Laplace(L1)/NoPreProc":        make_l1clip_laplace_pa_no_pre_proc(**anr_kw),
        "PrivUnit2/NoPreProc":          make_l1clip_privunit2_pa_no_pre_proc(**anr_kw),
        "PrivUnitG/NoPreProc":          make_l1clip_privunitg_pa_no_pre_proc(**anr_kw),

        # ── NoPostProc ablations ──
        "Laplace+PA/NoPostProc":        make_laplace_pa_no_post_proc(**anr_kw),
        "PrivUnit2(Opt)+PA/NoPostProc": make_privunit2_opt_pa_no_post_proc(**anr_kw),
        "PrivUnitG(MC)+PA/NoPostProc":  make_privunitg_mc_no_post_proc(**anr_kw),

        # ── NoReshaping ablations ──
        "Laplace+PA/NoReshaping":        make_laplace_no_reshaping(**anr_kw),
        "PrivUnit2(Opt)/NoReshaping":    make_privunit2_opt_pa_no_reshaping(**anr_kw),
        "PrivUnitG(MC)/NoReshaping":     make_privunitg_mc_pa_no_reshaping(**anr_kw),
    }
