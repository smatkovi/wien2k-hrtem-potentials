#!/usr/bin/env python3
"""
plot_kurasch_fig3.py — reproduce Kurasch et al. 2011, Fig. 3 from a lapw5 plane

Input is a lapw5 `case.rho` of an x-z plane through a nucleus: x laterally at
a fine rate (a multiple of 30 ppÅ, here 600 ppÅ over ±0.1167 Å), z along the
beam over the full cell at the production rate of 30 ppÅ, with the nucleus on
a grid point in both directions.  The lapw5 input used for graphene is in
reference/lapw5_fig3/ (units ATU -> Ry, NONORTHO).

The figure follows the definitions of the paper (p. 397 and the caption):

green : projected potential V_z(x) = sum_z V_s(x,z) dz at the fine lateral rate
        with z at the production rate of 30 ppÅ ("a constant sampling rate of
        30 ppÅ in the z-direction ..., but a much higher rate in the
        perpendicular direction")
red   : the same with the nearest-neighbour cutoff, applied for every lateral
        position x on the 30 ppÅ grid through that column (neighbours at
        x ± 1/30 Å and z ± 1/30 Å; "highest value", see vcoul2tiff_ulm.py,
        apply_nn_cutoff).  Only columns whose z = 0 voxel lies within ~2 mÅ of
        the nucleus exceed the cutoff, hence the narrow notch.
blue  : "the mean value of the green line at a resolution of 30 ppÅ" (caption),
        i.e. the 1-D mean of the green curve over each 30 ppÅ pixel.  The z = 0
        sample of the green curve diverges like 1/|x| at the core (lapw5 holds
        V constant below R0), so this mean depends on the lateral sampling; it
        is evaluated in the continuum limit.
        Outline (not in the paper): the true mean of the projected potential
        over the pixel column (x, y inside the pixel, z over the cell).  Near
        the core the spherical L=0 part of V_s inside the muffin-tin sphere
        (from case.vcoul) is integrated finely along z; the smooth remainder
        (non-spherical part, interstitial) is taken from the lapw5 plane.
black : Doyle–Turner (1968) projected potential of carbon (dotted, as in the
        paper) and Kirkland's (dash-dot, for comparison)

Usage:
    python3 plot_kurasch_fig3.py Graphene.rho_fig3 kurasch_fig3_graphene.png [case.vcoul case.struct]
    (case.vcoul / case.struct default to <case>.vcoul / <case>.struct next to the rho file)
"""
import re
import sys
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import k0
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RY_TO_V = 13.605693122994
BOHR = 0.529177210903
A0E = 0.529177210903 * 14.399645  # a0*e  [V*A^2]
CUTOFF_V = 50000.0

rho_path = sys.argv[1]
out_png = sys.argv[2]
_case = Path(rho_path).parent / Path(rho_path).name.split('.rho')[0]
vcoul_path = sys.argv[3] if len(sys.argv) > 3 else f"{_case}.vcoul"
struct_path = sys.argv[4] if len(sys.argv) > 4 else f"{_case}.struct"

# ---------------------------------------------------------------- read rho
with open(rho_path) as f:
    head = f.readline().split()
    vals = np.array([float(x) for line in f for x in line.split()])
npx, npy = int(head[0]), int(head[1])
relx, rely = float(head[2]) * BOHR, float(head[3]) * BOHR
vals = vals[:npx * npy]
rough = lambda d: np.mean(np.abs(np.diff(d, axis=0))) + np.mean(np.abs(np.diff(d, axis=1)))
dC = vals.reshape((npx, npy), order='C')
dF = vals.reshape((npx, npy), order='F')
V = dC if rough(dC) <= rough(dF) else dF
print(f"plane {npx} x {npy}, extent {relx:.4f} x {rely:.4f} A, order {'C' if V is dC else 'F'}")
x = np.linspace(-relx / 2, relx / 2, npx)
z = np.linspace(-rely / 2, rely / 2, npy)
dx, dz = x[1] - x[0], z[1] - z[0]
print(f"dx = {dx*1000:.3f} mA ({1/dx:.1f} ppA), dz = {dz*1000:.3f} mA ({1/dz:.1f} ppA)")

imin = np.unravel_index(np.argmin(V), V.shape)
print(f"most negative voxel {V.min():.1f} Ry = {V.min()*RY_TO_V:.0f} V at index {imin} -> x={x[imin[0]]:.4f} z={z[imin[1]]:.4f} A")
print(f"vacuum level (max) {V.max():.5f} Ry = {V.max()*RY_TO_V:.3f} V")

# paper normalisation: V_s = -V_C, shifted so the smallest value (vacuum) is 0
Vs = (V.max() - V) * RY_TO_V                     # [V], >= 0
Vs = Vs[:, :npy - 1]                             # drop duplicated z end plane
z = z[:npy - 1]
nz = Vs.shape[1]
iz0 = int(np.argmin(np.abs(z)))
ix0 = int(np.argmin(np.abs(x)))
print(f"nucleus column ix0={ix0} (x={x[ix0]:.4f}), iz0={iz0} (z={z[iz0]:.4f}); V_s there = {Vs[ix0, iz0]:.0f} V")
print(f"z-neighbours of nucleus: {Vs[ix0, iz0-1]:.0f} / {Vs[ix0, iz0+1]:.0f} V ; x-neighbours at 30 ppA: {Vs[ix0-20, iz0]:.0f} / {Vs[ix0+20, iz0]:.0f} V; at 600 ppA: {Vs[ix0-1, iz0]:.0f} V")

# ---------------------------------------------------------------- green
Vz_green = Vs.sum(axis=1) * dz                   # [V*A]
print(f"green: V_z(0) = {Vz_green[ix0]:.0f} V*A  (without nucleus voxel: {(Vz_green[ix0] - Vs[ix0, iz0]*dz):.0f} V*A)")

# ---------------------------------------------------------------- blue boxes (30 ppA pixels centred on 0, 1/30, 2/30, 3/30)
px = 1.0 / 30                                     # production pixel (30 ppÅ)
sub = int(round(px / dx))                         # fine samples per pixel
assert abs(sub * dx - px) < 1e-6, "lateral rate must be a multiple of 30 ppÅ"
edges = [(-px / 2, px / 2), (px / 2, 3 * px / 2), (3 * px / 2, 5 * px / 2), (5 * px / 2, 7 * px / 2)]

# (a) paper-style pixel value = area mean of the green curve over the 30 ppA pixel.
# V_z(r) is taken from the green curve (radially symmetric this close to the core);
# the sample at r = 0 is the WIEN2k r0 cusp and is not representative of the pixel,
# so for r < dx the log-divergence  V_z ~ A - B ln r  is fitted to the first samples.
# For r < dz the green samples are dominated by the single z = 0 plane, so this
# overestimates the central pixel; it is only kept for comparison with the paper.
rpos = x[ix0 + 1:]; vpos = Vz_green[ix0 + 1:]
B_, A_ = np.polyfit(np.log(rpos[:4]), vpos[:4], 1)  # V = A + B ln r (B < 0)
print(f"log fit near core: V_z(r) = {A_:.0f} + ({B_:.0f})*ln r  ->  V_z(dx)={A_+B_*np.log(dx):.0f} (sample {vpos[0]:.0f})")


def vz_radial(r):
    r = np.asarray(r, float)
    out = np.interp(r, rpos, vpos)
    small = r < rpos[0]
    out[small] = A_ + B_ * np.log(np.maximum(r[small], 1e-7))
    return out


# (b) ideal pixel value = true mean of the projected potential over the pixel
# column.  Inside the muffin-tin sphere of the nucleus, V_s = f(r) + remainder with
#   f(r) = [V_C,00(RMT) - V_C,00(r)] Y_00   (Ry -> V; V_s = -V_C, f(RMT) = 0),
# the spherical L=0 channel of case.vcoul (stored as V_00*r^2 on the radial mesh of
# case.struct).  f carries the whole nuclear cusp; it does not depend on the vacuum
# shift and is continuous at the sphere boundary.  The remainder (non-spherical
# part, constant V_s,00(RMT), interstitial) is smooth, so its z-sum on the lapw5
# plane is accurate: R(rho) = green(rho) - sum_z f dz.  f itself is integrated
# along z exactly, with z = rho*sinh(t) (dz = r dt) removing the 1/r cusp.
def read_struct_mt(path):
    """(NPT, R0, RMT) of every inequivalent atom of a WIEN2k case.struct."""
    return [(int(l[15:20]), float(l[25:35]), float(l[40:50]))
            for l in Path(path).read_text().splitlines() if 'NPT=' in l]


def read_vcoul_l0(path, npts):
    """V_00(r)*r^2 [Ry*bohr^2] (L=0, M=0 channel) of every atom in case.vcoul."""
    lines = Path(path).read_text().splitlines()
    if not any('r**2' in l for l in lines[:5]):
        sys.exit(f"Error: {path} does not look like a case.vcoul (VC(L,M)*r**2 header)")
    out, iat, i = {}, None, 0
    while i < len(lines):
        m = re.search(r'ATOMNUMBER\s*=\s*(\d+)', lines[i])
        if m:
            iat = int(m.group(1))
        elif (iat is not None and iat not in out
              and re.search(r'VLM\(R\)\s+FOR\s+L\s+0\s+M=\s*0\b', lines[i])):
            vals = []
            i += 1
            while len(vals) < npts[iat - 1]:
                vals += [float(v) for v in re.findall(r'[-+]?\d*\.\d+E[-+]\d+', lines[i])]
                i += 1
            out[iat] = np.array(vals[:npts[iat - 1]])
            continue
        i += 1
    return [out[k] for k in sorted(out)]


mt = read_struct_mt(struct_path)
v00r2_all = read_vcoul_l0(vcoul_path, [n for n, _, _ in mt])
Y00 = 1.0 / np.sqrt(4 * np.pi)
# the plane's nucleus voxel is lapw5's value at r = R0 -> identifies the atom
iat = int(np.argmin([abs(Y00 * t[0] / r0_**2 - V.min()) for t, (_, r0_, _) in zip(v00r2_all, mt)]))
npt_mt, r0_mt, rmt_mt = mt[iat]
v00r2 = v00r2_all[iat]
r_mesh = r0_mt * np.exp(np.arange(npt_mt) * np.log(rmt_mt / r0_mt) / (npt_mt - 1))   # [bohr]
spl00 = CubicSpline(r_mesh, v00r2)
VC_rmt = Y00 * v00r2[-1] / rmt_mt**2                                             # [Ry]
print(f"L=0 sphere part: atom {iat + 1} of {vcoul_path} (RMT = {rmt_mt * BOHR:.4f} A), "
      f"V_C(R0) = {Y00 * v00r2[0] / r0_mt**2:.1f} Ry vs plane nucleus voxel {V.min():.1f} Ry")


def v00r2_at(rb):
    """V_00*r^2 at r [bohr]; below R0 the Coulomb form V_00*r^2 ~ r."""
    rb = np.asarray(rb, float)
    return np.where(rb >= r0_mt, spl00(np.maximum(rb, r0_mt)), v00r2[0] * rb / r0_mt)


def f_sph(r):
    """f(r) [V] on the lapw5 plane (r [A]); lapw5 evaluates r < R0 at R0."""
    rb = np.maximum(np.asarray(r, float) / BOHR, r0_mt)
    return np.where(rb < rmt_mt, (VC_rmt - Y00 * v00r2_at(rb) / rb**2) * RY_TO_V, 0.0)


def f_proj(rho):
    """Exact integral of f along z through the sphere at lateral distance rho [A] -> [V*A]."""
    rmt_A = rmt_mt * BOHR
    t = np.linspace(0.0, np.arcsinh(np.sqrt(rmt_A**2 - rho**2) / rho), 20001)
    r = rho * np.cosh(t)
    rb = r / BOHR
    fr = (VC_rmt - Y00 * v00r2_at(rb) / rb**2) * RY_TO_V * r
    return 2 * np.trapezoid(fr, t)


X_, Z_ = np.meshgrid(x - x[ix0], z - z[iz0], indexing='ij')
R_rem = Vz_green - f_sph(np.sqrt(X_**2 + Z_**2)).sum(axis=1) * dz
rR, R_pos = x[ix0:] - x[ix0], R_rem[ix0:]
print(f"smooth remainder R = green - z-sum of f: {R_pos[0]:.2f} (r=0), {R_pos[1]:.2f} (r=dx), "
      f"{R_pos[sub]:.2f} (r=1/30 A) V*A")
rho_max = np.hypot(edges[-1][1], px / 2)
assert rho_max < rmt_mt * BOHR, "pixels must lie inside the muffin-tin sphere"
lrho = np.linspace(np.log(1e-6), np.log(rho_max * 1.01), 600)
fz_tab = CubicSpline(lrho, [f_proj(np.exp(v)) for v in lrho])   # smooth in ln(rho)


def vz_ideal(r):
    """True projected potential V_z(r) [V*A]; R is held constant beyond the plane edge."""
    r = np.asarray(r, float)
    return np.interp(r, rR, R_pos) + fz_tab(np.log(np.clip(r, 1e-6, None)))


print(f"ideal V_z(dx..4dx) = {np.round(vz_ideal(np.arange(1, 5) * dx), 0)} V*A "
      f"(green samples {np.round(Vz_green[ix0 + 1:ix0 + 5], 0)})")

line_mean, blue, blue_green = [], [], []
for lo, hi in edges:
    m = (np.round((x - lo) / dx) >= 0) & (np.round((x - hi) / dx) < 0)   # half-open pixel, in samples
    line_mean.append(Vz_green[m].mean())
    xx = np.linspace(lo, hi, 801)[:-1] + (hi - lo) / 1600
    yy = np.linspace(-px / 2, px / 2, 801)[:-1] + px / 1600
    XX, YY = np.meshgrid(xx, yy)
    blue_green.append(vz_radial(np.sqrt(XX**2 + YY**2)).mean())
    blue.append(vz_ideal(np.sqrt(XX**2 + YY**2)).mean())
print("1-D line means incl. r=0 sample [V*A]:", np.round(line_mean, 0))
print("ideal pixel-column means (fine z integration) [V*A]:              ", np.round(blue, 0))
print("2-D pixel-area means of the green curve (30 ppA z) [V*A]:          ", np.round(blue_green, 0))

# paper's blue boxes: 1-D mean of the green line over each pixel, continuum limit.
# Between the lapw5 samples the green curve is the smooth remainder R (interpolated)
# plus the z-sum of the spherical part f, which carries the 1/|x| spike of the z = 0
# sample and is evaluated exactly (held constant below R0, as lapw5 does).
zk_mt = (z - z[iz0])[np.abs(z - z[iz0]) < rmt_mt * BOHR]


def green_cont(xx):
    xx = np.atleast_1d(np.asarray(xx, float))
    r = np.sqrt(xx[:, None]**2 + zk_mt[None, :]**2)
    return np.interp(np.abs(xx), rR, R_pos) + f_sph(r).sum(axis=1) * dz


print(f"continuous green curve reproduces the lapw5 samples to "
      f"{np.max(np.abs(green_cont(x - x[ix0]) - Vz_green)):.1e} V*A")
r0_A = r0_mt * BOHR
line_cont = []
for lo, hi in edges:
    if lo < 0 < hi:                                   # centre pixel, mirror-symmetric
        xs = np.concatenate([np.linspace(0, r0_A, 101), np.geomspace(r0_A, hi, 40001)[1:]])
        line_cont.append(np.trapezoid(green_cont(xs), xs) / hi)
    else:
        xs = np.linspace(lo, hi, 20001)
        line_cont.append(np.trapezoid(green_cont(xs), xs) / (hi - lo))
print("paper's blue = 1-D mean of the green line per pixel, continuum limit [V*A]:", np.round(line_cont, 0))
for rate in (600, 3000, 6000, 30000):
    xs = np.arange(-int(round(px / 2 * rate)), int(round(px / 2 * rate)) + 1) / rate
    xs = xs[(xs >= -px / 2 - 1e-12) & (xs < px / 2 - 1e-12)]
    print(f"    centre pixel, green line sampled at {rate:5d} ppA: mean {green_cont(xs).mean():.0f} V*A")

# ---------------------------------------------------------------- red: 30 ppA lateral subgrid + nn cutoff
cols = np.arange(ix0 % sub, npx, sub)             # x = ..., -1/30, 0, 1/30, ...
G = Vs[cols, :].copy()                            # [ncol, nz] at 30 ppA isotropic
xg = x[cols]
print(f"30 ppA subgrid: {G.shape}, x = {np.round(xg, 4)}")
bad = G > CUTOFF_V
print(f"voxels above cutoff {CUTOFF_V:.0f} V: {bad.sum()} -> value(s) {G[bad]}")


def nn_replace(G, rule):
    """Replace every voxel above the cutoff by a value taken from its
    face neighbours (4 in this 2-D plane; periodic in z, clipped in x).
    rule='max' : highest neighbour value in the paper's positive convention
                 (= largest |V|, i.e. the neighbour closest to the nucleus)
    rule='min' : lowest neighbour value (what maximum_filter on WIEN2k-sign data does)"""
    H = G.copy()
    bad = H > CUTOFF_V
    for i, j in zip(*np.nonzero(bad)):
        nb = []
        if i > 0:            nb.append(H[i - 1, j])
        if i < H.shape[0]-1: nb.append(H[i + 1, j])
        nb.append(H[i, (j - 1) % H.shape[1]])
        nb.append(H[i, (j + 1) % H.shape[1]])
        nb = [v for v in nb if v <= CUTOFF_V]
        H[i, j] = max(nb) if rule == 'max' else min(nb)
    return H


results = {}
for rule in ('max', 'min'):
    H = nn_replace(G, rule)
    Vz = H.sum(axis=1) * dz
    results[rule] = Vz
    ic = int(np.argmin(np.abs(xg)))
    print(f"red[{rule}]: centre pixel V_z = {Vz[ic]:.0f} V*A, next pixels {Vz[ic+1]:.0f}, {Vz[ic+2]:.0f}; "
          f"ratio ideal/red = {blue[0]/Vz[ic]:.2f} (paper-style/red = {blue_green[0]/Vz[ic]:.2f})")
Vz_nocut = G.sum(axis=1) * dz
ic = int(np.argmin(np.abs(xg)))
print(f"no cutoff (30 ppA): centre pixel V_z = {Vz_nocut[ic]:.0f} V*A, ratio to ideal = {Vz_nocut[ic]/blue[0]:.1f} "
      f"(to paper-style = {Vz_nocut[ic]/blue_green[0]:.1f})")

# ---------------------------------------------------------------- red as in the paper: fine lateral line with nn cutoff
# For every lateral position x the 30 ppA grid is laid through that column: z
# neighbours at z +- 1/30 A, x neighbours at x +- 1/30 A (sub fine samples away).
# The y neighbours (normal to this plane, 1/30 A away) are not in the plane; they
# never win, since the near x neighbour (1/30 A - |x| from the nucleus) is closer
# than the y and z neighbours (sqrt(x^2 + (1/30 A)^2)).
red_fine = Vz_green.copy()
flag_cols = []
for i in range(npx):
    colv = Vs[i].copy()
    for j in np.nonzero(colv > CUTOFF_V)[0]:
        nb = [colv[(j - 1) % nz], colv[(j + 1) % nz]]
        if i - sub >= 0:
            nb.append(Vs[i - sub, j])
        if i + sub < npx:
            nb.append(Vs[i + sub, j])
        colv[j] = max(v for v in nb if v <= CUTOFF_V)
    if not np.array_equal(colv, Vs[i]):
        red_fine[i] = colv.sum() * dz
        flag_cols.append(i)
print(f"red (fine): columns above {CUTOFF_V:.0f} V at x = {np.round((x[flag_cols] - x[ix0]) * 1000, 2)} mA "
      f"-> V_z = {np.round(red_fine[flag_cols], 0)} V*A")
print(f"centre pixel: paper's blue / red = {line_cont[0] / red_fine[ix0]:.2f}, "
      f"ideal / red = {blue[0] / red_fine[ix0]:.2f}  (paper: 'within a factor of three')")

# ---------------------------------------------------------------- Kirkland projected potential for C (reference)
# fparams.dat layout per element: a1 b1 a2 b2 / a3 b3 c1 d1 / c2 d2 c3 d3
KC = np.array([[float(v) for v in l.split()] for l in """
  2.53148417e-001  2.08761775e-001  1.66953010e-001  5.69804854e+000
  1.60250013e-001  2.08774166e-001  1.41633354e-001  1.34112859e+000
  3.60244465e-001  3.81665326e+000  8.66192838e-004  4.12204523e-002
""".strip().splitlines()]).ravel()   # Z=6 from Kirkland fparams.dat
a = KC[[0, 2, 4]]; b = KC[[1, 3, 5]]; c = KC[[6, 8, 10]]; d = KC[[7, 9, 11]]


def kirk_vz(r):
    r = np.asarray(r, float)
    return (4 * np.pi**2 * A0E * sum(a[i] * k0(2 * np.pi * r * np.sqrt(b[i])) for i in range(3))
            + 2 * np.pi**2 * A0E * sum(c[i] / d[i] * np.exp(-np.pi**2 * r**2 / d[i]) for i in range(3)))


# pixel mean of the Kirkland projected potential over the central 30 ppA pixel
xx = np.linspace(-px / 2, px / 2, 401)
XX, YY = np.meshgrid(xx, xx)
RR = np.sqrt(XX**2 + YY**2); RR[RR < 1e-6] = 1e-6
print(f"Kirkland C: pixel mean over central 30 ppA pixel = {kirk_vz(RR).mean():.0f} V*A; line mean over x-pixel at y=0: {kirk_vz(np.abs(xx[np.abs(xx)>1e-6])).mean():.0f} V*A")

# ---------------------------------------------------------------- Doyle-Turner projected potential for C (paper's reference)
# Doyle & Turner, Acta Cryst. A24, 390 (1968): f_e(s) = sum a_i exp(-b_i s^2), s = sin(theta)/lambda [1/A];
# projected: V_z(r) = 8 pi^2 a0 e sum (a_i / b_i) exp(-4 pi^2 r^2 / b_i)
DT_A = np.array([0.7307, 1.1951, 0.4563, 0.1247])
DT_B = np.array([36.9951, 11.2966, 2.8139, 0.3456])


def dt_vz(r):
    r = np.asarray(r, float)
    return 8 * np.pi**2 * A0E * sum(DT_A[i] / DT_B[i] * np.exp(-4 * np.pi**2 * r**2 / DT_B[i]) for i in range(4))


print(f"Doyle-Turner C: V_z(0) = {float(dt_vz(0.0)):.0f} V*A")

# ---------------------------------------------------------------- plot (layout of the paper's Fig. 3)
fig, ax = plt.subplots(figsize=(7.5, 5))
for k, ((lo, hi), val, ideal) in enumerate(zip(edges, line_cont, blue)):
    ax.add_patch(plt.Rectangle((lo, 100), hi - lo, val - 100, color='#6fb7ff', alpha=0.55, zorder=0,
                               label='mean of the green line per 30 ppÅ pixel' if k == 0 else None))
    ax.add_patch(plt.Rectangle((lo, 100), hi - lo, ideal - 100, fill=False, ec='#1f4e9a', ls='--', lw=1.0,
                               zorder=1, label='true pixel-column mean (not in the paper)' if k == 0 else None))
xr = x - x[ix0]
ax.semilogy(xr, Vz_green, 'g-', lw=1.2, label='no cutoff')
ax.semilogy(xr, red_fine, 'r-', lw=1.2, label=f'nn cutoff {CUTOFF_V:.0f}')
rr = np.linspace(-0.02, 0.09, 1101)
ax.semilogy(rr, dt_vz(np.abs(rr)), 'k:', lw=1.2, label='DT projected')
ax.semilogy(rr, kirk_vz(np.maximum(np.abs(rr), 1e-4)), color='0.45', ls='-.', lw=0.9, label='Kirkland projected')
for k in range(1, 4):
    ax.axvline((2 * k - 1) * px / 2, color='k', lw=0.6, ls=':')
    ax.text((2 * k - 1) * px / 2, 1.08e5, f'pixel{k}', ha='center', va='bottom', fontsize=9)
ax.set_xlim(-px / 2 - 0.0008, 0.09); ax.set_ylim(100, 1e5)
ax.text(0.088, 2.2e3, f"centre pixel [V·Å]: no cutoff {Vz_green[ix0]:.0f}, nn cutoff {red_fine[ix0]:.0f},\n"
        f"line mean {line_cont[0]:.0f}, true pixel-column mean {blue[0]:.0f}", fontsize=7.5, va='top', ha='right')
ax.set_xlabel('distance from core in Å'); ax.set_ylabel('projected potential in V·Å')
ax.set_title('Graphene, WIEN2k vcoul via lapw5 — cf. Kurasch et al. 2011, Fig. 3', fontsize=10, pad=18)
ax.legend(fontsize=7.5, loc='upper right')
fig.tight_layout(); fig.savefig(out_png, dpi=150)
print("saved", out_png)
