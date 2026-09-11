#!/usr/bin/env python3
"""
plot_kurasch_fig3.py — reproduce Kurasch et al. 2011, Fig. 3 from a lapw5 plane

Input is a lapw5 `case.rho` of an x-z plane through a nucleus: x laterally at
a fine rate (a multiple of 30 ppÅ, here 600 ppÅ over ±0.1167 Å), z along the
beam over the full cell at the production rate of 30 ppÅ, with the nucleus on
a grid point in both directions.  The lapw5 input used for graphene is in
reference/lapw5_fig3/ (units ATU -> Ry, NONORTHO).

green : projected potential V_z(x) = sum_z V_s(x,z) dz at the fine lateral rate
blue  : 2-D pixel-area mean of V_z over 30 ppÅ pixels (the "true" pixel value)
red   : V_z on the 30 ppÅ sub-grid after the nearest-neighbour cutoff, for both
        readings of "highest value in the neighboring pixels" (see
        vcoul2tiff_ulm.py, apply_nn_cutoff)
black : Kirkland IAM projected potential of carbon for comparison

Usage:
    python3 plot_kurasch_fig3.py Graphene.rho_fig3 kurasch_fig3_graphene.png
"""
import sys
import numpy as np
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
# "Ideal" pixel value = area mean of V_z over the 30 ppA pixel.  V_z(r) is taken
# from the green curve (radially symmetric this close to the core); the sample
# at r = 0 is the WIEN2k r0 cusp and is not representative of the pixel, so for
# r < dx the log-divergence  V_z ~ A - B ln r  is fitted to the first samples.
px = 1.0 / 30                                     # production pixel (30 ppÅ)
sub = int(round(px / dx))                         # fine samples per pixel
assert abs(sub * dx - px) < 1e-6, "lateral rate must be a multiple of 30 ppÅ"
edges = [(-px / 2, px / 2), (px / 2, 3 * px / 2), (3 * px / 2, 5 * px / 2), (5 * px / 2, 7 * px / 2)]
rpos = x[ix0 + 1:]; vpos = Vz_green[ix0 + 1:]
B_, A_ = np.polyfit(np.log(rpos[:4]), vpos[:4], 1)  # V = A + B ln r (B < 0)
print(f"log fit near core: V_z(r) = {A_:.0f} + ({B_:.0f})*ln r  ->  V_z(dx)={A_+B_*np.log(dx):.0f} (sample {vpos[0]:.0f})")


def vz_radial(r):
    r = np.asarray(r, float)
    out = np.interp(r, rpos, vpos)
    small = r < rpos[0]
    out[small] = A_ + B_ * np.log(np.maximum(r[small], 1e-7))
    return out


line_mean, blue = [], []
for lo, hi in edges:
    m = (x >= lo - 1e-9) & (x < hi - 1e-9)
    line_mean.append(Vz_green[m].mean())
    xx = np.linspace(lo, hi, 801)[:-1] + (hi - lo) / 1600
    yy = np.linspace(-px / 2, px / 2, 801)[:-1] + px / 1600
    XX, YY = np.meshgrid(xx, yy)
    blue.append(vz_radial(np.sqrt(XX**2 + YY**2)).mean())
print("1-D line means incl. r=0 sample [V*A]:", np.round(line_mean, 0))
print("blue = 2-D pixel-area means [V*A]:   ", np.round(blue, 0))

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
          f"ratio ideal/red = {blue[0]/Vz[ic]:.2f}")
Vz_nocut = G.sum(axis=1) * dz
ic = int(np.argmin(np.abs(xg)))
print(f"no cutoff (30 ppA): centre pixel V_z = {Vz_nocut[ic]:.0f} V*A, ratio to ideal = {Vz_nocut[ic]/blue[0]:.1f}")

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

# ---------------------------------------------------------------- plot
fig, ax = plt.subplots(figsize=(7.5, 5))
for (lo, hi), val in zip(edges, blue):
    ax.add_patch(plt.Rectangle((lo, 100), hi - lo, val - 100, color='#6fb7ff', alpha=0.6, zorder=0))
ax.semilogy(x, Vz_green, 'g-', lw=1.2, label='no cutoff (600 ppÅ lateral, 30 ppÅ z)')
ax.step(np.append(xg - px / 2, xg[-1] + px / 2), np.append(results['max'], results['max'][-1]), where='post',
        color='r', lw=1.4, label="nn cutoff 50 kV, 'highest' = largest |V| neighbour")
ax.step(np.append(xg - px / 2, xg[-1] + px / 2), np.append(results['min'], results['min'][-1]), where='post',
        color='m', lw=1.0, ls='--', label="nn cutoff 50 kV, smallest |V| neighbour (old code)")
rr = np.linspace(1e-3, 0.12, 300)
ax.semilogy(rr, kirk_vz(rr), 'k:', lw=1.2, label='Kirkland IAM projected (C)')
for k in range(1, 4):
    ax.axvline((2 * k - 1) * px / 2, color='gray', lw=0.6, ls=':')
    ax.text((2 * k - 1) * px / 2, 115, f'pixel{k}', ha='center', fontsize=8)
ax.set_xlim(-0.02, 0.09); ax.set_ylim(100, 2e5)
ax.text(0.088, 1.5e4, f"centre pixel: no cutoff {Vz_nocut[ic]:.0f}, nn cutoff {results['max'][ic]:.0f}, "
        f"pixel-area mean {blue[0]:.0f} V·Å", fontsize=7.5, va='top', ha='right')
ax.set_xlabel('distance from core [Å]'); ax.set_ylabel('projected potential [V·Å]')
ax.set_title('Graphene C, WIEN2k vcoul via lapw5 — cf. Kurasch et al. 2011 Fig. 3', fontsize=10)
ax.legend(fontsize=7, loc='upper right')
fig.tight_layout(); fig.savefig(out_png, dpi=150)
print("saved", out_png)
