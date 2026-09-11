#!/usr/bin/env python3
"""
vcoul2tiff.py — Analytical projection of WIEN2k vcoul → ixchel TIF

Reads case.vcoul_DFT and case.vcoul_IAM in the native WIEN2k format and
generates projected potentials as ixchel-compatible TIFF files.

METHOD:
  V_proj = V_Kirkland + ΔV_proj
  where ΔV = vcoul_DFT − vcoul_IAM is projected analytically:
    - Interstitial: selection of the plane waves with G·ê_beam = 0
    - Muffin-Tin:   Abel transform of the (l,m) channels on the radial grid

No 3ddens grid, no separate Poisson solver, no nuclear singularity.

Usage:
    python3 vcoul2tiff.py \\
        --vcoul-dft Graphene.vcoul_DFT --vcoul-iam Graphene.vcoul_IAM \\
        --struct Graphene.struct --fparams fparams.dat \\
        --cfg Graphene_ixchel_001.cfg --miller 0 0 1 \\
        -o wien2k_analytical_001
"""

import argparse, re, sys, math
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import lpmv
import tifffile


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

BOHR      = 0.529177210903
BOHR3     = BOHR**3
RY_TO_EV  = 13.605693122994
H2_2PI_ME = 47.88

IXCHEL_MASS = {
     1:  1.00794,    2:  4.002602,   3:  6.941,       4:  9.012182,
     5: 10.811,      6: 12.0107,     7: 14.0067,      8: 15.9994,
     9: 18.9984032, 10: 20.1791,    11: 22.98976928, 12: 24.305,
    13: 26.9815386, 14: 28.0855,    15: 30.973762,   16: 32.065,
    17: 35.453,     18: 39.948,     19: 39.0983,     20: 40.078,
    21: 44.955912,  22: 47.867,     23: 50.9415,     24: 51.9961,
    25: 54.938045,  26: 55.845,     27: 58.933195,   28: 58.6934,
    29: 63.546,     30: 65.38,      31: 69.723,      32: 72.63,
    33: 74.9216,    34: 78.96,      35: 79.904,      36: 83.798,
    37: 85.4678,    38: 87.62,      39: 88.90585,    40: 91.224,
    41: 92.90638,   42: 95.96,      43: 98.0,        44: 101.07,
    45: 102.9055,   46: 106.42,     47: 107.8682,    48: 112.411,
    49: 114.818,    50: 118.71,     51: 121.76,      52: 127.6,
    53: 126.90447,  54: 131.293,    55: 132.9054519, 56: 137.327,
    57: 138.90547,  58: 140.116,    59: 140.90765,   60: 144.242,
    61: 145.0,      62: 150.36,     63: 151.964,     64: 157.25,
    65: 158.92535,  66: 162.5,      67: 164.93032,   68: 167.259,
    69: 168.93421,  70: 173.054,    71: 174.9668,    72: 178.49,
    73: 180.94788,  74: 183.84,     75: 186.207,     76: 190.23,
    77: 192.217,    78: 195.084,    79: 196.966569,  80: 200.59,
    81: 204.3833,   82: 207.2,      83: 208.9804,    84: 209.0,
    85: 210.0,      86: 222.0,      87: 223.0,       88: 226.0,
    89: 227.0,      90: 232.03806,  91: 231.03586,   92: 238.02891,
    93: 237.0,      94: 244.0,      95: 243.0,       96: 247.0,
}
M_SI = IXCHEL_MASS[14]

def ixchel_B(Z):
    M = IXCHEL_MASS.get(Z, M_SI)
    return 0.45 * M_SI / M


# ═══════════════════════════════════════════════════════════════
# I/O: vcoul, struct, fparams, cfg
# ═══════════════════════════════════════════════════════════════

def parse_vcoul(filename):
    """Reads case.vcoul in the native WIEN2k format."""
    with open(filename) as f:
        lines = f.readlines()
    atoms = []
    pw_data = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if 'ATOMNUMBER' in line:
            m = re.search(r'ATOMNUMBER\s*=\s*(\d+)', line)
            atom_idx = int(m.group(1))
            i += 1
            m = re.search(r'NUMBER OF LM\s*(\d+)', lines[i].strip())
            n_lm = int(m.group(1))
            i += 1
            lm_channels = []
            for _ in range(n_lm):
                while i < len(lines):
                    if 'FOR L' in lines[i]: break
                    i += 1
                m = re.search(r'L\s+(-?\d+)\s+M=\s*(-?\d+)', lines[i])
                l_val, m_val = int(m.group(1)), int(m.group(2))
                i += 1
                while i < len(lines) and lines[i].strip() == '':
                    i += 1
                vals = []
                while i < len(lines) and len(vals) < 10000:
                    raw = lines[i]
                    if raw.strip() == '' or 'VLM' in raw or 'CLM' in raw or \
                       'ATOM' in raw or 'COULOMB' in raw or 'MIXED' in raw:
                        break
                    pos = 3
                    while pos < len(raw) - 1:
                        chunk = raw[pos:pos+19]
                        if chunk.strip():
                            vals.append(float(chunk.replace('D', 'E')))
                        pos += 19
                    i += 1
                lm_channels.append({'l': l_val, 'm': m_val, 'data': np.array(vals)})
            atoms.append({'atom_idx': atom_idx, 'n_lm': n_lm, 'lm_channels': lm_channels})
            continue
        if 'NUMBER OF PW' in line:
            m = re.search(r'(\d+)\s*NUMBER OF PW', line)
            n_pw = int(m.group(1))
            i += 1
            G_list, coeff_list = [], []
            for _ in range(n_pw):
                while i < len(lines) and lines[i].strip() == '':
                    i += 1
                raw = lines[i]
                ga = int(raw[0:8]); gb = int(raw[8:13]); gc = int(raw[13:18])
                re_v = float(raw[18:37].replace('D', 'E'))
                im_v = float(raw[37:56].replace('D', 'E'))
                G_list.append([ga, gb, gc])
                coeff_list.append(complex(re_v, im_v))
                i += 1
            pw_data = {'n_pw': n_pw, 'G': np.array(G_list, dtype=int),
                       'coeff': np.array(coeff_list, dtype=complex)}
            continue
        i += 1
    return {'atoms': atoms, 'pw': pw_data}


def read_struct(filename):
    """Reads case.struct: lattice parameters and atoms."""
    with open(filename) as f:
        lines = f.readlines()
    lattice_type = lines[1][:3].strip()
    parts = lines[3].split()
    a, b, c = float(parts[0]), float(parts[1]), float(parts[2])
    atoms = []
    i = 4
    while i < len(lines):
        if lines[i].startswith('ATOM'):
            m = re.search(r'X=\s*([\d.]+)\s*Y=\s*([\d.]+)\s*Z=\s*([\d.]+)', lines[i])
            x0, y0, z0 = float(m.group(1)), float(m.group(2)), float(m.group(3))
            m2 = re.search(r'MULT=\s*(\d+)', lines[i+1])
            mult = int(m2.group(1))
            positions = [(x0, y0, z0)]
            for k in range(1, mult):
                m3 = re.search(r'X=\s*([\d.]+)\s*Y=\s*([\d.]+)\s*Z=\s*([\d.]+)',
                              lines[i+1+k])
                positions.append((float(m3.group(1)), float(m3.group(2)), float(m3.group(3))))
            elem_line = lines[i + 1 + mult]
            em = re.match(r'(\w+)', elem_line.strip())
            npt_m = re.search(r'NPT=\s*(\d+)', elem_line)
            r0_m  = re.search(r'R0=\s*([\d.]+)', elem_line)
            rmt_m = re.search(r'RMT=\s*([\d.]+)', elem_line)
            z_m   = re.search(r'Z:\s*([\d.]+)', elem_line)
            atoms.append({
                'element': em.group(1) if em else '??',
                'positions': positions, 'mult': mult,
                'NPT': int(npt_m.group(1)),
                'R0': float(r0_m.group(1)),
                'RMT': float(rmt_m.group(1)),
                'Z': int(float(z_m.group(1))),
            })
            i += 2 + mult
        else:
            i += 1
    return {'lattice_type': lattice_type, 'a': a, 'b': b, 'c': c, 'atoms': atoms}


def read_fparams(filename):
    """Reads Kirkland fparams.dat."""
    params = {}
    with open(filename) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        m = re.match(r'Z=\s*(\d+)', lines[i].strip())
        if m and i + 3 < len(lines):
            Z = int(m.group(1))
            v = [float(x) for ln in lines[i+1:i+4] for x in ln.split()]
            params[Z] = {'a': [v[0], v[2], v[4]], 'b': [v[1], v[3], v[5]],
                         'c': [v[6], v[8], v[10]], 'd': [v[7], v[9], v[11]]}
            i += 4
        else:
            i += 1
    return params


def parse_cfg(cfg_path):
    """Reads ixchel crystruct.cfg (layer geometry)."""
    text = Path(cfg_path).read_text()
    sm  = re.search(r'stack_seq\s*=\s*\"\d+\(([^)]+)\)\"', text)
    seq = list(sm.group(1)) if sm else []

    lm = re.search(r'layers\s*=\s*\{', text)
    depth = 0; j = lm.end() - 1
    while j < len(text):
        if text[j] == '{': depth += 1
        elif text[j] == '}':
            depth -= 1
            if depth == 0: break
        j += 1
    layers_text = text[lm.end():j]

    layer_defs = {}
    pos = 0
    while pos < len(layers_text):
        lbm = re.search(r'\b([a-z])\s*=\s*\{', layers_text[pos:])
        if not lbm: break
        label = lbm.group(1)
        bstart = pos + lbm.end() - 1
        d = 0; p2 = bstart
        while p2 < len(layers_text):
            if layers_text[p2] == '{': d += 1
            elif layers_text[p2] == '}':
                d -= 1
                if d == 0: break
            p2 += 1
        block = layers_text[bstart+1:p2]
        su = re.search(
            r'size_uc\s*=\s*\{[^}]*x\s*=\s*([0-9.E+\-]+)[^}]*'
            r'y\s*=\s*([0-9.E+\-]+)[^}]*z\s*=\s*([0-9.E+\-]+)', block)
        fn = re.search(r'filename\s*=\s*\"([^\"]+)\"', block).group(1)
        atoms_list = []
        asec_m = re.search(r'atoms\s*=\s*\(', block)
        if asec_m:
            ap0 = asec_m.end() - 1; d = 0; ae = ap0
            while ae < len(block):
                if block[ae] == '(': d += 1
                elif block[ae] == ')':
                    d -= 1
                    if d == 0: break
                ae += 1
            asec = block[ap0+1:ae]
            ap = 0
            while ap < len(asec):
                ob = asec.find('{', ap)
                if ob < 0: break
                d = 0; p3 = ob
                while p3 < len(asec):
                    if asec[p3] == '{': d += 1
                    elif asec[p3] == '}':
                        d -= 1
                        if d == 0: break
                    p3 += 1
                atom_block = asec[ob+1:p3]
                nm = re.search(r'(?<!file)\bname\s*=\s*\"([^\"]+)\"', atom_block)
                zm = re.search(r'\bZ\s*=\s*(\d+)', atom_block)
                pm = re.search(r'positions\s*=\s*\((.*?)\)', atom_block, re.DOTALL)
                dw_m = re.search(r'\bDW\s*=\s*([0-9.E+\-]+)', atom_block)
                if nm and zm and pm:
                    Z_val = int(zm.group(1))
                    B_val = float(dw_m.group(1)) if dw_m else ixchel_B(Z_val)
                    for pp in re.finditer(
                            r'\[\s*([0-9.E+\-]+)\s*,\s*([0-9.E+\-]+)\s*\]',
                            pm.group(1)):
                        atoms_list.append({
                            'name': nm.group(1), 'Z': Z_val, 'B': B_val,
                            'proj_x': float(pp.group(1)), 'proj_y': float(pp.group(2)),
                        })
                ap = p3 + 1
        layer_defs[label] = {
            'ax': float(su.group(1)), 'by': float(su.group(2)),
            'cz': float(su.group(3)), 'filename': fn, 'atoms': atoms_list
        }
        pos = p2 + 1

    layers = []
    z0 = 0.0
    for label in seq:
        ld = layer_defs[label]
        layers.append({
            'label': label, 'ax': ld['ax'], 'by': ld['by'],
            'cz': ld['cz'], 'z0': z0, 'filename': ld['filename'],
            'atoms': ld['atoms'],
        })
        z0 += ld['cz']
    return {'layers': layers, 'total_z': z0}


# ═══════════════════════════════════════════════════════════════
# Kirkland IAM (Fourier space)
# ═══════════════════════════════════════════════════════════════

def featom_k2(Z, k2, fp):
    p = fp[Z]
    res = np.zeros_like(k2, dtype=float)
    for a_, b_ in zip(p['a'], p['b']):
        res += a_ / (k2 + b_)
    for c_, d_ in zip(p['c'], p['d']):
        res += c_ * np.exp(-d_ * k2)
    return res


def kirkland_layer_fp(atoms_layer, ax, by, Nu, Nv, fp):
    """Kirkland IAM in Fourier space [1/Å]."""
    ixmid, iymid = Nu // 2, Nv // 2
    kxv = np.array([i if i < ixmid else i - Nu for i in range(Nu)], float)
    kyv = np.array([i if i < iymid else i - Nv for i in range(Nv)], float)
    KX, KY = np.meshgrid(kxv, kyv, indexing='ij')
    k2 = KX**2 / ax**2 + KY**2 / by**2
    k2max = 1.0 / max(2*ax/Nu, 2*by/Nv)**2
    mask = k2 <= k2max
    scale = float(Nu * Nv) / (ax * by)
    pix = np.zeros((Nu, Nv), dtype=complex)
    for at in atoms_layer:
        Z = at['Z']
        if Z not in fp:
            print(f"    WARNING: Z={Z} not in fparams — skipped")
            continue
        B = at['B']
        fe = featom_k2(Z, k2, fp) * np.exp(-B * k2)
        phase = np.exp(-2j * np.pi * (KX * at['proj_x'] + KY * at['proj_y']))
        pix += mask * scale * fe * phase
    return np.real(np.fft.ifft2(pix))


# ═══════════════════════════════════════════════════════════════
# Analytical ΔV projection
# ═══════════════════════════════════════════════════════════════


def primitive_vectors(struct):
    """Primitive lattice vectors in Bohr (WIEN2k convention)."""
    a_b, b_b, c_b = struct['a'], struct['b'], struct['c']
    lt = struct['lattice_type']
    if lt == 'P':
        return np.array([[a_b, 0, 0], [0, b_b, 0], [0, 0, c_b]])
    elif lt == 'F':
        h2 = a_b / 2
        return np.array([[h2, h2, 0], [h2, 0, h2], [0, h2, h2]])
    elif lt in ('B', 'I'):
        h2 = a_b / 2
        return np.array([[-h2, h2, h2], [h2, -h2, h2], [h2, h2, -h2]])
    elif lt in ('CXY', 'C'):
        return np.array([[a_b/2, -b_b/2, 0], [a_b/2, b_b/2, 0], [0, 0, c_b]])
    elif lt == 'CXZ':
        return np.array([[a_b/2, 0, -c_b/2], [0, b_b, 0], [a_b/2, 0, c_b/2]])
    elif lt == 'CYZ':
        return np.array([[a_b, 0, 0], [0, b_b/2, -c_b/2], [0, b_b/2, c_b/2]])
    raise NotImplementedError(f"Lattice type '{lt}' not implemented")


def project_dV_layer(vcoul_dft, vcoul_iam, struct, miller,
                     z0_ang, cz_ang, layer_atoms_frac_z, nx, ny):
    """
    Projects ΔV = vcoul_DFT − vcoul_IAM analytically for ONE layer.

    Interstitial: ALL PW with partial z-integral
      ∫_{z0}^{z0+cz} exp(iKz·z) dz = cz·sinc(Kz·cz/2)·exp(iKz·(z0+cz/2))

    Muffin-Tin: only atoms whose z-position lies within the layer.

    Parameters:
        z0_ang, cz_ang: layer boundaries in Å
        layer_atoms_frac_z: list of z-positions (fractional) of the atoms in this layer
    """
    h, k, l = miller
    a_b, b_b, c_b = struct['a'], struct['b'], struct['c']
    conv = np.array([[a_b, 0, 0], [0, b_b, 0], [0, 0, c_b]])

    prim = primitive_vectors(struct)
    V_cell = np.dot(prim[0], np.cross(prim[1], prim[2]))
    rec = np.array([
        2 * np.pi * np.cross(prim[1], prim[2]) / V_cell,
        2 * np.pi * np.cross(prim[2], prim[0]) / V_cell,
        2 * np.pi * np.cross(prim[0], prim[1]) / V_cell,
    ])

    # Layer boundaries in Bohr
    z0_b = z0_ang / BOHR
    cz_b = cz_ang / BOHR

    # ── Interstitial: ALL PW with partial z-integral ──
    pw_d, pw_i = vcoul_dft['pw'], vcoul_iam['pw']
    dV_pw = pw_d['coeff'] - pw_i['coeff']

    G_all = pw_d['G']
    G_cart_all = (G_all[:, 0:1] * rec[0] +
                  G_all[:, 1:2] * rec[1] +
                  G_all[:, 2:3] * rec[2])
    Kz_all = G_cart_all[:, 2]  # z-component of each G-vector

    # Partial z-integral: ∫_{z0}^{z0+cz} exp(iKz·z) dz
    z_mid = z0_b + cz_b / 2
    z_phase = np.zeros(len(Kz_all), dtype=complex)
    for j in range(len(Kz_all)):
        kz = Kz_all[j]
        if abs(kz) < 1e-12:
            z_phase[j] = cz_b
        else:
            z_phase[j] = (np.exp(1j * kz * (z0_b + cz_b)) -
                          np.exp(1j * kz * z0_b)) / (1j * kz)

    # Real-space grid
    x = np.linspace(0, a_b, nx, endpoint=False)
    y = np.linspace(0, b_b, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    dV_proj_pw = np.zeros((nx, ny))
    n_significant = 0
    for j in range(len(dV_pw)):
        coeff = dV_pw[j] * z_phase[j]
        if abs(coeff) < 1e-15:
            continue
        Gx, Gy = G_cart_all[j, 0], G_cart_all[j, 1]
        phase = Gx * X + Gy * Y
        dV_proj_pw += np.real(coeff * np.exp(1j * phase))
        n_significant += 1

    print(f"    Interstitial: {n_significant} PW significant (of {len(dV_pw)})")

    # ── Muffin-Tin: only atoms within the layer ──
    dV_proj_mt = np.zeros((nx, ny))
    c_ang = c_b * BOHR

    for iat, at_s in enumerate(struct['atoms']):
        R0, RMT, NPT = at_s['R0'], at_s['RMT'], at_s['NPT']
        dr_log = np.log(RMT / R0) / (NPT - 1)
        r_grid = R0 * np.exp(np.arange(NPT) * dr_log)

        ch_dft = vcoul_dft['atoms'][iat]['lm_channels']
        ch_iam = vcoul_iam['atoms'][iat]['lm_channels']

        # Precomputation: Abel projections for all (l,m) channels
        abel_cache = {}
        for cd, ci in zip(ch_dft, ch_iam):
            lv_raw, mv = cd['l'], cd['m']
            lv = abs(lv_raw)
            use_sin = (lv_raw < 0)
            dVlm_r2 = cd['data'] - ci['data']
            if np.max(np.abs(dVlm_r2)) < 1e-12:
                continue

            n_rho = 200
            rho_grid = np.linspace(0.001, RMT * 0.999, n_rho)
            dV_r2_interp = interp1d(r_grid, dVlm_r2, kind='cubic',
                                    fill_value=0, bounds_error=False)
            V_proj_rho = np.zeros(n_rho)

            for i, rho in enumerate(rho_grid):
                t_max = np.sqrt(RMT**2 - rho**2)
                if t_max < 1e-10: continue
                t_n, t_w = np.polynomial.legendre.leggauss(64)
                t_n = 0.5 * t_max * (t_n + 1)
                t_w = 0.5 * t_max * t_w
                r_n = np.sqrt(t_n**2 + rho**2)
                dV_r = dV_r2_interp(r_n) / r_n**2
                cos_t = t_n / r_n

                if mv == 0:
                    norm = np.sqrt((2*lv+1) / (4*np.pi))
                    Yp = norm * lpmv(0, lv, cos_t)
                    Ym = norm * lpmv(0, lv, -cos_t)
                else:
                    norm = np.sqrt((2*lv+1)/(4*np.pi) *
                                  math.factorial(lv-mv)/math.factorial(lv+mv)) * np.sqrt(2)
                    Yp = norm * lpmv(mv, lv, cos_t)
                    Ym = norm * lpmv(mv, lv, -cos_t)

                V_proj_rho[i] = np.sum(dV_r * (Yp + Ym) * t_w)

            if np.max(np.abs(V_proj_rho)) < 1e-12: continue

            V_interp = interp1d(rho_grid, V_proj_rho, kind='cubic',
                                fill_value=0, bounds_error=False)
            abel_cache[(lv_raw, mv)] = (V_interp, use_sin)

        if not abel_cache:
            continue

        # Only atoms whose z lies within the layer: z_rel in [0, cz)
        n_added = 0
        for pos in at_s['positions']:
            z_at_ang = pos[2] * c_ang
            z_rel = (z_at_ang - z0_ang) % c_ang
            if z_rel >= cz_ang - 1e-4:  # exclude upper boundary (→ next layer)
                continue

            x_at = pos[0] * conv[0,0] + pos[1] * conv[1,0] + pos[2] * conv[2,0]
            y_at = pos[0] * conv[0,1] + pos[1] * conv[1,1] + pos[2] * conv[2,1]

            for (lv_raw, mv), (V_interp, use_sin) in abel_cache.items():
                for px in range(-1, 2):
                    for py in range(-1, 2):
                        dx = X - (x_at + px * a_b)
                        dy = Y - (y_at + py * b_b)
                        rho = np.sqrt(dx**2 + dy**2)
                        if mv == 0:
                            dV_proj_mt += V_interp(rho)
                        else:
                            phi = np.arctan2(dy, dx)
                            if use_sin:
                                dV_proj_mt += V_interp(rho) * np.sin(mv * phi)
                            else:
                                dV_proj_mt += V_interp(rho) * np.cos(mv * phi)
            n_added += 1

        if n_added > 0:
            print(f"    Atom {iat+1} ({at_s['element']}): {n_added} positions in layer")

    # Ry·Bohr → 1/Å
    to_kirk = RY_TO_EV * BOHR / H2_2PI_ME
    dV_kirk = (dV_proj_pw + dV_proj_mt) * to_kirk

    print(f"    ΔV: [{dV_kirk.min():.4f}, {dV_kirk.max():.4f}] 1/Å")
    return dV_kirk



# ═══════════════════════════════════════════════════════════════
# TIFF output (ixchel format)
# ═══════════════════════════════════════════════════════════════

def write_ixchel_tiff(filename, data_yx):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    data = data_yx.astype(np.float32)
    vmin, vmax = float(data.min()), float(data.max())
    if vmax == vmin: vmax = vmin + 1e-6
    preview = np.clip((data-vmin)/(vmax-vmin)*255, 0, 255).astype(np.uint8)
    preview_rgb = np.stack([preview, preview, preview], axis=-1)
    meta = np.zeros((1, 2), dtype=np.uint32)
    with tifffile.TiffWriter(filename) as tfw:
        tfw.write(preview_rgb, photometric='rgb',        contiguous=False)
        tfw.write(data,        photometric='minisblack', contiguous=False)
        tfw.write(meta,        photometric='minisblack', contiguous=False)
    print(f"    → {filename}  [{vmin:.4g} … {vmax:.4g}] 1/Å")


# ═══════════════════════════════════════════════════════════════
# Main program
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='WIEN2k native vcoul → ixchel TIF (analytical projection)')
    ap.add_argument('--vcoul-dft', required=True, help='case.vcoul_DFT')
    ap.add_argument('--vcoul-iam', required=True, help='case.vcoul_IAM')
    ap.add_argument('--struct',    required=True, help='case.struct')
    ap.add_argument('--fparams',   required=True, help='fparams.dat')
    ap.add_argument('--cfg',       required=True, help='ixchel crystruct.cfg')
    ap.add_argument('--miller', type=int, nargs=3, default=[0,0,1],
                    metavar=('H','K','L'))
    ap.add_argument('--bfactors', nargs='+', default=[], metavar='Elem:B')
    ap.add_argument('-o', '--output', default=None)
    args = ap.parse_args()

    h, k, l = args.miller
    out_dir = args.output or f"wien2k_analytical_{''.join(str(abs(x)) for x in [h,k,l])}"

    # ── Reading input ──
    print(f"\n[1] Reading struct: {args.struct}")
    struct = read_struct(args.struct)
    print(f"    {struct['lattice_type']}: "
          f"a={struct['a']*BOHR:.4f}, b={struct['b']*BOHR:.4f}, "
          f"c={struct['c']*BOHR:.4f} Å")

    print(f"\n[2] Reading native vcoul files...")
    vcoul_dft = parse_vcoul(args.vcoul_dft)
    vcoul_iam = parse_vcoul(args.vcoul_iam)
    n_pw = vcoul_dft['pw']['n_pw']
    print(f"    DFT: {len(vcoul_dft['atoms'])} atoms, {n_pw} PW")
    print(f"    IAM: {len(vcoul_iam['atoms'])} atoms, {vcoul_iam['pw']['n_pw']} PW")

    print(f"\n[3] Reading fparams: {args.fparams}")
    fp = read_fparams(args.fparams)

    print(f"\n[4] Reading cfg: {args.cfg}")
    cfg = parse_cfg(args.cfg)
    layers = cfg['layers']
    print(f"    {len(layers)} layers, total thickness {cfg['total_z']:.3f} Å")

    # B-factor overrides
    bfactor_override = {}
    for tok in args.bfactors:
        el, val = tok.split(':')
        bfactor_override[el.strip()] = float(val)
    if bfactor_override:
        for ly in layers:
            for at in ly['atoms']:
                if at['name'] in bfactor_override:
                    at['B'] = bfactor_override[at['name']]

    # Display B-factors
    seen = {}
    for ly in layers:
        for at in ly['atoms']:
            key = (at['name'], at['Z'])
            if key not in seen:
                seen[key] = at['B']
    for (sym, Z), B in sorted(seen.items(), key=lambda x: x[1]):
        print(f"    {sym}(Z={Z}): B={B:.3f} Å²  σ_DW={np.sqrt(B/(8*np.pi**2))*1000:.1f} mÅ")

    # ── Projection per layer ──
    print(f"\n[5] Analytical ΔV projection [{h}{k}{l}] per layer...")
    dx_target = 0.025  # ~25 mÅ

    outs = []
    for ly in layers:
        ax_l, by_l = ly['ax'], ly['by']
        cz_l, z0_l = ly['cz'], ly['z0']
        nx_l = max(16, round(ax_l / dx_target))
        ny_l = max(16, round(by_l / dx_target))

        syms = '+'.join(sorted(set(a['name'] for a in ly['atoms'])))
        print(f"\n  Layer '{ly['label']}' ({syms}): "
              f"z0={z0_l:.3f}, cz={cz_l:.3f} Å, {nx_l}×{ny_l}px")

        # Kirkland IAM
        kirk = kirkland_layer_fp(ly['atoms'], ax_l, by_l, nx_l, ny_l, fp)

        # Analytical ΔV projection for this layer
        dpot = project_dV_layer(vcoul_dft, vcoul_iam, struct,
                                (h, k, l), z0_l, cz_l, None, nx_l, ny_l)

        tot = kirk + dpot
        print(f"    Kirk=[{kirk.min():.3f},{kirk.max():.3f}]  "
              f"ΔV=[{dpot.min():.3f},{dpot.max():.3f}]  "
              f"tot=[{tot.min():.3f},{tot.max():.3f}] 1/Å")

        write_ixchel_tiff(f"{out_dir}/{ly['filename']}", tot.T)
        outs.append(f"{out_dir}/{ly['filename']}")

    print(f"\nDone — {len(outs)} files in '{out_dir}/':")
    for f in outs:
        print(f"  {f}")


if __name__ == '__main__':
    main()
