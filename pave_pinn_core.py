"""PAVE-PINN inference in pure NumPy (no PyTorch needed).

Reproduces the trained 5-member deep ensemble exported from the training notebook:
feature layer (43 raw + 17 mechanistic descriptors) -> residual MLP -> hard-constrained
output layer (6 layers x 11 channels) -> mechanistic-empirical layer (eps_t, eps_c, Nf, Nr).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"

# feature indices in the 43-parameter input vector
I_E = [22, 19, 16, 13, 10, 7]   # E_WC, E_ACB, E_AggB, E_SB, E_Fill, E_SG
I_T = [5, 4, 3, 2, 1, 0]        # t_WC, t_ACB, t_AggB, t_SB, t_Fill, t_SG
I_P = 24                        # tyre pressure
I_EACB = 19                     # asphalt base modulus (MPa)


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


class PavePINN:
    def __init__(self, folder: Path | str = MODEL_DIR):
        folder = Path(folder)
        self.meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        w = np.load(folder / "pave_pinn_weights.npz")
        self.members = []
        for k in range(self.meta["n_members"]):
            g = lambda name: w[f"m{k}_{name}"].astype(np.float64)
            self.members.append({
                "kind": g("kind"), "sign": g("sign"), "t_mu": g("t_mu"), "t_sd": g("t_sd"),
                "f_mu": g("f_mu"), "f_sd": g("f_sd"),
                "inp": (g("inp_weight"), g("inp_bias")),
                "blocks": [tuple(g(f"blocks_{b}_{n}") for n in
                                 ("n1_weight", "n1_bias", "l1_weight", "l1_bias", "n2_weight", "n2_bias", "l2_weight", "l2_bias"))
                           for b in range(4)],
                "head": (g("head_0_weight"), g("head_0_bias"), g("head_2_weight"), g("head_2_bias")),
            })
        self.features = self.meta["features"]
        self.x_min = np.array(self.meta["x_min"], float)
        self.x_max = np.array(self.meta["x_max"], float)
        self.x_default = np.array(self.meta["x_default"], float)
        self.pi_scale = np.array(self.meta["pi_scale"], float)
        self.pi_floor = np.array(self.meta["pi_floor"], float)

    # ------------------------------------------------------------------ core
    @staticmethod
    def _features(x):
        E, t = x[:, I_E], x[:, I_T]
        lE = np.log(E)
        he = (t[:, :5] * np.exp((lE[:, :5] - lE[:, 5:6]) / 3)).sum(1, keepdims=True) * 0.9   # Odemark, 5 upper layers
        heb = (t[:, :2] * np.exp((lE[:, :2] - lE[:, 5:6]) / 3)).sum(1, keepdims=True) * 0.9  # Odemark, asphalt courses
        tb = t[:, :2].sum(1, keepdims=True)
        ts = t[:, :4].sum(1, keepdims=True)
        rig = lE[:, 1:2] + 3 * np.log(tb)
        ratios = lE[:, :5] - lE[:, 1:]
        return np.concatenate([x, lE, ratios, he, heb, tb, ts, rig, np.log(x[:, I_P:I_P + 1])], 1)

    def _member(self, m, x):
        h = (self._features(x) - m["f_mu"]) / m["f_sd"]
        W, b = m["inp"]
        h = h @ W.T + b
        for n1w, n1b, l1w, l1b, n2w, n2b, l2w, l2b in m["blocks"]:
            u = _silu(_layer_norm(h, n1w, n1b))
            u = _silu(_layer_norm(u @ l1w.T + l1b, n2w, n2b))
            h = h + (u @ l2w.T + l2b)
        hw, hb, ow, ob = m["head"]
        z = (_silu(_layer_norm(h, hw, hb)) @ ow.T + ob).reshape(-1, 6, 11)
        T = z * m["t_sd"] + m["t_mu"]
        e = np.exp(np.minimum(T, 20.0))
        Y = np.where(m["kind"] == 1, m["sign"] * e, np.where(m["kind"] == 3, np.maximum(e - 1.0, 0.0), T))
        et = np.maximum(Y[:, 1, 5] + Y[:, 1, 10], 1e-3)          # EE11 + CE11, asphalt base course
        ec = np.maximum(-Y[:, 4, 6], 1e-3)                       # |EE22|, subgrade fill
        lNf = -1.0990869322623309 - 3.291 * (np.log10(et) - 6) - 0.854 * np.log10(x[:, I_EACB] * 145.038)
        lNr = -8.8648673486232248 - 4.477 * (np.log10(ec) - 6)
        return Y, np.stack([np.log10(et), np.log10(ec), lNf, lNr], 1)

    def raw(self, X):
        """X: (n, 43) array. Returns ensemble-mean responses Y (n,6,11), mean log10 [et, ec, Nf, Nr] and member spread."""
        X = np.atleast_2d(np.asarray(X, float))
        Ys, Ds = zip(*(self._member(m, X) for m in self.members))
        Y = np.mean(Ys, 0)
        Dl = np.stack(Ds)
        return Y, Dl.mean(0), Dl.std(0)

    # ----------------------------------------------------------- user level
    def predict(self, designs, design_esals: float | None = None, fields: bool = False) -> pd.DataFrame:
        if isinstance(designs, dict):
            designs = pd.DataFrame([designs])
        df = pd.DataFrame(designs).copy()
        df.columns = [str(c).strip() for c in df.columns]
        for f, d in zip(self.features, self.x_default):
            if f not in df.columns:
                df[f] = d
        X = df[self.features].to_numpy(float)
        n = float(design_esals or self.meta["design_esals"])
        Y, mu, sd = self.raw(X)
        hw = self.pi_scale * np.sqrt(sd ** 2 + self.pi_floor ** 2)       # calibrated 95% half-width, log10 units
        Nf, Nr = 10 ** mu[:, 2], 10 ** mu[:, 3]
        span = self.x_max - self.x_min
        out_mask = (X < self.x_min - 1e-9 * span) | (X > self.x_max + 1e-9 * span)
        res = pd.DataFrame({
            "et_microstrain": 10 ** mu[:, 0], "ec_microstrain": 10 ** mu[:, 1],
            "et_low": 10 ** (mu[:, 0] - hw[:, 0]), "et_high": 10 ** (mu[:, 0] + hw[:, 0]),
            "ec_low": 10 ** (mu[:, 1] - hw[:, 1]), "ec_high": 10 ** (mu[:, 1] + hw[:, 1]),
            "Nf": Nf, "Nr": Nr,
            "Nf_low": 10 ** (mu[:, 2] - hw[:, 2]), "Nf_high": 10 ** (mu[:, 2] + hw[:, 2]),
            "Nr_low": 10 ** (mu[:, 3] - hw[:, 3]), "Nr_high": 10 ** (mu[:, 3] + hw[:, 3]),
            "Df": n / Nf, "Dr": n / Nr, "D": n / np.minimum(Nf, Nr),
            "design_life": np.minimum(Nf, Nr),
            "governing_mode": np.where(Nr < Nf, "rutting", "fatigue"),
            "U2_surface_mm": np.abs(Y[:, 0, self.meta["channels"].index("U2")]),
            "inputs_outside_training_range": [", ".join(f for f, b in zip(self.features, r) if b) for r in out_mask],
        })
        if fields:
            for l, lk in enumerate(self.meta["layers"]):
                for c, ch in enumerate(self.meta["channels"]):
                    res[f"{ch}_{lk}"] = Y[:, l, c]
        return res

    def sensitivities(self, x, js):
        """d log10(target)/d log10(x_j) by central differences (2% of range).
        targets: [et, ec, |U2| surface, governing life]"""
        x = np.asarray(x, float)
        iu = self.meta["channels"].index("U2")

        def tg(X):
            Y, mu, _ = self.raw(X)
            return np.stack([mu[:, 0], mu[:, 1], np.log10(np.abs(Y[:, 0, iu])), np.minimum(mu[:, 2], mu[:, 3])], 1)

        A, B = [], []
        for j in js:
            st = 0.02 * (self.x_max[j] - self.x_min[j])
            a, b = x.copy(), x.copy()
            a[j] = max(x[j] - st, self.x_min[j]); b[j] = min(x[j] + st, self.x_max[j])
            if b[j] <= a[j]:
                a[j], b[j] = x[j] * 0.99, x[j] * 1.01
            A.append(a); B.append(b)
        A, B = np.array(A), np.array(B)
        ta, tb = tg(A), tg(B)
        dl = (np.log10(B[np.arange(len(js)), js]) - np.log10(A[np.arange(len(js)), js]))[:, None]
        return {j: (tb[i] - ta[i]) / dl[i] for i, j in enumerate(js)}

    def physics_checks(self, x):
        """The audited physics rules evaluated on one design: 29 sensitivity signs, surface traction, stress attenuation."""
        x = np.asarray(x, float)
        mono = self.meta["mono"]
        S = self.sensitivities(x, sorted({j for _, j, _ in mono}))
        names = ["εt", "εc", "U2"]
        fails = [f"{names[a]} vs {self.features[j]}" for a, j, sg in mono if -sg * S[j][a] > 1e-3]
        Y, _, _ = self.raw(x[None])
        s22 = np.abs(Y[0, :, self.meta["channels"].index("S22")])
        lo, hi = self.meta["bc_band"]
        tr = s22[0] / x[I_P]
        if tr < lo - 0.01 or tr > hi + 0.01:
            fails.append("surface traction")
        if any(s22[i] > s22[i - 1] * 1.001 for i in range(1, 6)):
            fails.append("stress attenuation with depth")
        return len(mono) + 2, fails
