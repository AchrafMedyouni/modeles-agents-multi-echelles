"""
PINN Fokker-Planck (méthode Chen, Yang, Duan, Karniadakis 2021) appliquée au modèle de Chiarella.

Etat X = (delta, M), delta = P - V (V constant : g = 0, sigma_V = 0) :
    d delta = (f(delta) + beta*tanh(gamma*M)) dt + sigma dW
    d M     = alpha*(f(delta) + beta*tanh(gamma*M) - M) dt + alpha*sigma dW
  f(delta) = demande des fondamentalistes.

Mode "params" : f(delta) = -kappa*delta ; kappa, beta, sigma inconnus (scalaires entraînables).
Mode "nn"     : vraie f = -kappa*delta - kappa3*delta^3, apprise par un réseau f_theta(delta)
                (le "terme manquant"), beta et sigma inconnus aussi.

Données : uniquement des nuages de N agents/réalisations observés à quelques instants.
"""
import sys, time, json
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(1)
MODE = sys.argv[1] if len(sys.argv) > 1 else "params"
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 12000

# ---------------- vrai modèle ----------------
ALPHA, GAMMA = 1.0, 2.0                 # supposés connus
TRUE = dict(kappa=1.0, kappa3=0.5 if MODE == "nn" else 0.0, beta=0.5, sigma=0.5)
T_OBS = [0.0, 0.5, 1.0, 2.0, 3.0]
N_SAMPLES = 2000

def f_true(d):
    return -TRUE["kappa"] * d - TRUE["kappa3"] * d**3

def simulate(n, t_obs, dt=1e-3):
    d = np.random.normal(1.0, 0.3, n); M = np.random.normal(0.0, 0.3, n)
    snaps, t, k = {}, 0.0, 0
    steps = int(round(max(t_obs) / dt))
    for s in range(steps + 1):
        t = s * dt
        if k < len(t_obs) and abs(t - t_obs[k]) < dt / 2:
            snaps[t_obs[k]] = np.stack([d, M], 1).copy(); k += 1
        dW = np.random.normal(0, np.sqrt(dt), n)
        drift = f_true(d) + TRUE["beta"] * np.tanh(GAMMA * M)
        d, M = d + drift * dt + TRUE["sigma"] * dW, M + ALPHA * (drift - M) * dt + ALPHA * TRUE["sigma"] * dW
    return snaps

snaps = simulate(N_SAMPLES, T_OBS)
allx = np.concatenate(list(snaps.values()))
lo, hi = allx.min(0) - 0.5, allx.max(0) + 0.5
LO, HI = torch.tensor(lo, dtype=torch.float32), torch.tensor(hi, dtype=torch.float32)
T1 = max(T_OBS); VOL = float(np.prod(hi - lo))
data = {t: torch.tensor(v, dtype=torch.float32) for t, v in snaps.items()}

# ---------------- réseaux et paramètres ----------------
def mlp(i, o, w=32, depth=4):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, w), nn.Tanh()]; d = w
    layers.append(nn.Linear(d, o)); return nn.Sequential(*layers)

class PDF(nn.Module):              # p~(delta, M, t) > 0
    def __init__(s): super().__init__(); s.net = mlp(3, 1)
    def forward(s, x, t):
        xn = 2 * (x - LO) / (HI - LO) - 1
        tn = 2 * t / T1 - 1
        return nn.functional.softplus(s.net(torch.cat([xn, tn], 1)))

p_net = PDF()
raw = {"beta": torch.tensor(np.log(0.1), requires_grad=True),     # init arbitraire
       "sigma": torch.tensor(np.log(1.0), requires_grad=True)}
if MODE == "params":
    raw["kappa"] = torch.tensor(np.log(0.3), requires_grad=True)
    f_net = None
else:
    f_net = mlp(1, 1, w=20, depth=3)

def P(name): return torch.exp(raw[name])   # positivité

def f_model(d):
    if f_net is None: return -P("kappa") * d
    return f_net(d / 2.0)                   # réseau libre pour la demande des fondamentalistes

params = list(p_net.parameters()) + list(raw.values()) + (list(f_net.parameters()) if f_net else [])
opt = torch.optim.Adam(params, lr=1e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=ITERS // 3, gamma=0.3)

def rand_x(n): return LO + (HI - LO) * torch.rand(n, 2)

def residual(n):
    x = rand_x(n).requires_grad_(True); t = (T1 * torch.rand(n, 1)).requires_grad_(True)
    p = p_net(x, t)
    a1 = f_model(x[:, :1]) + P("beta") * torch.tanh(GAMMA * x[:, 1:2])
    a2 = ALPHA * (a1 - x[:, 1:2])
    s2 = P("sigma") ** 2
    g = lambda y, z: torch.autograd.grad(y, z, torch.ones_like(y), create_graph=True)[0]
    p_t = g(p, t)
    flux1, flux2 = a1 * p, a2 * p
    div = g(flux1, x)[:, :1] + g(flux2, x)[:, 1:2]
    px = g(p, x)
    p11 = g(px[:, :1], x)[:, :1]; p12 = g(px[:, :1], x)[:, 1:2]; p22 = g(px[:, 1:2], x)[:, 1:2]
    diff = 0.5 * s2 * (p11 + 2 * ALPHA * p12 + ALPHA**2 * p22)
    return ((p_t + div - diff) ** 2).mean()

def data_loss():
    L = 0.0
    for t, X in data.items():
        tt = torch.full((X.shape[0], 1), t)
        xi = rand_x(2000); ti = torch.full((2000, 1), t)
        L = L + (-torch.log(p_net(X, tt) + 1e-8).mean() + VOL * p_net(xi, ti).mean())
    return L

TAU = 10.0
hist = []
t0 = time.time()
for it in range(ITERS + 1):
    opt.zero_grad()
    Lr, Ld = residual(2000), data_loss()
    loss = TAU * Lr + Ld
    loss.backward(); opt.step(); sched.step()
    if it % 100 == 0:
        rec = {"it": it, "Lpde": Lr.item(), "Ldata": Ld.item(),
               "beta": P("beta").item(), "sigma": P("sigma").item()}
        if MODE == "params": rec["kappa"] = P("kappa").item()
        hist.append(rec)
        if it % 1000 == 0:
            print(json.dumps(rec), f"{time.time()-t0:.0f}s", flush=True)

out = {"mode": MODE, "true": TRUE, "hist": hist, "lo": lo.tolist(), "hi": hi.tolist(), "t_obs": T_OBS}
if f_net is not None:
    dg = torch.linspace(-1.0, 2.0, 200).unsqueeze(1)
    with torch.no_grad():
        out["f_grid"] = dg.squeeze().tolist(); out["f_learned"] = f_net(dg / 2.0).squeeze().tolist()
    out["delta_range_data"] = [float(np.percentile(allx[:, 0], 1)), float(np.percentile(allx[:, 0], 99))]
# densité apprise vs histogramme à un instant NON observé (t = 1.5)
test = simulate(N_SAMPLES * 5, [1.5])[1.5]
gx = torch.linspace(lo[0], hi[0], 120)
with torch.no_grad():
    mg = torch.linspace(lo[1], hi[1], 120)
    D, Mm = torch.meshgrid(gx, mg, indexing="ij")
    pg = p_net(torch.stack([D.flatten(), Mm.flatten()], 1), torch.full((120 * 120, 1), 1.5)).reshape(120, 120)
    marg = (pg.sum(1) * (mg[1] - mg[0])).tolist()
out["test_t"] = 1.5; out["grid_delta"] = gx.tolist(); out["marg_pinn"] = marg
out["test_samples_delta"] = test[:, 0].tolist()
json.dump(out, open(f"result_{MODE}.json", "w"))
print("done", time.time() - t0)
