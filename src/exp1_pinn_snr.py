#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento 1 — PINN para Predição de SNR em Canal VLC
=======================================================
Aprende o modelo de radiação Lambertiana para predição de SNR
em função da distância e ângulo de incidência.

Rodar: python src/exp1_pinn_snr.py
Saída: assets/resultado_exp1_snr_pinn.png
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ─── 1. Parâmetros Físicos do Canal VLC ─────────────────────────────────────
m_lambert  = 1.0        # Ordem Lambertiana (FOV ~60°)
A_det      = 1e-4       # Área do fotodetector [m²]
rho        = 0.53       # Responsividade [A/W]
B_bw       = 200e6      # Largura de banda [Hz]
N0         = 1e-21      # Densidade espectral de ruído [W/Hz]
q_elec     = 1.6e-19    # Carga do elétron [C]
Pt         = 1.0        # Potência transmitida [W]


# ─── 2. Formulação Analítica Lambertiana ─────────────────────────────────────
def channel_gain_lambertian(d, theta, m=m_lambert, A=A_det):
    phi = theta
    H = ((m + 1) * A) / (2 * np.pi * d**2)
    H *= np.cos(phi)**m * np.cos(theta)
    return H


def compute_snr(H, Pt=Pt, rho=rho, B=B_bw, N0=N0, q=q_elec):
    signal_power  = (rho * Pt * H)**2
    noise_shot    = 2 * q * rho * Pt * H * B
    noise_thermal = (N0 / 2) * B
    return 10 * np.log10(signal_power / (noise_shot + noise_thermal))


# Geração do grid de dados
d_vals     = np.linspace(0.5, 5.0, 50)
theta_vals = np.linspace(0, np.pi / 3, 30)
D, T       = np.meshgrid(d_vals, theta_vals)
H_grid     = channel_gain_lambertian(D, T)
SNR_grid   = compute_snr(H_grid)

d_norm     = (D.flatten() - 0.5) / 4.5
theta_norm = T.flatten() / (np.pi / 3)
snr_norm   = (SNR_grid.flatten() - SNR_grid.min()) / (SNR_grid.max() - SNR_grid.min())


# ─── 3. Arquitetura PINN ─────────────────────────────────────────────────────
class VLC_PINN(nn.Module):
    """Physics-Informed Neural Network para predição de SNR em canal VLC."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),   nn.Tanh(),
            nn.Linear(64, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


X = torch.tensor(np.stack([d_norm, theta_norm], axis=1), dtype=torch.float32).to(device)
Y = torch.tensor(snr_norm.reshape(-1, 1), dtype=torch.float32).to(device)

model   = VLC_PINN().to(device)
optim   = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()


# ─── 4. Loop de Treinamento ──────────────────────────────────────────────────
losses = []
print("Treinando PINN (Experimento 1)...")
for epoch in range(2001):
    model.train()
    pred = model(X)
    loss = loss_fn(pred, Y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    losses.append(loss.item())
    if epoch % 500 == 0:
        print(f"  Época {epoch:4d} | MSE Loss: {loss.item():.6f}")


# ─── 5. Visualização e Salvamento ────────────────────────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(losses, color='tab:blue')
plt.xlabel("Época")
plt.ylabel("MSE Loss")
plt.title("Curva de Aprendizado — VLC PINN")
plt.yscale("log")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
d_test     = np.linspace(0.5, 5.0, 100)
theta_test = np.zeros(100)
d_t  = torch.tensor((d_test - 0.5) / 4.5, dtype=torch.float32).reshape(-1, 1).to(device)
th_t = torch.tensor(theta_test, dtype=torch.float32).reshape(-1, 1).to(device)

model.eval()
with torch.no_grad():
    pred_snr = model(torch.cat([d_t, th_t], dim=1)).cpu().numpy()
pred_snr = pred_snr * (SNR_grid.max() - SNR_grid.min()) + SNR_grid.min()
real_snr = compute_snr(channel_gain_lambertian(d_test, theta_test))

plt.plot(d_test, real_snr, "b-",  label="Modelo Analítico")
plt.plot(d_test, pred_snr, "r--", label="PINN Predição")
plt.xlabel("Distância [m]")
plt.ylabel("SNR [dB]")
plt.title("SNR vs Distância — θ=0°")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("assets/resultado_exp1_snr_pinn.png", dpi=150, bbox_inches='tight')
plt.show()
print("Experimento 1 concluído! → assets/resultado_exp1_snr_pinn.png")
