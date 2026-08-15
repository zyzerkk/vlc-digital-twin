#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento 3.2 — Gêmeo Digital com Sombreamento e Physics-Weighted Loss
=========================================================================
Insere um obstáculo físico (LOS Blockage) e força a rede a aprender a
descontinuidade abrupta via função de custo com penalidade 10x na zona de sombra.

Rodar: python src/exp3_2_shadow_loss.py
Saída: assets/resultado_exp3_2_sombreamento.png
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Inicializando Modelo com Sombreamento no dispositivo: {device}")

# ─── 1. Espaço com Obstáculo Quadrado Físico ─────────────────────────────────
room_size = 3.0
x = np.linspace(-room_size / 2, room_size / 2, 100)
y = np.linspace(-room_size / 2, room_size / 2, 100)
X, Y = np.meshgrid(x, y)

x_flat = X.flatten().reshape(-1, 1)
y_flat = Y.flatten().reshape(-1, 1)

d_sq           = x_flat**2 + y_flat**2 + 2.5**2
intensity_base = 2.0 / (d_sq**1.5)

# Obstáculo físico central: bloco de [-0.5m, 0.5m] em X e Y
obstacle_mask  = (np.abs(x_flat) < 0.5) & (np.abs(y_flat) < 0.5)
intensity_real = np.copy(intensity_base)
intensity_real[obstacle_mask] = 0.0   # Bloqueio total da luz (sombra perfeita)
intensity_norm = intensity_real / intensity_real.max()


# ─── 2. Rede Neural com Fourier de Alta Frequência ───────────────────────────
class ShadowDigitalTwin(nn.Module):
    """
    Gêmeo Digital 2D com Fourier Features de alta frequência (σ=5.0).
    σ maior captura a transição abrupta na borda da sombra.
    """
    def __init__(self, mapping_size=64):
        super().__init__()
        self.B = (torch.randn((2, mapping_size)) * 5.0).to(device)
        self.net = nn.Sequential(
            nn.Linear(mapping_size * 2, 256), nn.GELU(),
            nn.Linear(256, 256),              nn.GELU(),
            nn.Linear(256, 128),              nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, xy):
        x_proj   = 2.0 * np.pi * xy @ self.B
        features = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.net(features)


model     = ShadowDigitalTwin().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ─── 3. Função de Custo Informada pela Física ────────────────────────────────
def custom_physics_loss(preds, targets, mask, shadow_weight=10.0):
    """
    MSE com penalidade amplificada na região de sombra.
    Garante que a rede priorize aprender a fronteira sombra/luz.
    """
    base_mse       = (preds - targets) ** 2
    physics_weight = torch.ones_like(targets)
    physics_weight[mask] = shadow_weight
    return torch.mean(base_mse * physics_weight)


inputs      = torch.tensor(np.hstack((x_flat, y_flat)), dtype=torch.float32).to(device)
targets     = torch.tensor(intensity_norm, dtype=torch.float32).to(device)
mask_tensor = torch.tensor(obstacle_mask, dtype=torch.bool).to(device)


# ─── 4. Treinamento ──────────────────────────────────────────────────────────
losses = []
print("Treinando Modelo de Sombreamento com Physics-Weighted Loss...")
for epoch in range(1001):
    optimizer.zero_grad()
    preds = model(inputs)
    loss  = custom_physics_loss(preds, targets, mask_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 500 == 0:
        print(f"  Época {epoch:4d} | Loss: {loss.item():.6f}")


# ─── 5. Visualização Comparativa ─────────────────────────────────────────────
with torch.no_grad():
    I_pred = model(inputs).cpu().numpy().reshape(100, 100)
    I_real = targets.cpu().numpy().reshape(100, 100)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.contourf(X, Y, I_real, levels=50, cmap='inferno')
plt.title("Mundo Real (Sombra Real)")
plt.xlabel("X (m)"); plt.ylabel("Y (m)")

plt.subplot(1, 2, 2)
plt.contourf(X, Y, I_pred, levels=50, cmap='inferno')
plt.title("Previsão IA (Sombra Aprendida)")
plt.xlabel("X (m)"); plt.ylabel("Y (m)")

plt.tight_layout()
plt.savefig("assets/resultado_exp3_2_sombreamento.png", dpi=150, bbox_inches='tight')
plt.show()
print("Experimento 3.2 concluído! → assets/resultado_exp3_2_sombreamento.png")
