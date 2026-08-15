#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento 3.1 — Gêmeo Digital 3D com Fourier Positional Encoding
====================================================================
Resolve o Spectral Bias com Fourier Feature Mapping (inspirado em NeRF).
Redes MLP convencionais suavizam campos de luz e perdem picos de intensidade;
o Positional Encoding força a rede a aprender componentes de alta frequência.

Código atualizado conforme versão executada no Google Colab (agosto/2026).

Rodar: python src/exp3_1_digital_twin_3d.py
Saída: assets/resultado_exp3_1_digital_twin_3d.png
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Inicializando o Gêmeo Digital no dispositivo: {device}")

# ─── 1. Configurações Iniciais ────────────────────────────────────────────────
room_size = {'x': 2.5, 'y': 2.5, 'z': 3.0}
led_pos   = {'x': 0.0, 'y': 0.0, 'z': 3.0}
m_lambert = 1.0

# ─── 2. Geração do Grid Espacial e Física ────────────────────────────────────
x = np.linspace(-room_size['x'] / 2, room_size['x'] / 2, 40)
y = np.linspace(-room_size['y'] / 2, room_size['y'] / 2, 40)
z = np.linspace(0.5, room_size['z'], 30)
X, Y, Z = np.meshgrid(x, y, z)
x_flat = X.flatten().reshape(-1, 1)
y_flat = Y.flatten().reshape(-1, 1)
z_flat = Z.flatten().reshape(-1, 1)

d_squared      = (x_flat - led_pos['x'])**2 + (y_flat - led_pos['y'])**2 + (z_flat - led_pos['z'])**2
cos_theta      = np.abs(z_flat - led_pos['z']) / np.sqrt(d_squared)
intensity_real = (m_lambert + 1) / (2 * np.pi) * cos_theta**m_lambert / d_squared
intensity_norm = (intensity_real - intensity_real.min()) / (intensity_real.max() - intensity_real.min())


# ─── 3. Arquitetura Neural com Fourier Feature Mapping ───────────────────────
class DigitalTwin3D_Fourier(nn.Module):
    """
    Gêmeo Digital 3D com Fourier Positional Encoding.
    σ=2.0 captura frequências médias do campo de luz Lambertiano.
    """
    def __init__(self, mapping_size=64):
        super().__init__()
        self.B = (torch.randn((3, mapping_size)) * 2.0).to(device)
        self.net = nn.Sequential(
            nn.Linear(mapping_size * 2, 128), nn.Tanh(),
            nn.Linear(128, 128),              nn.Tanh(),
            nn.Linear(128, 128),              nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, xyz):
        x_proj   = 2.0 * np.pi * xyz @ self.B
        features = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.net(features)


vlc_net   = DigitalTwin3D_Fourier().to(device)
optimizer = torch.optim.Adam(vlc_net.parameters(), lr=1e-3)
loss_fn   = nn.MSELoss()

inputs  = torch.tensor(np.hstack((x_flat, y_flat, z_flat)), dtype=torch.float32).to(device)
targets = torch.tensor(intensity_norm, dtype=torch.float32).to(device)


# ─── 4. Loop de Treinamento ──────────────────────────────────────────────────
losses = []
print("\nTreinando Modelo 3D Espacial (1000 Épocas)...")
for epoch in range(1001):
    optimizer.zero_grad()
    preds = vlc_net(inputs)
    loss  = loss_fn(preds, targets)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 500 == 0:
        print(f"  Época {epoch:4d} | Loss: {loss.item():.6f}")


# ─── 5. Renderização e Plotagem ──────────────────────────────────────────────
print("\nGerando gráficos de saída...")
Z_plane  = 0.5
X_plane, Y_plane = np.meshgrid(
    np.linspace(-1.25, 1.25, 100),
    np.linspace(-1.25, 1.25, 100)
)
Z_arr = np.full_like(X_plane, Z_plane)

inputs_plane = torch.tensor(
    np.column_stack((X_plane.flatten(), Y_plane.flatten(), Z_arr.flatten())),
    dtype=torch.float32
).to(device)

with torch.no_grad():
    I_pred = vlc_net(inputs_plane).cpu().numpy().reshape(100, 100)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses, color='purple')
plt.title("Convergência Espacial (Fourier)")
plt.yscale("log")
plt.xlabel("Épocas")
plt.ylabel("MSE Loss")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
contour = plt.contourf(X_plane, Y_plane, I_pred, levels=50, cmap='magma')
plt.colorbar(contour, label='Intensidade Luminosa Predita')
plt.title(f"Mapa de Calor IA (Plano Z={Z_plane}m)")
plt.xlabel("Eixo X (m)")
plt.ylabel("Eixo Y (m)")

plt.tight_layout()
plt.savefig("assets/resultado_exp3_1_digital_twin_3d.png", dpi=150, bbox_inches='tight')
plt.show()
print("Experimento 3.1 concluído! → assets/resultado_exp3_1_digital_twin_3d.png")
