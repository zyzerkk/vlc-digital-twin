#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento 3.3 — Gêmeo Digital MIMO-VLC Avançado (4 LEDs + PSNR)
===================================================================
Simula uma matriz 2x2 de 4 transmissores LED e treina um Gêmeo Digital
para aprender o campo de luz sobreposto. Monitora PSNR durante treinamento.

Configuração: 4 LEDs em posições ±1m do centro, a 3m de altura.
Campo total: superposição linear H_total = Σ H_i(x,y,z) para i=1..4

Rodar: python src/exp3_3_mimo_vlc.py
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Inicializando Gêmeo Digital MIMO-VLC no dispositivo: {device}")

# ─── 1. Configuração MIMO ────────────────────────────────────────────────────
room_size = {'x': 4.0, 'y': 4.0, 'z': 3.0}
m_lambert = 1.0

leds = [
    {'x': -1.0, 'y': -1.0, 'z': 3.0},
    {'x':  1.0, 'y': -1.0, 'z': 3.0},
    {'x': -1.0, 'y':  1.0, 'z': 3.0},
    {'x':  1.0, 'y':  1.0, 'z': 3.0},
]

# ─── 2. Campo de Luz Sobreposto (superposição linear) ────────────────────────
x = np.linspace(-room_size['x'] / 2, room_size['x'] / 2, 40)
y = np.linspace(-room_size['y'] / 2, room_size['y'] / 2, 40)
z = np.linspace(0.5, room_size['z'], 20)
X, Y, Z = np.meshgrid(x, y, z)
x_flat = X.flatten().reshape(-1, 1)
y_flat = Y.flatten().reshape(-1, 1)
z_flat = Z.flatten().reshape(-1, 1)

intensity_real = np.zeros_like(x_flat)
for led in leds:
    d_sq  = (x_flat - led['x'])**2 + (y_flat - led['y'])**2 + (z_flat - led['z'])**2
    cos_t = np.abs(z_flat - led['z']) / np.sqrt(d_sq)
    intensity_real += ((m_lambert + 1) / (2 * np.pi)) * (cos_t**m_lambert / d_sq)

intensity_norm = (intensity_real - intensity_real.min()) / (intensity_real.max() - intensity_real.min())


# ─── 3. Arquitetura MIMO Digital Twin ────────────────────────────────────────
class MIMO_DigitalTwin(nn.Module):
    """
    Gêmeo Digital para campo MIMO-VLC.
    σ=3.0 equilibra captura de variações suaves e picos locais.
    Otimizador AdamW oferece melhor generalização por weight decay desacoplado.
    """
    def __init__(self, mapping_size=64):
        super().__init__()
        self.B = (torch.randn((3, mapping_size)) * 3.0).to(device)
        self.net = nn.Sequential(
            nn.Linear(mapping_size * 2, 128), nn.GELU(),
            nn.Linear(128, 128),              nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, xyz):
        x_proj   = 2.0 * np.pi * xyz @ self.B
        features = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.net(features)


vlc_net   = MIMO_DigitalTwin().to(device)
optimizer = torch.optim.AdamW(vlc_net.parameters(), lr=2e-3)
loss_fn   = nn.MSELoss()

inputs  = torch.tensor(np.hstack((x_flat, y_flat, z_flat)), dtype=torch.float32).to(device)
targets = torch.tensor(intensity_norm, dtype=torch.float32).to(device)


# ─── 4. Treinamento com Monitoramento de PSNR ────────────────────────────────
print("Treinando Gêmeo Digital MIMO-VLC...")
for epoch in range(1001):
    optimizer.zero_grad()
    preds = vlc_net(inputs)
    loss  = loss_fn(preds, targets)
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        psnr = 10 * np.log10(1.0 / loss.item())
        print(f"  Época {epoch:4d} | MSE: {loss.item():.6f} | PSNR: {psnr:.2f} dB")

print("MIMO Digital Twin concluído!")
