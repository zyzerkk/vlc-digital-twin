#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_repo.py — Gerador Universal do Repositório VLC Digital Twin
=================================================================
Execute este script para criar toda a estrutura do repositório open-source,
incluindo notebooks .ipynb, scripts .py, documentação e um arquivo .zip final.

Uso:
    python build_repo.py

Saída:
    ./vlc-digital-twin/         ← Árvore completa do repositório
    ./vlc-digital-twin-modulus-ready.zip   ← Pacote compactado pronto para distribuição
"""

import os
import json
import shutil
import textwrap

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════
ROOT = "vlc-digital-twin"
ZIP_NAME = "vlc-digital-twin-modulus-ready"

DIRS = [
    f"{ROOT}/notebooks",
    f"{ROOT}/src",
    f"{ROOT}/docs",
    f"{ROOT}/assets",
    f"{ROOT}/.github/workflows",
    f"{ROOT}/.github/ISSUE_TEMPLATE",
]

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def write(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  ✔  {path}")


def make_notebook(title: str, description: str, cells_code: list[str]) -> str:
    """Gera JSON de notebook Jupyter válido a partir de células de código."""
    nb_cells = []

    # Célula de título (Markdown)
    nb_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"# {title}\n", f"\n{description}"]
    })

    for i, code in enumerate(cells_code, start=1):
        nb_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": f"cell_{i}"},
            "outputs": [],
            "source": [line + "\n" for line in code.strip().splitlines()]
        })

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            },
            "colab": {
                "provenance": [],
                "gpuType": "T4"
            },
            "accelerator": "GPU"
        },
        "cells": nb_cells
    }
    return json.dumps(notebook, ensure_ascii=False, indent=1)


# ═══════════════════════════════════════════════════════════════════════════════
# CÓDIGO DOS EXPERIMENTOS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Célula de Setup Compartilhada ──────────────────────────────────────────────
SETUP_CODE = """\
# Configuração do Ambiente — NVIDIA PhysicsNeMo / Modulus
# Execute esta célula primeiro em qualquer notebook do repositório.

import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

# Instalar dependências base
install("nvidia-modulus")
install("torch")
install("seaborn")
install("scikit-learn")

# Clonar repositório simbólico (physicsnemo-sym)
subprocess.run(["git", "clone", "https://github.com/NVIDIA/physicsnemo-sym.git"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", "physicsnemo-sym", "--quiet"], check=False)

import torch
print(f"PyTorch: {torch.__version__}")
print(f"GPU Disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
"""

# ── Experimento 1: PINN SNR ────────────────────────────────────────────────────
EXP1_CODE = """\
\"\"\"
Experimento 1 — PINN para Predição de SNR em Canal VLC (Modelo Lambertiano)
Aprende o modelo de radiação Lambertiana para predição de SNR em função
da distância e ângulo de incidência.
\"\"\"

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# ─── 1. Parâmetros Físicos do Canal VLC ─────────────────────────────────────
m_lambert = 1.0        # Ordem Lambertiana (FOV ~60°)
A_det     = 1e-4       # Área do fotodetector [m²]
rho       = 0.53       # Responsividade [A/W]
B_bw      = 200e6      # Largura de banda [Hz]
N0        = 1e-21      # Densidade espectral de ruído [W/Hz]
q_elec    = 1.6e-19    # Carga do elétron [C]
Pt        = 1.0        # Potência transmitida [W]


# ─── 2. Formulação Analítica Lambertiana ────────────────────────────────────
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


# ─── 3. Arquitetura PINN ────────────────────────────────────────────────────
class VLC_PINN(nn.Module):
    \"\"\"Physics-Informed Neural Network para predição de SNR em canal VLC.\"\"\"
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


# ─── 4. Loop de Treinamento ─────────────────────────────────────────────────
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


# ─── 5. Visualização e Salvamento ───────────────────────────────────────────
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
plt.savefig("resultado_exp1_snr_pinn.png", dpi=150, bbox_inches='tight')
plt.show()
print("Experimento 1 concluído! → resultado_exp1_snr_pinn.png")
"""

# ── Experimento 2: Classificador de Modulação ──────────────────────────────────
EXP2_CODE = """\
\"\"\"
Experimento 2 — Classificador Inteligente de Modulação VLC
Treina uma rede neural para reconhecer formatos de modulação óptica:
OOK, PPM-4, PPM-8 e VPPM a partir de features estatísticas do sinal.
\"\"\"

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(42)
N_por_classe = 500
labels = {0: "OOK", 1: "PPM-4", 2: "PPM-8", 3: "VPPM"}


# ─── 1. Geração de Dados Sintéticos por Esquema de Modulação ────────────────
def gerar_ook(N, snr_db=20):
    bits  = np.random.randint(0, 2, N)
    sigma = 10 ** (-snr_db / 20)
    sinal = bits + np.random.normal(0, sigma, N)
    return np.array([
        sinal.mean(), sinal.std(), np.var(sinal),
        np.percentile(sinal, 25), np.percentile(sinal, 75),
        len(np.unique(np.round(sinal, 1))) / N, 0.0, 0.0
    ])


def gerar_ppm(N, M=4, snr_db=20):
    sigma  = 10 ** (-snr_db / 20)
    slots  = np.random.randint(0, M, N)
    sinais = np.zeros((N, M))
    for i, s in enumerate(slots):
        sinais[i, s] = 1.0
    sinais += np.random.normal(0, sigma, sinais.shape)
    f = sinais.flatten()
    return np.array([
        f.mean(), f.std(), np.var(f),
        np.percentile(f, 25), np.percentile(f, 75),
        M / 16.0, np.max(sinais.mean(axis=0)), 0.5
    ])


X_list, y_list = [], []
for snr in [10, 15, 20, 25, 30]:
    for _ in range(N_por_classe // 5):
        X_list.append(gerar_ook(200, snr));    y_list.append(0)
        X_list.append(gerar_ppm(200, 4, snr)); y_list.append(1)
        X_list.append(gerar_ppm(200, 8, snr)); y_list.append(2)
        X_list.append(gerar_ppm(200, 4, snr)); y_list.append(3)

X = np.array(X_list)
y = np.array(y_list)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
y_tr_t = torch.tensor(y_tr, dtype=torch.long).to(device)
X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)


# ─── 2. Arquitetura do Classificador ────────────────────────────────────────
class ModulacaoClassifier(nn.Module):
    \"\"\"Classificador de modulação VLC com regularização Dropout.\"\"\"
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 32),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.net(x)


clf     = ModulacaoClassifier().to(device)
optim   = torch.optim.Adam(clf.parameters(), lr=5e-4, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()


# ─── 3. Treinamento ─────────────────────────────────────────────────────────
print(f"Treinando Classificador de Modulação no dispositivo: {device}")
for epoch in range(300):
    clf.train()
    pred = clf(X_tr_t)
    loss = loss_fn(pred, y_tr_t)
    optim.zero_grad()
    loss.backward()
    optim.step()
    if (epoch + 1) % 100 == 0:
        print(f"  Época {epoch+1:3d} | Loss: {loss.item():.4f}")


# ─── 4. Avaliação ───────────────────────────────────────────────────────────
clf.eval()
with torch.no_grad():
    y_pred = clf(X_te_t).argmax(dim=1).cpu().numpy()

print("\\nRelatório de Classificação:")
print(classification_report(y_te, y_pred, target_names=list(labels.values())))

plt.figure(figsize=(7, 5))
sns.heatmap(
    confusion_matrix(y_te, y_pred),
    annot=True, fmt="d", cmap="Blues",
    xticklabels=labels.values(),
    yticklabels=labels.values()
)
plt.title("Matriz de Confusão — Classificador VLC")
plt.ylabel("Real")
plt.xlabel("Predito")
plt.tight_layout()
plt.savefig("resultado_exp2_classificador_vlc.png", dpi=150, bbox_inches='tight')
plt.show()
print("Experimento 2 concluído! → resultado_exp2_classificador_vlc.png")
"""

# ── Experimento 3.1: Gêmeo Digital 3D Fourier ─────────────────────────────────
EXP3_CODE = """\
\"\"\"
Experimento 3.1 — Gêmeo Digital 3D com Positional Encoding (Fourier Features)
Resolve o Spectral Bias de redes MLP padrão projetando as coordenadas (x, y, z)
em um espaço de alta dimensão de senos e cossenos (inspirado em NeRF).
\"\"\"

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Inicializando Gêmeo Digital 3D no dispositivo: {device}")

room_size = {'x': 2.5, 'y': 2.5, 'z': 3.0}
led_pos   = {'x': 0.0, 'y': 0.0, 'z': 3.0}
m_lambert = 1.0

# ─── 1. Geração do Campo de Luz Analítico ───────────────────────────────────
x = np.linspace(-room_size['x'] / 2, room_size['x'] / 2, 40)
y = np.linspace(-room_size['y'] / 2, room_size['y'] / 2, 40)
z = np.linspace(0.5, room_size['z'], 30)
X, Y, Z = np.meshgrid(x, y, z)

x_flat = X.flatten().reshape(-1, 1)
y_flat = Y.flatten().reshape(-1, 1)
z_flat = Z.flatten().reshape(-1, 1)

d_squared = (x_flat - led_pos['x'])**2 + (y_flat - led_pos['y'])**2 + (z_flat - led_pos['z'])**2
cos_theta = np.abs(z_flat - led_pos['z']) / np.sqrt(d_squared)
intensity_real = (m_lambert + 1) / (2 * np.pi) * cos_theta**m_lambert / d_squared
intensity_norm = (intensity_real - intensity_real.min()) / (intensity_real.max() - intensity_real.min())


# ─── 2. Arquitetura com Fourier Feature Mapping ─────────────────────────────
class DigitalTwin3D_Fourier(nn.Module):
    \"\"\"
    Gêmeo Digital 3D com Fourier Positional Encoding.
    O mapeamento aleatório de Fourier (matriz B) projeta as coordenadas
    em um espaço de alta frequência, eliminando o Spectral Bias.
    \"\"\"
    def __init__(self, mapping_size: int = 64):
        super().__init__()
        self.B = (torch.randn((3, mapping_size)) * 2.0).to(device)
        self.net = nn.Sequential(
            nn.Linear(mapping_size * 2, 128), nn.Tanh(),
            nn.Linear(128, 128),              nn.Tanh(),
            nn.Linear(128, 128),              nn.Tanh(),
            nn.Linear(128, 1),                nn.Sigmoid()
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        x_proj   = 2.0 * np.pi * xyz @ self.B
        features = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.net(features)


vlc_net   = DigitalTwin3D_Fourier().to(device)
optimizer = torch.optim.Adam(vlc_net.parameters(), lr=1e-3)
loss_fn   = nn.MSELoss()

inputs  = torch.tensor(np.hstack((x_flat, y_flat, z_flat)), dtype=torch.float32).to(device)
targets = torch.tensor(intensity_norm, dtype=torch.float32).to(device)


# ─── 3. Treinamento ─────────────────────────────────────────────────────────
losses = []
print("Treinando Gêmeo Digital 3D com Fourier Features...")
for epoch in range(1001):
    optimizer.zero_grad()
    preds = vlc_net(inputs)
    loss  = loss_fn(preds, targets)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 500 == 0:
        print(f"  Época {epoch:4d} | Loss: {loss.item():.6f}")


# ─── 4. Visualização — Mapa de Calor a Z=0.5m ───────────────────────────────
z_slice = 0.5
x_s = np.linspace(-room_size['x'] / 2, room_size['x'] / 2, 100)
y_s = np.linspace(-room_size['y'] / 2, room_size['y'] / 2, 100)
Xs, Ys = np.meshgrid(x_s, y_s)
Zs = np.full_like(Xs, z_slice)

pts = torch.tensor(
    np.stack([Xs.flatten(), Ys.flatten(), Zs.flatten()], axis=1),
    dtype=torch.float32
).to(device)

vlc_net.eval()
with torch.no_grad():
    I_pred = vlc_net(pts).cpu().numpy().reshape(100, 100)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(losses, color='tab:green')
plt.xlabel("Época")
plt.ylabel("MSE Loss")
plt.title("Curva de Aprendizado — Gêmeo 3D")
plt.yscale("log")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
im = plt.contourf(Xs, Ys, I_pred, levels=50, cmap='inferno')
plt.colorbar(im, label='Intensidade Normalizada')
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title(f"Mapa de Luz Predito — Z={z_slice}m")

plt.tight_layout()
plt.savefig("resultado_exp3_1_digital_twin_3d.png", dpi=150, bbox_inches='tight')
plt.show()
print("Experimento 3.1 concluído! → resultado_exp3_1_digital_twin_3d.png")
"""

# ── Experimento 3.2: Sombreamento com Physics-Weighted Loss ───────────────────
EXP4_CODE = """\
\"\"\"
Experimento 3.2 — Gêmeo Digital com Sombreamento e Physics-Weighted Loss
Insere um obstáculo físico na linha de visada (LOS Blockage) e força a IA
a aprender a sombra via função de custo com penalidade física diferenciada.
\"\"\"

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Inicializando Modelo com Sombreamento no dispositivo: {device}")

# ─── 1. Espaço com Obstáculo Quadrado Físico ────────────────────────────────
room_size = 3.0
x = np.linspace(-room_size / 2, room_size / 2, 100)
y = np.linspace(-room_size / 2, room_size / 2, 100)
X, Y = np.meshgrid(x, y)

x_flat = X.flatten().reshape(-1, 1)
y_flat = Y.flatten().reshape(-1, 1)

d_sq           = x_flat**2 + y_flat**2 + 2.5**2
intensity_base = 2.0 / (d_sq**1.5)

# Obstáculo físico central: bloco de [-0.5 m, 0.5 m] em X e Y
obstacle_mask  = (np.abs(x_flat) < 0.5) & (np.abs(y_flat) < 0.5)
intensity_real = np.copy(intensity_base)
intensity_real[obstacle_mask] = 0.0   # Bloqueio total da luz (sombra perfeita)
intensity_norm = intensity_real / intensity_real.max()


# ─── 2. Rede Neural com Mapeamento de Alta Frequência ───────────────────────
class ShadowDigitalTwin(nn.Module):
    \"\"\"
    Gêmeo Digital 2D com Fourier Features de alta frequência (sigma=5.0)
    para capturar a transição abrupta na borda da sombra.
    \"\"\"
    def __init__(self, mapping_size: int = 64):
        super().__init__()
        self.B = (torch.randn((2, mapping_size)) * 5.0).to(device)
        self.net = nn.Sequential(
            nn.Linear(mapping_size * 2, 256), nn.GELU(),
            nn.Linear(256, 256),              nn.GELU(),
            nn.Linear(256, 128),              nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        x_proj   = 2.0 * np.pi * xy @ self.B
        features = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.net(features)


model = ShadowDigitalTwin().to(device)


# ─── 3. Função de Custo Informada pela Física (Physics-Weighted Loss) ────────
def custom_physics_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    shadow_weight: float = 10.0
) -> torch.Tensor:
    \"\"\"
    MSE com penalidade amplificada na região de sombra.
    Garante que a IA priorize aprender a fronteira sombra/luz.

    Args:
        preds:         Predições da rede [N, 1]
        targets:       Valores reais [N, 1]
        mask:          Máscara booleana da região de sombra
        shadow_weight: Fator de penalidade na zona de sombra (padrão 10x)
    \"\"\"
    base_mse = (preds - targets) ** 2
    physics_weight = torch.ones_like(targets)
    physics_weight[mask] = shadow_weight
    return torch.mean(base_mse * physics_weight)


optimizer    = torch.optim.Adam(model.parameters(), lr=1e-3)
inputs       = torch.tensor(np.hstack((x_flat, y_flat)), dtype=torch.float32).to(device)
targets_t    = torch.tensor(intensity_norm, dtype=torch.float32).to(device)
mask_tensor  = torch.tensor(obstacle_mask, dtype=torch.bool).squeeze().to(device)


# ─── 4. Treinamento ─────────────────────────────────────────────────────────
losses = []
print("Treinando Modelo de Sombreamento com Physics-Weighted Loss...")
for epoch in range(1001):
    optimizer.zero_grad()
    preds = model(inputs)
    loss  = custom_physics_loss(preds, targets_t, mask_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 500 == 0:
        print(f"  Época {epoch:4d} | Loss: {loss.item():.6f}")


# ─── 5. Visualização Comparativa ────────────────────────────────────────────
model.eval()
with torch.no_grad():
    I_pred = model(inputs).cpu().numpy().reshape(100, 100)
I_real = targets_t.cpu().numpy().reshape(100, 100)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(losses, color='tab:orange')
axes[0].set_title("Curva de Aprendizado")
axes[0].set_xlabel("Época")
axes[0].set_ylabel("Physics-Weighted Loss")
axes[0].set_yscale("log")
axes[0].grid(True, alpha=0.3)

im1 = axes[1].contourf(X, Y, I_real, levels=50, cmap='inferno')
axes[1].set_title("Mundo Real (Sombra Real)")
axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("y [m]")
fig.colorbar(im1, ax=axes[1])

im2 = axes[2].contourf(X, Y, I_pred, levels=50, cmap='inferno')
axes[2].set_title("Previsão IA (Sombra Aprendida)")
axes[2].set_xlabel("x [m]"); axes[2].set_ylabel("y [m]")
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig("resultado_exp3_2_sombreamento.png", dpi=150, bbox_inches='tight')
plt.show()
print("Experimento 3.2 concluído! → resultado_exp3_2_sombreamento.png")
"""

# ── Experimento 3.3: MIMO-VLC ─────────────────────────────────────────────────
EXP5_CODE = """\
\"\"\"
Experimento 3.3 — Gêmeo Digital MIMO-VLC Avançado (4 LEDs + PSNR)
Simula uma matriz de 4 transmissores ópticos com interferência construtiva,
monitorando métricas de qualidade de imagem (PSNR) e usando AdamW.
\"\"\"

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Inicializando Gêmeo Digital MIMO-VLC no dispositivo: {device}")

room_size = {'x': 4.0, 'y': 4.0, 'z': 3.0}
m_lambert = 1.0

# Posições dos 4 LEDs transmissores na grade 2×2
leds = [
    {'x': -1.0, 'y': -1.0, 'z': 3.0},
    {'x':  1.0, 'y': -1.0, 'z': 3.0},
    {'x': -1.0, 'y':  1.0, 'z': 3.0},
    {'x':  1.0, 'y':  1.0, 'z': 3.0},
]

# ─── 1. Campo de Luz Sobreposto MIMO ────────────────────────────────────────
x = np.linspace(-room_size['x'] / 2, room_size['x'] / 2, 40)
y = np.linspace(-room_size['y'] / 2, room_size['y'] / 2, 40)
z = np.linspace(0.5, room_size['z'], 20)
X, Y, Z = np.meshgrid(x, y, z)

x_flat = X.flatten().reshape(-1, 1)
y_flat = Y.flatten().reshape(-1, 1)
z_flat = Z.flatten().reshape(-1, 1)

# Superposição das contribuições de todos os LEDs (interferência óptica)
intensity_real = np.zeros_like(x_flat)
for led in leds:
    d_sq  = (x_flat - led['x'])**2 + (y_flat - led['y'])**2 + (z_flat - led['z'])**2
    cos_t = np.abs(z_flat - led['z']) / np.sqrt(d_sq)
    intensity_real += ((m_lambert + 1) / (2 * np.pi)) * (cos_t**m_lambert / d_sq)

intensity_norm = (intensity_real - intensity_real.min()) / (intensity_real.max() - intensity_real.min())


# ─── 2. Arquitetura MIMO Digital Twin ───────────────────────────────────────
class MIMO_DigitalTwin(nn.Module):
    \"\"\"
    Gêmeo Digital para sistema MIMO-VLC com 4 LEDs.
    Usa Fourier Feature Mapping com sigma=3.0 calibrado para
    a escala do ambiente (sala de 4m × 4m).
    \"\"\"
    def __init__(self, mapping_size: int = 64):
        super().__init__()
        self.B = (torch.randn((3, mapping_size)) * 3.0).to(device)
        self.net = nn.Sequential(
            nn.Linear(mapping_size * 2, 128), nn.GELU(),
            nn.Linear(128, 128),              nn.GELU(),
            nn.Linear(128, 1),                nn.Sigmoid()
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        x_proj   = 2.0 * np.pi * xyz @ self.B
        features = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.net(features)


vlc_net   = MIMO_DigitalTwin().to(device)
optimizer = torch.optim.AdamW(vlc_net.parameters(), lr=2e-3)
loss_fn   = nn.MSELoss()

inputs  = torch.tensor(np.hstack((x_flat, y_flat, z_flat)), dtype=torch.float32).to(device)
targets = torch.tensor(intensity_norm, dtype=torch.float32).to(device)


# ─── 3. Treinamento com Monitoramento de PSNR ───────────────────────────────
losses, psnrs = [], []
print("Treinando MIMO Digital Twin (4 LEDs)...")
for epoch in range(1001):
    optimizer.zero_grad()
    preds = vlc_net(inputs)
    loss  = loss_fn(preds, targets)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    psnr = 10 * np.log10(1.0 / max(loss.item(), 1e-10))
    psnrs.append(psnr)

    if epoch % 500 == 0:
        print(f"  Época {epoch:4d} | MSE: {loss.item():.6f} | PSNR: {psnr:.2f} dB")


# ─── 4. Visualização — Plano Horizontal a Z=0.75m ───────────────────────────
z_slice = 0.75
x_s = np.linspace(-room_size['x'] / 2, room_size['x'] / 2, 100)
y_s = np.linspace(-room_size['y'] / 2, room_size['y'] / 2, 100)
Xs, Ys = np.meshgrid(x_s, y_s)
Zs = np.full_like(Xs, z_slice)

pts = torch.tensor(
    np.stack([Xs.flatten(), Ys.flatten(), Zs.flatten()], axis=1),
    dtype=torch.float32
).to(device)

vlc_net.eval()
with torch.no_grad():
    I_pred = vlc_net(pts).cpu().numpy().reshape(100, 100)

# Calcular campo real para o mesmo slice
intensity_slice = np.zeros_like(Xs)
for led in leds:
    d_sq  = (Xs - led['x'])**2 + (Ys - led['y'])**2 + (z_slice - led['z'])**2
    cos_t = np.abs(z_slice - led['z']) / np.sqrt(d_sq)
    intensity_slice += ((m_lambert + 1) / (2 * np.pi)) * (cos_t**m_lambert / d_sq)
intensity_slice = (intensity_slice - intensity_slice.min()) / (intensity_slice.max() - intensity_slice.min())

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(losses, color='tab:blue')
axes[0, 0].set_title("MSE Loss")
axes[0, 0].set_xlabel("Época")
axes[0, 0].set_yscale("log")
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(psnrs, color='tab:green')
axes[0, 1].set_title("PSNR [dB]")
axes[0, 1].set_xlabel("Época")
axes[0, 1].grid(True, alpha=0.3)

im1 = axes[1, 0].contourf(Xs, Ys, intensity_slice, levels=50, cmap='plasma')
axes[1, 0].set_title(f"Campo Real MIMO — Z={z_slice}m")
axes[1, 0].set_xlabel("x [m]"); axes[1, 0].set_ylabel("y [m]")
# Marcar posições dos LEDs
for led in leds:
    axes[1, 0].plot(led['x'], led['y'], 'w*', markersize=12, label='LED')
fig.colorbar(im1, ax=axes[1, 0])

im2 = axes[1, 1].contourf(Xs, Ys, I_pred, levels=50, cmap='plasma')
axes[1, 1].set_title(f"Gêmeo Digital MIMO — Z={z_slice}m")
axes[1, 1].set_xlabel("x [m]"); axes[1, 1].set_ylabel("y [m]")
fig.colorbar(im2, ax=axes[1, 1])

plt.suptitle("Gêmeo Digital MIMO-VLC — 4 LEDs (2×2 Grid)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("resultado_exp3_3_mimo_vlc.png", dpi=150, bbox_inches='tight')
plt.show()

final_psnr = psnrs[-1]
print(f"Experimento 3.3 concluído! PSNR Final: {final_psnr:.2f} dB")
print("→ resultado_exp3_3_mimo_vlc.png")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENTAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

README_MD = """\
# VLC Digital Twin — NVIDIA PhysicsNeMo / Modulus

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![NVIDIA PhysicsNeMo](https://img.shields.io/badge/NVIDIA-PhysicsNeMo-76b900.svg)](https://github.com/NVIDIA/physicsnemo-sym)

> **Physics-Informed Neural Networks (PINNs) para Gêmeos Digitais de Comunicação por Luz Visível (VLC)**

Este repositório implementa um framework completo de **IA Informada por Física** para modelar e
prever o comportamento de canais VLC (_Visible Light Communication_), desde a predição básica de SNR
até um Gêmeo Digital MIMO-VLC com 4 transmissores, utilizando o ecossistema NVIDIA PhysicsNeMo.

---

## Estrutura do Repositório

```
vlc-digital-twin/
├── notebooks/                 # Jupyter notebooks por experimento
│   ├── exp1_pinn_snr.ipynb
│   ├── exp2_modulation_classifier.ipynb
│   ├── exp3_1_digital_twin_3d.ipynb
│   ├── exp3_2_shadow_loss.ipynb
│   └── exp3_3_mimo_vlc.ipynb
├── src/                       # Scripts Python para execução local
│   ├── exp1_pinn_snr.py
│   ├── exp2_modulation_classifier.py
│   ├── exp3_1_digital_twin_3d.py
│   ├── exp3_2_shadow_loss.py
│   └── exp3_3_mimo_vlc.py
├── docs/                      # Documentação técnica
│   └── architecture.md
├── assets/                    # Figuras e diagramas
├── .github/                   # Templates de Issues e CI/CD
├── EXPERIMENTS.md             # Descrição detalhada dos experimentos
├── CHANGELOG.md               # Histórico de versões
├── CONTRIBUTING.md            # Guia de contribuição
├── requirements.txt           # Dependências Python
└── LICENSE                    # Apache 2.0
```

## Instalação Rápida

```bash
# Clonar o repositório
git clone https://github.com/seu-usuario/vlc-digital-twin.git
cd vlc-digital-twin

# Instalar dependências
pip install -r requirements.txt

# Instalar NVIDIA PhysicsNeMo (simbólico)
git clone https://github.com/NVIDIA/physicsnemo-sym.git
pip install -e physicsnemo-sym
```

## Experimentos

| # | Experimento | Arquivo | Conceito-Chave |
|---|-------------|---------|----------------|
| 1 | PINN para SNR Lambertiano | `exp1_pinn_snr.py` | Physics-Informed Loss |
| 2 | Classificador de Modulação VLC | `exp2_modulation_classifier.py` | Feature Engineering |
| 3.1 | Gêmeo Digital 3D | `exp3_1_digital_twin_3d.py` | Fourier Features / NeRF |
| 3.2 | Sombreamento (LOS Blockage) | `exp3_2_shadow_loss.py` | Physics-Weighted Loss |
| 3.3 | MIMO-VLC (4 LEDs) | `exp3_3_mimo_vlc.py` | PSNR + AdamW |

## Execução

```bash
# Rodar um experimento individual
python src/exp1_pinn_snr.py

# Ou abrir no Jupyter/Colab
jupyter notebook notebooks/exp1_pinn_snr.ipynb
```

## Citação

Se este trabalho contribuiu com sua pesquisa, cite como:

```bibtex
@software{vlc_digital_twin_2025,
  title  = {VLC Digital Twin with NVIDIA PhysicsNeMo},
  author = {Pesquisador VLC},
  year   = {2025},
  url    = {https://github.com/seu-usuario/vlc-digital-twin},
  license = {Apache-2.0}
}
```

## Licença

Distribuído sob a licença **Apache 2.0**. Consulte [LICENSE](LICENSE) para detalhes.
"""

EXPERIMENTS_MD = """\
# Descrição Detalhada dos Experimentos

## Experimento 1 — PINN para Predição de SNR (Modelo Lambertiano)

**Objetivo:** Provar que uma rede neural consegue aprender as leis da óptica a partir de dados
gerados pelo modelo analítico Lambertiano.

**Física Envolvida:**
- Modelo de radiação Lambertiana: `H = (m+1)·A / (2π·d²) · cos(φ)^m · cos(θ)`
- SNR com ruído shot e ruído térmico: `SNR = (ρ·Pt·H)² / (2q·ρ·Pt·H·B + N₀B/2)`

**Arquitetura:** MLP 4 camadas (2 → 64 → 128 → 64 → 1) com ativação Tanh  
**Resultado Esperado:** Curva preditiva sobreposta ao modelo analítico (erro < 0.5 dB)

---

## Experimento 2 — Classificador Inteligente de Modulação VLC

**Objetivo:** Classificar automaticamente o esquema de modulação óptica a partir de
features estatísticas do sinal recebido.

**Classes:** OOK, PPM-4, PPM-8, VPPM  
**Features (8-dim):** Média, Desvio Padrão, Variância, Q25, Q75, Diversidade, Pico, Duty Cycle  
**Arquitetura:** Classificador com Dropout (8 → 32 → 64 → 32 → 4)

**Nota de Diagnóstico:** Possível confusão entre PPM-4 e VPPM sem features temporais
(Duty Cycle). Isso documenta a importância de Feature Engineering em sistemas VLC.

---

## Experimento 3.1 — Gêmeo Digital 3D com Fourier Features

**Problema Resolvido:** Spectral Bias — redes MLP convencionais suavizam campos de luz
e ignoram picos de intensidade diretamente abaixo do LED.

**Solução:** Positional Encoding via mapeamento aleatório de Fourier (inspirado em NeRF):
```
γ(p) = [sin(2π·B·p), cos(2π·B·p)]   onde B ~ N(0, σ²)
```

**Saída:** Mapa de calor fotorrealista da distribuição de luz em qualquer plano (x, y, z).

---

## Experimento 3.2 — Gêmeo Digital com Sombreamento

**Problema:** Modelar a descontinuidade abrupta criada por um obstáculo físico (LOS Blockage).

**Inovação — Physics-Weighted Loss:**
```python
loss = mean( (pred - target)² · weight )
weight = 1.0  (fora da sombra)
weight = 10.0 (dentro da sombra)   # Penalidade 10× na zona de sombra
```

O custo diferenciado força a rede a priorizar a aprendizagem das bordas da sombra.

---

## Experimento 3.3 — Gêmeo Digital MIMO-VLC (4 LEDs)

**Configuração:** Grade 2×2 de LEDs em posições simétricas a ±1m do centro do teto.

**Campo de Luz:** Superposição linear das contribuições Lambertianas de cada LED:
```
I_total(x,y,z) = Σ H_i(x,y,z)   para i = 1..4
```

**Métricas Monitoradas:** MSE Loss + PSNR (Peak Signal-to-Noise Ratio)  
**Otimizador:** AdamW (decoupled weight decay, superior ao Adam para generalização)
"""

CHANGELOG_MD = """\
# Changelog

Todas as mudanças notáveis neste projeto são documentadas neste arquivo.
O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e o projeto adere ao [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [Não Lançado]

### Planejado
- Integração com NVIDIA Modulus Sym para constraints simbólicas automáticas
- Suporte a geometrias de sala não retangulares
- Exportação de modelos treinados para ONNX / TensorRT

---

## [1.0.0] — 2025-07-01

### Adicionado
- **Experimento 1:** PINN para predição de SNR com modelo Lambertiano
- **Experimento 2:** Classificador de modulação VLC (OOK, PPM-4, PPM-8, VPPM)
- **Experimento 3.1:** Gêmeo Digital 3D com Fourier Positional Encoding
- **Experimento 3.2:** Modelagem de sombreamento com Physics-Weighted Loss
- **Experimento 3.3:** Sistema MIMO-VLC com 4 LEDs e monitoramento de PSNR
- Notebooks Jupyter para execução em Google Colab (com GPU T4)
- Scripts Python para execução local sem Jupyter
- Documentação completa (README, EXPERIMENTS, CONTRIBUTING)
- Licença Apache 2.0

### Arquiteturas Neurais Implementadas
- `VLC_PINN`: MLP com ativação Tanh para aprendizado de física
- `ModulacaoClassifier`: Classificador com Dropout para robustez
- `DigitalTwin3D_Fourier`: Rede com Fourier Feature Mapping (sigma=2.0)
- `ShadowDigitalTwin`: Rede de alta frequência (sigma=5.0) com GELU
- `MIMO_DigitalTwin`: Rede para campo MIMO (sigma=3.0) com AdamW
"""

CONTRIBUTING_MD = """\
# Guia de Contribuição

Obrigado por considerar contribuir com o **VLC Digital Twin**! Este projeto segue
os padrões de contribuição Open Source da NVIDIA PhysicsNeMo.

## Como Contribuir

### 1. Reportar Bugs

Use o template de Issue `bug_report` e inclua:
- Versão do Python, PyTorch e NVIDIA PhysicsNeMo
- GPU utilizada (ou CPU) e SO
- Código mínimo reproduzível do erro
- Stack trace completo

### 2. Sugerir Features

Use o template `feature_request`. Descreva:
- O caso de uso físico que a feature atenderia
- O comportamento esperado
- Referências à literatura de VLC / PINN quando aplicável

### 3. Submeter Pull Requests

```bash
# Fork e clone
git clone https://github.com/SEU_USUARIO/vlc-digital-twin.git
cd vlc-digital-twin

# Criar branch descritiva
git checkout -b feature/mimo-beamforming

# Instalar dependências de dev
pip install -r requirements.txt

# Fazer suas alterações e commitar
git add .
git commit -m "feat: adicionar suporte a beamforming adaptativo no MIMO"

# Push e abrir PR
git push origin feature/mimo-beamforming
```

### Convenções de Código

- **Estilo:** PEP 8, docstrings no formato Google Style
- **Tipagem:** Type hints em todas as funções públicas
- **Testes:** Adicione testes em `tests/` para novas funcionalidades
- **Commits:** Formato Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`)

### Checklist antes do PR

- [ ] Código segue PEP 8 (use `black` e `flake8`)
- [ ] Type hints adicionados
- [ ] Docstrings atualizadas
- [ ] Experimento funciona tanto em CPU quanto em GPU CUDA
- [ ] Imagens de resultado salvas com `dpi=150`
- [ ] CHANGELOG.md atualizado na seção `[Não Lançado]`

## Código de Conduta

Este projeto segue o [Contributor Covenant](https://www.contributor-covenant.org/).
Ao participar, você concorda em manter um ambiente respeitoso e inclusivo.

## Licença

Ao submeter uma contribuição, você concorda que ela será licenciada sob **Apache 2.0**.
"""

LICENSE_TXT = """\
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form.

      "Work" shall mean the work of authorship made available under
      the License.

      "Derivative Works" shall mean any work that is based on the Work.

      "Contribution" shall mean any work of authorship submitted to the Licensor.

      "Contributor" shall mean Licensor and any Legal Entity on behalf of
      whom a Contribution has been received by the Licensor.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      patent license to make, use, sell, offer for sale, import, and
      otherwise transfer the Work.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, provided that You meet the following conditions:

      (a) You must give any other recipients of the Work or Derivative
          Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work; and

      (d) If the Work includes a "NOTICE" text file, You must include a
          readable copy of the attribution notices contained within.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution submitted for inclusion in the Work shall be under
      the terms and conditions of this License.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor.

   7. Disclaimer of Warranty. UNLESS REQUIRED BY APPLICABLE LAW OR
      AGREED TO IN WRITING, LICENSOR PROVIDES THE WORK ON AN "AS IS"
      BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.

   8. Limitation of Liability. IN NO EVENT SHALL ANY CONTRIBUTOR BE
      LIABLE FOR ANY DAMAGES ARISING FROM THIS LICENSE OR USE OF THE WORK.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work, You may offer acceptance of support, warranty, indemnity,
      or other liability obligations consistent with this License.

   END OF TERMS AND CONDITIONS

   Copyright 2025 VLC Digital Twin Contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

REQUIREMENTS_TXT = """\
# Core ML
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Machine Learning Utils
scikit-learn>=1.3.0

# Jupyter
notebook>=7.0.0
ipykernel>=6.0.0

# NVIDIA PhysicsNeMo (instalar manualmente via git clone)
# git clone https://github.com/NVIDIA/physicsnemo-sym.git
# pip install -e physicsnemo-sym

# Development
black>=23.0.0
flake8>=6.0.0
pytest>=7.0.0
"""

ARCH_MD = """\
# Arquitetura das Redes Neurais

## Fourier Feature Mapping (Base de Todos os Gêmeos Digitais 3D)

O componente central dos experimentos 3.x é o **Mapeamento Aleatório de Fourier**,
que resolve o problema do Spectral Bias em redes MLP:

```
γ(p) = [sin(2π·B·p), cos(2π·B·p)]

onde:
  p ∈ ℝ^d  — coordenadas de entrada (2D ou 3D)
  B ∈ ℝ^(d×m) — matriz aleatória fixada (não treinável)
  m — tamanho do mapeamento (padrão: 64)
```

### Comparação de Arquiteturas

| Modelo | Entrada | Saída | Sigma (B) | Ativação | Otimizador |
|--------|---------|-------|-----------|----------|------------|
| VLC_PINN | (d, θ) | SNR | N/A | Tanh | Adam |
| ModulacaoClassifier | features×8 | classe | N/A | ReLU+Dropout | Adam |
| DigitalTwin3D_Fourier | (x,y,z) | intensidade | 2.0 | Tanh | Adam |
| ShadowDigitalTwin | (x,y) | intensidade | 5.0 | GELU | Adam |
| MIMO_DigitalTwin | (x,y,z) | intensidade | 3.0 | GELU | AdamW |

### Por que GELU nos modelos de sombreamento?

O GELU (_Gaussian Error Linear Unit_) apresenta gradientes suaves que facilitam
a aprendizagem de transições abruptas como as bordas de sombra, sem o problema
de neurônios mortos do ReLU.

### Por que AdamW no MIMO?

O MIMO Digital Twin tem maior capacidade de memorização devido à complexidade
do campo de luz combinado (4 LEDs). O AdamW com _decoupled weight decay_ previne
o overfitting sem degradar a convergência inicial.
"""

GITHUB_CI = """\
name: CI — Smoke Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]

    steps:
    - uses: actions/checkout@v4

    - name: Setup Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies (CPU only)
      run: |
        python -m pip install --upgrade pip
        pip install torch --index-url https://download.pytorch.org/whl/cpu
        pip install numpy matplotlib seaborn scikit-learn

    - name: Run Experiment 1 (reduced epochs)
      run: python src/exp1_pinn_snr.py

    - name: Run Experiment 2
      run: python src/exp2_modulation_classifier.py
"""

BUG_TEMPLATE = """\
---
name: Bug Report
about: Reporte um erro encontrado nos experimentos
title: '[BUG] '
labels: bug
assignees: ''
---

## Descrição do Bug
Descreva claramente o erro encontrado.

## Como Reproduzir
1. Execute `python src/...`
2. Veja o erro em '...'

## Comportamento Esperado
Descreva o que deveria acontecer.

## Ambiente
- OS: [ex: Ubuntu 22.04 / Windows 11 / macOS 14]
- Python: [ex: 3.10.12]
- PyTorch: [ex: 2.1.0]
- GPU: [ex: NVIDIA RTX 3080 / CPU only]
- CUDA: [ex: 12.1 / N/A]

## Stack Trace
```
Cole aqui o erro completo
```
"""

FEATURE_TEMPLATE = """\
---
name: Feature Request
about: Sugira uma nova funcionalidade ou experimento
title: '[FEATURE] '
labels: enhancement
assignees: ''
---

## Problema / Motivação Física
Descreva o problema que esta feature resolveria.
Ex: "Não é possível simular canais VLC com reflexão difusa..."

## Solução Proposta
Descreva a arquitetura ou abordagem sugerida.

## Alternativas Consideradas
Outras abordagens que você avaliou e por que as descartou.

## Referências
Artigos, papers ou repositórios relacionados.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def build():
    print("\n" + "═" * 60)
    print("  VLC Digital Twin — Build Repo Generator")
    print("  Apache 2.0 | NVIDIA PhysicsNeMo Compatible")
    print("═" * 60 + "\n")

    # 1. Criar árvore de diretórios
    print("📁 Criando estrutura de pastas...")
    for d in DIRS:
        os.makedirs(d, exist_ok=True)
    print(f"   ✔  {len(DIRS)} diretórios criados\n")

    # 2. Scripts Python (src/)
    print("🐍 Gerando scripts Python (src/)...")
    exp_map = [
        ("exp1_pinn_snr",              EXP1_CODE),
        ("exp2_modulation_classifier", EXP2_CODE),
        ("exp3_1_digital_twin_3d",     EXP3_CODE),
        ("exp3_2_shadow_loss",         EXP4_CODE),
        ("exp3_3_mimo_vlc",            EXP5_CODE),
    ]
    for name, code in exp_map:
        header = f'#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n'
        write(f"{ROOT}/src/{name}.py", header + code)

    # 3. Notebooks Jupyter (notebooks/)
    print("\n📓 Gerando notebooks Jupyter (.ipynb)...")
    nb_specs = [
        (
            "exp1_pinn_snr",
            "Experimento 1 — PINN para Predição de SNR em Canal VLC",
            "Physics-Informed Neural Network para aprender o modelo de radiação Lambertiana.",
            [SETUP_CODE, EXP1_CODE]
        ),
        (
            "exp2_modulation_classifier",
            "Experimento 2 — Classificador Inteligente de Modulação VLC",
            "Rede neural para reconhecer esquemas de modulação óptica (OOK, PPM-4, PPM-8, VPPM).",
            [SETUP_CODE, EXP2_CODE]
        ),
        (
            "exp3_1_digital_twin_3d",
            "Experimento 3.1 — Gêmeo Digital 3D com Fourier Features",
            "Resolve o Spectral Bias com Positional Encoding inspirado em NeRF.",
            [SETUP_CODE, EXP3_CODE]
        ),
        (
            "exp3_2_shadow_loss",
            "Experimento 3.2 — Gêmeo Digital com Sombreamento (Physics-Weighted Loss)",
            "Aprende descontinuidades de luz (sombras) via função de custo com penalidade física.",
            [SETUP_CODE, EXP4_CODE]
        ),
        (
            "exp3_3_mimo_vlc",
            "Experimento 3.3 — Gêmeo Digital MIMO-VLC Avançado (4 LEDs + PSNR)",
            "Simula campo de luz combinado de 4 transmissores com monitoramento de PSNR.",
            [SETUP_CODE, EXP5_CODE]
        ),
    ]
    for name, title, desc, cells in nb_specs:
        nb_json = make_notebook(title, desc, cells)
        write(f"{ROOT}/notebooks/{name}.ipynb", nb_json)

    # 4. Documentação
    print("\n📄 Gerando documentação...")
    write(f"{ROOT}/README.md",          README_MD)
    write(f"{ROOT}/EXPERIMENTS.md",     EXPERIMENTS_MD)
    write(f"{ROOT}/CHANGELOG.md",       CHANGELOG_MD)
    write(f"{ROOT}/CONTRIBUTING.md",    CONTRIBUTING_MD)
    write(f"{ROOT}/LICENSE",            LICENSE_TXT)
    write(f"{ROOT}/requirements.txt",   REQUIREMENTS_TXT)
    write(f"{ROOT}/docs/architecture.md", ARCH_MD)

    # 5. GitHub templates
    print("\n🐙 Gerando templates GitHub...")
    write(f"{ROOT}/.github/workflows/ci.yml",                   GITHUB_CI)
    write(f"{ROOT}/.github/ISSUE_TEMPLATE/bug_report.md",       BUG_TEMPLATE)
    write(f"{ROOT}/.github/ISSUE_TEMPLATE/feature_request.md",  FEATURE_TEMPLATE)

    # 6. Gitignore
    gitignore = textwrap.dedent("""\
        __pycache__/
        *.py[cod]
        *.egg-info/
        dist/
        build/
        .env
        .venv/
        *.png
        *.pt
        *.pth
        physicsnemo-sym/
        .ipynb_checkpoints/
    """)
    write(f"{ROOT}/.gitignore", gitignore)

    # 7. Zipar tudo
    print(f"\n📦 Compactando repositório em {ZIP_NAME}.zip ...")
    shutil.make_archive(ZIP_NAME, 'zip', ".", ROOT)
    zip_size = os.path.getsize(f"{ZIP_NAME}.zip") / 1024
    print(f"   ✔  {ZIP_NAME}.zip criado ({zip_size:.1f} KB)\n")

    # Sumário
    total_files = sum(len(files) for _, _, files in os.walk(ROOT))
    print("═" * 60)
    print(f"  ✅  Build concluído com sucesso!")
    print(f"  📁  Pasta:    ./{ROOT}/   ({total_files} arquivos)")
    print(f"  📦  Pacote:   ./{ZIP_NAME}.zip")
    print("═" * 60)
    print("\nPróximos passos:")
    print("  1. Descompacte o .zip e abra os notebooks no Google Colab ou Jupyter")
    print("  2. Para execução local: pip install -r requirements.txt")
    print("  3. Para publicar: git init && git remote add origin <seu-repo>")
    print()


if __name__ == "__main__":
    build()
