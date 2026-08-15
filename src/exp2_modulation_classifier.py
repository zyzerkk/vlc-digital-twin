#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento 2 — Classificador Inteligente de Modulação VLC
===========================================================
Treina uma rede neural para reconhecer esquemas de modulação óptica
(OOK, PPM-4, PPM-8, VPPM) a partir de features estatísticas do sinal.

Nota de diagnóstico: observou-se confusão entre PPM-4 e VPPM devido à
ausência de features temporais (Duty Cycle) — documentado em EXPERIMENTS.md.

Rodar: python src/exp2_modulation_classifier.py
Saída: assets/resultado_exp2_classificador.png
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {device}")

np.random.seed(42)
N_por_classe = 500
labels = {0: "OOK", 1: "PPM-4", 2: "PPM-8", 3: "VPPM"}


# ─── 1. Geração de Dados Sintéticos ──────────────────────────────────────────
def gerar_ook(N, snr_db=20):
    bits  = np.random.randint(0, 2, N)
    sigma = 10 ** (-snr_db / 20)
    sinal = bits + np.random.normal(0, sigma, N)
    features = [sinal.mean(), sinal.std(), np.var(sinal),
                np.percentile(sinal, 25), np.percentile(sinal, 75),
                len(np.unique(np.round(sinal, 1))) / N, 0.0, 0.0]
    return np.array(features)


def gerar_ppm(N, M=4, snr_db=20):
    sigma  = 10 ** (-snr_db / 20)
    slots  = np.random.randint(0, M, N)
    sinais = np.zeros((N, M))
    for i, s in enumerate(slots):
        sinais[i, s] = 1.0
    sinais += np.random.normal(0, sigma, sinais.shape)
    f = sinais.flatten()
    features = [f.mean(), f.std(), np.var(f),
                np.percentile(f, 25), np.percentile(f, 75),
                M / 16.0, np.max(sinais.mean(axis=0)), 0.5]
    return np.array(features)


X_list, y_list = [], []
for snr in [10, 15, 20, 25, 30]:
    for _ in range(N_por_classe // 5):
        X_list.append(gerar_ook(200, snr));    y_list.append(0)
        X_list.append(gerar_ppm(200, 4, snr)); y_list.append(1)
        X_list.append(gerar_ppm(200, 8, snr)); y_list.append(2)
        X_list.append(gerar_ppm(200, 4, snr)); y_list.append(3)  # VPPM ~ PPM-4 sem Duty Cycle

X = np.array(X_list)
y = np.array(y_list)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y)

X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
y_tr_t = torch.tensor(y_tr, dtype=torch.long).to(device)
X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)


# ─── 2. Arquitetura do Classificador ─────────────────────────────────────────
class ModulacaoClassifier(nn.Module):
    """Classificador de modulação VLC com Dropout para robustez a ruído."""
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


# ─── 3. Treinamento ───────────────────────────────────────────────────────────
print(f"Treinando Classificador (Experimento 2)...")
for epoch in range(300):
    clf.train()
    pred = clf(X_tr_t)
    loss = loss_fn(pred, y_tr_t)
    optim.zero_grad()
    loss.backward()
    optim.step()
    if epoch % 100 == 0:
        print(f"  Época {epoch:3d} | Loss: {loss.item():.4f}")


# ─── 4. Avaliação ─────────────────────────────────────────────────────────────
clf.eval()
with torch.no_grad():
    y_pred = clf(X_te_t).argmax(dim=1).cpu().numpy()

print("\nRelatório de Classificação:")
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
plt.savefig("assets/resultado_exp2_classificador.png", dpi=150, bbox_inches='tight')
plt.show()
print("Experimento 2 concluído! → assets/resultado_exp2_classificador.png")
