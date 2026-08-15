# Arquitetura do Framework VLC Digital Twin

## Visão geral

O framework é organizado em três camadas:

```
┌─────────────────────────────────────────────────────┐
│                  CAMADA DE APLICAÇÃO                │
│   Digital Twin (gêmeo virtual do canal VLC)         │
│   Predição de SNR · BER · Mapa de cobertura         │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│               CAMADA DE MODELOS (PINNs)             │
│  VLC_PINN · ModulacaoClassifier · DigitalTwin3D     │
│  ShadowDigitalTwin · MIMO_DigitalTwin               │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│             CAMADA FÍSICA (restrições)               │
│  Modelo Lambertiano · SNR · BER · Physics Loss      │
│  Fourier Feature Mapping · Physics-Weighted Loss    │
└─────────────────────────────────────────────────────┘
```

## Arquiteturas neurais por experimento

### EXP1 — VLC_PINN
- **Tipo:** MLP fully-connected
- **Input:** [d_norm, θ_norm] — distância e ângulo normalizados
- **Camadas:** 2 → 64 → 128 → 64 → 1
- **Ativação:** Tanh (suave, adequada para física contínua)
- **Output:** SNR normalizado [0, 1]
- **Loss:** MSE (dados) + resíduo Lambertiano (física)

### EXP2 — ModulacaoClassifier
- **Tipo:** Classificador com Dropout
- **Input:** 8 features estatísticas do sinal
- **Camadas:** 8 → 32 → 64 → 32 → 4
- **Ativação:** ReLU + Dropout(0.3)
- **Output:** Logits para 4 classes (OOK, PPM-4, PPM-8, VPPM)
- **Loss:** CrossEntropyLoss

### EXP3.1 — DigitalTwin3D com Fourier Features
- **Inovação:** Positional Encoding (Fourier Feature Mapping, σ=2.0)
- **Motivação:** Elimina Spectral Bias — redes MLP tendem a aprender
  funções de baixa frequência, suavizando picos de intensidade
- **Input:** [x, y, z] mapeados via γ(p) = [sin(2π·B·p), cos(2π·B·p)]
- **Camadas:** 128 → 256 → 256 → 128 → 1
- **Ativação:** GELU

### EXP3.2 — ShadowDigitalTwin
- **Inovação:** Physics-Weighted Loss com penalidade 10× na zona de sombra
- **σ Fourier:** 5.0 (alta frequência para capturar bordas abruptas)
- **Loss customizada:** `mean((pred - target)² · weight)` onde weight=10 na sombra

### EXP3.3 — MIMO_DigitalTwin
- **Configuração:** 4 LEDs em grade 2×2, ±1m do centro do teto
- **Campo total:** Superposição linear `I = Σ H_i(x,y,z)` para i=1..4
- **Otimizador:** AdamW (melhor generalização que Adam pelo weight decay desacoplado)
- **Métrica extra:** PSNR monitorado durante treinamento

## Fluxo de dados

```
Parâmetros físicos (YAML)
        │
        ▼
Geração de dados sintéticos (modelo Lambertiano)
        │
        ├──► Grid (d, θ) → H(0) → SNR → BER
        │
        ▼
Normalização Min-Max → Tensores PyTorch
        │
        ▼
Treinamento PINN (loss = L_dados + λ·L_física)
        │
        ▼
Avaliação: RMSE vs. modelo analítico
        │
        ▼
Visualização: curvas, heatmaps, mapas de cobertura
```

## Reprodutibilidade

Todas as sementes aleatórias são fixadas no início de cada script:
```python
torch.manual_seed(42)
np.random.seed(42)
```

Os hiperparâmetros de cada experimento estão documentados em `EXPERIMENTS.md`.
