# Arquitetura do Framework VLC Digital Twin

## Visão geral

O framework é organizado em três camadas conceituais:

```
┌─────────────────────────────────────────────────────┐
│                  CAMADA DE APLICAÇÃO                │
│   Digital Twin (gêmeo virtual do canal VLC)         │
│   Predição de SNR · Classificação · Mapa de cobertura│
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│               CAMADA DE MODELOS (PINNs)             │
│  VLC_PINN · ModulacaoClassifier · DigitalTwin3D     │
│  ShadowDigitalTwin · MIMO_DigitalTwin               │
│  (v2: FullyConnected e FNO nativos do PhysicsNeMo)  │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│             CAMADA FÍSICA (restrições)               │
│  Modelo Lambertiano · SNR · BER · Physics Loss      │
│  Fourier Feature Mapping · Physics-Weighted Loss    │
└─────────────────────────────────────────────────────┘
```

Este documento cobre as arquiteturas dos 12 experimentos distribuídos nos três notebooks do
repositório. Detalhes de treinamento (épocas, hiperparâmetros, resultados) estão em
[`EXPERIMENTS.md`](../EXPERIMENTS.md).

---

## Notebook 01 — Experimentos Fundamentais (PyTorch puro)

### EXP1 — VLC_PINN
- **Tipo:** MLP fully-connected
- **Input:** [d_norm, θ_norm] — distância e ângulo normalizados
- **Camadas:** 2 → 64 → 128 → 64 → 1
- **Ativação:** Tanh (suave, adequada para física contínua)
- **Loss:** MSE (dados) + resíduo Lambertiano (física)

### EXP2 — ModulacaoClassifier
- **Tipo:** Classificador com Dropout
- **Input:** 8 features estatísticas do sinal
- **Camadas:** 8 → 32 → 64 → 32 → 4
- **Ativação:** ReLU + Dropout(0.3)
- **Loss:** CrossEntropyLoss

### EXP3.1 — DigitalTwin3D_Fourier
- **Inovação:** Positional Encoding (Fourier Feature Mapping, σ=2.0)
- **Motivação:** eliminar Spectral Bias — redes MLP tendem a aprender funções de baixa
  frequência, suavizando picos de intensidade
- **Input:** [x, y, z] mapeados via γ(p) = [sin(2π·B·p), cos(2π·B·p)]
- **Camadas:** 128 → 256 → 256 → 128 → 1
- **Ativação:** GELU

### EXP3.2 — ShadowDigitalTwin
- **Inovação:** Physics-Weighted Loss com penalidade 10× na zona de sombra
- **σ Fourier:** 5.0 (alta frequência para capturar bordas abruptas)

### EXP3.3 — MIMO_DigitalTwin
- **Configuração:** 4 LEDs em grade 2×2, ±1 m do centro do teto
- **Campo total:** superposição linear `I = Σ H_i(x,y,z)` para i=1..4
- **Otimizador:** AdamW (melhor generalização por weight decay desacoplado)

---

## Notebook 02 — Experimentos Avançados

### EXP4 — ModulacaoClassifier v2
- **Diferença do EXP2:** 12 features (8 originais + autocorrelação, entropia, taxa de
  zero-crossing, variação média entre amostras) + BatchNorm
- **Correção de causa raiz:** gerador de dados sintéticos de VPPM reescrito para produzir duty
  cycle genuinamente distinto de PPM-4 (ver `EXPERIMENTS.md`)

### EXP5 — Ablation (PINN vs. MLP)
- Duas instâncias da mesma arquitetura MLP (idêntica ao EXP1), treinadas em paralelo: uma com
  `loss = MSE`, outra com `loss = MSE + resíduo físico`

### EXP6 — Curriculum Learning
- Mesma arquitetura do EXP1, mas com dataset particionado em 3 fases de dificuldade crescente
  apresentadas sequencialmente ao otimizador

### EXP7 — Transfer Learning
- Reaproveita os pesos do EXP1; congela as camadas iniciais; fine-tuning apenas nas camadas
  finais sobre dados de um canal com interferência de luz ambiente

---

## Notebook 03 — PhysicsNeMo v2.0 Nativo

### EXP1-v2 / EXP2-v2 — `physicsnemo.models.FullyConnected`
- Substitui o `nn.Sequential` manual pelo MLP oficial do PhysicsNeMo
- Inicialização de pesos otimizada para PINNs
- EXP2-v2 adiciona `skip_connections=True` (conexões residuais)
- Suporte nativo a exportação ONNX/TensorRT e `DistributedDataParallel`

### EXP3.1-v2 / EXP3.3-v2 — `physicsnemo.models.FNO` (Fourier Neural Operator)
- Substitui o Fourier Feature Mapping manual por um operador neural que aprende no espaço de
  frequências via FFT
- **Status conhecido:** resultado visual dos campos espaciais preditos não corresponde à física
  esperada, apesar de métricas de perda favoráveis — ver limitação documentada em
  `EXPERIMENTS.md`. Suspeita-se de um problema de adaptação da representação de entrada
  (coordenadas pontuais) ao formato de grid estruturado nativamente esperado pelo FNO.

### EXP3.2-v2 — Peso Adaptativo
- Mesma arquitetura de Fourier Features do EXP3.2 (v1), mas com peso de penalidade na zona de
  sombra crescendo progressivamente de 1× para 20× ao longo do treinamento, em vez de um valor
  fixo

---

## Fluxo de dados (típico, notebooks 01 e 02)

```
Parâmetros físicos (definidos em código, início do notebook)
        │
        ▼
Geração de dados sintéticos (modelo Lambertiano ou sinais modulados)
        │
        ├──► Grid (d, θ) → H(0) → SNR → BER
        │
        ▼
Normalização Min-Max / StandardScaler → Tensores PyTorch
        │
        ▼
Treinamento (loss = L_dados + λ·L_física, quando aplicável)
        │
        ▼
Avaliação: RMSE / acurácia / PSNR vs. referência analítica ou conjunto de teste
        │
        ▼
Visualização: curvas de convergência, heatmaps, matrizes de confusão
```

## Reprodutibilidade

O uso de sementes aleatórias **não é uniforme entre os três notebooks** — este é um ponto de
atenção documentado, não uma garantia geral:

| Notebook | Seed fixada |
|---|---|
| `01_experimentos_fundamentais.ipynb` | Apenas no EXP2 (`np.random.seed(42)`). EXP1, EXP3.1, EXP3.2 e EXP3.3 não fixam seed. |
| `02_experimentos_avancados.ipynb` | Global, no início do notebook (`torch.manual_seed(42)` + `np.random.seed(42)`) |
| `03_physicsnemo_v2_nativo.ipynb` | Global, no início do notebook (`torch.manual_seed(42)` + `np.random.seed(42)`) |

Reexecuções dos experimentos do notebook 01 (exceto EXP2) podem produzir curvas ligeiramente
diferentes das publicadas em `assets/`, embora o comportamento qualitativo — incluindo as
limitações documentadas, como a convergência incompleta do EXP3.1 — seja consistente entre
execuções.
