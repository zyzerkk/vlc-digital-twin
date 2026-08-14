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
