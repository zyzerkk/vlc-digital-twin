# Descrição Detalhada dos Experimentos

---

## Experimento 1 — PINN para Predição de SNR (Modelo Lambertiano)

**Objetivo:** Provar que uma rede neural consegue aprender as leis da óptica a partir de dados
gerados pelo modelo analítico Lambertiano, respeitando a física durante o treinamento.

**Física envolvida:**
- Modelo de radiação Lambertiana: `H(0) = (m+1)·A / (2π·d²) · cosᵐ(φ) · cos(θ)`
- SNR com ruído shot e ruído térmico: `SNR = (ρ·Pt·H)² / (2q·ρ·Pt·H·B + N₀·B/2)`

**Arquitetura:** MLP 4 camadas (2 → 64 → 128 → 64 → 1), ativação Tanh
**Otimizador:** Adam, lr=1e-3, 2000 épocas
**Input:** [d_norm, θ_norm] — distância e ângulo normalizados para [0,1]
**Output:** SNR normalizado

**Resultado obtido:** Curva predita (linha vermelha tracejada) sobreposta ao modelo analítico (azul) com excelente aderência em toda a faixa 0,5–5,0m. MSE Loss convergiu para ~10⁻⁵ com oscilações de learning rate documentadas. Imagem: `assets/resultado_exp1_snr_pinn.png`

---

## Experimento 2 — Classificador Inteligente de Modulação VLC

**Objetivo:** Identificar automaticamente o esquema de modulação óptica a partir de
features estatísticas do sinal recebido — simulando um receptor VLC inteligente.

**Classes:** OOK, PPM-4, PPM-8, VPPM
**Features (8 dimensões):** Média, Desvio Padrão, Variância, Q25, Q75, Diversidade, Pico, Duty Cycle
**Arquitetura:** Classificador com Dropout (8 → 32 → 64 → 32 → 4)
**Otimizador:** Adam, lr=5e-4, Dropout=0.3
**Loss:** CrossEntropyLoss

**Nota de diagnóstico documentada:** Durante os experimentos, observou-se possível confusão
entre PPM-4 e VPPM sem features temporais (Duty Cycle). Isso motivou a inclusão do Duty Cycle
como 8ª feature, ilustrando a importância do Feature Engineering em sistemas VLC.

**Resultado obtido:** OOK 100% · PPM-8 100% · PPM-4 68% (32% confundido com VPPM) · VPPM 38% (62% confundido com PPM-4). Confusão PPM-4↔VPPM confirmada experimentalmente — justifica extensão futura com features de Duty Cycle. Imagem: `assets/resultado_exp2_classificador.png`

---

## Experimento 3.1 — Gêmeo Digital 3D com Fourier Features

**Problema resolvido:** Spectral Bias — redes MLP convencionais suavizam campos de luz
e ignoram picos de intensidade diretamente abaixo do LED (região de maior irradiância).

**Causa do Spectral Bias:** Redes MLP com ativações suaves (Tanh, ReLU) convergem
prioritariamente para funções de baixa frequência espacial, perdendo detalhes locais agudos.

**Solução:** Positional Encoding via Fourier Feature Mapping (inspirado em NeRF):
```
γ(p) = [sin(2π·B·p), cos(2π·B·p)]   onde B ~ N(0, σ²), σ=2.0
```

**Arquitetura:** 128 → 256 → 256 → 128 → 1, ativação GELU
**Saída:** Mapa de calor da distribuição de luz em qualquer plano (x, y, z)

**Resultado obtido:** Loss convergiu de ~2×10⁻¹ para ~2×10⁻⁴ em 1000 épocas. Mapa de calor no plano Z=0,5m apresenta padrão de manchas irregulares — indica que o modelo ainda não convergiu completamente para o campo analítico. Recomenda-se aumentar épocas para 3000+ ou ajustar σ. Imagem: `assets/resultado_exp3_1_digital_twin_3d.png`

---

## Experimento 3.2 — Gêmeo Digital com Sombreamento (LOS Blockage)

**Problema:** Modelar a descontinuidade abrupta criada por um obstáculo físico (LOS Blockage).
Redes convencionais tendem a suavizar a transição, produzindo sombras com bordas borradas.

**Inovação — Physics-Weighted Loss:**
```python
loss = mean( (pred - target)² · weight )
weight = 1.0   # fora da sombra
weight = 10.0  # dentro da sombra — penalidade 10× na zona bloqueada
```

O custo diferenciado força a rede a priorizar o aprendizado da fronteira sombra/luz,
capturando a descontinuidade física do bloqueio de LOS.

**Parâmetros do obstáculo:** bloco quadrado de [-0,5m, 0,5m] em X e Y, bloqueio total (I=0)
**σ Fourier:** 5.0 (alta frequência, adequada para bordas abruptas)

**Saída gerada:** `resultado_exp3_2_sombra.png`

---

## Experimento 3.3 — Gêmeo Digital MIMO-VLC (4 LEDs)

**Configuração:** Grade 2×2 de LEDs em posições simétricas a ±1m do centro do teto,
altura de 2,5m do plano do chão.

**Campo de luz total:** Superposição linear das contribuições Lambertianas:
```
I_total(x,y,z) = Σ H_i(x,y,z)   para i = 1..4
```

**Arquitetura:** Fourier Feature Mapping (σ=3.0) → 128 → 256 → 256 → 128 → 1
**Otimizador:** AdamW (weight decay desacoplado → melhor generalização que Adam)
**Métricas monitoradas:** MSE Loss + PSNR (Peak Signal-to-Noise Ratio)

**Por que PSNR?** Métrica padrão em reconstrução de imagem/campo — permite comparar
a qualidade do campo reconstruído pelo Digital Twin com o campo analítico real,
expressando o erro em escala logarítmica (dB).

**Saída gerada:** `resultado_exp3_3_mimo_vlc.png`

---

## Registro de Obstáculos e Soluções (Pesquisa Original 2023)

Esta seção documenta os problemas reais encontrados durante a pesquisa exploratória de 2023
no Google Colab. Incluída como contribuição metodológica para pesquisadores que trabalham
com NVIDIA Modulus/PhysicsNeMo em ambientes similares.

### Obstáculo 1 — Repositório GitLab da NVIDIA offline
**Contexto:** O tutorial oficial indicava clonagem via `gitlab.com/nvidia/modulus/modulus.git`.
**Problema:** Retornava erro 404 — repositório estava inacessível ou privado.
**Tentativa 1:** Busca por mirrors do repositório (sem sucesso direto).
**Solução adotada:** Localização via fóruns da NVIDIA e comunidade GitHub. O repositório
havia migrado para `github.com/NVIDIA/modulus`. Clonagem com token de acesso pessoal
funcionou para a versão disponível à época.
**Status atual (2026):** Repositório disponível publicamente em `github.com/NVIDIA/physicsnemo`.
Instalação simplificada: `pip install nvidia-physicsnemo`.

### Obstáculo 2 — Incompatibilidade de versão Python no Colab
**Contexto:** Google Colab usava Python 3.11 como padrão em 2023.
**Problema:** Modulus 22.09 exigia Python ≤ 3.10; instalação falhava com erros de dependência.
**Tentativa 1:** Downgrade via `pyenv` dentro do Colab (instável, runtime reiniciava).
**Tentativa 2:** Ajuste manual do `PATH` para apontar para Python 3.10 alternativo.
**Solução adotada:** Ajuste das dependências e uso de versões compatíveis das bibliotecas.
**Lição:** Para projetos com dependências rígidas de versão, documentar e fixar desde o início
com `pip freeze > requirements.txt` e especificar a versão Python no README.

### Obstáculo 3 — Instalação das dependências do Modulus
**Problema:** `python setup.py install` dentro do repositório clonado falhava com erros
de compilação de extensões C.
**Solução:** Instalação via `pip install -e .` (modo editable) contornava os erros de build.
