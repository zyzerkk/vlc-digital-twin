# Descrição Detalhada dos Experimentos

Este documento descreve os 12 experimentos realizados nos três notebooks do repositório,
incluindo objetivo, arquitetura, resultado obtido e — de forma deliberada — os casos em que o
resultado não correspondeu ao esperado. A transparência sobre limitações é tratada aqui como
parte do relato metodológico.

---

# Notebook 01 — `01_experimentos_fundamentais.ipynb`

Implementação em **PyTorch puro**, independente de versão do PhysicsNeMo — escolha deliberada
para isolar a validação conceitual de PINNs da complexidade de uma dependência externa em
evolução rápida.

## Experimento 1 — PINN para Predição de SNR (Modelo Lambertiano)

**Objetivo:** provar que uma rede neural consegue aprender as leis da óptica a partir de dados
gerados pelo modelo analítico Lambertiano, respeitando a física durante o treinamento.

**Física envolvida:**
- Modelo de radiação Lambertiana: `H(0) = (m+1)·A / (2π·d²) · cosᵐ(φ) · cos(θ)`
- SNR com ruído shot e ruído térmico: `SNR = (ρ·Pt·H)² / (2q·ρ·Pt·H·B + N₀·B/2)`

**Arquitetura:** MLP 4 camadas (2 → 64 → 128 → 64 → 1), ativação Tanh
**Otimizador:** Adam, lr=1e-3, 2000 épocas
**Input:** [d_norm, θ_norm] — distância e ângulo normalizados para [0,1]

**Resultado obtido:** curva predita sobreposta ao modelo analítico com excelente aderência em
toda a faixa 0,5–5,0 m. MSE Loss convergiu para ~10⁻⁵. Imagem: `assets/01_exp1_snr_pinn.png`

**Limitação observada:** a curva de perda apresenta picos periódicos recorrentes durante o
treinamento, associados à ausência de um scheduler de taxa de aprendizado — corrigido no EXP1-v2
(notebook 03) com `CosineAnnealingLR`. Este experimento não fixa seed aleatória.

---

## Experimento 2 — Classificador Inteligente de Modulação VLC

**Objetivo:** identificar automaticamente o esquema de modulação óptica a partir de features
estatísticas do sinal recebido — simulando um receptor VLC inteligente.

**Classes:** OOK, PPM-4, PPM-8, VPPM
**Features (8 dimensões):** Média, Desvio Padrão, Variância, Q25, Q75, Diversidade, Pico, Duty Cycle
**Arquitetura:** Classificador com Dropout (8 → 32 → 64 → 32 → 4)
**Otimizador:** Adam, lr=5e-4, Dropout=0.3
**Seed:** `np.random.seed(42)` fixada neste experimento.

**Resultado obtido:** OOK 100% · PPM-8 100% · PPM-4 68% (32% confundido com VPPM) · VPPM 38%
(62% confundido com PPM-4). Imagem: `assets/01_exp2_classificador.png`

**Causa raiz identificada (diagnóstico técnico):** a função de geração de dados sintéticos usada
para VPPM nesta versão chama, na prática, a mesma rotina de geração de PPM-4
(`gerar_ppm(N, M=4)`), produzindo duas classes estatisticamente indistinguíveis nas 8 features
utilizadas. O problema não é apenas a ausência de informação temporal no classificador — o
próprio gerador de dados não diferenciava as duas modulações. Corrigido no EXP4 (notebook 02).

---

## Experimento 3.1 — Gêmeo Digital 3D com Fourier Features

**Problema resolvido:** Spectral Bias — redes MLP convencionais suavizam campos de luz
e ignoram picos de intensidade diretamente abaixo do LED (região de maior irradiância).

**Solução:** Positional Encoding via Fourier Feature Mapping (inspirado em NeRF):
```
γ(p) = [sin(2π·B·p), cos(2π·B·p)]   onde B ~ N(0, σ²), σ=2.0
```

**Arquitetura:** 128 → 256 → 256 → 128 → 1, ativação GELU
**Saída:** Mapa de calor da distribuição de luz em qualquer plano (x, y, z)

**Resultado obtido:** Loss convergiu de ~2×10⁻¹ para ~2×10⁻⁴ em 1000 épocas.
Imagem: `assets/01_exp3_1_digital_twin_3d.png`

**Limitação observada:** apesar do MSE em patamar baixo, o mapa de calor gerado no plano
Z=0,5 m apresenta manchas irregulares distribuídas pelo plano, sem um pico dominante claro sob
o LED — o modelo não convergiu completamente para o campo analítico esperado. Recomenda-se
aumentar épocas para 3000+ ou ajustar σ. Este experimento não fixa seed aleatória.

---

## Experimento 3.2 — Gêmeo Digital com Sombreamento (LOS Blockage)

**Problema:** modelar a descontinuidade abrupta criada por um obstáculo físico (LOS Blockage).
Redes convencionais tendem a suavizar a transição, produzindo sombras com bordas borradas.

**Inovação — Physics-Weighted Loss:**
```python
loss = mean( (pred - target)² · weight )
weight = 1.0   # fora da sombra
weight = 10.0  # dentro da sombra — penalidade 10× na zona bloqueada
```

**Parâmetros do obstáculo:** bloco quadrado de [-0,5 m, 0,5 m] em X e Y, bloqueio total (I=0)
**σ Fourier:** 5.0 (alta frequência, adequada para bordas abruptas)

**Resultado obtido:** resultado visual praticamente indistinguível do campo real — a rede
reproduziu com fidelidade tanto a região de bloqueio total quanto o gradiente de iluminância ao
redor do obstáculo. Melhor aderência qualitativa entre os experimentos deste notebook.
Imagem: `assets/01_exp3_2_sombreamento.png`

---

## Experimento 3.3 — Gêmeo Digital MIMO-VLC (4 LEDs)

**Configuração:** grade 2×2 de LEDs em posições simétricas a ±1 m do centro do teto,
altura de 2,5 m do plano do chão.

**Campo de luz total:** superposição linear das contribuições Lambertianas:
```
I_total(x,y,z) = Σ H_i(x,y,z)   para i = 1..4
```

**Arquitetura:** Fourier Feature Mapping (σ=3.0) → 128 → 256 → 256 → 128 → 1
**Otimizador:** AdamW

**Resultado obtido:** MSE de 2,56×10⁻⁴ e PSNR final de **35,91 dB** após 1000 épocas — a melhor
métrica quantitativa entre os experimentos espaciais deste notebook.

**Nota de transparência:** esta execução específica não gerou nem salvou uma imagem de mapa de
calor (apenas as métricas de treinamento foram registradas no console). Não há, portanto,
`assets/01_exp3_3_*.png` para este experimento — mencionamos isso explicitamente para não
sugerir uma evidência visual que não existe. O PSNR de ~36 dB permanece como ponto de comparação
quantitativo relevante frente ao EXP3.3-v2 (notebook 03).

---

# Notebook 02 — `02_experimentos_avancados.ipynb`

Motivado pelas limitações identificadas no notebook 01 — especialmente a confusão PPM-4/VPPM do
EXP2 — este notebook aprofunda o diagnóstico com quatro experimentos adicionais. Seed fixada
globalmente no início do notebook (`torch.manual_seed(42)`, `np.random.seed(42)`).

## Experimento 4 — Classificador v2: Resolvendo a Confusão PPM-4 ↔ VPPM

**Correção aplicada:** reescrita da função geradora de VPPM (`gerar_vppm_v2`) para produzir
pulsos com duty cycle variável e genuinamente distinto de PPM-4, e ampliação do vetor de features
de 8 para 12 dimensões (adicionando autocorrelação, entropia, taxa de zero-crossing e variação
média entre amostras).

**Arquitetura:** MLP com BatchNorm (12 → ... → 4)

**Resultado obtido:** acurácia geral de **95,6%** (vs. ~76% no baseline), com a confusão
PPM-4/VPPM praticamente eliminada (96,2% de acerto em VPPM).
Imagem: `assets/02_exp4_classificador_v2.png`

**Observação metodológica:** a correção efetiva dependeu tanto da engenharia de features quanto
da correção do próprio gerador de dados sintéticos — corrigir apenas um dos dois lados não teria
resolvido o problema.

---

## Experimento 5 — Ablation Study: PINN com Física vs. MLP Puro

**Pergunta de pesquisa:** a restrição física embutida na PINN realmente contribui para a
qualidade do modelo, ou uma rede convencional treinada apenas com dados chegaria a um resultado
equivalente?

**Método:** dois modelos com arquitetura idêntica treinados em paralelo sobre o mesmo conjunto de
dados reduzido (420 amostras de treino) — um com perda puramente MSE, outro com MSE mais o
resíduo da equação Lambertiana como termo de regularização física.

**Resultado obtido:** RMSE PINN = 0,1561 dB vs. RMSE MLP Puro = 0,1938 dB — uma redução de
**19,4%** no erro com a restrição física. Imagem: `assets/02_exp5_ablation_pinn_vs_mlp.png`

**Análise:** o ganho da restrição física não é uniforme — concentra-se nas regiões intermediárias
de distância, onde a densidade de dados de treinamento era menor. Evidência de que a física atua
como regularizador, ajudando o modelo a generalizar melhor em regiões com poucos exemplos.

---

## Experimento 6 — Curriculum Learning para PINN de Canal VLC

**Hipótese testada:** apresentar exemplos de treinamento em ordem crescente de dificuldade
(distâncias curtas e ângulos pequenos primeiro, depois progressivamente mais difíceis) produziria
convergência mais estável do que a ordem aleatória padrão.

**Definição de dificuldade:** Fácil (0,5–2,0 m, θ≈0°) → Médio (2,0–3,5 m) → Difícil (3,5–5,0 m,
θ>45°).

**Resultado obtido — hipótese NÃO confirmada:** RMSE Curriculum = 0,245 dB vs. RMSE Aleatório =
0,143 dB. O curriculum learning produziu um resultado **pior** que a ordenação aleatória padrão.
Imagem: `assets/02_exp6_curriculum_learning.png`

**Análise:** o mapa de ganho por região mostra que o curriculum não é uniformemente pior — ajuda
em regiões específicas (ângulos altos, bordas do domínio) mas prejudica em outras. Isso sugere que
o cronograma de dificuldade utilizado não estava bem calibrado para este problema. Registrado como
resultado negativo genuíno, sem tentativa de reformulação favorável.

---

## Experimento 7 — Transfer Learning: Canal com Interferência de Luz Ambiente

**Cenário:** adaptar a PINN treinada no EXP1 (canal VLC ideal) para um cenário com interferência
de luz ambiente (lâmpadas fluorescentes, luz solar), reaproveitando pesos ao invés de treinar do
zero.

**Estratégia:** congelar as camadas iniciais (que aprenderam a física básica) e treinar apenas as
camadas finais (fine-tuning).

**Resultado obtido:** o modelo com transfer learning converge em poucas dezenas de épocas para o
mesmo patamar de perda que o treino do zero leva centenas de épocas para atingir — ambas as
abordagens chegam a qualidade final semelhante, mas o custo computacional do transfer learning é
uma fração do treino completo. Imagem: `assets/02_exp7_transfer_learning.png`

---

# Notebook 03 — `03_physicsnemo_v2_nativo.ipynb`

Reescrita do pipeline utilizando os módulos oficiais do **NVIDIA PhysicsNeMo v2.0**
(`physicsnemo.models.FullyConnected`, `physicsnemo.models.FNO`, `physicsnemo.sym`), avaliando se
as implementações manuais dos notebooks 01/02 poderiam ser substituídas com ganho de qualidade e
portabilidade. Seed fixada globalmente (`torch.manual_seed(42)`, `np.random.seed(42)`).

## EXP1-v2 — PINN com `FullyConnected` Nativo

**O que mudou:** substituição do `nn.Sequential` manual por
`physicsnemo.models.mlp.fully_connected.FullyConnected`, com inicialização de pesos otimizada
para PINNs e adição de `CosineAnnealingLR`.

**Resultado obtido:** RMSE 0,2483 dB — mesma ordem de grandeza da v1 —, mas com curva de perda
visivelmente mais estável, sem os picos periódicos da versão manual.
Imagem: `assets/03_exp1_v2_physicsnemo.png`

---

## EXP2-v2 — Classificador com Skip Connections

**O que mudou:** `FullyConnected` nativo com `skip_connections=True` (conexões residuais) e
normalização de entrada embutida, mantendo as 12 features temporais do EXP4.

**Resultado obtido:** acurácia de **95,9%**, equivalente ao EXP4, confirmando que a migração para
os módulos nativos preservou a qualidade obtida na correção manual, com ganho adicional em
portabilidade (exportação ONNX/TensorRT nativa, suporte a `DistributedDataParallel`).
Imagem: `assets/03_exp2_v2_physicsnemo.png`

---

## EXP3.1-v2 — Fourier Neural Operator (FNO)

**O que mudou:** substituição do Fourier Feature Mapping manual pelo FNO nativo do PhysicsNeMo —
uma arquitetura de operador neural que aprende no espaço de frequências via FFT.

**Resultado obtido:** MSE convergindo para ~10⁻⁷ e PSNR de até **71,71 dB** — métricas
numericamente excelentes, muito melhores que a v1.

**LIMITAÇÃO CRÍTICA (resultado mais importante deste notebook):** a inspeção visual do mapa de
calor gerado (`assets/03_exp3_1_v2_fno.png`) mostra um padrão de ruído **sem nenhuma estrutura
espacial coerente** — não há pico de intensidade sob o LED, nem o gradiente suave esperado do
modelo Lambertiano. A métrica de perda reportada durante o treinamento não reflete a qualidade
real do campo predito, indicando um problema de reshaping/normalização na adaptação da entrada
pontual (x, y, z) para o formato de grid estruturado que o FNO espera como operador neural.
**Problema em aberto, não resolvido neste repositório.** Registrado deliberadamente como exemplo
de por que uma métrica numérica favorável nunca deve ser aceita como validação suficiente sem
inspeção visual do resultado.

---

## EXP3.2-v2 — Sombreamento com Peso Adaptativo

**O que mudou:** peso fixo (10×) da Physics-Weighted Loss substituído por peso adaptativo,
crescendo progressivamente de 1× para 20× ao longo do treinamento.

**Resultado obtido:** resultado visualmente equivalente ao obtido na v1, confirmando que a
estratégia de peso adaptativo é, no mínimo, tão eficaz quanto o peso fixo, sem exigir a escolha
manual de um único valor de penalidade. Imagem: `assets/03_exp3_2_v2_adaptive_loss.png`

---

## EXP3.3-v2 — MIMO-VLC com FNO e Análise de Cobertura QoS

**O que mudou:** substituição da arquitetura Fourier Features (v1) pelo FNO nativo, com adição de
análise automática de cobertura de QoS (percentual da sala com boa cobertura / marginal / sem
cobertura) e monitoramento de PSNR e SSIM.

**Resultado obtido:** PSNR reportado de 27,64 dB durante o treinamento.

**LIMITAÇÃO CRÍTICA:** o mesmo problema do EXP3.1-v2 se repete de forma ainda mais visível — o
campo de luz predito (`assets/03_exp3_3_v2_mimo_coverage.png`) não apresenta nenhuma estrutura
reconhecível, os quatro LEDs não geram picos de intensidade visíveis, e a escala de valores
preditos (~10⁻⁶) é ordens de grandeza menor que o esperado fisicamente. Como consequência, a
análise de cobertura de QoS classifica **100% da sala como "sem cobertura"** — um resultado
fisicamente inconsistente com a configuração de 4 LEDs simulada. Comparado à v1 (PSNR 35,91 dB
com Fourier Features tradicionais, EXP3.3 no notebook 01), a migração para FNO neste experimento
representou uma **regressão de qualidade**, não uma evolução.

---

## Síntese — Lições Aprendidas

O padrão que se repete em EXP3.1-v2 e EXP3.3-v2 — perda numérica baixa acompanhada de resultado
visual fisicamente incoerente — é o achado metodológico mais importante deste conjunto de
experimentos: **nunca aceitar uma métrica de treinamento como validação suficiente sem inspeção
visual ou validação por métricas específicas de domínio.** Da mesma forma, o resultado negativo do
EXP6 (curriculum learning) é mantido no repositório sem tentativa de reformulação favorável —
documentar o que não funcionou é considerado parte da contribuição, não um problema a esconder.
