# VLC Digital Twin — Physics-Informed Neural Networks com NVIDIA PhysicsNeMo

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![NVIDIA PhysicsNeMo](https://img.shields.io/badge/NVIDIA-PhysicsNeMo-76b900.svg)](https://github.com/NVIDIA/physicsnemo-sym)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey.svg)](#citação)
[![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow.svg)](#)

> **Redes Neurais Informadas por Física (PINNs) aplicadas à modelagem de Gêmeos Digitais para
> canais de Comunicação por Luz Visível (Visible Light Communication — VLC)**

---

## Resumo

A Comunicação por Luz Visível (VLC) é uma tecnologia de acesso óptico sem fio que utiliza
diodos emissores de luz (LEDs) como transmissores de dados, explorando o espectro visível
(380–780 nm) como meio de propagação. A modelagem precisa do canal VLC — em particular a relação
sinal-ruído (SNR), o bloqueio de linha de visada (LOS) e a interferência entre múltiplos
transmissores em arranjos MIMO — é um pré-requisito para o projeto de sistemas de comunicação
óptica robustos.

Este repositório apresenta um framework de **Gêmeo Digital (Digital Twin)** para canais VLC
baseado em **Redes Neurais Informadas por Física (Physics-Informed Neural Networks — PINNs)**,
construído sobre o ecossistema [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo-sym)
(anteriormente Modulus). O framework combina dados observacionais com restrições físicas
derivadas do modelo de canal Lambertiano generalizado, permitindo que a rede neural respeite
leis físicas conhecidas mesmo em regiões do domínio com poucos dados de treinamento.

São apresentados cinco experimentos progressivos, do problema de predição de SNR em espaço livre
até um Gêmeo Digital MIMO-VLC completo com quatro transmissores, sombreamento e reconstrução de
campo de iluminância 3D.

---

## Motivação e Contexto

Modelos analíticos clássicos de canal VLC (e.g., modelo Lambertiano de ordem *m*) fornecem boa
aproximação em cenários idealizados, mas degradam-se na presença de obstruções, reflexões
multipercurso e geometrias não triviais de transmissores/receptores. Simulações puramente
numéricas (ray tracing, método de elementos finitos) são precisas, porém computacionalmente
custosas para uso em tempo real ou em loops de otimização (e.g., posicionamento ótimo de LEDs).

PINNs oferecem um meio-termo: aproximam a solução por uma rede neural treinada para minimizar
simultaneamente (i) o erro em relação a dados observados/simulados e (ii) o resíduo de equações
físicas ou restrições de domínio (regularização física), resultando em modelos que generalizam
melhor fora da distribuição de treinamento e que são ordens de magnitude mais rápidos que
simulações numéricas completas, uma vez treinados.

---

## Estrutura do Repositório

```
vlc-digital-twin/
├── notebooks/                 # Jupyter notebooks por experimento (exploração e visualização)
│   ├── exp1_pinn_snr.ipynb
│   ├── exp2_modulation_classifier.ipynb
│   ├── exp3_1_digital_twin_3d.ipynb
│   ├── exp3_2_shadow_loss.ipynb
│   └── exp3_3_mimo_vlc.ipynb
├── src/                        # Scripts Python para execução local/reprodutível
│   ├── exp1_pinn_snr.py
│   ├── exp2_modulation_classifier.py
│   ├── exp3_1_digital_twin_3d.py
│   ├── exp3_2_shadow_loss.py
│   └── exp3_3_mimo_vlc.py
├── docs/                       # Documentação técnica
│   └── architecture.md
├── assets/                     # Figuras, diagramas e resultados gráficos
├── results/                    # Métricas, checkpoints e logs de treinamento (gerado)
├── .github/                    # Templates de Issues, PRs e workflows de CI/CD
├── EXPERIMENTS.md              # Descrição detalhada de cada experimento
├── CHANGELOG.md                # Histórico de versões (Keep a Changelog / SemVer)
├── CONTRIBUTING.md             # Guia de contribuição
├── CITATION.cff                # Metadados de citação (Citation File Format)
├── requirements.txt            # Dependências Python fixadas por versão
└── LICENSE                     # Apache 2.0
```

---

## Fundamentação Teórica

### Modelo de canal Lambertiano generalizado

A irradiância recebida em um fotodetector a partir de um LED transmissor, assumindo apenas
componente de linha de visada (LOS), é modelada como:

```
H(0) = [(m+1) A / (2π d²)] · cosᵐ(φ) · Ts(ψ) · g(ψ) · cos(ψ)
```

onde `m` é a ordem lambertiana de emissão (função do semiângulo de meia potência do LED),
`A` é a área ativa do fotodetector, `d` a distância transmissor–receptor, `φ` o ângulo de
irradiância, `ψ` o ângulo de incidência, `Ts(ψ)` o ganho do filtro óptico e `g(ψ)` o ganho do
concentrador óptico.

### Formulação como problema informado por física

A rede neural `f_θ(x, y, z)` é treinada para aproximar a distribuição de SNR (ou irradiância)
no espaço, minimizando uma função de perda composta:

```
L(θ) = λ_data · L_data(θ) + λ_phys · L_phys(θ) + λ_bc · L_bc(θ)
```

- **L_data**: erro quadrático médio entre a predição da rede e amostras observadas/simuladas;
- **L_phys**: resíduo da equação de canal Lambertiano (ou de suas derivadas, quando aplicável),
  penalizando soluções fisicamente inconsistentes;
- **L_bc**: penalização de condições de contorno (e.g., bloqueio total em regiões de sombra,
  continuidade nas bordas do ambiente simulado).

Os pesos `λ_data`, `λ_phys`, `λ_bc` são hiperparâmetros ajustados por experimento (detalhes em
[`EXPERIMENTS.md`](EXPERIMENTS.md)).

---

## Experimentos

| # | Experimento | Arquivo | Conceito-Chave | Métrica de Avaliação |
|---|-------------|---------|-----------------|------------------------|
| 1 | PINN para SNR Lambertiano | `exp1_pinn_snr.py` | Physics-Informed Loss | RMSE, MAE vs. modelo analítico |
| 2 | Classificador de Modulação VLC | `exp2_modulation_classifier.py` | Feature Engineering | Acurácia, F1-score |
| 3.1 | Gêmeo Digital 3D | `exp3_1_digital_twin_3d.py` | Fourier Features / NeRF | PSNR, SSIM |
| 3.2 | Sombreamento (LOS Blockage) | `exp3_2_shadow_loss.py` | Physics-Weighted Loss | RMSE em regiões de sombra |
| 3.3 | MIMO-VLC (4 LEDs) | `exp3_3_mimo_vlc.py` | PSNR + AdamW | PSNR, tempo de inferência |

Descrições completas de arquitetura de rede, hiperparâmetros, conjuntos de dados sintéticos e
protocolo experimental de cada item estão em [`EXPERIMENTS.md`](EXPERIMENTS.md).

---

## Instalação e Reprodutibilidade

### Requisitos de ambiente

- Python ≥ 3.10
- PyTorch 2.x (com suporte a CUDA recomendado para treinamento em GPU)
- NVIDIA PhysicsNeMo (símbolico) — `physicsnemo-sym`

### Instalação

```bash
# Clonar o repositório
git clone https://github.com/seu-usuario/vlc-digital-twin.git
cd vlc-digital-twin

# (Recomendado) Criar ambiente virtual isolado
python -m venv .venv
source .venv/bin/activate

# Instalar dependências fixadas
pip install -r requirements.txt

# Instalar NVIDIA PhysicsNeMo (simbólico)
git clone https://github.com/NVIDIA/physicsnemo-sym.git
pip install -e physicsnemo-sym
```

### Execução

```bash
# Rodar um experimento individual
python src/exp1_pinn_snr.py

# Ou explorar interativamente via Jupyter/Colab
jupyter notebook notebooks/exp1_pinn_snr.ipynb
```

### Reprodutibilidade

Para garantir resultados reprodutíveis entre execuções:

- Sementes aleatórias (`seed`) fixadas em todos os scripts de `src/`;
- Versões de dependências fixadas em `requirements.txt`;
- Configurações de treinamento (hiperparâmetros, arquitetura, otimizador) documentadas por
  experimento em `EXPERIMENTS.md`;
- Checkpoints e logs de treinamento salvos em `results/` para auditoria posterior.

---

## Resultados

*(Seção a preencher com tabelas/figuras consolidadas de métricas por experimento — recomenda-se
incluir aqui gráficos de convergência da função de perda, comparações contra baseline analítico
e visualizações do campo de irradiância reconstruído, referenciando os arquivos em `assets/`.)*

---

## Limitações e Trabalhos Futuros

- Os experimentos atuais consideram ambientes estáticos sem mobilidade do receptor;
- O modelo de sombreamento assume obstáculos com geometria simplificada;
- Extensões futuras incluem: canais dinâmicos (receptor móvel), múltiplos materiais de reflexão,
  e validação contra medições experimentais em bancada óptica.

---

## Como Citar

Se este repositório for utilizado em trabalho acadêmico, por favor cite:

```bibtex
@misc{vlc_digital_twin_2026,
  author       = {Luís Otávio Guero},
  title        = {VLC Digital Twin: Physics-Informed Neural Networks para
                   Comunicação por Luz Visível},
  year         = {2026},
  howpublished = {\url{https://github.com/zyzerkk/vlc-digital-twin}},
  note         = {Acessado em: Agosto/26}
}
```

Metadados estruturados também estão disponíveis em [`CITATION.cff`](CITATION.cff).

---

## Contribuição

Contribuições são bem-vindas. Consulte [`CONTRIBUTING.md`](CONTRIBUTING.md) para diretrizes de
estilo de código, processo de *pull request* e relato de problemas.

## Licença

Distribuído sob a licença **Apache 2.0**. Consulte [`LICENSE`](LICENSE) para o texto completo.

## Contato

Dúvidas, sugestões ou relatos de erro: abra uma *issue* no repositório ou entre em contato via
[luis.guero@acad.ufsm.br].
