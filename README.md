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



## Licença

Distribuído sob a licença **Apache 2.0**. Consulte [LICENSE](LICENSE) para detalhes.
