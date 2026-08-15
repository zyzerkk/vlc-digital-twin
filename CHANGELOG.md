# Changelog

Todas as mudanças notáveis neste projeto são documentadas neste arquivo.
O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e o projeto adere ao [Versionamento Semântico](https://semver.org/lang/pt-BR/).

---

## [Não Lançado]

### Planejado
- Preencher seção de Resultados com métricas reais após execução local dos experimentos
- Integração com API simbólica do NVIDIA PhysicsNeMo para constraints automáticas via PDEs
- Suporte a geometrias de sala não retangulares
- Exportação de modelos treinados para ONNX / TensorRT
- Validação contra dados experimentais de bancada óptica

---

## [1.0.0] — 2026-08-15

> **Contexto:** Esta versão formaliza uma pesquisa exploratória iniciada em julho de 2023.
> Os experimentos foram originalmente desenvolvidos no Google Colab. O repositório foi
> estruturado e publicado em agosto de 2026 como parte do processo de submissão de artigo.

### Adicionado
- **Experimento 1:** PINN para predição de SNR com modelo Lambertiano
- **Experimento 2:** Classificador de modulação VLC (OOK, PPM-4, PPM-8, VPPM)
- **Experimento 3.1:** Gêmeo Digital 3D com Fourier Positional Encoding
- **Experimento 3.2:** Modelagem de sombreamento com Physics-Weighted Loss
- **Experimento 3.3:** Sistema MIMO-VLC com 4 LEDs e monitoramento de PSNR
- Notebooks Jupyter para execução em Google Colab (com GPU T4)
- Scripts Python para execução local sem Jupyter
- `build_repo.py` — gerador da estrutura completa do repositório
- `docs/architecture.md` — documentação das arquiteturas neurais e fluxo de dados
- `CITATION.cff` — metadados de citação no formato Citation File Format
- `.github/workflows/ci.yml` — CI automático com verificação de imports e lint
- `.github/ISSUE_TEMPLATE/` — templates de bug report e feature request
- Documentação completa (README, EXPERIMENTS, CONTRIBUTING)
- Licença Apache 2.0

### Arquiteturas neurais implementadas
- `VLC_PINN` — MLP com ativação Tanh para aprendizado de física contínua
- `ModulacaoClassifier` — classificador com Dropout para robustez a ruído
- `DigitalTwin3D_Fourier` — Fourier Feature Mapping (σ=2.0) para resolver Spectral Bias
- `ShadowDigitalTwin` — alta frequência (σ=5.0) com GELU para bordas de sombra
- `MIMO_DigitalTwin` — campo MIMO com superposição linear (σ=3.0) e AdamW

### Problemas documentados (pesquisa 2023)
- Link GitLab da NVIDIA estava offline; repositório migrou para GitHub
- Incompatibilidade entre Python 3.11 (Colab padrão) e Modulus 22.09 (exigia ≤ 3.10)
- Solução: ajuste manual do ambiente e fixação de versões no requirements.txt
