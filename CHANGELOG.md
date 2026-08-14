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
