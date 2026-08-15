# assets/

Esta pasta contém as figuras e gráficos gerados pelos experimentos.

Os arquivos são gerados automaticamente ao rodar os scripts em `src/`:

| Arquivo | Gerado por | Descrição |
|---------|-----------|-----------|
| `resultado_exp1_snr_pinn.png` | `src/exp1_pinn_snr.py` | Curva de aprendizado + SNR analítico vs. PINN |
| `resultado_exp2_classificador.png` | `src/exp2_modulation_classifier.py` | Matriz de confusão |
| `resultado_exp3_1_digital_twin_3d.png` | `src/exp3_1_digital_twin_3d.py` | Heatmap de irradiância 3D |
| `resultado_exp3_2_sombra.png` | `src/exp3_2_shadow_loss.py` | Canal real vs. Twin com sombra |
| `resultado_exp3_3_mimo.png` | `src/exp3_3_mimo_vlc.py` | Campo MIMO-VLC com 4 LEDs |

**Para gerar:** rode `python src/<experimento>.py` com o ambiente ativo.
