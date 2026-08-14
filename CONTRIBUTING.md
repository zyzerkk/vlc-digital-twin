# Guia de Contribuição

Obrigado por considerar contribuir com o **VLC Digital Twin**! Este projeto segue
os padrões de contribuição Open Source da NVIDIA PhysicsNeMo.

## Como Contribuir

### 1. Reportar Bugs

Use o template de Issue `bug_report` e inclua:
- Versão do Python, PyTorch e NVIDIA PhysicsNeMo
- GPU utilizada (ou CPU) e SO
- Código mínimo reproduzível do erro
- Stack trace completo

### 2. Sugerir Features

Use o template `feature_request`. Descreva:
- O caso de uso físico que a feature atenderia
- O comportamento esperado
- Referências à literatura de VLC / PINN quando aplicável

### 3. Submeter Pull Requests

```bash
# Fork e clone
git clone https://github.com/SEU_USUARIO/vlc-digital-twin.git
cd vlc-digital-twin

# Criar branch descritiva
git checkout -b feature/mimo-beamforming

# Instalar dependências de dev
pip install -r requirements.txt

# Fazer suas alterações e commitar
git add .
git commit -m "feat: adicionar suporte a beamforming adaptativo no MIMO"

# Push e abrir PR
git push origin feature/mimo-beamforming
```

### Convenções de Código

- **Estilo:** PEP 8, docstrings no formato Google Style
- **Tipagem:** Type hints em todas as funções públicas
- **Testes:** Adicione testes em `tests/` para novas funcionalidades
- **Commits:** Formato Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`)

### Checklist antes do PR

- [ ] Código segue PEP 8 (use `black` e `flake8`)
- [ ] Type hints adicionados
- [ ] Docstrings atualizadas
- [ ] Experimento funciona tanto em CPU quanto em GPU CUDA
- [ ] Imagens de resultado salvas com `dpi=150`
- [ ] CHANGELOG.md atualizado na seção `[Não Lançado]`

## Código de Conduta

Este projeto segue o [Contributor Covenant](https://www.contributor-covenant.org/).
Ao participar, você concorda em manter um ambiente respeitoso e inclusivo.

## Licença

Ao submeter uma contribuição, você concorda que ela será licenciada sob **Apache 2.0**.
