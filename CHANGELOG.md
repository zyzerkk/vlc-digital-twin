# Changelog

Todas as mudanças notáveis neste projeto são documentadas neste arquivo.
O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e o projeto adere ao [Versionamento Semântico](https://semver.org/lang/pt-BR/).

---

## [2.0.0] — 2026-08-20

> **Contexto:** esta versão reestrutura o repositório para ser 100% orientado a execução via
> Google Colab, elimina inconsistências entre documentação e código identificadas em auditoria
> interna, e incorpora os experimentos avançados e a migração para PhysicsNeMo v2.0 nativo, que
> antes existiam apenas fora do repositório público.

### Alterado (breaking changes)
- **Removidos `src/*.py` e `build_repo.py`.** A versão 1.0.0 mantinha dois conjuntos de código
  divergentes para os mesmos experimentos (os scripts commitados em `src/` e o código gerado por
  `build_repo.py`), que produziam resultados e nomes de arquivo diferentes entre si. Os notebooks
  em `notebooks/` passam a ser a única fonte de verdade — cada um instala suas próprias
  dependências e roda de forma independente no Google Colab, sem necessidade de ambiente local.
- **Removida a pasta `.github/`** (workflows de CI e templates de issue) — estava documentada no
  README e no `docs/architecture.md` da v1.0.0, mas nunca havia sido de fato publicada no
  repositório.
- `requirements.txt` removido — cada notebook já declara e instala suas próprias dependências.

### Adicionado
- **`notebooks/02_experimentos_avancados.ipynb`** — EXP4 (classificador corrigido), EXP5
  (ablation study PINN vs. MLP), EXP6 (curriculum learning) e EXP7 (transfer learning).
  Anteriormente existiam apenas fora do repositório público.
- **`notebooks/03_physicsnemo_v2_nativo.ipynb`** — migração de todo o pipeline para a API oficial
  do NVIDIA PhysicsNeMo v2.0 (`FullyConnected`, `FNO`, `physicsnemo.sym`). Anteriormente existia
  apenas fora do repositório público.
- `assets/` reorganizado com prefixo por notebook de origem (`01_`, `02_`, `03_`) e ampliado de 4
  para 13 figuras, cobrindo todos os experimentos com resultado visual salvo.
- `EXPERIMENTS.md` reescrito para cobrir os 12 experimentos, incluindo diagnóstico de causa raiz
  (EXP2) e registro explícito de limitações não resolvidas (EXP3.1-v2, EXP3.3-v2, EXP6).
- `docs/architecture.md` ampliado para cobrir os modelos dos três notebooks e substituída a
  afirmação genérica "sementes fixadas em todos os scripts" por uma tabela precisa de quais
  notebooks efetivamente fixam seed.
- Seção "Como executar via Google Colab" no README, com badges de abertura direta por notebook.

### Corrigido
- README v1.0.0 descrevia uma árvore de arquivos (5 notebooks individuais + `.github/`) que não
  existia no repositório publicado — corrigido para refletir a estrutura real.
- `EXPERIMENTS.md` v1.0.0 referenciava `resultado_exp3_2_sombra.png` (nome incorreto; o arquivo
  real era `resultado_exp3_2_sombreamento.png`) e `resultado_exp3_3_mimo_vlc.png` (arquivo que
  nunca existiu, pois o script correspondente não salvava imagem) — ambos corrigidos ou removidos
  com nota explícita quando a evidência visual de fato não existe (ver EXP3.3 no notebook 01).
- Removida a alegação de que todos os scripts fixavam `seed=42` — falsa para 4 dos 5 experimentos
  do antigo `src/` (agora notebook 01); substituída por tabela precisa por notebook.

---

## [1.0.0] — 2026-08-15

> **Contexto:** esta versão formalizou uma pesquisa exploratória iniciada em julho de 2023.
> Os experimentos foram originalmente desenvolvidos no Google Colab.

### Adicionado
- Experimentos 1, 2, 3.1, 3.2 e 3.3 (PINN SNR, classificador de modulação, gêmeo digital 3D,
  sombreamento, MIMO-VLC), implementados em PyTorch puro.
- Scripts Python (`src/`) e gerador de estrutura (`build_repo.py`).
- Documentação inicial (README, EXPERIMENTS, CONTRIBUTING, CITATION.cff).
- Licença Apache 2.0.

### Problemas documentados (pesquisa 2023)
- Link GitLab da NVIDIA estava offline; repositório migrou para GitHub.
- Incompatibilidade entre Python 3.11 (Colab padrão) e Modulus 22.09 (exigia ≤ 3.10).
