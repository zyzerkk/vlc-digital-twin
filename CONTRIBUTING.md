# Guia de Contribuição

Obrigado por considerar contribuir com o **VLC Digital Twin**!

Este repositório é 100% orientado a notebooks executados via Google Colab — não há scripts
locais, pipeline de CI ou ambiente de desenvolvimento a configurar. Contribuir aqui significa,
na prática, editar ou adicionar notebooks.

## Como Contribuir

### 1. Reportar Bugs ou Resultados Inconsistentes

Abra uma *issue* descrevendo:
- Qual notebook e qual célula/experimento apresentou o problema
- Se possível, um link para uma cópia do notebook no seu Google Drive/Colab reproduzindo o erro
- GPU utilizada (ou CPU) — visível na primeira célula de cada notebook
- Comportamento esperado vs. observado

Se o problema for uma inconsistência entre o que está documentado (`README.md`,
`EXPERIMENTS.md`, `docs/architecture.md`) e o que o notebook realmente produz, isso também é
bem-vindo como *issue* — este projeto trata esse tipo de divergência como prioridade.

### 2. Sugerir Novos Experimentos

Descreva:
- O caso de uso físico ou de engenharia que o experimento atenderia
- Por que ele se encaixa na estrutura atual (extensão de um notebook existente) ou justifica um
  novo notebook
- Referências à literatura de VLC / PINN quando aplicável

### 3. Submeter Pull Requests

```bash
# Fork e clone
git clone https://github.com/SEU_USUARIO/vlc-digital-twin.git
cd vlc-digital-twin

# Criar branch descritiva
git checkout -b fix/exp3-1-fno-reshape
```

**Antes de abrir o PR:**
- Rode o notebook completo no Google Colab (`Ambiente de execução → Executar tudo`) do início ao
  fim, com runtime limpo, e confirme que todas as células executam sem erro.
- Se o PR altera resultados de algum experimento, atualize também as figuras correspondentes em
  `assets/` (mesmo nome de arquivo, prefixado pelo notebook de origem — ex.: `01_exp1_...png`) e
  os números citados em `README.md` e `EXPERIMENTS.md`. Documentação e notebook devem permanecer
  consistentes entre si.
- Se corrigir um problema documentado como "limitação conhecida" (ex.: o problema de FNO no
  EXP3.1-v2/EXP3.3-v2), atualize a seção correspondente em `EXPERIMENTS.md` para refletir a
  correção, em vez de simplesmente remover o registro do problema anterior — preferimos manter o
  histórico de diagnóstico visível.

### 4. Padrão de Documentação

Este projeto trata a documentação honesta de limitações e resultados negativos como parte da
contribuição científica, não como algo a ser omitido. Ao adicionar um experimento novo, inclua
tanto o que funcionou quanto o que não funcionou como esperado.

## Código de Conduta

Seja respeitoso e construtivo. Pesquisa envolve tentativa e erro — o objetivo deste repositório é
documentar esse processo com transparência, não apresentar apenas resultados perfeitos.
