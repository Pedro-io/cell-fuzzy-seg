# Perdas do módulo de segmentação celular

Este documento descreve a implementação atual do módulo de perdas em `src/losses` e reflete a versão que está ativa no pipeline. Algumas variantes antigas foram ajustadas, reorganizadas ou removidas para manter a base consistente com o código que hoje realmente é usado.

> Observação importante: a documentação foi reduzida para o português e para o conjunto real de perdas implementadas no repositório. Itens obsoletos ou versões antigas não presentes na estrutura atual foram omitidos.

---

## 1. Visão geral

O módulo reúne funções de perda e regularização usadas para treinar modelos de segmentação celular. Em geral, as perdas se dividem em duas categorias:

- Perdas de segmentação: medem a aderência da previsão ao ground truth.
- Perdas de regularização: controlam propriedades geométricas e espaciais da predição, como suavidade, tamanho e presença de artefatos.

A estrutura atual inclui os seguintes arquivos:

```text
src/losses/
├── __init__.py
├── border_loss.py
├── distance_map_loss.py
├── loss_composer.py
├── loss_term.py
├── not_too_thin_loss.py
├── object_size_loss.py
├── rmse_accuracy.py
├── rmse_loss.py
├── soft_dice_loss.py
├── terms.py
├── total_variation_loss.py
└── README.md
```

---

## 2. Implementações ativas e mudanças recentes

A base atual foi simplificada para manter apenas as perdas que fazem sentido para a arquitetura do projeto:

- `SoftDiceLoss`: perda principal de similaridade entre previsão e máscara.
- `RMSELoss`: penalização de erros grandes, útil como termo complementar.
- `ObjectSizeLoss`: preserva a massa total da segmentação.
- `TotalVariationLoss`: melhora suavidade espacial.
- `DistanceMapLoss`: orienta os marcadores para o interior das células.
- `BorderLoss`: reduz falsos positivos nas bordas da imagem.
- `NotTooThinLoss`: elimina estruturas finas e filamentosas.

Além disso, o módulo expõe o padrão de composição `LossComposer` e os wrappers `LossTerm`, `SizeTerm`, `TVTerm`, `DMapTerm`, `BorderTerm`, `DiceTerm`, `RMSETerm` e `NotTooThinTerm` para combinar termos sem alterar a lógica central do treino.

> Algumas implementações antigas ou variantes experimentais foram removidas da documentação porque não fazem parte da API atual do módulo. Isso evita confusão com componentes que não estão mais presentes no código do pipeline.

---

## 3. Explicação teórica das perdas implementadas

### 3.1 `SoftDiceLoss`

Arquivo: `soft_dice_loss.py`

A perda de Dice é baseada na similaridade entre a previsão e o alvo. O coeficiente clássico de Dice é dado por:

$$
\mathrm{Dice} = \frac{2 \sum p_i g_i + \epsilon}{\sum p_i^2 + \sum g_i^2 + \epsilon}
$$

A perda usada no código é a forma diferenciável:

$$
\mathcal{L}_{dice} = 1 - \mathrm{Dice}
$$

#### Por que é usada?

Ela é robusta em cenários de desbalanceamento, muito comuns em segmentação celular, onde o fundo ocupa grande parte da imagem e as células são pequenas e esparsas. Em vez de focar no erro absoluto, a perda compara a sobreposição estrutural entre previsão e máscara.

#### Nível de peso recomendado

- Como perda principal: 1.0
- Em combinação com termos de regularização: 0.5 a 1.0

#### Observação da implementação atual

A classe não recebe um parâmetro de peso interno; o peso real fica no conjunto de termos ou na composição do treino. Em geral, ela deve ficar como termo dominante quando a qualidade estrutural da segmentação é o objetivo principal.

---

### 3.2 `RMSELoss`

Arquivo: `rmse_loss.py`

A perda RMSE é calculada como:

$$
\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_i (p_i - g_i)^2}
$$

#### Por que é usada?

Ela penaliza de forma forte erros grandes. Isso é útil quando a rede precisa produzir valores mais calibrados e a diferença de magnitude entre predição e alvo importa. Em segmentação, costuma ser mais útil como termo complementar do que como perda principal, porque ela não favorece diretamente a sobreposição geométrica da mesma maneira que Dice.

#### Nível de peso recomendado

- Como termo auxiliar: 0.05 a 0.5
- Como perda principal: só se houver necessidade de controlar a escala do erro com muita força, e ainda assim com cautela

#### Observação da implementação atual

A classe não possui parâmetro de peso embutido; quando usada no `LossComposer`, o efeito de peso deve ser controlado pela composição do termo ou por uma camada externa de escala.

---

### 3.3 `ObjectSizeLoss`

Arquivo: `object_size_loss.py`

A ideia da perda é comparar a massa total da previsão com a massa total do ground truth:

$$
\mathcal{L}_{size} = w \cdot \frac{\sum p_i}{\sum g_i}
$$

Quando o ground truth é vazio, a implementação retorna a soma absoluta da predição para evitar que a rede produza massa residual inútil.

#### Por que é usada?

Em segmentações celulares, a rede pode gerar marcadores excessivamente pequenos, demasiadamente grandes ou mal distribuídos. Essa perda atua como uma regularização de massa, incentivando que a previsão tenha uma área total compatível com a referência.

#### Nível de peso recomendado

- Valor típico: 0.05 a 0.2
- Se o problema é subsegmentação persistente: 0.1 a 0.2
- Se a segmentação já está estável: 0.02 a 0.08

#### Observação da implementação atual

Essa perda é muito útil como correção de escala durante o treinamento, especialmente quando a rede aprende a produzir objetos de forma muito pouco densa ou muito dispersa.

---

### 3.4 `TotalVariationLoss`

Arquivo: `total_variation_loss.py`

A variação total penaliza diferenças abruptas entre vizinhos na imagem:

$$
\mathcal{L}_{TV} = w \cdot \frac{\sum |p_{i,j} - p_{i,j-1}| + \sum |p_{i,j} - p_{i-1,j}|}{\sqrt{\sum g_i}}
$$

A normalização pela raiz da massa do ground truth evita que a perda dependa apenas da densidade global dos objetos.

#### Por que é usada?

Ela reduz saltos bruscos e ruído espacial, incentivando mapas mais suaves. Em segmentação, isso ajuda a evitar padrões granulares ou regiões irregulares e fragmentadas.

#### Nível de peso recomendado

- Valor típico: 0.01 a 0.1
- Em imagens com ruído espacial forte: 0.05 a 0.1
- Em segmentações já muito suaves: 0.01 a 0.03

#### Observação da implementação atual

É uma regularização leve e geralmente segura; o principal risco é “apagar” detalhes se o peso for alto demais.

---

### 3.5 `DistanceMapLoss`

Arquivo: `distance_map_loss.py`

A perda do mapa de distância envolve a multiplicação da previsão por um mapa de distância pré-calculado:

$$
\mathcal{L}_{dmap} = w \cdot \frac{\sum p_i \cdot d_i}{\sum g_i}
$$

Onde `d_i` representa a distância do pixel a bordas ou a regiões de menor confiança.

#### Por que é usada?

Ela empurra os marcadores para o interior das estruturas, diminuindo a resposta nas fronteiras entre células. Esse tipo de sinal é especialmente útil em segmentação celular, onde a borda costuma ser o local com maior ambiguidade e maior chance de falso positivo.

#### Nível de peso recomendado

- Valor típico: 0.05 a 0.2
- Se o problema principal for bordas mal posicionadas: 0.1 a 0.2
- Se a rede já está estável: 0.02 a 0.08

#### Observação da implementação atual

Essa é uma boa perda complementar para supervisão direta de marcadores, especialmente quando se trabalha com uma MarkerNet e mapas de distância bem calibrados.

---

### 3.6 `BorderLoss`

Arquivo: `border_loss.py`

A perda penaliza as ativações próximas às bordas da imagem via uma máscara de borda com largura configurável. A operação básica é:

$$
\mathcal{L}_{border} = w \cdot \frac{\sum p_i \cdot m_i}{N \cdot C \cdot H \cdot W}
$$

Onde `m_i = 1` para pixels localizados na região de borda e `m_i = 0` no restante da imagem.

#### Por que é usada?

A borda da imagem normalmente contém menos contexto e mais artefatos de aquisição. Em segmentações celulares, é comum haver falsos positivos ou respostas fracas perto das extremidades da imagem. A perda de borda reduz esse problema.

#### Nível de peso recomendado

- Valor típico: 0.01 a 0.1
- Se há muitos falsos positivos nas bordas: 0.05 a 0.1
- Em imagens bem centradas e sem artefatos: 0.01 a 0.03

#### Observação da implementação atual

A largura da borda é controlada por `border_size`. Quanto maior esse valor, mais agressiva a penalização executa nas extremidades.

---

### 3.7 `NotTooThinLoss`

Arquivo: `not_too_thin_loss.py`

Esse termo usa uma abertura morfológica (erosão seguida de dilatação) para identificar partes finas da predição que seriam removidas por um filtro estrutural. Em termos práticos, a perda tenta reconhecer estruturas que são muito finas e penaliza a presença delas.

#### Por que é usada?

Predições finas e filamentares são frequentes em imagens com ruído ou com estruturas muito elongadas. Em segmentação celular, isso pode produzir artefatos tipo linha, ramificações espúrias ou regiões incompletas. A perda força a rede a preferir objetos mais robustos e menos “finos”.

#### Nível de peso recomendado

- Valor típico: 0.05 a 0.5
- Se os artefatos finos forem frequentes: 0.2 a 0.5
- Se a predição já está bem estável: 0.05 a 0.1

#### Observação da implementação atual

A perda depende fortemente do kernel morfológico. Kernels maiores aumentam a sensibilidade à presença de estruturas finas, enquanto kernels menores são mais suaves.

---

## 4. Padrão de composição das perdas

O módulo também expõe a interface `LossTerm` e o `LossComposer`, que permitem combinar múltiplas perdas sem trocar a lógica principal do treino.

### `LossTerm`

Arquivo: `loss_term.py`

Classe base para qualquer termo que será composto no treinamento. Ela define a interface:

- `name`: identificador do termo no log
- `compute(ctx)`: calcula o valor da perda a partir do contexto do batch

### `LossComposer`

Arquivo: `loss_composer.py`

O `LossComposer` acumula diversos termos em um único valor total:

```python
from src.losses import LossComposer, SizeTerm, TVTerm, DMapTerm

composer = LossComposer([
    SizeTerm(weight=0.1),
    TVTerm(weight=0.05),
    DMapTerm(weight=0.1),
])

total_loss, loss_log = composer(prediction, distance_maps, gt_masks, markers=markers)
```

O retorno inclui:

- `total_loss`: soma dos termos ativos
- `loss_log`: dicionário com cada termo e o valor correspondente

### Termos wrappers disponíveis

Arquivo: `terms.py`

Os termos ativos atualmente incluem:

- `SizeTerm(weight)`
- `TVTerm(weight, power)`
- `DMapTerm(weight)`
- `BorderTerm(weight, border_size)`
- `DiceTerm(epsilon)`
- `RMSETerm()`
- `NotTooThinTerm(kernel, weight)`

---

## 5. Combinações recomendadas

A escolha da combinação depende da fase do treinamento e do tipo de artefato que aparece na predição.

### Configuração conservadora para segmentação estável

```python
LossComposer([
    SizeTerm(0.1),
    TVTerm(0.05),
    DMapTerm(0.1),
])
```

Use quando a rede já está relativamente estável e o objetivo é refinar a segmentação sem exagerar na regularização.

### Configuração para bordas e falsas detecções

```python
LossComposer([
    SoftDiceLoss(),
    ObjectSizeLoss(0.1),
    BorderLoss(weight=0.05, border_size=30),
])
```

Use quando há excesso de respostas nas bordas da imagem.

### Configuração para estruturas finas ou filamentosas

```python
LossComposer([
    SoftDiceLoss(),
    ObjectSizeLoss(0.08),
    NotTooThinLoss(kernel=torch.ones(5, 5), weight=0.2),
])
```

Use quando a rede produz linhas finas ou artefatos morfológicos desnecessários.

---

## 6. Resumo prático

Em geral, a estratégia mais segura é:

1. usar `SoftDiceLoss` como base estrutural;
2. combinar com `ObjectSizeLoss` para controlar a massa total;
3. adicionar `TotalVariationLoss` para suavizar a predição;
4. incluir `DistanceMapLoss`, `BorderLoss` ou `NotTooThinLoss` somente quando houver artefatos típicos do problema específico.

Essa combinação mantém a perda principal focada na segmentação correta e usa regularizações para controlar fenômenos geométricos e de ruído.

---

## 7. Observação final sobre a base atual

A implementação atual foi organizada para refletir somente aquilo que está efetivamente presente em `src/losses`. Isso significa que:

- perdas experimentais ou versões antigas foram removidas da documentação;
- o foco está em um conjunto coerente de perdas de segmentação e regularização;
- os pesos recomendados são valores de referência empírica, e não regras absolutas.

Em treinamento real, os melhores valores costumam depender do conjunto de dados, da resolução da imagem, da densidade de células e da intensidade do ruído.

from src.losses import (
    LossTerm,       # base interface
    LossComposer,   # composition
    SizeTerm, TVTerm, DMapTerm,
    BorderTerm, DiceTerm, RMSETerm, NotTooThinTerm,
)
```
