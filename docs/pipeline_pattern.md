# Padrão de pipeline usado no projeto

Este projeto usa um padrão de projeto baseado em pipeline para organizar o processamento de imagens em uma sequência lógica de etapas. A ideia central é simples: cada etapa recebe um conjunto de dados, faz uma transformação específica e passa o resultado para a próxima etapa.

Esse padrão é útil porque separa responsabilidades, facilita a manutenção e permite montar diferentes fluxos sem reescrever todo o código.

## O que é um pipeline

Imagine um processo industrial em que cada etapa recebe um material, transforma esse material e entrega para a próxima estação. No caso deste projeto, a "matéria-prima" é um dicionário contendo dados de imagem e, em cada etapa, algo novo é acrescentado ou refinado.

Em vez de colocar tudo em uma função gigante, o fluxo é dividido em pequenos blocos. Cada bloco tem uma responsabilidade bem definida.

Exemplo mental:

1. Entrar com uma imagem
2. Gerar uma segmentação
3. Converter a imagem para um formato visualizado com transparência
4. Gerar marcadores para refinar a segmentação
5. Entregar o resultado final

Cada passo acima pode ser tratado como uma etapa do pipeline.

## Por que esse padrão é usado aqui

O uso desse padrão faz sentido neste projeto por vários motivos:

- O processamento é composto por etapas distintas, como segmentação, conversão de formato e geração de marcadores.
- Cada etapa depende de informações produzidas pela etapa anterior.
- O código fica mais organizado, porque cada etapa é uma unidade pequena e isolada.
- É mais fácil testar e evoluir o sistema, porque mudanças em uma etapa têm menos chances de afetar o restante.
- O mesmo fluxo pode ser reutilizado em diferentes partes do projeto, como treino, pré-processamento e inferência.

Em resumo, o padrão pipeline ajuda a transformar uma sequência complexa de operações em uma estrutura mais compreensível.

## Conceitos principais

### 1. Step

Uma step é uma unidade de processamento. Ela representa uma ação específica do fluxo.

No projeto, todas as steps herdam de uma classe base chamada PipelineStep. Essa classe define um contrato comum:

- recebe um dicionário com dados
- executa uma transformação
- retorna um novo dicionário com os dados atualizados

Essa estrutura torna todas as etapas intercambiáveis. Ou seja, o pipeline não precisa saber detalhes internos de cada step. Ele apenas sabe que cada uma aceita a mesma entrada e produz a mesma forma de saída.

### 2. Data dictionary

Em vez de passar vários argumentos separados, o fluxo usa um dicionário. Esse dicionário é o meio de comunicação entre as etapas.

Exemplo de conteúdo que pode existir nesse dicionário:

- image: a imagem de entrada
- segmentation: a máscara de segmentação
- flows: dados intermediários produzidos pela segmentação
- styles: características extraídas pelo modelo
- rgba: imagem no formato RGB com canal alfa
- markers: marcadores derivados da segmentação

Cada step pode acrescentar novas chaves ao dicionário. Isso é importante, porque uma etapa posterior pode depender dessas informações.

### 3. Pipeline orchestrator

O orchestrator, ou coordenador, é quem controla a ordem das etapas. Ele percorre as steps em sequência e passa o mesmo dicionário de dados por todas elas.

O papel do coordenador é simples:

- executar a primeira step
- pegar o resultado
- enviar esse resultado para a segunda step
- continuar até o fim

Isso cria um fluxo linear, mas flexível.

## Como o projeto implementa isso

A implementação do padrão está concentrada em dois pontos principais:

- a classe base das steps
- os pipelines que coordenam a execução

### Classe base de step

A base de todas as steps está em src/pipeline/steps/base_step.py.

Ela define um método chamado forward, que é o ponto onde a lógica principal da step é implementada. Além disso, a classe permite que a step seja chamada como uma função, por meio do método __call__.

Isso torna o uso mais natural:

- a step pode ser invocada diretamente
- o pipeline consegue chamar cada step de forma uniforme

### Pipeline de pré-processamento

O fluxo de pré-processamento está definido em src/pipeline/preprocessing_pipeline.py.

Esse pipeline é usado para executar etapas que não fazem parte do treinamento em si, mas que preparam o dado para o restante do sistema. A lógica é simples:

1. o pipeline recebe um dicionário com dados
2. executa cada step em ordem
3. retorna o dicionário atualizado

Esse tipo de pipeline é particularmente útil quando uma etapa é cara computacionalmente, como a geração de segmentações. Em vez de recalcular tudo sempre, os resultados podem ser reutilizados em outros pontos do fluxo.

No projeto, isso é feito de forma concreta com o `SaveResultsStep`: o pré-processamento roda **uma única vez** (pelo notebook `notebooks/preprocessing/preprocessamento_monuseg_persistido.ipynb`), persiste os resultados em `data_source/MoNuSegPreprocessed/` e os notebooks de experimento (ex.: `notebooks/experiments/experiment_3.ipynb`) apenas **carregam** os dados já processados — sem reprocessar o Cellpose a cada novo experimento.

## As steps concretas do projeto

### CellposeStep

A step CellposeStep é responsável por aplicar a segmentação da imagem usando o modelo Cellpose.

Ela recebe a imagem e acrescenta ao dicionário:

- segmentation
- flows
- styles

Esses resultados representam a saída da segmentação. A partir daí, outras etapas podem usar essa informação para produzir visualizações ou refinamentos adicionais.

### RGBAStep

A step RGBAStep transforma a imagem e a segmentação em uma representação RGBA.

Essa representação é útil porque ela junta:

- a imagem em formato RGB
- a máscara de segmentação como canal alfa

Em outras palavras, a imagem ganha uma camada de transparência que indica onde há objeto segmentado e onde há fundo.

Essa etapa prepara os dados para a próxima fase de processamento, especialmente para inferência visual e geração de marcadores.

### SaveResultsStep

A step SaveResultsStep persiste o resultado do pré-processamento em disco, delegando a gravação ao `src/io/output_writer.py`.

Ela recebe o dicionário já enriquecido pelas steps anteriores e grava cada chave configurada como `output_dir/<chave>/<id>.npy`:

- `image` — a imagem RGB
- `segmentation` — a máscara do Cellpose
- `rgba` — a imagem RGBA de 4 canais
- `ground_truth` — a máscara de referência
- `distance_map` — o mapa de distância

O formato `.npy` preserva a precisão `float32` de `rgba` e `distance_map`. Como o Cellpose é caro, essa step é o que permite rodar vários experimentos **sem reprocessar os dados**: o pré-processamento roda uma vez, fica em disco (`data_source/MoNuSegPreprocessed/`) e os notebooks de treino carregam o resultado.

> **Importante:** o `SaveResultsStep` pertence exclusivamente ao `PreprocessingPipeline` — ele nunca participa do `TrainingPipeline` (que encadeia as redes treináveis).

### MarkerStep

A step MarkerStep gera marcadores a partir da imagem RGBA.

Ela usa a informação criada pela etapa anterior para produzir uma saída chamada markers. Esse resultado pode ser usado para refinar a segmentação ou para servir como entrada para etapas posteriores.

Em alguns cenários, se o modelo não estiver disponível, a step pode fazer um fallback simples e usar a segmentação como base para os marcadores.

## Fluxo de execução típico

Um fluxo típico neste projeto pode ser visto como uma cadeia assim:

1. O sistema recebe uma imagem
2. A step de segmentação gera uma máscara
3. A step de conversão para RGBA prepara a imagem para o próximo estágio
4. A step de marcadores produz uma representação refinada
5. O dicionário final contém todos os resultados intermediários e finais

Essa forma de processamento permite que cada etapa trabalhe com a saída da anterior sem necessidade de reprocessar tudo do zero.

## Como o projeto se encaixa na ideia de pipeline

O padrão se encaixa bem porque o projeto trabalha com uma sequência contínua de transformações sobre dados visuais. Em vez de tratar cada operação de forma isolada, o sistema organiza tudo em um fluxo que cresce de forma modular.

A estrutura atual reflete isso em arquivos como:

- src/pipeline/preprocessing_pipeline.py
- src/pipeline/model_pipeline.py
- src/pipeline/steps/base_step.py
- src/pipeline/steps/preprocessing/cellpose_step.py
- src/pipeline/steps/preprocessing/rgba_step.py
- src/pipeline/steps/persistence/save_results_step.py
- src/pipeline/steps/inference/marker_step.py

Cada um desses módulos representa um pedaço do modelo geral:

- o coordenador de execução
- a definição da interface das steps
- uma etapa específica de processamento

## Vantagens práticas deste desenho

### Modularidade

Cada etapa tem uma função bem delimitada. Isso reduz o acoplamento entre partes diferentes do sistema.

### Reutilização

Uma step pode ser usada em diferentes fluxos, sem precisar ser reescrita para cada caso.

### Extensibilidade

Novas etapas podem ser adicionadas no futuro sem quebrar o restante do pipeline. Basta criar uma nova step e inseri-la na sequência correta.

### Facilidade de depuração

Como cada etapa executa uma tarefa específica, fica mais simples localizar onde um problema ocorreu. Se uma etapa falhar, o sistema consegue registrar qual step foi responsável pelo erro.

### Clareza de fluxo

Mesmo para quem não conhece o padrão, a estrutura se torna mais intuitiva quando se entende que cada etapa recebe um resultado e entrega outro resultado para a etapa seguinte.

## Regras de design importantes

Para que esse padrão funcione bem, algumas convenções são importantes:

- cada step deve receber o mesmo tipo de entrada: um dicionário com dados
- cada step deve retornar um dicionário atualizado
- as etapas devem preferencialmente acrescentar informações em vez de sobrescrever tudo sem necessidade
- o pipeline deve executar as etapas em ordem definida
- uma step deve depender de chaves esperadas no dicionário, como image, segmentation ou rgba

Essas regras ajudam a manter o fluxo previsível.

## Exemplo de pensamento mental

Quando você lê o código, pense assim:

- o dicionário é o "estado atual" do processamento
- cada step lê esse estado e pode adicioná-lo ou melhorá-lo
- o pipeline é o gerente do fluxo

Portanto, o sistema não opera como uma única função que faz tudo de uma vez. Ele opera como uma cadeia de pequenas transformações organizadas.

## Resumo

O padrão pipeline utilizado neste projeto é uma forma estruturada de organizar tarefas de processamento de imagem em etapas sequenciais. Ele transforma um problema complexo em uma série de pequenos blocos bem definidos.

Esse desenho é vantajoso porque:

- organiza melhor o código
- facilita o entendimento do fluxo
- torna o sistema expansível
- melhora a reutilização de componentes
- ajuda na manutenção e na depuração

Para quem está vendo esse padrão pela primeira vez, a melhor forma de entender é pensar em cada etapa como uma estação de um processo: cada estação recebe algo, transforma e entrega para a próxima.
