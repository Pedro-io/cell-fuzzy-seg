# 12. Decisões de projeto

Este capítulo reúne num só lugar as decisões que definem o projeto, cada uma com o motivo e o lugar onde ela se aplica. Os
outros capítulos descrevem o estado atual e citam as decisões por aqui quando o motivo importa.

**Como usar.** Cada decisão tem um identificador `D-nn`, que não é reaproveitado. Uma decisão revista não é apagada: ela fica
marcada como **substituída por D-nn**, para que o caminho até o estado atual continue legível. Decisões em aberto ficam na seção
12.8. Uma decisão nova entra no fim do tema correspondente, com o próximo número livre (hoje, D-41).

As perguntas e respostas originais que deram origem às decisões de 2026-09-27 estão nas seções "Decisões" e "Perguntas
originais" dos documentos `docs/estudo/00` a `06`, no commit `b5455ad`. As decisões de 2026-09-28 em diante estão registradas
em [08-pendencias.md](../estudo/08-pendencias.md) e [09-correcoes-pontuais.md](../estudo/09-correcoes-pontuais.md).

## 12.1 Escopo e pergunta

| ID | Data | Decisão | Motivo | Onde |
|---|---|---|---|---|
| D-01 | 2026-09-27 | Não usar o FMBS. A pergunta do trabalho é se o pipeline da tese funciona, e talvez funcione melhor, com outra camada de segmentação. | Pergunta do autor. | Cap. 2, seção 2.5 |
| D-02 | 2026-08-01 | A camada de segmentação é o ScribblePrompt-UNet congelado. Outras redes podem ser testadas se ele não der resultado. | O motivo da escolha não foi registrado. Argumentos a favor, para o texto: é uma rede de segmentação interativa feita para imagens biomédicas, treinada também com núcleos, e diferenciável por ser uma rede neural. Antes dela foram testados um watershed por marcadores (junho de 2026) e uma U-Net congelada (julho de 2026), ambos descartados por resultados muito ruins, sem números registrados. | Cap. 7 |
| D-03 | 2026-09-27 | O ScribblePrompt-SAM fica adiado. | Ele converte cada pixel rabiscado num clique dado por coordenadas, o que não é diferenciável em relação ao marcador. | Cap. 7, seção 7.1; PD-39 |
| D-04 | 2026-09-27 | Avaliação semântica, por pixel (Dice e IoU); métricas por instância (AJI, PQ) não são exigidas. | Sugestão do coordenador; a saída do método é binária. | Cap. 1, seções 1.5 e 1.6 |
| D-05 | 2026-09-27 | O trabalho será publicado. Todo resultado novo é salvo e reproduzível: métricas por imagem, configuração, commit, modelos, dispersão e comparação pareada com o Cellpose. Os oráculos entram como resultado. | Decisão do autor. | Cap. 9, seção 9.6; PD-47 |
| D-06 | 2026-09-27 | Os experimentos novos rodam antes de levar os resultados ao orientador; a decisão de rumo é do autor. | Decisão do autor. | Cap. 11 |

## 12.2 Dados

| ID | Data | Decisão | Motivo | Onde |
|---|---|---|---|---|
| D-07 | 2026-09-29 | Usar o pacote baixado de novo da página oficial: 37 imagens de treino e 14 de teste. | A cópia anterior tinha só 30 das 37 imagens de treino. | Cap. 3, seção 3.1; PD-23 |
| D-08 | 2026-09-29 | Rasterizar os contornos pelo centro do pixel, descartar regiões degeneradas e, na sobreposição, deixar o menor núcleo vencer. Gerar também a máscara por instância. | O preenchimento pelo OpenCV aumentava a área em cerca de 10%; a máscara por instância permite o mapa de distância por núcleo. | Cap. 3, seção 3.3; PD-25, PD-07 |
| D-09 | 2026-09-29 | Remover as pastas de máscaras prontas (`Binary_masks`, `Binary_masks_instance`). | Não fazem parte do download oficial e não eram usadas. | PD-26 |
| D-10 | 2026-09-27 | Manter os dados no repositório por enquanto. | Sem problemas até agora. | PD-21 (adiada) |
| D-11 | 2026-09-27 | Registrar a origem, a licença e as citações exigidas do MoNuSeg em `data_source/README.md`. | O repositório é público e redistribui dados sob CC BY-NC-SA 4.0. Falta escolher a licença do código. | PD-27 |
| D-12 | 2026-09-27 | Manter a divisão oficial; o teste não entra na escolha de configurações. | Evitar escolher configurações pelo teste. O autor não quis tirar imagens do treino para uma validação fixa, o que levou à validação cruzada (D-33). | Cap. 3, seção 3.5; PD-02 |

## 12.3 Segmentação inicial (Cellpose)

| ID | Data | Decisão | Motivo | Onde |
|---|---|---|---|---|
| D-13 | 2026-09-27 | Usar o modelo `cpsam` do Cellpose 4.1.1, congelado e calculado uma vez. | É o único modelo pré-treinado do Cellpose 4. Calculá-lo uma vez barateia os experimentos. | Cap. 5 |
| D-14 | antes de 2026-09 | Limiares `flow_threshold` 0,2 e `min_size` 4. **Substituída por D-15.** | Testes do autor, sem números registrados. | cap. 5, seção 5.3 |
| D-15 | 2026-10-06 | Limiares padrão da biblioteca (`flow_threshold` 0,4, `min_size` 15) e sem reescala (`diameter=None`). | Medido: sobe o Dice no treino de 0,814 para 0,845, melhor em 37 de 37 imagens; reescalar pelo diâmetro quase não muda e dobra o tempo. | Cap. 5, seção 5.3; PD-30 |
| D-16 | 2026-10-01 | Um nome de modelo desconhecido do Cellpose levanta erro. | Sem isso, a biblioteca troca o modelo silenciosamente e muda a entrada do pipeline. | Cap. 5, seção 5.2; PD-29 |
| D-17 | 2026-09-27 | Remover o código morto (`ModelPipeline`, métodos não usados do `OutputWriter` e outros) e renomear a saída final para `final_segmentation`. Ainda não aplicada. | A chave `segmentation` significa duas coisas; o código morto confunde. | PD-31, PD-33 |

## 12.4 Rede de marcadores

| ID | Data | Decisão | Motivo | Onde |
|---|---|---|---|---|
| D-18 | maio de 2026 | MarkerUNet: U-Net com um codificador ResNet34 do ImageNet e entrada de 4 canais (fusão precoce). | O motivo original não foi registrado. A escolha coincide com a prova de conceito da tese, com o codificador do modelo de objeto; sem hierarquia, não haveria o terceiro codificador. | Cap. 6 |
| D-19 | 2026-09-27 | Testar juntos a padronização do ImageNet e o congelamento do codificador, numa ablação de 5 variantes, depois de escolhida a configuração principal. | 24,4 milhões de parâmetros treinados com 37 imagens; entrada sem a padronização que os pesos esperam. | Cap. 6, seções 6.2 e 6.5; PD-36, PD-38 |
| D-20 | 2026-10-01 | A probabilidade do Cellpose fica disponível como quarto canal, para uma ablação. | Carrega mais informação que a máscara binária. | Cap. 6, seção 6.2; PD-34 |

## 12.5 Camada de segmentação

| ID | Data | Decisão | Motivo | Onde |
|---|---|---|---|---|
| D-21 | antes de 2026-09 | Canal negativo = complemento do marcador (modos `sharpened` e `dense_soft`). **Substituída por D-22.** | O autor não lembra o motivo e considera provável que tenha sido um erro. | Cap. 7, seção 7.3; PD-04 |
| D-22 | 2026-09-27 | Usar só o canal positivo. | O complemento informa o interior não marcado dos núcleos como fundo; no oráculo, derruba os marcadores pequenos (miolo de 0,754 para 0,561). | Cap. 7, seção 7.3; cap. 10; PD-04 |
| D-23 | 2026-10-07 | Converter o marcador com um corte: $\max(m - \tau, 0)/(1 - \tau)$, $\tau = 0{,}5$ (modo `positive`). | Um fundo de $10^{-3}$ no canal positivo já faz a rede segmentar a mais; a sigmoide da MarkerUNet nunca dá zero. | Cap. 7, seção 7.3; PD-49 |
| D-24 | 2026-10-15 | Oferecer como opção um canal de fundo tirado do Cellpose (pixels a mais de 3 px de qualquer núcleo), calculado na leitura dos dados para acompanhar a aumentação. | No oráculo, a informação de borda vem do negativo; sem retreinar, o canal sobe o pipeline de 0,773 para 0,802. | Cap. 7, seção 7.4; 09 §18.15 |
| D-25 | 2026-09-27 | Manter a redução sem *antialias* do pacote do ScribblePrompt. | No oráculo, o *antialias* piora o resultado. | Cap. 7, seção 7.5; PD-37 |
| D-26 | 2026-10-17 | Conferir o SHA-256 do checkpoint do ScribblePrompt no download, gravando antes num arquivo temporário. | O pacote abre o arquivo com `torch.load` sem restrição; um arquivo adulterado executaria código. | Cap. 7, seção 7.7; PD-15 |

## 12.6 Perdas

| ID | Data | Decisão | Motivo | Onde |
|---|---|---|---|---|
| D-27 | 2026-05-19 | Não implementar o $L_{comp}$ da tese. | Complexo e pensado junto com o FMBS; a editabilidade por um humano não é objetivo do trabalho; e ele favorece um componente conexo, quando aqui o marcador deve ter um por núcleo. | Cap. 2, seção 2.4; cap. 8, seção 8.3 |
| D-28 | 2026-08-30 | O $L_{size}$ é $\sum m / \sum y$, como na tese (equação 5.5), sem módulo. É intencional; não trocar por $\lvert$razão $- 1\rvert$. | Alinhar com a tese. Os testes antigos ainda esperam a versão simétrica. | Cap. 8, seção 8.3; PD-11 |
| D-29 | 2026-09-27 | Usar o Dice ou o RMSE como perda de segmentação, nunca os dois. | Decisão do autor; o motivo não foi registrado. | Cap. 8, seção 8.2 |
| D-30 | 2026-10-06 | Mapa de distância normalizado por núcleo, a partir da máscara por instância, antes de mexer no peso do $L_{Dmap}$. | Com o máximo da imagem, o centro de 43,5% dos núcleos do treino era punido como borda. | Cap. 8, seção 8.3; PD-06 |
| D-31 | 2026-09-27 | O termo de borda da imagem fica fora do experimento-base e só volta se provar ganho. | O motivo original do termo é desconhecido, e ele pune 16 a 17% dos pixels de núcleo. Os termos de borda, "não muito fino" e RMSE vieram do orientador. | Cap. 8, seção 8.4; PD-41 |
| D-32 | 2026-09-27 | Experimento-base só com Dice e TV, ScribblePrompt em 256², só positivo, sem Size, Dmap nem borda; depois, reintroduzir as regularizações uma a uma. | Os oráculos mostram que Size, Dmap e borda puxam contra o Dice com esta camada. | Cap. 8, seção 8.6; cap. 11; PD-44 |

## 12.7 Treino e protocolo

| ID | Data | Decisão | Motivo | Onde |
|---|---|---|---|---|
| D-33 | 2026-09-27 | Validação cruzada em 5 *folds* nas 37 imagens de treino, por imagem, sem estratificação. | Escolher configurações sem usar o teste e sem tirar imagens do treino. | Cap. 9, seção 9.2; PD-02 |
| D-34 | 2026-09-27 | Guardar o modelo da melhor época de validação e parar depois de 20 validações sem melhora. | Avaliar a última época não escolhe o melhor modelo. | Cap. 9, seção 9.3; PD-46 |
| D-35 | 2026-09-27 | Aumentação sorteada a cada época e ordem dos lotes embaralhada, num módulo em `src/` usado por todos os experimentos. | A aumentação era sorteada uma vez só, e as épocas repetiam os mesmos lotes. | Cap. 3, seção 3.6; PD-08, PD-19 |
| D-36 | 2026-10-17 | Os notebooks de experimento fazem só a validação cruzada; um notebook final treina a configuração escolhida com as 37 imagens e avalia o teste uma única vez. | Não olhar o teste enquanto se escolhem configurações. | Cap. 9, seção 9.4; PD-02 |
| D-37 | 2026-10-17 | Gravar os modelos de cada *fold* e o final no Google Drive de quem roda, em float32, com leitura restrita a pesos. | Refazer a predição de cada imagem pelo modelo que não a viu e permitir o *ensemble*. | Cap. 9, seção 9.6; PD-10 |
| D-38 | 2026-10-17 | Os notebooks já rodados na estrutura antiga vão para `legado/`, com uma célula que diz o que não têm, o resultado e o que se aprendeu. | Manter os resultados sem misturar estruturas. | Cap. 11, seção 11.1 |
| D-39 | 2026-10-15 | Recortes no pipeline inteiro: treino com recortes de 256 px (MarkerUNet sem redução, ScribblePrompt em 128²) e avaliação da imagem inteira em blocos sobrepostos. | O oráculo de resolução apontou esse formato como equilibrado; o autor também não queria reduzir tanto a imagem na MarkerUNet. A alternativa descartada era usar blocos só na entrada do ScribblePrompt. | Cap. 4, seção 4.3; 09 §18.15 |
| D-40 | 2026-10-15 | Ordem dos próximos passos, uma mudança por vez: canal de fundo do Cellpose, depois recortes sem ele, depois os dois juntos. | Medir o efeito de cada mudança separadamente. | Cap. 11, seção 11.5 |

## 12.8 Em aberto

| Assunto | Opções | Onde |
|---|---|---|
| Épocas do treino final | Mediana das melhores épocas dos *folds* (atual) ou a época de maior Dice na curva média dos *folds*. | Cap. 9, seção 9.4 |
| Rumo depois dos próximos experimentos | Entre outras, o "caminho d": a MarkerUNet passa a corrigir a segmentação do Cellpose. Decisão do autor, depois dos experimentos do bloco B. | Cap. 11, seção 11.6; PD-40 |
| Enquadramento pela seção 7.1.2 da tese | Ainda não discutido com o orientador. | Cap. 2, seção 2.4 |
| Comparação com um modelo totalmente supervisionado | Proposta, não decidida: a tese (seção 7.1.2) sugere essa comparação, e um revisor pode pedi-la. | Cap. 2, seção 2.4 |
| Faixa livre do canal de fundo | 3 px foi escolhido a dedo. Testar 1 ou 2 px, ou um raio proporcional ao diâmetro dos núcleos de cada imagem? | Cap. 7, seção 7.4 |
| Licença do código | A escolher. | PD-27 |
| Sobreposição de núcleos na máscara por instância | Hoje o menor núcleo vence; os autores do MoNuSeg deram o pixel ao maior. Só afeta o mapa de distância. | Cap. 3, seção 3.3 |
| Efeito de o Cellpose-SAM ter visto as imagens de treino | Medir com um modelo que não viu o MoNuSeg, ou registrar só como limitação. | Cap. 5, seção 5.4 |

## Para o artigo

- **Métodos:** as decisões D-04, D-08, D-15, D-22 a D-24, D-27 a D-30, D-33 a D-36 e D-39 aparecem como escolhas de método, com o
  motivo em uma frase quando ele não for óbvio.
- **Discussão:** D-02 (as camadas descartadas antes do ScribblePrompt, se houver números para mostrar), D-22 e D-27.

## Referências

As referências estão nos capítulos citados em cada decisão.
