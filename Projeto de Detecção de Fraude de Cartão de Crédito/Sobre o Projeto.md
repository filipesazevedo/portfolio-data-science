# Projeto de Detecção de Fraude de Cartão de Crédito

Um problema recorrente tanto para a sociedade quanto para as instituições financeiras, como bancos e fintechs, são as fraudes de cartões de crédito. Estima-se que quase 20% da população brasileira já teve seu cartão de crédito utilizado por terceiros sem qualquer autorização. Isso é algo preocupante, ainda mais se tratando de um dos meios de pagamento mais utilizados no Brasil.

Essas fraudes geram danos financeiros tanto para consumidores de cartões de crédito quanto para os bancos e fintechs. Também, outro fator relevante a se considerar é a quantidade de falsos positivos (quando o cliente tem seu cartão bloqueado preventivamente pela instituição financeira) que ocorrem para tentar impedir possíveis fraudes, o que se torna uma dor de cabeça no dia a dia do consumidor.

Por conta de todos esses motivos, as empresas acabam investindo cada vez mais na detecção de fraudes utilizando a Inteligência Artificial. Isso significa que utilizando os conhecimentos de Ciência de Dados, principalmente modelos de *Machine Learning*, as empresas podem gerar uma significativa economia com problemas de fraudes.

Este projeto mostrará a criação de uma solução de Ciência de Dados para detecção de fraudes, sendo útil para a redução das fraudes de cartões de crédito.

Fonte: [2 em cada 10 brasileiros já sofreram fraudes de cartão de crédito](https://blog.idwall.co/fraudes-de-cartao-de-credito/)

---

## Dados do Projeto

Neste projeto foram utilizados dados que apresentam transações financeiras que aconteceram em dois dias, disponibilizados por algumas empresas européias de cartões de crédito. Este *dataset* apresenta 492 fraudes em 284807 transações, o que denota que um grande desbalanceamento dos dados, pois as fraudes representam 0,172% do total de registros.

É informado na página original do *dataset* que os dados possuem apenas variáveis numéricas por causa de uma transformação PCA (Principal Component Analysis - em português, Análise de Componentes Principais). Assim, a maioria das variáveis passaram por essa transformação para fins de confidencialidade dos dados, resultando em colunas com nomes de V1, V2, V3, ..., v28. 

Para acessar a página que contém os dados utilizados no projeto e mais detalhes sobre eles, [clique aqui](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

---

## Análise Exploratória dos Dados

Inicialmente, os dados foram carregados e colocados em um *dataframe* para que fosse possível fazer a sua análise exploratória. Essa análise inicial dos dados é o que permite descobrir mais sobre os dados e como prepará-los para uso de modelo de *machine learning*.

A primeira etapa da análise foi descobrir como os dados estavam dispostos, visualizando os cinco primeiros registros com o método `head()`. Foi visto que o *dataset* tem as colunas transformadas V1, V2, V3, ..., v28 e as colunas não transformadas `Time`, `Amount` e `Class`. É por meio da coluna `Class` que se sabe quais são as transações normais (com valor 0) e as transações com fraude (com valor 1).

Depois foi verificado o resumo estatístico dos dados com o método `descibre()`. As variáveis com transformação PCA não tem qualquer problema aparente, assim como a `Time`. A partir da variável `Amount`, percebe-se que a média de transações é de 88.34, mediana de 22.00, desvio padrão de 250.12 e o valor máximo é de 25691,16. Assim, entende-se que a maior parte de todas as transações financeiras são de valores pequenos.

Também, foi verificado se o *dataset* possui valores ausentes com o comando `df.isnull().sum().max()`. Não existem dados faltantes, pois o maior valor nulo encontrado é zero. Além do mais, este conjunto de dados não apresenta a necessidade de um processo de limpeza.

Outra informação verificada foi o balanceamento dos dados, ou seja, a proporção de transações fraudulentas em relação ao total de transações. Conforme mencionado pela página dos dados, apenas 0,172% das transações são fraude. Isso pode ser melhor visualizado através do gráfico de barras gerado com a distribuição entre as duas classes (transações normais e transações com fraude).

![Gráfico 1 - Distribuição das Classes]([https://drive.google.com/file/d/1oVPQlNHIJY9_tTTuhkYQUsnwCUl2TSt4/view?usp=sharing](https://github.com/filipesazevedo/portifolio-data-science/blob/af089798098953e214af0cffdd9360811b778bf1/Projeto%20de%20Detec%C3%A7%C3%A3o%20de%20Fraude%20de%20Cart%C3%A3o%20de%20Cr%C3%A9dito/imagens/grafico1.png)https://github.com/filipesazevedo/portifolio-data-science/blob/af089798098953e214af0cffdd9360811b778bf1/Projeto%20de%20Detec%C3%A7%C3%A3o%20de%20Fraude%20de%20Cart%C3%A3o%20de%20Cr%C3%A9dito/imagens/grafico1.png)

---
