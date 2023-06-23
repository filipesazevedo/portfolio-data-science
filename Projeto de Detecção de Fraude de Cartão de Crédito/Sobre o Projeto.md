# Projeto de Detecção de Fraude de Cartão de Crédito

Um problema recorrente tanto para a sociedade quanto para as instituições financeiras, como bancos e fintechs, são as fraudes de cartões de crédito. Estima-se que quase 20% da população brasileira já teve seu cartão de crédito utilizado por terceiros sem qualquer autorização. Isso é algo preocupante, ainda mais se tratando de um dos meios de pagamento mais utilizados no Brasil.

Essas fraudes geram danos financeiros tanto para consumidores de cartões de crédito quanto para os bancos e fintechs. Também, outro fator relevante a se considerar é a quantidade de falsos positivos (quando o cliente tem seu cartão bloqueado preventivamente pela instituição financeira) que ocorrem para tentar impedir possíveis fraudes, o que se torna uma dor de cabeça no dia a dia do consumidor.

Por conta de todos esses motivos, as empresas acabam investindo cada vez mais na detecção de fraudes utilizando a Inteligência Artificial. Isso significa que utilizando os conhecimentos de Ciência de Dados, principalmente modelos de *Machine Learning*, as empresas podem gerar uma significativa economia com problemas de fraudes.

Fonte: [2 em cada 10 brasileiros já sofreram fraudes de cartão de crédito](https://blog.idwall.co/fraudes-de-cartao-de-credito/)

Este projeto mostrará a criação de uma solução de Ciência de Dados para detecção de fraudes, sendo útil para a redução das fraudes de cartões de crédito. 

Para acessar o código deste projeto em Python, ### [clique aqui](https://github.com/filipesazevedo/portifolio-data-science/blob/main/Projeto%20de%20Detec%C3%A7%C3%A3o%20de%20Fraude%20de%20Cart%C3%A3o%20de%20Cr%C3%A9dito/projeto-fraude-cartao-credito.py).

---

## Dados do Projeto

Neste projeto foram utilizados dados que apresentam transações financeiras que aconteceram em dois dias, disponibilizados por algumas empresas européias de cartões de crédito. Este *dataset* apresenta 492 fraudes em 284807 transações, o que denota que um grande desbalanceamento dos dados, pois as fraudes representam 0,172% do total de registros.

É informado na página original do *dataset* que os dados possuem apenas variáveis numéricas por causa de uma transformação PCA (Principal Component Analysis - em português, Análise de Componentes Principais). Assim, a maioria das variáveis passaram por essa transformação para fins de confidencialidade dos dados, resultando em colunas com nomes de V1, V2, V3, ..., v28. 

Para acessar a página que contém os dados utilizados no projeto e mais detalhes sobre eles, [clique aqui](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

---

## Análise Exploratória dos Dados

Inicialmente, os dados foram carregados e colocados em um *dataframe* para que fosse possível fazer a sua análise exploratória. Essa análise inicial dos dados é o que permite descobrir mais sobre os dados e como prepará-los para uso de modelo de *machine learning*.

A primeira etapa da análise foi descobrir como os dados estavam dispostos, visualizando os cinco primeiros registros com o método `head()`. Foi visto que o *dataset* tem as colunas transformadas V1, V2, V3, ..., v28 e as colunas não transformadas `Time`, `Amount` e `Class`. É por meio da coluna `Class` que se sabe quais são as transações normais (com valor 0) e as transações com fraude (com valor 1).

Depois foi verificado o resumo estatístico dos dados com o método `descibre()`. As variáveis com transformação PCA não tem qualquer problema aparente, assim como a `Time`. A partir da variável `Amount`, percebe-se que a média de transações é de 88.34, mediana de 22.00, desvio padrão de 250.12 e o valor máximo é de 25691.16. Assim, entende-se que a maior parte de todas as transações financeiras são de valores pequenos.

Também, foi verificado se o *dataset* possui valores ausentes com o comando `df.isnull().sum().max()`. Não existem dados faltantes, pois o maior valor nulo encontrado é zero. Além do mais, este conjunto de dados não apresenta a necessidade de um processo de limpeza.

Outra informação verificada foi o balanceamento dos dados, ou seja, a proporção de transações fraudulentas em relação ao total de transações. Conforme mencionado pela página dos dados, apenas 0,172% das transações são fraude. Isso pode ser melhor visualizado através do gráfico de barras gerado com a distribuição entre as duas classes (transações normais e transações com fraude) abaixo. Essa grande desproporcionalidade entre as classes 0 e 1 interfere bastante no desemmpenho de modelos de *machine learning*, portanto é essêncial realizar o balanceamento dos dados neste projeto.

![Gráfico 1 - Distribuição das Classes](https://raw.githubusercontent.com/filipesazevedo/portifolio-data-science/main/Projeto%20de%20Detec%C3%A7%C3%A3o%20de%20Fraude%20de%20Cart%C3%A3o%20de%20Cr%C3%A9dito/imagens/grafico1.png)

Em seguida, foram gerados dois gráficos para comparar as distribuições das duas classes em função do tempo. Entretanto, nenhuma informação nova foi identificada a partir das distribuições abaixo.

![Gráfico 2 - Distribuição das Classes ao longo do tempo](https://raw.githubusercontent.com/filipesazevedo/portifolio-data-science/main/Projeto%20de%20Detec%C3%A7%C3%A3o%20de%20Fraude%20de%20Cart%C3%A3o%20de%20Cr%C3%A9dito/imagens/grafico2.png)

Também, foram verificados os *boxplots* de cada classe para ver se há alguma diferença no padrão das transações financeiras em relação à dimensão `Amount`. Conforme pode ser vista na imagem abaixo, os boxplots evidenciam uma distribuição diferente para as classes, o que pode influenciar no treinamento do modelo de *machine learning*.

![Gráfico 3 - Boxplots das Classes em relação a 'Amount'](https://raw.githubusercontent.com/filipesazevedo/portifolio-data-science/main/Projeto%20de%20Detec%C3%A7%C3%A3o%20de%20Fraude%20de%20Cart%C3%A3o%20de%20Cr%C3%A9dito/imagens/grafico3.png)

Outra ponto interessante verificado foram as informações estatísticas para a classe das transações com fraude. Neste resumo, a média é de 118.13 e a mediana de 9.21, uma diferença considerável entre valores de média e mediana.

Um ponto relevante para se visualizar são as distribuições de cada uma das variáveis para cada classe. Com isso, pretende-se ver quais são as variáveis mais importantes para detecção de fraudes, ou seja, as que se diferem uma da outra. Observe, por exemplo, que as variáveis V3 e V4 são bem diferentes.

![Gráfico 4 - Distribuição das Variáveis para cada Classe](https://raw.githubusercontent.com/filipesazevedo/portifolio-data-science/main/Projeto%20de%20Detec%C3%A7%C3%A3o%20de%20Fraude%20de%20Cart%C3%A3o%20de%20Cr%C3%A9dito/imagens/grafico4.png)

---

## Preparo dos Dados para o Modelo

Uma parte importante deste projeto é o preparo dos dados para o modelo de *machine learning*, pois os dados precisam estar adequados para isso. Uma vez que se entende o *dataset* na análise exploratória, fica fácil saber como prepará-lo para o modelo.

Primeiro foi feita a padronização das colunas `Amount` e `Time` utilizando a classe `StandardScaler` para deixá-las na mesma escala das outras colunas do *dataframe*, visto que uma diferença de escala pode enviesar o treino do modelo.

Após a padronização, os dados foram divididos entre conjunto de treino e conjunto de teste com a função `train_test_split`. Essa parte é muito importante para o uso de *machine learning*, porque os dados de treino e teste não podem ser os mesmos. É importante ressaltar que foi passado o parâmetro `stratify=y` para que a proporção das classes seja a mesma e, assim, evitar a sub-representação de uma delas.

Uma etapa que não poderia faltar é o balanceamento dos dados. Nesta parte do preparo de dados, foi utilizada a técnica de *under-sampling* para que a classe minoritária seja preservada. Também, foi gerado um gráfico para verificar a nova distribuição das classes, conforme pode ser visto na imagem abaixo.

![Gráfico 5 - Nova Distribuição Classes](https://raw.githubusercontent.com/filipesazevedo/portifolio-data-science/main/Projeto%20de%20Detec%C3%A7%C3%A3o%20de%20Fraude%20de%20Cart%C3%A3o%20de%20Cr%C3%A9dito/imagens/grafico5.png)

Após o balanceamento, foram geradas duas matrizes de correlação, uma mostrando as correlações das variáveis antes do balanceamento e outra fazendo o mesmo com os dados balanceados. Com as matrizes abaixo,  é possível ver a nítida diferença entre a matriz de correlação com dados desbalanceados e a mesma com dados balanceados, sendo que a segunda apresenta muito melhor as correlações entre as variáveis.

![Gráfico 6 - Matriz de Correlação: Dados Desbalanceados X Dados Balanceados](https://raw.githubusercontent.com/filipesazevedo/portifolio-data-science/main/Projeto%20de%20Detec%C3%A7%C3%A3o%20de%20Fraude%20de%20Cart%C3%A3o%20de%20Cr%C3%A9dito/imagens/grafico6.png)

---

## Modelo de *Machine Learning*

Após o preparo dos dados, pode-se finalmente construir um modelo de *machine learning*. Como neste projeto a proposta é fazer uma solução que permite detectar fraudes em transações financeiras de cartões de crédito, foi utilizado um modelo de Regressão Logística. A regressão logística é um algoritmo usado para problemas de classificação binária em que o objetivo é prever uma das duas classes possíveis com base em dados de entrada. Como este problema possui uma classe para transações normais (valor 0) e outra para transações com fraude (valor 1), este modelo de classificação cumpre bem o papel de detectar as fraudes.

O modelo de regressão logística instanciado foi treinado com os dados contidos em `X_rus` e `y_rus`. Com o modelo treinado, foram feitas as previsões com os dados de teste.

Após os testes, foi feita a avaliação de desempenho do modelo. Apesar da acurácia do modelo obtido ser de 96,71%, esta não foi a única métrica observada. Depender somente da acurácia pode ser problemático, especialmente em casos de dados originalmente desbalanceados. Portanto, também foi gerado um relatório de classificação com a precisão (precision), recall, f1-score e support do modelo. A tabela abaixo apresenta melhor essas informações do relatório de classificação.

### Relatório de Classificação
| Class | Precision | Recall | F1-Score | Support |
|--- |--- |--- |--- |--- |
| 0 | 0.9998 | 0.9673 | 0.9833 | 71079 |
| 1 | 0.0455 | 0.9024 | 0.0867 | 123 |

Outra métrica interessante para é a AUC ROC, também chamada de área sob a curva ROC (Receiver Operating Characteristic). Esta é uma métrica que mede a capacidade do modelo de distinguir corretamente as classes. No caso desse modelo obteve-se 95% de AUC.

No final, foi gerada a matriz de confusão que mostra a taxa de acertos para transações com fraude, comparando os valores reais com os valores previstos. Na matriz de confusão abaixo, pode-se perceber os quadrantes em azul escuro com os acertos do modelo tanto para os casos de transações normas quanto para os casos com fraude. Além disso, nos quadrantes em cor branca, tem-se os erros para ambos.

![Gráfico 7 - Matriz de Confusão](https://raw.githubusercontent.com/filipesazevedo/portifolio-data-science/main/Projeto%20de%20Detec%C3%A7%C3%A3o%20de%20Fraude%20de%20Cart%C3%A3o%20de%20Cr%C3%A9dito/imagens/grafico7.png)

---

## Conclusão

Após a execução deste projeto, pode-se dizer que, mesmo com dados sem valores ausentes e bem tratados, ainda é demandado um pouco de trabalho para adequá-los, por conta do desbalanceamento e da transformação PCA. Foi necessário fazer o balanceamento dos dados e a padronização de duas variáveis para igualá-las com as variáveis que passaram pela transformação PCA, assim deixando o *dataset* pronto para a aplicação de modelo de *machine learning*.

Quanto ao desempenho do modelo, alcançou-se uma boa taxa de acerto. O uso de regressão logística resolveu bem o problema de detectar fraudes de transações de cartão de crédito, porém ainda há espaço para otimização do modelo e, até mesmo, testar outros tipos diferentes de modelo de classificação, a fim de verificar seus desempenhos.

---
