"""   Projeto de detecção de fraude de cartão de crédito   """


# Imports necessários
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.metrics import confusion_matrix

from imblearn.under_sampling import RandomUnderSampler

# Configurando o estilo de gráfico do Seaborn
sns.set_style('dark')

# Carregando o dataset
df = pd.read_csv('dados\creditcard.csv')



"""   Análise Exploratória dos Dados   """

# Verificando os primeiros registros e o tamanho do dataframe
print('\nShape do dataframe:', df.shape)
print('Primeiras entradas do dataframe')
print(df.head())

# Verificando o resumo estatístico
print('\nResumo estatístico:')
print(df.describe())

# Verificando se há valores ausentes
print('\nQuantidade de valores ausentes:')
print(df.isnull().sum().max())

# Verificando o balanceamento das classes
print('\nQuantidade de registros para cada classe:')
print(df.Class.value_counts())
print('\nPorcentagem de fraudes no dataset:')
print((df[df.Class == 1].shape[0] / df.shape[0]) * 100)

# Plotando o gráfico de barras das classes
fig, ax = plt.subplots()
sns.countplot(x='Class', data=df, ax=ax)
ax.set_title('Distribuição das Classes')
plt.savefig('imagens/grafico1.png')
plt.show()

# Plotando dois gráficos para comparar as distribuições das duas classes
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,6))

num_bins = 40

ax[0].hist(df.Time[df.Class == 0], bins=num_bins)
ax[0].set_title('Normal')

ax[1].hist(df.Time[df.Class == 1], bins=num_bins)
ax[1].set_title('Fraude')

plt.xlabel('Tempo (segundos)')
plt.ylabel('Transações')
plt.tight_layout()
plt.savefig('imagens/grafico2.png')
plt.show()

# Plotando os boxsplots das classes em relação à dimensão "Amount"
fig, ax = plt.subplots(figsize=(6,10), sharex=True)

sns.boxplot(x=df['Class'], y=df['Amount'], showmeans=True, ax=ax)
plt.ylim((-20, 400))
plt.xticks([0, 1], ['Normal', 'Fraude'])

plt.tight_layout()
plt.savefig('imagens/grafico3.png')
plt.show()

# Verificando informações estatísticas
print('\nInformações estatísticas:')
print(df[(df.Class == 1) & (df.Amount < 2000)]['Amount'].describe())

# Plotando gráficos de densidade com a distribuição de cada variável para cada classe
column_names = df.drop(['Class', 'Amount', 'Time'], axis=1).columns
num_plots = len(column_names)
df_class_0 = df[df.Class == 0]
df_class_1 = df[df.Class == 1]

fig, ax = plt.subplots(nrows=7, ncols=4, figsize=(18,18))
fig.subplots_adjust(hspace=1, wspace=1)

idx = 0
for col in column_names:
    idx += 1
    plt.subplot(7, 4, idx)
    sns.kdeplot(df_class_0[col], label="Class 0", fill=True)
    sns.kdeplot(df_class_1[col], label="Class 1", fill=True)
    plt.title(col, fontsize=10)
plt.tight_layout()
plt.savefig('imagens/grafico4.png')
plt.show()



"""   Preparando os Dados para o Modelo   """

# Padronizando as colunas "Time" e "Amount"
df_clean = df.copy()

std_scaler = StandardScaler()
df_clean['std_amount'] = std_scaler.fit_transform(df_clean['Amount'].values.reshape(-1, 1))
df_clean['std_time'] = std_scaler.fit_transform(df_clean['Time'].values.reshape(-1, 1))

df_clean.drop(['Time', 'Amount'], axis=1, inplace=True)

# Verificando as primeiras entradas
print('\nPrimeiras entradas do novo dataframe')
print('\n',df_clean.head())

# Separando variáveis entre X e y
X = df_clean.drop('Class', axis=1)
y = df['Class']

# Dividindo o dataset entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True)

# Utilizando a técnica de under-sampling para balancear as classes
rus = RandomUnderSampler()
X_rus, y_rus = rus.fit_resample(X_train, y_train)

# Verificando o balanceamento das classes
print("Contagem de amostras antes do balanceamento:")
print(pd.Series(y_train).value_counts())

print("\nContagem de amostras após o balanceamento:")
print(pd.Series(y_rus).value_counts())

# Plotando a nova distribuição de classes
class_counts = pd.Series(y_rus).value_counts()
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.xlabel('Classe')
plt.ylabel('Contagem')
plt.title('Nova Distribuição das Classes')
plt.savefig('imagens/grafico5.png')
plt.show()

# Plotando a matriz de correlação de antes e depois do balanceamento
corr = X_train.corr()
corr_rus = pd.DataFrame(X_rus).corr()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (18,8))
fig.suptitle('Matriz de Correlação')

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.1, cmap="coolwarm", ax=ax[0])
ax[0].set_title('Desbalanceado')

sns.heatmap(corr_rus, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.1, cmap="coolwarm", ax=ax[1])
ax[1].set_title('Balanceado')

plt.savefig('imagens/grafico6.png')
plt.show()



"""   Modelo de Classificação - Regressão Logística   """

# Criação e treinamento do modelo de Regressão Logística
np.random.seed(2)
model = LogisticRegression()
model.fit(X_rus, y_rus)

# Fazendo previsões a partir dos dados de teste
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Calculando a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Normalizando a matriz de confusão
cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]

# Plotando a matriz de confusão normalizada
sns.heatmap(cm_norm, annot=True, cmap='Blues', fmt='.2f', cbar=True)
plt.xlabel('Classe Prevista')
plt.ylabel('Classe Real')
plt.title('Matriz de Confusão Normalizada')
plt.savefig('imagens/grafico7.png')
plt.show()

# Imprimindo relatório de classificação
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, digits=4))

# Imprimindo a acurácia do modelo
print("\nAcurácia:", accuracy_score(y_test, y_pred))

# Imprimindo a área sob da curva
print("\nAUC:", roc_auc_score(y_test, y_pred))
