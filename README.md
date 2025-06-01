Análise de Tendências de Câncer de Pulmão
Este projeto realiza uma análise exploratória de dados (EDA) em um conjunto de dados simulado sobre tendências de câncer de pulmão. O objetivo é limpar os dados, realizar engenharia de features e gerar visualizações para identificar padrões e insights relevantes.

Sobre o Projeto
O script Python (main.py) processa o arquivo Lung_Cancer_Trends_Realistic.csv para:

Carregar e Inspecionar os Dados: Verifica a estrutura inicial, tipos de dados e estatísticas descritivas.

Limpeza e Pré-processamento:

Trata valores ausentes e linhas duplicadas.

Padroniza colunas categóricas (ex: Gender).

Converte colunas binárias (ex: Family_History 'Sim'/'Não') para formato numérico (1/0).

Mapeia colunas ordinais (ex: Air_Pollution_Level, Income_Level) para representações numéricas.

Engenharia de Features:

Cria a feature Age_Group a partir da coluna Age.

Calcula Pack_Years (maços-ano) com base nos dados de tabagismo.

Cria a feature numérica Survived a partir da coluna Survival_Status.

Análise Exploratória de Dados (EDA) com Visualizações:

Distribuição de diagnósticos ao longo dos anos.

Distribuição de idade no momento do diagnóstico.

Status de sobrevivência por grupo etário, gênero e estágio do câncer.

Impacto do status de tabagismo e maços-ano no estágio do câncer.

Heatmap de correlação entre features numéricas e ordinais.

Influência de fatores socioeconômicos (nível de renda) e ambientais (nível de poluição do ar) no status de sobrevivência ou estágio do câncer.

Distribuição do IMC (Índice de Massa Corporal).

Análise da exposição ocupacional e região em relação ao estágio do câncer ou status de sobrevivência.

Conjunto de Dados
O conjunto de dados utilizado é o Lung_Cancer_Trends_Realistic.csv. Este arquivo deve estar no mesmo diretório que o script Python para que ele seja executado corretamente.


Certifique-se de ter o Python instalado. As seguintes bibliotecas Python são necessárias:

pandas

numpy

matplotlib

seaborn

