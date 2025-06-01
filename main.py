# %% Importando Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# %% Lendo os dados
df = pd.read_csv('/Users/felipesilveira/PycharmProjects/Estudo_cancer_pulmão/data/Lung_Cancer_Trends_Realistic.csv')

# %% Explorando os dados
print("5 primeiros registros:")
print(df.head())
print("Informações do DataFrame:")
df.info()
print("Estatísticas descritivas para colunas númericas:")
print(df.describe())
print("Estatísticas descritivas para colunas categóricas:")
print(df.describe(include='object'))
print("Verificando valores ausentes:")
print(df.isnull().sum())
print("Verificando se existem linhas duplicadas:")
df.duplicated().sum()

# %% Detecção de Outliers
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=df['Age'])
plt.title('Age Boxplot')
plt.subplot(1, 2, 2)
sns.boxplot(y=df['Years_Smoking'])
plt.title('Years Smoking Boxplot')
plt.tight_layout()
plt.show()

# %% Distribuição da Análise(exemplo com 'Age', 'Years_Smoking', and 'Gender')
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution')
plt.subplot(1, 3, 2)
sns.histplot(df['Years_Smoking'], kde=True)
plt.title('Years Smoking Distribution')
plt.subplot(1, 3, 3)
sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution')
plt.tight_layout()
plt.show()

# %% Correlação das colunas númericas
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap (Numeric Columns)')
plt.show()

# Limpeza e pré-processamento dos dados

# %% Verificando se Patient_ID possui valores únicos
df.Patient_ID.nunique()

# %% Verificando se Patient_ID possui valores duplicados
if df.Patient_ID.duplicated().sum() == 0:
    print("Não existem valores duplicados")

# %% Padronizando os generos
df["Gender"] = df["Gender"].replace({
    "Male": "M", "Female": "F", "Non-Binary": "Other", "Unknown": "Other", "Prefer not to answer": "Other"})

# %% Convertendo colunas (Yes/No para 1/0) para facilitar a análise
col_bin = ["Family_History", "Genetic_Markers_Positive", "Chronic_Lung_Disease"]
for col in col_bin:
    df[col] = df[col].replace({"Yes": 1, "No": 0})

# %% Convertendo colunas ordinais para numericas
ord_cols = {
'Secondhand_Smoke_Exposure': {'Low': 0, 'Medium': 1, 'High': 2},
    'Air_Pollution_Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'Physical_Activity_Level': {'Low': 0, 'Moderate': 1, 'High': 2},
    'Alcohol_Consumption': {'None': 0, 'Low': 1, 'Moderate': 2, 'High': 3},
    'Diet_Quality': {'Poor': 0, 'Average': 1, 'Good': 2},
    'Income_Level': {'Low': 0, 'Middle': 1, 'High': 2},
    'Education_Level': {'Primary': 0, 'Secondary': 1, 'Tertiary': 2},
    'Access_to_Healthcare': {'Poor': 0, 'Average': 1, 'Good': 2},
    'Screening_Frequency': {'Never': 0, 'Occasionally': 1, 'Bi-annually':2, 'Annually': 3, 'Regularly': 4}
}

# %%Check for combinations of smoking status and years smoked
df = df[~((df['Smoking_Status'] == 'Never') & (df['Years_Smoking'] > 0))]
df = df[~((df['Smoking_Status'] == 'Former') & (df['Years_Smoking'] == 0))]

display(df.head())

# %% Variável alvo
df['Survived'] = df['Survival_Status'].map({'Alive': 1, 'Deceased': 0}).fillna(-1)

# Criando recursos

# %% Categorizando "Age" em grupos
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 25, 35, 45, 55, 65, 100], labels=['0-18', '18-25', '25-35', '35-45', '45-55', '55-65', '65+'])

# %% Medindo a exposição ao fumo
df['Pack_Years'] = (df['Cigarettes_Per_Day'] / 20) * df['Years_Smoking']
# "Never" para alguém que nunca fumou
#  "Former" para alguém que tem o histórico de fumo
df.loc[df['Smoking_Status'] == 'Never', 'Pack_Years'] = 0
print("'Pack_Years' calculated. Min:", df['Pack_Years'].min(), "Max:", df['Pack_Years'].max())

# %% Checando se restou alguma coluna com valores faltando
print("Colunas com valores faltando:")
print(df.isnull().sum())

# %% Salvando o arquivo com dados limpos e preparados para análise e reportes
cleaned_file_path = "data/Lung_Cancer_Data_Cleaned_Prepared.csv"
df.to_csv(cleaned_file_path, index=False) # 'index=False' avoids writing the DataFrame index as a column
print(f"Cleaned and prepared data saved to '{"/Users/felipesilveira/PycharmProjects/Estudo_cancer_pulmão/data}")


# EDA

# %% Função para salvar gráficos 

def save_plot(figure_object, filename_with_extension):
    figure_object.tight_layout()
    full_path = os.path.join(output_plot_dir, filename_with_extension)
    figure_object.savefig(full_path)
    print(f"Plot saved: {'/Users/felipesilveira/PycharmProjects/Estudo_cancer_pulmão/plot'}")
    plt.close(figure_object)

# %% Distribuição de diagnósticos através dos anos
fig1, ax1 = plt.subplots()
sns.countplot(data=df, x='Diagnosis_Year', palette='viridis', ax=ax1)
ax1.set_title('Number of Lung Cancer Diagnoses per Year', fontsize=16)
ax1.set_xlabel('Diagnosis Year', fontsize=12)
ax1.set_ylabel('Number of Diagnoses', fontsize=12)
ax1.tick_params(axis='x', rotation=45)
fig1.savefig("01_diagnoses_per_year.png")

# %% Diagnóstico por idade
fig2, ax2 = plt.subplots()
sns.histplot(df['Age'], kde=True, bins=30, color='skyblue', ax=ax2)
ax2.set_title('Age Distribution at Diagnosis', fontsize=16)
ax2.set_xlabel('Age', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
fig2.savefig("02_age_distribution.png")

# %% Status de sobrevivência agrupado por idade
fig3, ax3 = plt.subplots()
sns.countplot(data=df, x='Age_Group', hue='Survived', palette='viridis', ax=ax3)
ax3.set_title('Survival Status by Age Group', fontsize=16)
ax3.set_xlabel('Age Group', fontsize=12)
ax3.set_ylabel('Number of Patients', fontsize=12)
ax3.tick_params(axis='x', rotation=45)
fig3.savefig("03_survival_status_by_age_group.png")

# %% Status de sobrevivência agrupado por genero
fig4, ax4 = plt.subplots()
sns.countplot(data=df, x='Gender', hue='Survived', palette='viridis', ax=ax4)
ax4.set_title('Survival Status by Gender', fontsize=16)
ax4.set_xlabel('Gender', fontsize=12)
ax4.set_ylabel('Number of Patients', fontsize=12)
fig4.savefig("04_survival_status_by_gender.png")

# %% Status de sobrevivência por estágio do câncer
def stage_sort_key(stage_string_value):
    if pd.isna(stage_string_value) or stage_string_value == "None": return (0, stage_string_value) # Handle None/NaN
    if "Stage IV" in stage_string_value: return (4, stage_string_value) # Assign higher numbers to later stages
    if "Stage III" in stage_string_value: return (3, stage_string_value)
    if "Stage II" in stage_string_value: return (2, stage_string_value)
    if "Stage I" in stage_string_value: return (1, stage_string_value)
    return (5, stage_string_value) # Other or unknown stages last

unique_stages_in_data = df['Lung_Cancer_Stage'].dropna().unique() # Get unique, non-null stages
stage_plot_order = sorted(unique_stages_in_data, key=stage_sort_key) # Sort them

fig5, ax5 = plt.subplots()
sns.countplot(data=df, x='Lung_Cancer_Stage', hue='Survival_Status', order=stage_plot_order, palette={'Alive': 'lightgreen', 'Deceased': 'salmon'}, ax=ax5)
ax5.set_title('Survival Status by Lung Cancer Stage', fontsize=16)
ax5.set_xlabel('Lung Cancer Stage', fontsize=12)
ax5.set_ylabel('Count', fontsize=12)
ax5.tick_params(axis='x', rotation=45)
ax5.legend(title='Survival Status')
fig4.savefig("05_survival_status_by_lung_cancer_stage")