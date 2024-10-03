import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ler o arquivo CSV
df = pd.read_csv('ecommerce_estatistica.csv')

# Exibir as primeiras linhas do dataset
print(df.head())

# Resumo estatístico dos dados
print(df.describe())

# Informações sobre o tipo de dados
print(df.info())

# --- Gráficos ---

# 1. Gráfico de Histograma - Distribuição do Preço
plt.figure(figsize=(10, 6))
plt.hist(df['Preço'], bins=20, color='skyblue', alpha=0.7)
plt.title('Distribuição do Preço dos Produtos', fontsize=14)
plt.xlabel('Preço', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.grid(True)
plt.legend(['Frequência'])
plt.show()

# 2. Gráfico de Dispersão - Nota vs Preço
plt.figure(figsize=(10, 6))
plt.scatter(df['Nota'], df['Preço'], color='purple', alpha=0.7)
plt.title('Dispersão entre Nota e Preço dos Produtos', fontsize=14)
plt.xlabel('Nota (Escala de 1 a 5)', fontsize=12)
plt.ylabel('Preço (R$)', fontsize=12)
plt.grid(True)
plt.legend(['Nota vs Preço'])
plt.show()

# 3. Mapa de Calor - Correlações entre variáveis numéricas
# Selecionar apenas colunas numéricas
df_numerico = df.select_dtypes(include=['float64', 'int64'])

# Calcular e visualizar as correlações
plt.figure(figsize=(10, 6))
correlacoes = df_numerico.corr()
sns.heatmap(correlacoes, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Mapa de Calor das Correlações entre Variáveis', fontsize=14)
plt.show()

# 4. Gráfico de Barra - Número de Produtos por Marca
plt.figure(figsize=(12, 6))
df['Marca'].value_counts().head(10).plot(kind='bar', color='lightgreen')
plt.title('Número de Produtos por Marca (Top 10)', fontsize=14)
plt.xlabel('Marca', fontsize=12)
plt.ylabel('Número de Produtos', fontsize=12)
plt.xticks(rotation=45)  # Rotaciona os rótulos do eixo x para melhor legibilidade
plt.grid(True)
plt.legend(['Número de Produtos'])
plt.show()

# 5. Gráfico de Pizza - Distribuição dos Materiais
plt.figure(figsize=(8, 8))
df['Material'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightblue', 'orange', 'lightgreen'])
plt.title('Distribuição dos Materiais dos Produtos', fontsize=14)
plt.ylabel('')
plt.axis('equal')  # Para deixar o gráfico de pizza com aspecto circular
plt.legend(df['Material'].value_counts().index, title="Materiais")
plt.show()

# 6. Gráfico de Densidade - Distribuição de Densidade do Preço
plt.figure(figsize=(10, 6))
sns.kdeplot(df['Preço'], shade=True, color='red', bw_adjust=0.5)  # bw_adjust ajusta a suavização
plt.title('Distribuição de Densidade do Preço dos Produtos', fontsize=14)
plt.xlabel('Preço (R$)', fontsize=12)
plt.ylabel('Densidade', fontsize=12)
plt.grid(True)
plt.legend(['Densidade'])
plt.show()

# 7. Gráfico de Regressão - Nota vs Preço
plt.figure(figsize=(10, 6))
sns.regplot(x='Nota', y='Preço', data=df, color='blue', line_kws={'color': 'red'})
plt.title('Regressão entre Nota e Preço dos Produtos', fontsize=14)
plt.xlabel('Nota (Escala de 1 a 5)', fontsize=12)
plt.ylabel('Preço (R$)', fontsize=12)
plt.grid(True)
plt.legend(['Regressão'])
plt.show()
