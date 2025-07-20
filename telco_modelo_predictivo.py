# Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

# Configuración de paleta de colores
# Paleta principal
PASTEL_COLORS = [
    "#FBC6D4", "#B9AEDC", "#A0E7E5", "#FFB7B2", "#FFDAC1", "#B5EAD7",
    "#FFFFB5", "#C7CEEA", "#93C6E0", "#D6C1E6", "#E2F0CB", "#A6DCEF"
]
# Para gráficos con dos categorías
DUAL_COLORS = [PASTEL_COLORS[3], PASTEL_COLORS[1]]
# Para fondos, bordes y textos
BORDER_COLOR = "#B9AEDC"
TITLE_COLOR = "#3F4156"

sns.set_palette(PASTEL_COLORS)
plt.rcParams['axes.facecolor'] = '#fcfcfc'
plt.rcParams['figure.facecolor'] = '#fcfcfc'
plt.rcParams['axes.edgecolor'] = BORDER_COLOR
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'

# Carga de dataset
df = pd.read_csv('/Users/kely/PycharmProjects/Proyecto_final_modelos_predictivos_Kely_Feng/data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Exploración inicial
print("Información general del DataFrame:")
print(df.info())
print("\nPrimeros registros del DataFrame:")
print(df.head())
print("\nColumnas del DataFrame:")
print(df.columns.values)
print("\nTipos de datos en el DataFrame:")
print(df.dtypes)

# Limpieza y transformación de datos
# Eliminar columna de customer ID
df = df.drop(['customerID'], axis=1)

# Convertir 'TotalCharges' a numérico, forzando errores a NaN
df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')

# Visualizar valores nulos antes de limpiar
print("\nValores nulos antes de limpieza:")
print(df.isnull().sum())

# Eliminar registros con tenure igual a 0 (clientes muy nuevos o registros erróneos)
df = df[df['tenure'] != 0]

# Rellenar valores nulos en 'TotalCharges' con la media
df['TotalCharges'].fillna(df["TotalCharges"].mean(), inplace=True)

# Cambiar codificación de SeniorCitizen para hacerla más comprensible
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

# Definir columnas numéricas para análisis y escalado
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Visualización de valores nulos
msno.matrix(df, color=(185/255, 174/255, 220/255))
plt.title('Valores nulos', fontsize=16, fontweight='bold', color='#7F53AC', pad=12)
plt.tight_layout()
plt.show()

# Pie charts
# Visualización circular con paleta pastel
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(
    labels=['Male', 'Female'],
    values=df['gender'].value_counts(),
    marker=dict(colors=PASTEL_COLORS[:2], line=dict(color='#fcfcfc', width=2)),
    name="Gender",
    textinfo='percent+label',
    insidetextorientation='radial',
    textfont_size=16
), 1, 1)
fig.add_trace(go.Pie(
    labels=['No', 'Yes'],
    values=df['Churn'].value_counts(),
    marker=dict(colors=[PASTEL_COLORS[2], PASTEL_COLORS[7]], line=dict(color='#fcfcfc', width=2)),
    name="Churn",
    textinfo='percent+label',
    insidetextorientation='radial',
    textfont_size=16
), 1, 2)
fig.update_traces(hole=0.55)
fig.update_layout(
    title={'text': 'Género y Churn', 'x': 0.5, 'font': {'size': 20, 'color': TITLE_COLOR}},
    annotations=[
        dict(text='Género', x=0.18, y=0.5, font_size=18, showarrow=False, font_color='#7F53AC'),
        dict(text='Churn', x=0.83, y=0.5, font_size=18, showarrow=False, font_color='#7F53AC')
    ],
    showlegend=False, margin=dict(t=60, b=0, l=0, r=0), paper_bgcolor='#fcfcfc'
)
fig.show()

# KDE e histogramas para variables numéricas
# KDE: cargos mensuales vs churn
plt.figure(figsize=(9, 5))
sns.kdeplot(df.MonthlyCharges[df["Churn"] == 'No'], color=PASTEL_COLORS[1], shade=True, label="No Churn", alpha=0.6, linewidth=2)
sns.kdeplot(df.MonthlyCharges[df["Churn"] == 'Yes'], color=PASTEL_COLORS[2], shade=True, label="Churn", alpha=0.8, linewidth=2)
plt.ylabel('Densidad')
plt.xlabel('Cargos Mensuales')
plt.title('Distribución de Cargos Mensuales por Churn', fontweight='bold', color=TITLE_COLOR, pad=12)
plt.legend(frameon=False)
plt.grid(axis='y', alpha=0.15)
plt.tight_layout()
plt.show()

# Histograma de tenure
plt.figure(figsize=(8,4))
plt.hist(df['tenure'], bins=20, color=PASTEL_COLORS[5], edgecolor='#fcfcfc')
plt.title('Distribución de Tenure', fontweight='bold', color='#7F53AC')
plt.xlabel('Meses con la compañía')
plt.ylabel('Clientes')
plt.grid(axis='y', alpha=0.11)
plt.tight_layout()
plt.show()

# Boxplot de tenure
fig = px.box(df, x='Churn', y='tenure', color='Churn',
             color_discrete_sequence=[PASTEL_COLORS[1], PASTEL_COLORS[3]],
             points="all", title='Tenure vs Churn')
fig.update_traces(marker=dict(size=6, line=dict(width=1, color=BORDER_COLOR)))
fig.update_layout(paper_bgcolor='#fcfcfc', plot_bgcolor='#fcfcfc', title_font=dict(size=20, color=TITLE_COLOR))
fig.show()

# Matriz de correlación
plt.figure(figsize=(13, 9))
corr = df.apply(lambda x: pd.factorize(x)[0]).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask,
    cmap=sns.color_palette(["#FBC6D4", "#B9AEDC", "#FFFACD", "#A0E7E5", "#FFB7B2"], as_cmap=True),
    xticklabels=corr.columns, yticklabels=corr.columns,
    annot=True, linewidths=.2, vmin=-1, vmax=1,
    annot_kws={"color":TITLE_COLOR},
    cbar_kws={"shrink": 0.7, 'orientation': 'horizontal'})
plt.title('Matriz de correlación', fontweight='bold', color=TITLE_COLOR, pad=12)
plt.tight_layout()
plt.show()

# Visualización de variables vs churn
# 1. Variables categóricas vs Churn: gráfico de barras

# Seleccionar las variables categóricas relevantes
cat_cols = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

for i, col in enumerate(cat_cols):
    n_categories = df[col].nunique()
    palette = PASTEL_COLORS[:n_categories]
    plt.figure(figsize=(6,4))
    ax = sns.countplot(data=df, x=col, hue='Churn', palette=DUAL_COLORS)
    plt.title(f'{col} vs Churn', fontweight='bold', color=TITLE_COLOR)
    plt.xlabel(col)
    plt.ylabel('Cantidad de clientes')
    plt.xticks(rotation=15)
    plt.legend(title='Churn', loc='upper right', frameon=False)
    plt.tight_layout()
    plt.show()

# 2. Variables numéricas vs Churn: boxplot y KDE
for col, main_color in zip(numerical_cols, [PASTEL_COLORS[2], PASTEL_COLORS[3], PASTEL_COLORS[4]]):
    # Boxplot comparando la variable numérica según churn
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Churn', y=col, data=df, palette=DUAL_COLORS)
    plt.title(f'{col} según Churn', fontweight='bold', color=TITLE_COLOR)
    plt.xlabel('Churn')
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()
    # KDE para ver forma y separación
    plt.figure(figsize=(8,5))
    for val, c in zip(['No', 'Yes'], DUAL_COLORS):
        sns.kdeplot(df[col][df['Churn'] == val], color=c, shade=True, label=f"Churn {val}", linewidth=2, alpha=0.7)
    plt.title(f'Distribución de {col} por Churn', fontweight='bold', color=TITLE_COLOR)
    plt.xlabel(col)
    plt.ylabel('Densidad')
    plt.legend(frameon=False)
    plt.grid(axis='y', alpha=0.13)
    plt.tight_layout()
    plt.show()

# Codificar todas las variables categóricas a números para análisis de correlación y modelado
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])

# Análisis de correlación variable objetivo (churn)
# Crear copia codificada para correlación
correlations = df.corr()['Churn'].sort_values(ascending=False)

print("\nCorrelación de todas las variables con 'Churn':")
print(correlations)

plt.figure(figsize=(10,5))
correlations.drop('Churn').plot(kind='bar', color="#B9AEDC", edgecolor="#7F53AC")
plt.title("Correlación de variables con 'Churn'", fontsize=16, fontweight='bold', color='#7F53AC', pad=12)
plt.xlabel("Variable", fontsize=13)
plt.ylabel("Correlación", fontsize=13)
plt.grid(axis='y', alpha=0.18)
plt.tight_layout()
plt.show()

# Preprocesamiento y escalado
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])
X = df.drop('Churn', axis=1)
y = df['Churn'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=40, stratify=y)
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Evaluación de modelos
def evaluar_modelo(modelo, X_tr, X_te, y_tr, y_te, nombre):
    modelo.fit(X_tr, y_tr)
    pred = modelo.predict(X_te)
    acc = accuracy_score(y_te, pred)
    print(f"\n--- {nombre} ---")
    print("Accuracy:", acc)
    print("Matriz de confusión:\n", confusion_matrix(y_te, pred))
    print("Reporte de clasificación:\n", classification_report(y_te, pred))
    return acc

# Entrenamiento y evaluación de modelos
acc_dt = evaluar_modelo(DecisionTreeClassifier(random_state=42), X_train, X_test, y_train, y_test, "Árbol de Decisión")
acc_rf = evaluar_modelo(RandomForestClassifier(n_estimators=100, random_state=42), X_train, X_test, y_train, y_test, "Random Forest")
acc_lr = evaluar_modelo(LogisticRegression(max_iter=1000, random_state=42), X_train_scaled, X_test_scaled, y_train, y_test, "Regresión Logística")
acc_knn = evaluar_modelo(KNeighborsClassifier(n_neighbors=10), X_train_scaled, X_test_scaled, y_train, y_test, "K-Nearest Neighbors")
acc_svm = evaluar_modelo(SVC(kernel='rbf', probability=True, random_state=42), X_train_scaled, X_test_scaled, y_train, y_test, "SVM")

#Importancia de valores de Random Forest
# Entrenar un Random Forest para obtener importancia de variables (feature importance)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Extraer la importancia y ordena de mayor a menor
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

# Visualizar el top 10 de variables más importantes
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices][:10], y=features[indices][:10], palette=PASTEL_COLORS)
plt.title('Top 10 variables más importantes (Random Forest)', fontweight='bold', color=TITLE_COLOR)
plt.xlabel('Importancia')
plt.ylabel('Variable')
plt.grid(axis='x', alpha=0.16)
plt.tight_layout()
plt.show()

print("\nImportancia de variables según Random Forest:")
for f, imp in zip(features[indices], importances[indices]):
    print(f"{f:20}: {imp:.4f}")