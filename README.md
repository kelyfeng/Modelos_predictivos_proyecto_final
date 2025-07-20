# Proyecto final - Modelos predictivos
## Análisis predictivo de la deserción de clientes en el sector de telecomunicaciones utilizando técnicas de aprendizaje automático<img width="468" height="37" alt="image" src="https://github.com/user-attachments/assets/46deaf4f-30a8-4bb1-90b8-a9f64098af07" />

📚 Maestría en Analítica de Datos – Universidad Tecnológica de Panamá  
📅 23 de julio de 2025  
👩‍💻 Estudiante: Kely Feng  
👨‍🏫 Profesor: Juan Marcos Castillo, PhD

---

### Descripción general del proyecto

Este repositorio contiene el desarrollo de un modelo predictivo para anticipar el churn (abandono) de clientes en una empresa de telecomunicaciones. El proyecto integra análisis exploratorio de datos, selección de variables y entrenamiento de diferentes algoritmos de machine learning con el objetivo de identificar patrones y factores clave asociados al abandono de los clientes.

---

### Motivación
El churn representa uno de los mayores retos en la industria de telecomunicaciones, pues la pérdida de clientes impacta directamente en la rentabilidad y sostenibilidad del negocio. Predecir el abandono con anticipación permite a las empresas enfocar sus esfuerzos en retención, diseñar estrategias más efectivas y reducir costos relacionados con la adquisición de nuevos clientes. Este proyecto busca aprovechar el potencial de los datos y los modelos predictivos para ofrecer una herramienta de apoyo a la toma de decisiones comerciales.

---


### Descripción del dataset

El proyecto utiliza el conjunto de datos **Telco Customer Churn** proveniente de [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn), que contiene información real de clientes de una empresa de telecomunicaciones.

- **Observaciones:** 7,043 clientes
- **Variables:** 21 columnas, incluyendo datos demográficos, de servicios y facturación

### Principales variables

- `gender`: Género del cliente (`Male`, `Female`)
- `SeniorCitizen`: Si es adulto mayor (`1` = Sí, `0` = No)
- `Partner`: Tiene pareja (`Yes`, `No`)
- `Dependents`: Tiene dependientes (`Yes`, `No`)
- `tenure`: Meses como cliente
- `PhoneService`: Servicio telefónico (`Yes`, `No`)
- `MultipleLines`: Líneas múltiples (`Yes`, `No`, `No phone service`)
- `InternetService`: Tipo de Internet (`DSL`, `Fiber optic`, `No`)
- `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`: Servicios adicionales (`Yes`, `No`, `No internet service`)
- `Contract`: Tipo de contrato (`Month-to-month`, `One year`, `Two year`)
- `PaperlessBilling`: Facturación electrónica (`Yes`, `No`)
- `PaymentMethod`: Método de pago
- `MonthlyCharges`: Cargo mensual (numérico)
- `TotalCharges`: Cargo total acumulado (numérico)
- `Churn`: **Variable objetivo**, indica si el cliente abandonó la compañía (`Yes`, `No`)

> **Nota:**  
> - Se eliminó la columna `customerID` por ser un identificador único y no aportar valor predictivo.
> - Se corrigieron valores nulos y se ajustaron los tipos de datos para preparar el dataset al modelado.

Este dataset es ampliamente utilizado en la comunidad de ciencia de datos para analizar y predecir el abandono de clientes (**churn**) en empresas de servicios.

---

### Estructura del proyecto

1. **Preprocesamiento y limpieza:**  
   Transformación y depuración de los datos para asegurar su calidad.

2. **Análisis exploratorio:**  
   Visualizaciones y estadísticas para comprender la distribución y relaciones de las variables.

3. **Modelado predictivo:**  
   Implementación y evaluación de modelos de clasificación como regresión logística, árbol de decisión, Random Forest, KNN y SVM.

4. **Resultados y conclusiones:**  
   Comparación de métricas, interpretación de los resultados y hallazgos relevantes.

