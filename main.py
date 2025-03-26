import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# Sample dataset
data = {'Salary': [30000, 50000, 70000, 90000, 110000]}
df = pd.DataFrame(data)

# **Normalization (Min-Max Scaling)**
scaler = MinMaxScaler()
df['Salary_Normalized'] = scaler.fit_transform(df[['Salary']])

# **Standardization (Z-score Scaling)**
scaler = StandardScaler()
df['Salary_Standardized'] = scaler.fit_transform(df[['Salary']])

# **Visualization - Normalization vs. Standardization**
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Salary'], marker='o', label='Original Salary', linestyle='dashed')
plt.plot(df.index, df['Salary_Normalized'], marker='o', label='Normalized (0-1)')
plt.plot(df.index, df['Salary_Standardized'], marker='o', label='Standardized (Z-score)')
plt.xlabel("Index")
plt.ylabel("Values")
plt.title("Normalization vs. Standardization")
plt.legend()
plt.grid()
plt.show()

# **Categorical Data for One-Hot Encoding**
df_cat = pd.DataFrame({'City': ['New York', 'London', 'Paris', 'New York', 'Paris']})

# **Label Encoding**
label_encoder = LabelEncoder()
df_cat['City_LabelEncoded'] = label_encoder.fit_transform(df_cat['City'])

# **One-Hot Encoding**
df_one_hot = pd.get_dummies(df_cat, columns=['City'], prefix='City')

# **Visualization - One-Hot Encoding Heatmap**
plt.figure(figsize=(6, 4))
sns.heatmap(df_one_hot.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("One-Hot Encoding Heatmap")
plt.show()
