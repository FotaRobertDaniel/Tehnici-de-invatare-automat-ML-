import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Incarcarea datelor
train_file_path = '/kaggle/input/citire3/train.csv'
test_file_path = '/kaggle/input/citire3/test.csv'

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

# Pre-procesarea datelor
X_train = train_df.drop(['id', 'is_anomaly'], axis=1)
X_test = test_df.drop(['id'], axis=1)

# Definirea functiilor pentru calculul LOF

def calculate_distance_matrix(X):
    """Calculeaza o matrice de distanta pentru toate punctele."""
    n = X.shape[0]
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = np.linalg.norm(X.iloc[i] - X.iloc[j])
    return distance_matrix

def find_k_nearest_neighbors(distance_matrix, k):
    """Gaseste k vecini apropiati pentru fiecare punct."""
    neighbors = np.argsort(distance_matrix, axis=1)[:, 1:k+1]
    return neighbors

def calculate_local_density(distance_matrix, neighbors):
    """Calculeaza densitatea locala pentru fiecare punct."""
    density = np.zeros(distance_matrix.shape[0])
    for i in range(distance_matrix.shape[0]):
        density[i] = np.mean(distance_matrix[i, neighbors[i]])
    return density

def calculate_lof_scores(density, neighbors):
    """Calculeaza scorurile LOF pentru fiecare punct."""
    lof_scores = np.zeros(density.shape)
    for i in range(density.shape[0]):
        lof_scores[i] = np.mean(density[neighbors[i]]) / density[i]
    return lof_scores

# Setarea parametrului k
k = 20

# Calculul matricei de distante si al vecinilor pentru setul de antrenament
distance_matrix_train = calculate_distance_matrix(X_train)
neighbors_train = find_k_nearest_neighbors(distance_matrix_train, k)

# Calculul densitatilor locale si a scorurilor LOF pentru setul de antrenament
local_density_train = calculate_local_density(distance_matrix_train, neighbors_train)
lof_scores_train = calculate_lof_scores(local_density_train, neighbors_train)

# Calculul matricei de distante pentru setul de testare
distance_matrix_test = np.zeros((X_test.shape[0], X_train.shape[0]))
for i in range(X_test.shape[0]):
    for j in range(X_train.shape[0]):
        distance_matrix_test[i, j] = np.linalg.norm(X_test.iloc[i] - X_train.iloc[j])

# Calculul densitatilor locale si a scorurilor LOF pentru setul de testare
neighbors_test = find_k_nearest_neighbors(distance_matrix_test, k)
local_density_test = calculate_local_density(distance_matrix_test, neighbors_test)
lof_scores_test = calculate_lof_scores(local_density_test, neighbors_test)

# Identificarea anomaliilor (de exemplu, folosind un prag pentru scorurile LOF)
anomaly_threshold = 1.2
test_anomalies = lof_scores_test > anomaly_threshold

# Adaugarea predictiilor la setul de date de test
test_df['is_anomaly'] = test_anomalies.astype(int)

# Salvarea fisierului CSV final
final_csv_path = '/kaggle/working/incercare11.csv'
test_df[['id', 'is_anomaly']].to_csv(final_csv_path, index=False)

plt.figure(figsize=(10, 6))
plt.scatter(range(len(lof_scores_test)), lof_scores_test, c=test_anomalies, cmap='coolwarm', alpha=0.5)
plt.axhline(y=anomaly_threshold, color='r', linestyle='--')
plt.title('Distribuția Scorurilor LOF în Setul de Testare')
plt.xlabel('Indexul Probei')
plt.ylabel('Scorul LOF')
plt.show()