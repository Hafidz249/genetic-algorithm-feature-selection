import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# --- KONFIGURASI ALGORITMA GENETIKA ---
POPULATION_SIZE = 20    # Jumlah individu dalam populasi
GENERATIONS = 15        # Jumlah generasi (bisa ditambah agar hasil lebih maksimal)
CROSSOVER_RATE = 0.8    # Peluang crossover
MUTATION_RATE = 0.1     # Peluang mutasi

# --- 1. LOAD & PREPROCESSING DATASET (PERBAIKAN UTAMA DISINI) ---
try:
    print("Membaca dataset...")
    # Ganti 'data.csv' sesuai nama file kamu
    data = pd.read_csv('data.csv') 
    
    # Bersihkan kolom yang tidak berguna (ID dan kolom kosong di akhir)
    if 'id' in data.columns:
        data = data.drop(columns=['id'])
    if 'Unnamed: 32' in data.columns:
        data = data.drop(columns=['Unnamed: 32'])

    # Deteksi dan Encode Target (Diagnosis M/B -> 1/0)
    # Cek apakah ada kolom 'diagnosis' (standar dataset breast cancer)
    if 'diagnosis' in data.columns:
        le = LabelEncoder()
        y = le.fit_transform(data['diagnosis']) # Ubah M/B jadi 1/0
        X = data.drop(columns=['diagnosis']).values # Sisa kolom jadi fitur
    else:
        # Jika nama kolom beda, asumsikan kolom terakhir adalah target
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        # Jika targetnya berupa text, ubah jadi angka
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

    print(f"Dataset dimuat: {X.shape[0]} baris, {X.shape[1]} fitur.")

except Exception as e:
    print(f"Gagal membaca file CSV: {e}")
    print("Menggunakan dataset dummy (Breast Cancer) dari Scikit-Learn...")
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    X, y = d.data, d.target

# Split data jadi Training dan Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
num_features = X.shape[1]

# --- 2. INISIALISASI POPULASI ---
def create_individual():
    # Membuat satu kromosom acak (0 atau 1)
    return [random.randint(0, 1) for _ in range(num_features)]

def create_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]

# --- 3. FUNGSI FITNESS ---
def calculate_fitness(individual):
    # Cek fitur mana yang aktif (bernilai 1)
    cols = [i for i, x in enumerate(individual) if x == 1]
    
    # Jika tidak ada fitur terpilih, beri nilai buruk (0)
    if len(cols) == 0: 
        return 0
    
    # Filter dataset hanya kolom terpilih
    X_train_sel = X_train[:, cols]
    X_test_sel = X_test[:, cols]
    
    # Hitung akurasi menggunakan Decision Tree
    clf = DecisionTreeClassifier()
    clf.fit(X_train_sel, y_train)
    predictions = clf.predict(X_test_sel)
    
    return accuracy_score(y_test, predictions)

# --- 4. SELEKSI (ROULETTE WHEEL) ---
def selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    # Hindari pembagian dengan nol
    if total_fitness == 0: 
        return random.choice(population), random.choice(population)
    
    probs = [f / total_fitness for f in fitness_scores]
    # Pilih 2 orang tua berdasarkan probabilitas fitness
    parents = random.choices(population, weights=probs, k=2)
    return parents[0], parents[1]

# --- 5. CROSSOVER (ONE POINT) ---
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        # Tentukan titik potong acak
        point = random.randint(1, num_features - 1)
        # Tukar gen
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1, parent2

# --- 6. MUTASI (BIT FLIP) ---
def mutation(individual):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            # Ubah 0 jadi 1, atau 1 jadi 0
            individual[i] = 1 - individual[i]
    return individual

# --- MAIN PROGRAM ---
def run_genetic_algorithm():
    population = create_population()
    accuracy_history = [] # Untuk menyimpan data grafik
    
    print(f"\n--- Memulai Evolusi GA selama {GENERATIONS} Generasi ---\n")
    
    for gen in range(GENERATIONS):
        # Hitung fitness semua individu
        fitness_scores = [calculate_fitness(ind) for ind in population]
        
        # Cari yang terbaik di generasi ini
        max_fitness = max(fitness_scores)
        best_ind = population[fitness_scores.index(max_fitness)]
        
        # Simpan untuk grafik
        accuracy_history.append(max_fitness)
        
        # Tampilkan log
        fitur_terpakai = sum(best_ind)
        print(f"Generasi {gen+1} | Akurasi Terbaik: {max_fitness:.4f} | Fitur: {fitur_terpakai}/{num_features}")
        
        # Buat populasi baru
        new_population = []
        
        # Elitisme: Simpan juara bertahan tanpa diubah
        new_population.append(best_ind)
        
        # Generate sisanya
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = selection(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutation(child1)
            if len(new_population) < POPULATION_SIZE: new_population.append(child1)
            if len(new_population) < POPULATION_SIZE: 
                child2 = mutation(child2) 
                new_population.append(child2)
        
        population = new_population

    # --- HASIL AKHIR ---
    fitness_scores = [calculate_fitness(ind) for ind in population]
    best_fitness = max(fitness_scores)
    best_solution = population[fitness_scores.index(best_fitness)]
    
    print("\n" + "="*30)
    print("       HASIL AKHIR GA")
    print("="*30)
    print(f"Akurasi Tertinggi : {best_fitness:.4f}")
    print(f"Jumlah Fitur      : {sum(best_solution)} dari {num_features}")
    print(f"Kromosom Terbaik  : {best_solution}")
    
    # --- MENAMPILKAN GRAFIK ---
    print("\nMenampilkan grafik...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, GENERATIONS + 1), accuracy_history, marker='o', linestyle='-', color='b')
    plt.title('Grafik Optimasi Akurasi (Algoritma Genetika)')
    plt.xlabel('Generasi')
    plt.ylabel('Akurasi (Fitness)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_genetic_algorithm()