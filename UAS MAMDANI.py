import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd

# Membaca dataset diabetes
dataset = pd.read_csv('m/diabetes_dataset.csv')

# Membuat variabel masukan
glucose = ctrl.Antecedent(np.arange(0, 201, 1), 'glucose')
insulin = ctrl.Antecedent(np.arange(0, 101, 1), 'insulin')

# Membuat variabel keluaran
output = ctrl.Consequent(np.arange(0, 11, 1), 'output')

# Mendefinisikan fungsi keanggotaan untuk setiap variabel
glucose['low'] = fuzz.trimf(glucose.universe, [0, 0, 100])
glucose['medium'] = fuzz.trimf(glucose.universe, [25, 100, 175])
glucose['high'] = fuzz.trimf(glucose.universe, [100, 200, 200])

insulin['low'] = fuzz.trimf(insulin.universe, [0, 0, 50])
insulin['medium'] = fuzz.trimf(insulin.universe, [20, 50, 80])
insulin['high'] = fuzz.trimf(insulin.universe, [50, 100, 100])

output['low'] = fuzz.trimf(output.universe, [0, 0, 5])
output['medium'] = fuzz.trimf(output.universe, [2, 5, 8])
output['high'] = fuzz.trimf(output.universe, [5, 10, 10])

# Membuat aturan fuzzy
rule1 = ctrl.Rule(glucose['low'] & insulin['low'], output['low'])
rule2 = ctrl.Rule(glucose['low'] & insulin['medium'], output['low'])
rule3 = ctrl.Rule(glucose['low'] & insulin['high'], output['medium'])

rule4 = ctrl.Rule(glucose['medium'] & insulin['low'], output['low'])
rule5 = ctrl.Rule(glucose['medium'] & insulin['medium'], output['medium'])
rule6 = ctrl.Rule(glucose['medium'] & insulin['high'], output['high'])

rule7 = ctrl.Rule(glucose['high'] & insulin['low'], output['medium'])
rule8 = ctrl.Rule(glucose['high'] & insulin['medium'], output['high'])
rule9 = ctrl.Rule(glucose['high'] & insulin['high'], output['high'])

# Membuat sistem inferensi fuzzy
system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
simulator = ctrl.ControlSystemSimulation(system)

# Memberikan nilai masukan dari dataset
glucose_values = dataset['Glucose'].values
insulin_values = dataset['Insulin'].values

results = []

for i in range(len(glucose_values)):
    # Mengatur nilai masukan
    simulator.input['glucose'] = glucose_values[i]
    simulator.input['insulin'] = insulin_values[i]

    # Melakukan perhitungan fuzzy
    simulator.compute()

    # Mendapatkan hasil keluaran
    result = simulator.output['output']
    result/10
    results.append(result)

# Membaca kolom 'Outcome' dari dataset sebagai label yang diharapkan
labels = dataset['Outcome'].values

# Menampilkan hasil keluaran dengan label
classifications = []
correct_count = 0

for i, result in enumerate(results):
    if result <4 and labels[i] == 0:
        classifications.append("Tidak Kena Diabetes (Benar)")
        correct_count += 1
    elif result <8 and labels[i] == 1:
        classifications.append("Kena Diabetes (Benar)")
        correct_count += 1
    elif result <=10 and labels[i] == 1:
        classifications.append("Kena Diabetes (Benar)")
        correct_count += 1
    else:
        classifications.append("Perhitungan Data Salah")

print("Klasifikasi:")
print("\n".join(classifications))
print("Jumlah data benar:", correct_count)
percent = correct_count / len(dataset) * 100
print("Akurasi data benar : {}%".format(int(percent)))