import numpy as np
import pandas as pd

def tsukamoto_diabetes_detection(glucose_level, insulin_level):
    glucose_low = np.array([1, 1, 0])
    glucose_normal = np.array([0, 1, 0])
    glucose_high = np.array([0, 1, 1])
    
    insulin_low = np.array([1, 1, 0])
    insulin_normal = np.array([0, 1, 0])
    insulin_high = np.array([0, 1, 1])
    
    # rule
    rule1 = np.fmin(glucose_low, insulin_low)
    rule2 = np.fmin(glucose_normal, insulin_normal)
    rule3 = np.fmin(glucose_high, insulin_high)
    
    # fungsi buat gabungin rule-rule
    aggregated = np.fmax(rule1, np.fmax(rule2, rule3))
    
    # Menghitung nilai crisp dengan metode centroid
    output = np.sum(aggregated * np.array([0, 1, 2])) / np.sum(aggregated)
    
    # hitung kena-tidak
    if output < 1.5:
        return "Tidak Kena Diabetes"
    else:
        return "Kena Diabetes"


def main():
    dataset = pd.read_csv('m/diabetes_dataset.csv')
    
    glucose_levels = dataset['Glucose'].values
    insulin_levels = dataset['Insulin'].values
    
    classifications = []
    correct_count = 0
    
    for glucose, insulin, label in zip(glucose_levels, insulin_levels, dataset['Outcome']):
        result = tsukamoto_diabetes_detection(glucose, insulin)
        
        if result == "Tidak Kena Diabetes" and label == 0:
            classifications.append("Tidak Kena Diabetes (Benar)")
            correct_count += 1
        elif result == "Kena Diabetes" and label == 1:
            classifications.append("Kena Diabetes (Benar)")
            correct_count += 1
        else:
            classifications.append("Perhitungan Data salah")

    print("Hasil Klasifikasi :")
    for classification in classifications:
        print(classification)
    
    print("Jumlah data benar:", correct_count)
    percent = correct_count / len(dataset) * 100
    print("Akurasi data benar : {}%".format(int(percent)))
    

if __name__ == "__main__":
    main()
