cat > modules/training.py << 'EOF'
import tkinter as tk
from tkinter import messagebox, Toplevel, Text, Scrollbar
import numpy as np
import pandas as pd

class Classifier:
    """Класс классификатора на основе метода минимальных расстояний"""
    
    def __init__(self):
        self.class_centroids = {}
        self.class_names = []
        self.feature_count = 0
    
    def train(self, data):
        labels = data.iloc[:, 0]
        features = data.iloc[:, 1:]
        
        self.feature_count = features.shape[1]
        self.class_names = labels.unique()
        
        for class_name in self.class_names:
            class_data = features[labels == class_name]
            centroid = class_data.mean().values
            self.class_centroids[class_name] = centroid
        
        return self
    
    def predict(self, features):
        if len(features) != self.feature_count:
            raise ValueError(f"Ожидалось {self.feature_count} признаков")
        
        distances = {}
        for class_name, centroid in self.class_centroids.items():
            dist = np.linalg.norm(features - centroid)
            distances[class_name] = dist
        
        predicted_class = min(distances, key=distances.get)
        return predicted_class, distances

def train_classifier(parent, data):
    try:
        classifier = Classifier()
        classifier.train(data)
        
        # Показываем результаты
        window = Toplevel(parent)
        window.title("Результаты обучения")
        window.geometry("600x400")
        
        text_area = Text(window, wrap=tk.WORD, font=('Courier', 10))
        scrollbar = Scrollbar(window, command=text_area.yview)
        text_area.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        report = "=" * 50 + "\n"
        report += "РЕЗУЛЬТАТЫ ОБУЧЕНИЯ\n"
        report += f"Количество классов: {len(classifier.class_names)}\n"
        report += f"Количество признаков: {classifier.feature_count}\n"
        report += "=" * 50 + "\n\n"
        report += "ЦЕНТРОИДЫ КЛАССОВ:\n"
        
        for class_name, centroid in classifier.class_centroids.items():
            report += f"\nКласс '{class_name}':\n"
            for i, val in enumerate(centroid, 1):
                report += f"  Признак {i}: {val:.4f}\n"
        
        text_area.insert(tk.END, report)
        text_area.configure(state='disabled')
        
        messagebox.showinfo("Успех", "Модель успешно обучена!")
        return classifier
        
    except Exception as e:
        messagebox.showerror("Ошибка обучения", f"Не удалось обучить модель:\n{str(e)}")
        return None
EOF