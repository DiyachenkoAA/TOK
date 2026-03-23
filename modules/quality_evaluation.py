import tkinter as tk
from tkinter import Toplevel, Text, Scrollbar, messagebox
import numpy as np

def evaluate(parent, classifier, data):
    """Оценка качества распознавания"""
    try:
        labels = data.iloc[:, 0]
        features = data.iloc[:, 1:]  # Берем все признаки
        
        class_names = classifier.class_names
        n_classes = len(class_names)
        confusion_matrix = np.zeros((n_classes, n_classes))
        
        correct = 0
        for idx, row in features.iterrows():
            true_class = labels.iloc[idx]
            predicted, _ = classifier.predict(row.values)
            
            i = list(class_names).index(true_class)
            j = list(class_names).index(predicted)
            confusion_matrix[i, j] += 1
            
            if true_class == predicted:
                correct += 1
        
        accuracy = correct / len(data) * 100
        
        window = Toplevel(parent)
        window.title("Оценка качества")
        window.geometry("800x600")
        
        text_area = Text(window, wrap=tk.WORD, font=('Courier', 11))
        scrollbar = Scrollbar(window, command=text_area.yview)
        text_area.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        report = "=" * 70 + "\n"
        report += "ОЦЕНКА КАЧЕСТВА РАСПОЗНАВАНИЯ\n"
        report += "=" * 70 + "\n\n"
        report += f"📊 Метод обучения: {classifier.method}\n"
        report += f"📏 Метрика: {classifier.metric}\n"
        report += f"📋 Количество признаков: {features.shape[1]}\n"
        report += f"📋 Количество классов: {len(class_names)}\n"
        report += f"✅ Общая точность: {accuracy:.2f}%\n"
        report += f"❌ Количество ошибок: {len(data) - correct} из {len(data)}\n\n"
        
        report += "📋 МАТРИЦА ОШИБОК (Confusion Matrix):\n"
        report += "-" * 70 + "\n"
        
        max_name_len = max(len(str(name)) for name in class_names)
        col_width = max(10, max_name_len + 2)
        
        # Заголовки столбцов
        report += " " * (max_name_len + 2)
        for name in class_names:
            report += f"{str(name)[:10]:^{col_width}}"
        report += "\n" + "-" * 70 + "\n"
        
        for i, name in enumerate(class_names):
            report += f"{str(name):<{max_name_len+2}}"
            for j in range(n_classes):
                report += f"{int(confusion_matrix[i, j]):^{col_width}}"
            report += "\n"
        
        report += "\n📈 ДЕТАЛЬНАЯ СТАТИСТИКА ПО КЛАССАМ:\n"
        report += "-" * 70 + "\n"
        for i, name in enumerate(class_names):
            correct_count = confusion_matrix[i, i]
            total = confusion_matrix[i].sum()
            class_accuracy = (correct_count / total * 100) if total > 0 else 0
            report += f"  • {name}:\n"
            report += f"      Точность: {class_accuracy:.1f}%\n"
            report += f"      Правильно: {int(correct_count)}/{int(total)}\n"
            if total - correct_count > 0:
                report += f"      Ошибок: {int(total - correct_count)}\n"
            report += "\n"
        
        text_area.insert(tk.END, report)
        text_area.configure(state='disabled')
        
    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка оценки качества:\n{str(e)}")