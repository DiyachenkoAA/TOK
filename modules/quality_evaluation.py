cat > modules/quality_evaluation.py << 'EOF'
import tkinter as tk
from tkinter import Toplevel, Text, Scrollbar, messagebox
import numpy as np

def evaluate(parent, classifier, data):
    try:
        labels = data.iloc[:, 0]
        features = data.iloc[:, 1:]
        
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
        
        # Показываем результаты
        window = Toplevel(parent)
        window.title("Оценка качества")
        window.geometry("600x500")
        
        text_area = Text(window, wrap=tk.WORD, font=('Courier', 11))
        scrollbar = Scrollbar(window, command=text_area.yview)
        text_area.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        report = "=" * 60 + "\n"
        report += "ОЦЕНКА КАЧЕСТВА РАСПОЗНАВАНИЯ\n"
        report += f"Общая точность: {accuracy:.2f}%\n"
        report += "=" * 60 + "\n\n"
        report += "МАТРИЦА ОШИБОК:\n"
        
        for i, name in enumerate(class_names):
            report += f"\n{name}: "
            for j in range(len(class_names)):
                report += f"{int(confusion_matrix[i, j])} "
        
        text_area.insert(tk.END, report)
        text_area.configure(state='disabled')
        
    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка оценки качества:\n{str(e)}")
EOF