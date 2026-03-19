import tkinter as tk
from tkinter import Toplevel, Label, Entry, Button, messagebox
import numpy as np

def recognize_object(parent, classifier):
    window = Toplevel(parent)
    window.title("Распознавание нового объекта")
    window.geometry("500x400")
    
    Label(window, text="Введите признаки объекта", 
          font=('Arial', 14, 'bold')).pack(pady=10)
    
    input_frame = tk.Frame(window)
    input_frame.pack(pady=20)
    
    entries = []
    
    for i in range(classifier.feature_count):
        Label(input_frame, text=f"Признак {i+1}:", 
              font=('Arial', 12)).grid(row=i, column=0, padx=5, pady=5)
        entry = Entry(input_frame, font=('Arial', 12), width=15)
        entry.grid(row=i, column=1, padx=5, pady=5)
        entries.append(entry)
    
    result_frame = tk.Frame(window, relief=tk.SUNKEN, bd=2)
    result_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
    
    result_text = tk.Text(result_frame, height=10, font=('Arial', 11))
    result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def recognize():
        try:
            features = []
            for entry in entries:
                value = entry.get().strip()
                if not value:
                    messagebox.showwarning("Предупреждение", "Заполните все поля!")
                    return
                features.append(float(value))
            
            features_array = np.array(features)
            predicted_class, distances = classifier.predict(features_array)
            
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, "РЕЗУЛЬТАТ РАСПОЗНАВАНИЯ\n")
            result_text.insert(tk.END, "=" * 40 + "\n\n")
            result_text.insert(tk.END, f"Распознанный класс: {predicted_class}\n\n")
            result_text.insert(tk.END, "Расстояния до классов:\n")
            
            for class_name, dist in distances.items():
                result_text.insert(tk.END, f"  {class_name}: {dist:.4f}\n")
            
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные числовые значения!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка распознавания:\n{str(e)}")
    
    Button(window, text="Распознать", command=recognize,
           font=('Arial', 12, 'bold'), bg='lightgreen',
           padx=30, pady=10).pack(pady=10)
