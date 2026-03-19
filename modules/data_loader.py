import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import os

def load_data(parent):
    file_path = filedialog.askopenfilename(
        title="Выберите файл с данными",
        filetypes=[
            ("Excel files", "*.xlsx *.xls"),
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
    )
    
    if not file_path:
        return None
    
    try:
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        else:
            data = pd.read_excel(file_path)
        
        if data.shape[1] < 2:
            messagebox.showerror("Ошибка", "Файл должен содержать минимум 2 столбца")
            return None
        
        messagebox.showinfo("Успех", f"Данные загружены:\nФайл: {os.path.basename(file_path)}\nОбъектов: {len(data)}")
        return data
       
    except Exception as e:
        messagebox.showerror("Ошибка загрузки", f"Не удалось загрузить файл:\n{str(e)}")
        return None
