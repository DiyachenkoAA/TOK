cat > modules/data_loader.py << 'EOF'
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import os

def load_data(parent):
    """
    Загрузка данных из файла Excel или CSV
    """
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
        # Определяем тип файла по расширению
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        else:
            data = pd.read_excel(file_path)
        
        # Базовая валидация
        if data.shape[1] < 2:
            messagebox.showerror("Ошибка", "Файл должен содержать минимум 2 столбца (класс и признаки)")
            return None
        
        messagebox.showinfo("Успех", f"Данные загружены:\n"
                                      f"Файл: {os.path.basename(file_path)}\n"
                                      f"Объектов: {len(data)}\n"
                                      f"Признаков: {data.shape[1]-1}")
        
        return data
        
    except Exception as e:
        messagebox.showerror("Ошибка загрузки", f"Не удалось загрузить файл:\n{str(e)}")
        return None
EOF