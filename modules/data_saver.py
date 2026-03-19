cat > modules/data_saver.py << 'EOF'
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import os

def save_data(parent, data):
    """
    Сохранение данных в файл
    """
    if data is None:
        messagebox.showwarning("Нет данных", "Нет данных для сохранения")
        return
    
    file_path = filedialog.asksaveasfilename(
        title="Сохранить данные как",
        defaultextension=".xlsx",
        filetypes=[
            ("Excel files", "*.xlsx"),
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
    )
    
    if not file_path:
        return
    
    try:
        if file_path.endswith('.csv'):
            data.to_csv(file_path, index=False)
        else:
            data.to_excel(file_path, index=False)
        
        messagebox.showinfo("Успех", f"Данные сохранены:\n{os.path.basename(file_path)}")
        
    except Exception as e:
        messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить файл:\n{str(e)}")
EOF