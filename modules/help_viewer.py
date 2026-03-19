cat > modules/help_viewer.py << 'EOF'
import tkinter as tk
from tkinter import Toplevel, Text, Scrollbar

def show_help(parent):
    window = Toplevel(parent)
    window.title("Справка - HELP")
    window.geometry("700x600")
    
    text_area = Text(window, wrap=tk.WORD, font=('Arial', 11))
    scrollbar = Scrollbar(window, command=text_area.yview)
    text_area.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    help_text = """
    ========================================
    ПРОГРАММА РАСПОЗНАВАНИЯ ОБЪЕКТОВ
    Метод минимальных расстояний
    ========================================
    
    1. НАЗНАЧЕНИЕ ПРОГРАММЫ
    -----------------------
    Программа предназначена для классификации объектов 
    методом минимальных расстояний (ближайшего центроида).
    
    2. ФОРМАТ ВХОДНЫХ ДАННЫХ
    ------------------------
    Поддерживаются файлы Excel (.xlsx) и CSV (.csv).
    Первый столбец: метки классов (текст)
    Остальные столбцы: числовые признаки
    
    3. ПОРЯДОК РАБОТЫ
    -----------------
    1. Загрузите обучающую выборку
    2. Обучите классификатор
    3. Оцените качество
    4. Распознавайте новые объекты
    """
    
    text_area.insert(tk.END, help_text)
    text_area.configure(state='disabled')
EOF