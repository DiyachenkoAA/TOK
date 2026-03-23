# data_loader.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import os

def load_data(parent):
    """Загрузка данных из файла"""
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

def display_data_table(parent, data, table_frame, on_data_selected=None):
    """
    Отображает ВСЕ данные в табличном виде
    
    Args:
        parent: родительское окно
        data: pandas DataFrame с данными
        table_frame: фрейм для размещения таблицы
        on_data_selected: функция обратного вызова при выборе данных
    
    Returns:
        tree: объект Treeview с таблицей
    """
    if data is None or data.empty:
        messagebox.showwarning("Нет данных", "Нет данных для отображения")
        return None
    
    # Очищаем фрейм таблицы
    for widget in table_frame.winfo_children():
        widget.destroy()
    
    # Создаем контейнер для таблицы с прокруткой
    table_container = ttk.Frame(table_frame)
    table_container.pack(fill=tk.BOTH, expand=True)
    
    # Создаем скроллбары
    scrollbar_y = ttk.Scrollbar(table_container, orient=tk.VERTICAL)
    scrollbar_x = ttk.Scrollbar(table_container, orient=tk.HORIZONTAL)
    
    # Создаем таблицу - показываем ВСЕ строки
    columns = list(data.columns)
    
    # Вычисляем высоту таблицы
    table_height = min(len(data), 30)
    
    tree = ttk.Treeview(
        table_container,
        columns=columns,
        show='headings',
        yscrollcommand=scrollbar_y.set,
        xscrollcommand=scrollbar_x.set,
        height=table_height
    )
    
    # Настраиваем скроллбары
    scrollbar_y.config(command=tree.yview)
    scrollbar_x.config(command=tree.xview)
    
    # Размещаем элементы
    tree.grid(row=0, column=0, sticky='nsew')
    scrollbar_y.grid(row=0, column=1, sticky='ns')
    scrollbar_x.grid(row=1, column=0, sticky='ew')
    
    # Настраиваем веса
    table_container.grid_rowconfigure(0, weight=1)
    table_container.grid_columnconfigure(0, weight=1)
    
    # Настраиваем заголовки и ширину столбцов
    for col in columns:
        tree.heading(col, text=col, command=lambda c=col, t=tree: sort_column(t, c))
        # Автоматическая ширина
        max_width = max(len(str(col)) * 10, 100)
        for val in data[col].astype(str).head(500):
            width = len(str(val)) * 8
            if width > max_width:
                max_width = min(width, 300)
        tree.column(col, width=max_width, minwidth=50, anchor='w')
    
    # Заполняем данными - ВСЕ строки
    for idx, row in data.iterrows():
        values = []
        for col in columns:
            val = row[col]
            if isinstance(val, (int, float)):
                if isinstance(val, float):
                    values.append(f"{val:.4f}" if val != int(val) else f"{int(val)}")
                else:
                    values.append(str(val))
            else:
                values.append(str(val))
        
        if idx % 2 == 0:
            tree.insert('', 'end', values=values, tags=('evenrow',))
        else:
            tree.insert('', 'end', values=values)
    
    # Настройка цветов
    tree.tag_configure('evenrow', background='#f5f5f5')
    
    # Добавляем обработчик выделения
    if on_data_selected:
        tree.bind('<<TreeviewSelect>>', lambda e: on_data_selected(tree))
    
    # Добавляем информационную панель
    info_bar = ttk.Frame(table_frame)
    info_bar.pack(fill=tk.X, pady=(5, 0))
    
    ttk.Button(info_bar, text="📊 Общая статистика", 
              command=lambda: show_general_statistics(parent, data)).pack(side=tk.LEFT, padx=2)
    ttk.Button(info_bar, text="📋 Копировать выделенное", 
              command=lambda: copy_selected(parent, tree)).pack(side=tk.LEFT, padx=2)
    ttk.Button(info_bar, text="🔄 Обновить", 
              command=lambda: display_data_table(parent, data, table_frame, on_data_selected)).pack(side=tk.LEFT, padx=2)
    ttk.Button(info_bar, text="📤 Экспорт", 
              command=lambda: export_data(parent, data)).pack(side=tk.LEFT, padx=2)
    
    ttk.Label(info_bar, 
             text=f"📊 Всего: {len(data)} строк | {len(columns)} столбцов | Признаков: {len(columns)-1}",
             font=('Arial', 9),
             foreground='blue').pack(side=tk.RIGHT, padx=5)
    
    return tree

def sort_column(tree, col):
    """Сортировка таблицы по столбцу"""
    if tree is None:
        return
    
    items = [(tree.set(item, col), item) for item in tree.get_children('')]
    
    try:
        items.sort(key=lambda x: float(x[0]))
    except (ValueError, TypeError):
        items.sort(key=lambda x: x[0].lower())
    
    for index, (val, item) in enumerate(items):
        tree.move(item, '', index)

def show_general_statistics(parent, data):
    """Показывает общую статистику по ВСЕМ данным"""
    if data is None or data.empty:
        messagebox.showwarning("Нет данных", "Нет данных для статистики")
        return
    
    window = tk.Toplevel(parent)
    window.title("Общая статистика данных")
    window.geometry("800x700")
    
    text_frame = tk.Frame(window)
    text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    text_area = tk.Text(text_frame, wrap=tk.WORD, font=('Courier', 10))
    scrollbar = tk.Scrollbar(text_frame, command=text_area.yview)
    text_area.configure(yscrollcommand=scrollbar.set)
    
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    stats_text = "=" * 80 + "\n"
    stats_text += "ОБЩАЯ СТАТИСТИКА ДАННЫХ\n"
    stats_text += "=" * 80 + "\n\n"
    
    stats_text += "📊 ОБЩАЯ ИНФОРМАЦИЯ:\n"
    stats_text += f"  • Количество объектов: {len(data)}\n"
    stats_text += f"  • Количество признаков: {data.shape[1]-1}\n"
    stats_text += f"  • Количество столбцов: {data.shape[1]}\n"
    stats_text += f"  • Количество пропусков: {data.isnull().sum().sum()}\n\n"
    
    stats_text += "📋 ИНФОРМАЦИЯ О СТОЛБЦАХ:\n"
    for i, col in enumerate(data.columns):
        stats_text += f"  {i+1}. {col}\n"
        stats_text += f"     Тип: {data[col].dtype}\n"
        stats_text += f"     Уникальных: {data[col].nunique()}\n"
        stats_text += f"     Пропусков: {data[col].isnull().sum()}\n"
    
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        stats_text += "\n📈 СТАТИСТИКА ПО ЧИСЛОВЫМ ПРИЗНАКАМ:\n"
        for col in numeric_cols:
            stats_text += f"\n  {col}:\n"
            stats_text += f"    Мин: {data[col].min():.4f}\n"
            stats_text += f"    Макс: {data[col].max():.4f}\n"
            stats_text += f"    Среднее: {data[col].mean():.4f}\n"
            stats_text += f"    Медиана: {data[col].median():.4f}\n"
            stats_text += f"    Стандартное отклонение: {data[col].std():.4f}\n"
    
    text_area.insert(tk.END, stats_text)
    text_area.configure(state='disabled')
    
    button_frame = tk.Frame(window)
    button_frame.pack(pady=5)
    
    def save_statistics():
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(stats_text)
            messagebox.showinfo("Успех", f"Статистика сохранена в:\n{file_path}")
    
    ttk.Button(button_frame, text="💾 Сохранить статистику", 
              command=save_statistics).pack(side=tk.LEFT, padx=5)

def copy_selected(parent, tree):
    """Копирует выделенные строки в буфер обмена"""
    if tree is None:
        return
    
    selected = tree.selection()
    if not selected:
        messagebox.showinfo("Информация", "Выделите строки для копирования")
        return
    
    columns = tree['columns']
    copy_text = '\t'.join(columns) + '\n'
    
    for item in selected:
        values = tree.item(item)['values']
        copy_text += '\t'.join(str(v) for v in values) + '\n'
    
    parent.clipboard_clear()
    parent.clipboard_append(copy_text)
    messagebox.showinfo("Успех", f"Скопировано {len(selected)} строк(и)")

def export_data(parent, data):
    """Экспорт ВСЕХ данных в файл"""
    if data is None:
        messagebox.showwarning("Нет данных", "Нет данных для экспорта")
        return
    
    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx"),
            ("All files", "*.*")
        ]
    )
    
    if file_path:
        try:
            if file_path.endswith('.xlsx'):
                data.to_excel(file_path, index=False)
            else:
                data.to_csv(file_path, index=False, encoding='utf-8-sig')
            messagebox.showinfo("Успех", f"Все данные экспортированы в:\n{file_path}\n\nКоличество строк: {len(data)}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось экспортировать данные:\n{str(e)}")