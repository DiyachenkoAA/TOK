"""
Модуль для обработки выхода из программы
"""

import tkinter as tk
from tkinter import messagebox

def exit_app(parent):
    """
    Функция выхода из программы
    
    Аргументы:
        parent: родительское окно (обычно root)
    
    Возвращает:
        None
    """
    if messagebox.askyesno(
        "Подтверждение", 
        "Вы действительно хотите выйти из программы?",
        icon='question',
        parent=parent
    ):
        parent.quit()
        parent.destroy()