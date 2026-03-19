
import tkinter as tk
from tkinter import messagebox

def exit_app(parent):
    if messagebox.askyesno("Подтверждение", "Вы действительно хотите выйти?"):
        parent.quit()
        parent.destroy()
