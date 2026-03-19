import tkinter as tk
from tkinter import messagebox, Menu
import os

# Импортируем модули с функциями
from modules import data_loader, data_saver, training, recognition, quality_evaluation, help_viewer

class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title('Распознавание объектов методом минимальных расстояний')
        self.root.geometry("800x600")
        
        # Данные программы (будут заполняться при работе)
        self.data = None
        self.classifier = None
        
        self.setup_menu()  # Создаём меню
        self.setup_ui()    # Создаём интерфейс
    
    def setup_menu(self):
        """Создание главного меню программы"""
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        
        # ============================================================
        # 1. МЕНЮ "ФАЙЛ"
        # ============================================================
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="📁 Файл", menu=file_menu)
        file_menu.add_command(label="📂 Считать данные", command=self.load_data, accelerator="Ctrl+O")
        file_menu.add_command(label="💾 Сохранить данные", command=self.save_data, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="🚪 Выход", command=self.exit_app, accelerator="Ctrl+Q")
        
        # ============================================================
        # 2. МЕНЮ "ОБУЧЕНИЕ"
        # ============================================================
        train_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="🎓 Обучение", menu=train_menu)
        train_menu.add_command(label="🎓 Обучение модели", command=self.train_model, accelerator="Ctrl+T")
        train_menu.add_command(label="📊 Оценка качества", command=self.evaluate_quality, accelerator="Ctrl+E")
        
        # ============================================================
        # 3. МЕНЮ "РАСПОЗНАВАНИЕ"
        # ============================================================
        recognize_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="🔍 Распознавание", menu=recognize_menu)
        recognize_menu.add_command(label="🔍 Распознать объект", command=self.recognize_object, accelerator="Ctrl+R")
        
        # ============================================================
        # 4. МЕНЮ "СПРАВКА"
        # ============================================================
        help_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="❓ Справка", menu=help_menu)
        help_menu.add_command(label="📖 Справка HELP", command=self.show_help, accelerator="F1")
        help_menu.add_separator()
        help_menu.add_command(label="ℹ️ О программе", command=self.show_about)
        
        # ============================================================
        # Горячие клавиши
        # ============================================================
        self.root.bind('<Control-o>', lambda e: self.load_data())
        self.root.bind('<Control-s>', lambda e: self.save_data())
        self.root.bind('<Control-q>', lambda e: self.exit_app())
        self.root.bind('<Control-t>', lambda e: self.train_model())
        self.root.bind('<Control-e>', lambda e: self.evaluate_quality())
        self.root.bind('<Control-r>', lambda e: self.recognize_object())
        self.root.bind('<F1>', lambda e: self.show_help())
    
    def setup_ui(self):
        """Интерфейс главного окна"""
        # ============================================================
        # 1. НАЗВАНИЕ ПРОГРАММЫ
        # ============================================================
        title_label = tk.Label(
            self.root, 
            text='РАСПОЗНАВАНИЕ ОБЪЕКТОВ\nМетод минимальных расстояний',
            font=('Arial', 18, 'bold'),
            justify=tk.CENTER
        )
        title_label.pack(pady=(30, 10))
        
        # ============================================================
        # 2. ФАМИЛИЯ РАЗРАБОТЧИКА
        # ============================================================
        developer_label = tk.Label(
            self.root,
            text='Разработчик: Дьяченко А.А.',
            font=('Arial', 12)
        )
        developer_label.pack(pady=5)
        
        # ============================================================
        # 3. НАЗВАНИЕ ВУЗА
        # ============================================================
        university_label = tk.Label(
            self.root,
            text='СибГМУ\nКафедра медицинской кибернетики',
            font=('Arial', 12),
            justify=tk.CENTER
        )
        university_label.pack(pady=10)
        
        # ============================================================
        # 4. ПАНЕЛЬ БЫСТРОГО ДОСТУПА (КНОПКИ)
        # ============================================================
        button_frame = tk.Frame(self.root, bg='lightgray', relief=tk.RAISED, bd=2)
        button_frame.pack(pady=20, padx=20, fill=tk.X)
        
        buttons = [
            ("📂 Загрузить", self.load_data),
            ("💾 Сохранить", self.save_data),
            ("🎓 Обучение", self.train_model),
            ("🔍 Распознать", self.recognize_object),
            ("📊 Оценка", self.evaluate_quality),
            ("❓ Справка", self.show_help),
        ]
        
        for btn_text, btn_command in buttons:
            btn = tk.Button(
                button_frame,
                text=btn_text,
                command=btn_command,
                font=('Arial', 10, 'bold'),
                bg='white',
                relief=tk.RAISED,
                padx=15,
                pady=8,
                width=10
            )
            btn.pack(side=tk.LEFT, padx=5, pady=5, expand=True)
        
        # ============================================================
        # 5. РАБОЧАЯ ОБЛАСТЬ (ИНФОРМАЦИЯ)
        # ============================================================
        info_frame = tk.Frame(self.root, relief=tk.SUNKEN, bd=2)
        info_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
        
        self.info_text = tk.Text(info_frame, font=('Arial', 11), wrap=tk.WORD)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = tk.Scrollbar(info_frame, command=self.info_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        # Начальное сообщение
        self.info_text.insert(tk.END, "Добро пожаловать в программу распознавания объектов!\n\n")
        self.info_text.insert(tk.END, "Используйте меню 'Файл' для загрузки данных.\n")
        self.info_text.insert(tk.END, "После загрузки данных можно обучить модель и распознавать объекты.\n")
        self.info_text.configure(state='disabled')
        
        # ============================================================
        # 6. СТАТУСНАЯ СТРОКА
        # ============================================================
        self.status_label = tk.Label(
            self.root,
            text=" Готов к работе",
            font=('Arial', 10),
            bg='lightgray',
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    # ============================================================
    # МЕТОДЫ ДЛЯ ПУНКТОВ МЕНЮ
    # ============================================================
    
    def load_data(self):
        """Загрузка данных из файла"""
        result = data_loader.load_data(self.root)
        if result is not None:
            self.data = result
            self.update_status(f"Данные загружены: {len(self.data)} объектов, {self.data.shape[1]-1} признаков")
            self.show_info("✅ Данные успешно загружены!\n\n" + self.data.head().to_string())
    
    def save_data(self):
        """Сохранение данных в файл"""
        if self.data is not None:
            data_saver.save_data(self.root, self.data)
        else:
            messagebox.showwarning("Нет данных", "Сначала загрузите данные!")
    
    def train_model(self):
        """Обучение классификатора"""
        if self.data is not None:
            self.classifier = training.train_classifier(self.root, self.data)
            if self.classifier:
                self.update_status("Модель обучена")
        else:
            messagebox.showwarning("Нет данных", "Сначала загрузите данные!")
    
    def recognize_object(self):
        """Распознавание нового объекта"""
        if self.classifier is not None:
            recognition.recognize_object(self.root, self.classifier)
        else:
            messagebox.showwarning("Нет модели", "Сначала обучите модель!")
    
    def evaluate_quality(self):
        """Оценка качества распознавания"""
        if self.classifier is not None and self.data is not None:
            quality_evaluation.evaluate(self.root, self.classifier, self.data)
        else:
            messagebox.showwarning("Нет данных", "Загрузите данные и обучите модель!")
    
    def show_help(self):
        """Показать справку"""
        help_viewer.show_help(self.root)
    
    def show_about(self):
        """О программе"""
        about_text = """
        Программа распознавания объектов
        Метод минимальных расстояний
        
        Разработчик: Дьяченко А.А.
        СибГМУ, Кафедра медицинской кибернетики
        Версия: 1.0
        
        © 2026
        """
        messagebox.showinfo("О программе", about_text)
    
    def exit_app(self):
        """Выход из программы"""
        if messagebox.askyesno("Подтверждение", "Вы действительно хотите выйти?"):
            self.root.quit()
            self.root.destroy()
    
    def update_status(self, message):
        """Обновление статусной строки"""
        self.status_label.config(text=f" Статус: {message}")
    
    def show_info(self, message):
        """Показать информацию в рабочей области"""
        self.info_text.configure(state='normal')
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, message)
        self.info_text.configure(state='disabled')

# ============================================================
# ЗАПУСК ПРОГРАММЫ
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()