# main.py
import tkinter as tk
from tkinter import messagebox, Menu, ttk, filedialog
import os

# Импортируем модули с функциями
from modules import data_loader, data_saver, training, recognition, quality_evaluation, help_viewer, exit_handler

class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title('Распознавание объектов методом минимальных расстояний')
        self.root.geometry("1024x768")
        self.root.configure(bg='#f0f0f0')
        
        # Данные программы
        self.data = None
        self.classifier = None
        self.selected_features = None
        self.feature_names = []
        
        self.setup_menu()
        self.setup_ui()
    
    def setup_menu(self):
        """Создание главного меню программы"""
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        
        # МЕНЮ "ФАЙЛ"
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="📁 Файл", menu=file_menu)
        file_menu.add_command(label="📂 Считать данные", command=self.load_data, accelerator="Ctrl+O")
        file_menu.add_command(label="💾 Сохранить данные", command=self.save_data, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="🚪 Выход", command=self.exit_app, accelerator="Ctrl+Q")
        
        # МЕНЮ "ОБУЧЕНИЕ"
        train_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="🎓 Обучение", menu=train_menu)
        train_menu.add_command(label="🎓 Обучить модель", command=self.train_model, accelerator="Ctrl+T")
        train_menu.add_command(label="📊 Оценка качества", command=self.evaluate_quality, accelerator="Ctrl+E")
        
        # МЕНЮ "РАСПОЗНАВАНИЕ"
        recognize_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="🔍 Распознавание", menu=recognize_menu)
        recognize_menu.add_command(label="🔍 Распознать объект", command=self.recognize_object, accelerator="Ctrl+R")
        
        # МЕНЮ "СПРАВКА"
        help_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="❓ Справка", menu=help_menu)
        help_menu.add_command(label="📖 Справка HELP", command=self.show_help, accelerator="F1")
        help_menu.add_separator()
        help_menu.add_command(label="ℹ️ О программе", command=self.show_about)
        
        # Горячие клавиши
        self.root.bind('<Control-o>', lambda e: self.load_data())
        self.root.bind('<Control-s>', lambda e: self.save_data())
        self.root.bind('<Control-q>', lambda e: self.exit_app())
        self.root.bind('<Control-t>', lambda e: self.train_model())
        self.root.bind('<Control-e>', lambda e: self.evaluate_quality())
        self.root.bind('<Control-r>', lambda e: self.recognize_object())
        self.root.bind('<F1>', lambda e: self.show_help())
    
    def setup_ui(self):
        """Интерфейс главного окна"""
        
        # ВЕРХНЯЯ ПАНЕЛЬ
        header_frame = tk.Frame(self.root, bg='#f0f0f0')
        header_frame.pack(pady=(20, 10), fill=tk.X)
        
        title_label = tk.Label(
            header_frame, 
            text='РАСПОЗНАВАНИЕ ОБЪЕКТОВ',
            font=('Arial', 20, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=5)
        
        method_label = tk.Label(
            header_frame,
            text='Метод минимальных расстояний',
            font=('Arial', 12),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        method_label.pack()
        
        developer_label = tk.Label(
            header_frame,
            text='Разработчик: Дьяченко А.А.',
            font=('Arial', 11),
            bg='#f0f0f0',
            fg='#34495e'
        )
        developer_label.pack(pady=5)
        
        university_label = tk.Label(
            header_frame,
            text='СибГМУ | Кафедра медицинской кибернетики',
            font=('Arial', 11),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        university_label.pack(pady=5)
        
        # ПАНЕЛЬ ИНСТРУМЕНТОВ
        toolbar_frame = tk.Frame(self.root, bg='#e0e0e0', relief=tk.RAISED, bd=1)
        toolbar_frame.pack(pady=10, padx=20, fill=tk.X)
        
        buttons = [
            ("📂 Загрузить данные", self.load_data),
            ("💾 Сохранить данные", self.save_data),
            ("🎓 Обучить модель", self.train_model),
            ("🔍 Распознать", self.recognize_object),
            ("📊 Оценка качества", self.evaluate_quality),
            ("❓ Справка", self.show_help),
            ("🚪 Выход", self.exit_app),
        ]
        
        for btn_text, btn_command in buttons:
            btn = tk.Button(
                toolbar_frame,
                text=btn_text,
                command=btn_command,
                font=('Arial', 10),
                bg='white',
                fg='#2c3e50',
                relief=tk.RAISED,
                padx=12,
                pady=6,
                cursor='hand2'
            )
            btn.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
        
        # РАБОЧАЯ ОБЛАСТЬ (ВКЛАДКИ)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        # Вкладка 1: Данные
        self.data_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.data_frame, text="📋 Данные")
        self.setup_data_tab()
        
        # Вкладка 2: Характеристики классов
        self.stats_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.stats_frame, text="📊 Характеристики классов")
        self.setup_stats_tab()
        
        # Вкладка 3: Результаты распознавания
        self.results_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.results_frame, text="🎯 Распознавание")
        self.setup_results_tab()
        
        # Вкладка 4: Справка
        self.help_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.help_frame, text="📖 Справка")
        self.setup_help_tab()
        
        # СТАТУСНАЯ СТРОКА
        self.status_label = tk.Label(
            self.root,
            text="✅ Готов к работе",
            font=('Arial', 9),
            bg='#e0e0e0',
            relief=tk.SUNKEN,
            anchor=tk.W,
            padx=5
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_data_tab(self):
        """Настройка вкладки с данными"""
        # Создаем фрейм для таблицы с прокруткой
        table_container = tk.Frame(self.data_frame, bg='white')
        table_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Создаем фрейм для таблицы и скроллбаров
        table_frame = ttk.Frame(table_container)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Создаем скроллбары
        scrollbar_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL)
        scrollbar_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
        
        # Создаем таблицу (Treeview)
        self.data_tree = ttk.Treeview(
            table_frame,
            show='headings',
            yscrollcommand=scrollbar_y.set,
            xscrollcommand=scrollbar_x.set,
            height=20
        )
        
        scrollbar_y.config(command=self.data_tree.yview)
        scrollbar_x.config(command=self.data_tree.xview)
        
        self.data_tree.grid(row=0, column=0, sticky='nsew')
        scrollbar_y.grid(row=0, column=1, sticky='ns')
        scrollbar_x.grid(row=1, column=0, sticky='ew')
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Информационная панель
        info_frame = ttk.Frame(table_container)
        info_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.data_info_label = tk.Label(
            info_frame,
            text="📊 Данные не загружены",
            font=('Arial', 10),
            bg='white',
            fg='gray',
            anchor=tk.W
        )
        self.data_info_label.pack(side=tk.LEFT, padx=5)
    
    def setup_stats_tab(self):
        """Настройка вкладки с характеристиками классов"""
        # Создаем текстовое поле с прокруткой
        stats_text_frame = tk.Frame(self.stats_frame)
        stats_text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.stats_text = tk.Text(stats_text_frame, wrap=tk.WORD, font=('Courier', 10))
        scrollbar = tk.Scrollbar(stats_text_frame, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.stats_text.insert(tk.END, "Характеристики классов появятся здесь после обучения модели.")
        self.stats_text.configure(state='disabled')
    
    def setup_results_tab(self):
        """Настройка вкладки с результатами распознавания"""
        # Создаем фрейм для ввода
        input_frame = tk.Frame(self.results_frame, bg='white', relief=tk.SUNKEN, bd=1)
        input_frame.pack(pady=10, padx=10, fill=tk.X)
        
        tk.Label(input_frame, text="Ввод признаков объекта:", 
                font=('Arial', 12, 'bold'), bg='white').pack(pady=5)
        
        # Поля для ввода признаков
        self.entries = []
        self.entry_frame = tk.Frame(input_frame, bg='white')
        self.entry_frame.pack(pady=10)
        
        for i in range(4):
            tk.Label(self.entry_frame, text=f"Признак {i+1}:", bg='white').grid(row=i, column=0, padx=5, pady=3)
            entry = tk.Entry(self.entry_frame, width=20, font=('Arial', 10))
            entry.grid(row=i, column=1, padx=5, pady=3)
            self.entries.append(entry)
        
        # Кнопки
        button_frame = tk.Frame(input_frame, bg='white')
        button_frame.pack(pady=10)
        
        self.recognize_btn = tk.Button(
            button_frame,
            text="🔍 Распознать объект",
            command=self.recognize_current_object,
            font=('Arial', 10, 'bold'),
            bg='#3498db',
            fg='white',
            padx=20,
            pady=5,
            state='disabled'
        )
        self.recognize_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(
            button_frame,
            text="🗑 Очистить",
            command=self.clear_results,
            font=('Arial', 10),
            bg='#95a5a6',
            fg='white',
            padx=20,
            pady=5
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Область для вывода результатов
        results_frame = tk.Frame(self.results_frame)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, font=('Arial', 11))
        scrollbar = tk.Scrollbar(results_frame, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.results_text.insert(tk.END, "Результаты распознавания появятся здесь.\n\nСначала обучите модель.")
        self.results_text.configure(state='disabled')
    
    def setup_help_tab(self):
        """Настройка вкладки со справочной информацией"""
        help_text = tk.Text(self.help_frame, wrap=tk.WORD, font=('Arial', 11))
        help_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        help_content = """
        ╔══════════════════════════════════════════════════════════════════════════════╗
        ║                    МЕТОД МИНИМАЛЬНЫХ РАССТОЯНИЙ                             ║
        ╚══════════════════════════════════════════════════════════════════════════════╝
        
        1. ОСНОВНОЙ ПРИНЦИП
        -------------------
        Метод минимальных расстояний (Nearest Centroid Classifier) — это простой и эффективный 
        метод классификации, основанный на вычислении расстояния от нового объекта до центроидов 
        (средних точек) каждого класса.
        
        2. МЕТОДЫ ОБУЧЕНИЯ
        ------------------
        • Метод минимальных расстояний (центроиды):
          - Вычисляются центроиды (средние значения) для каждого класса
          - Новый объект относится к ближайшему центроиду
        
        • Метод минимального числа ошибок (персептрон):
          - Строится линейная дискриминантная функция f(x) = w·x + b
          - Веса подбираются для минимизации числа ошибок
        
        3. ПОРЯДОК РАБОТЫ
        -----------------
        1. Загрузите обучающую выборку (Файл → Считать данные)
        2. Обучите классификатор (Обучение → Обучить модель)
        3. Распознавайте новые объекты
        
        4. ГОРЯЧИЕ КЛАВИШИ
        -----------------
        Ctrl+O - Загрузить данные
        Ctrl+S - Сохранить данные
        Ctrl+T - Обучение модели
        Ctrl+E - Оценка качества
        Ctrl+R - Распознать объект
        Ctrl+Q - Выход
        F1 - Справка
        
        © 2026 СибГМУ
        """
        
        help_text.insert(tk.END, help_content)
        help_text.configure(state='disabled')
    
    # ============================================================
    # МЕТОДЫ ДЛЯ РАБОТЫ С ДАННЫМИ
    # ============================================================
    
    def load_data(self):
        """Загрузка данных из файла"""
        result = data_loader.load_data(self.root)
        if result is not None:
            self.data = result
            self.feature_names = list(self.data.columns[1:])
            
            self.update_status(f"✅ Данные загружены: {len(self.data)} объектов, {self.data.shape[1]-1} признаков")
            self.display_data_table()
            
            self.recognize_btn.config(state='disabled')
            self.notebook.select(0)
    
    def display_data_table(self):
        """Отображение данных в таблице"""
        if self.data is None:
            return
        
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        columns = list(self.data.columns)
        self.data_tree['columns'] = columns
        
        for col in columns:
            self.data_tree.heading(col, text=col)
            max_width = max(len(str(col)) * 10, 80)
            sample_values = self.data[col].astype(str).head(200)
            for val in sample_values:
                width = len(val) * 8
                if width > max_width:
                    max_width = min(width, 300)
            self.data_tree.column(col, width=max_width, anchor='center')
        
        for idx, row in self.data.iterrows():
            values = [str(row[col]) for col in columns]
            if idx % 2 == 0:
                self.data_tree.insert('', tk.END, values=values, tags=('evenrow',))
            else:
                self.data_tree.insert('', tk.END, values=values)
        
        self.data_tree.tag_configure('evenrow', background='#f5f5f5')
        
        unique_classes = self.data.iloc[:, 0].unique()
        self.data_info_label.config(
            text=f"📊 Всего: {len(self.data)} объектов | {self.data.shape[1]-1} признаков | "
                 f"Классы: {', '.join(map(str, unique_classes[:5]))}"
                 f"{'...' if len(unique_classes) > 5 else ''}"
        )
    
    def train_model(self):
        """Обучение классификатора"""
        if self.data is None:
            messagebox.showwarning("Нет данных", "Сначала загрузите данные!")
            return
        
        # Автоматически выбираем первые 4 признака
        if len(self.feature_names) >= 4:
            self.selected_features = self.feature_names[:4]
        else:
            self.selected_features = self.feature_names
        
        if len(self.selected_features) < 2:
            messagebox.showerror("Ошибка", f"Недостаточно признаков для обучения!\n\nТребуется минимум 2 признака.")
            return
        
        # Создаем DataFrame с выбранными признаками
        class_column = self.data.columns[0]
        train_data = self.data[[class_column] + self.selected_features].copy()
        
        self.update_status(f"🔄 Обучение модели на {len(self.selected_features)} признаках...")
        
        # Обучаем классификатор
        self.classifier = training.train_classifier_simple(self.root, train_data)
        
        if self.classifier:
            self.update_status(f"✅ Модель обучена")
            self.recognize_btn.config(state='normal')
            self.display_class_stats()
            self.notebook.select(1)
            
            messagebox.showinfo("Обучение завершено", 
                               f"Модель успешно обучена!\n\n"
                               f"Метод: минимальных расстояний\n"
                               f"Метрика: евклидово расстояние\n"
                               f"Классов: {len(self.classifier.class_names)}\n"
                               f"Признаков: {self.classifier.feature_count}")
        else:
            self.update_status("❌ Ошибка при обучении модели")
    
    def display_class_stats(self):
        """Отображение характеристик классов"""
        if self.classifier is None:
            return
        
        self.stats_text.configure(state='normal')
        self.stats_text.delete(1.0, tk.END)
        
        # Формируем отчет
        report = "╔" + "═" * 68 + "╗\n"
        report += "║" + "ХАРАКТЕРИСТИКИ КЛАССОВ".center(68) + "║\n"
        report += "╚" + "═" * 68 + "╝\n\n"
        
        report += f"📊 МЕТОД ОБУЧЕНИЯ: {self.classifier.method}\n"
        report += f"📏 МЕТРИКА РАССТОЯНИЯ: {self.classifier.metric}\n"
        if self.classifier.metric == 'euclidean':
            report += f"   (Евклидово расстояние: d = √[Σ(xi - yi)²])\n"
        report += f"📋 ИСПОЛЬЗОВАННЫЕ ПРИЗНАКИ: {', '.join(self.selected_features)}\n"
        report += f"📅 ДАТА ОБУЧЕНИЯ: {self.classifier.get_model_info()['training_date']}\n\n"
        
        # Таблица статистики классов
        report += "═" * 70 + "\n"
        report += "1. СТАТИСТИКА ПО КЛАССАМ\n"
        report += "═" * 70 + "\n\n"
        
        report += f"{'Класс':<15} {'Кол-во':<8} {'Центроид':<35} {'Внутрикласс. расстояние':<25}\n"
        report += "─" * 70 + "\n"
        
        for class_name in self.classifier.class_names:
            stats = self.classifier.class_stats.get(class_name, {})
            count = stats.get('count', 0)
            centroid = ', '.join([f'{v:.4f}' for v in stats.get('mean', [])])
            intra_dist = self.classifier.intraclass_distances.get(class_name, 0)
            report += f"{class_name:<15} {count:<8} {centroid:<35} {intra_dist:<25.4f}\n"
        
        # Дисперсии признаков
        report += "\n" + "═" * 70 + "\n"
        report += "2. ДИСПЕРСИИ ПРИЗНАКОВ ПО КЛАССАМ\n"
        report += "═" * 70 + "\n\n"
        
        header = f"{'Класс':<15}"
        for f in self.selected_features:
            header += f"{f[:12]:<13}"
        report += header + "\n"
        report += "─" * (15 + 13 * len(self.selected_features)) + "\n"
        
        for class_name in self.classifier.class_names:
            stats = self.classifier.class_stats.get(class_name, {})
            variances = stats.get('variance', [])
            row = f"{class_name:<15}"
            for v in variances:
                row += f"{v:<13.4f}"
            report += row + "\n"
        
        # Расстояния между классами
        if self.classifier.class_distances:
            report += "\n" + "═" * 70 + "\n"
            report += "3. РАССТОЯНИЯ МЕЖДУ КЛАССАМИ\n"
            report += "═" * 70 + "\n\n"
            
            report += f"{'Пары классов':<25} {'Евклидово расстояние':<25} {'Манхэттенское расстояние':<25}\n"
            report += "─" * 70 + "\n"
            
            for pair, dists in self.classifier.class_distances.items():
                report += f"{pair:<25} {dists['euclidean']:<25.4f} {dists['manhattan']:<25.4f}\n"
        
        # Пояснения
        report += "\n" + "═" * 70 + "\n"
        report += "📌 ПОЯСНЕНИЯ\n"
        report += "═" * 70 + "\n"
        report += "• Центроид - среднее значение признаков для каждого класса\n"
        report += "• Дисперсия - мера разброса значений признака относительно среднего\n"
        report += "• Внутриклассовое расстояние - среднее расстояние между объектами одного класса\n"
        report += "• Расстояния между классами - расстояние между центроидами классов\n"
        report += "• Используется евклидова метрика: d = √[(x₁-y₁)² + (x₂-y₂)² + ...]\n"
        
        self.stats_text.insert(tk.END, report)
        self.stats_text.configure(state='disabled')
    
    def clear_results(self):
        """Очистка полей ввода и результатов"""
        for entry in self.entries:
            entry.delete(0, tk.END)
        
        self.results_text.configure(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Введите значения признаков и нажмите 'Распознать объект'.")
        self.results_text.configure(state='disabled')
    
    def recognize_current_object(self):
        """Распознавание объекта из полей ввода"""
        if self.classifier is None:
            messagebox.showwarning("Нет модели", "Сначала обучите модель!")
            return
        
        for entry in self.entries:
            if not entry.get().strip():
                messagebox.showwarning("Предупреждение", "Заполните все поля!")
                return
        
        try:
            features = []
            for entry in self.entries:
                value = entry.get().strip()
                features.append(float(value))
            
            predicted_class, scores = self.classifier.predict(features)
            
            self.results_text.configure(state='normal')
            self.results_text.delete(1.0, tk.END)
            
            result = "═" * 60 + "\n"
            result += "РЕЗУЛЬТАТ РАСПОЗНАВАНИЯ\n"
            result += "═" * 60 + "\n\n"
            result += f"🎯 Распознанный класс: {predicted_class}\n\n"
            
            if isinstance(scores, dict):
                result += "📏 Расстояния до классов:\n"
                result += "─" * 40 + "\n"
                for class_name, score in scores.items():
                    result += f"   • {class_name}: {score:.4f}\n"
            
            result += "\n" + "─" * 60 + "\n"
            result += "💡 Объект отнесен к классу с минимальным расстоянием"
            
            self.results_text.insert(tk.END, result)
            self.results_text.configure(state='disabled')
            
            self.update_status(f"✅ Объект распознан как '{predicted_class}'")
            
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные числовые значения!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка распознавания:\n{str(e)}")
    
    def save_data(self):
        """Сохранение данных в файл"""
        if self.data is not None:
            data_saver.save_data(self.root, self.data)
        else:
            messagebox.showwarning("Нет данных", "Сначала загрузите данные!")
    
    def recognize_object(self):
        """Распознавание нового объекта (вызов отдельного окна)"""
        if self.classifier is not None:
            recognition.recognize_object(self.root, self.classifier)
        else:
            messagebox.showwarning("Нет модели", "Сначала обучите модель!")
    
    def evaluate_quality(self):
        """Оценка качества распознавания"""
        if self.classifier is not None and self.data is not None:
            if self.selected_features:
                class_column = self.data.columns[0]
                eval_data = self.data[[class_column] + self.selected_features].copy()
                quality_evaluation.evaluate(self.root, self.classifier, eval_data)
            else:
                quality_evaluation.evaluate(self.root, self.classifier, self.data)
        else:
            messagebox.showwarning("Нет данных", "Загрузите данные и обучите модель!")
    
    def show_help(self):
        """Показать справку"""
        help_viewer.show_help(self.root)
    
    def show_about(self):
        """О программе"""
        about_text = """
        ╔══════════════════════════════════════════════════════════╗
        ║     ПРОГРАММА РАСПОЗНАВАНИЯ ОБЪЕКТОВ                    ║
        ║     Метод минимальных расстояний                        ║
        ╚══════════════════════════════════════════════════════════╝
        
        Разработчик: Дьяченко А.А.
        СибГМУ, Кафедра медицинской кибернетики
        Версия: 2.0
        
        Функции:
        • Загрузка данных из Excel/CSV
        • Обучение модели (метод минимальных расстояний)
        • Распознавание новых объектов
        • Оценка качества классификации
        
        © 2026
        """
        messagebox.showinfo("ℹ️ О программе", about_text)
    
    def exit_app(self):
        """Выход из программы"""
        exit_handler.exit_app(self.root)
    
    def update_status(self, message):
        """Обновление статусной строки"""
        self.status_label.config(text=f" {message}")

# ============================================================
# ЗАПУСК ПРОГРАММЫ
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()