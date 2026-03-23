import tkinter as tk
from tkinter import messagebox, Toplevel, Text, Scrollbar, ttk
import numpy as np
import pandas as pd
import re
from datetime import datetime

class Classifier:
    """Класс классификатора с поддержкой различных методов обучения"""
    
    def __init__(self, method='min_distance', metric='euclidean'):
        self.method = method
        self.metric = metric
        
        self.class_centroids = {}
        self.class_names = []
        self.feature_count = 0
        
        self.weights = None
        self.bias = None
        self.class_encoding = {}
        
        self.class_stats = {}
        self.class_distances = {}
        self.intraclass_distances = {}
        
        self.learning_params = {
            'learning_rate': 0.01,
            'max_iterations': 1000,
            'convergence_threshold': 0.001
        }
    
    def calculate_euclidean_distance(self, point1, point2):
        return np.linalg.norm(point1 - point2)
    
    def calculate_manhattan_distance(self, point1, point2):
        return np.sum(np.abs(point1 - point2))
    
    def calculate_distance(self, point1, point2):
        if self.metric == 'euclidean':
            return self.calculate_euclidean_distance(point1, point2)
        elif self.metric == 'manhattan':
            return self.calculate_manhattan_distance(point1, point2)
        return self.calculate_euclidean_distance(point1, point2)
    
    def train_min_distance(self, features, labels):
        """Обучение методом минимальных расстояний"""
        print("\n" + "=" * 70)
        print("ОБУЧЕНИЕ МЕТОДОМ МИНИМАЛЬНЫХ РАССТОЯНИЙ")
        print("=" * 70)
        
        self.class_names = np.unique(labels)
        self.feature_count = features.shape[1]
        
        print(f"\n📊 Информация о данных:")
        print(f"  • Количество объектов: {len(features)}")
        print(f"  • Количество признаков: {self.feature_count}")
        print(f"  • Количество классов: {len(self.class_names)}")
        print(f"  • Метрика расстояния: {self.metric}")
        
        for class_name in self.class_names:
            class_data = features[labels == class_name]
            
            if len(class_data) == 0:
                continue
            
            centroid = class_data.mean().values
            self.class_centroids[class_name] = centroid
            
            self.class_stats[class_name] = {
                'count': len(class_data),
                'mean': centroid,
                'variance': class_data.var().values,
                'std': class_data.std().values,
                'min': class_data.min().values,
                'max': class_data.max().values
            }
            
            self.intraclass_distances[class_name] = self.calculate_intraclass_distance(class_data)
            
            print(f"\n📌 Класс '{class_name}':")
            print(f"  • Объектов: {len(class_data)}")
            print(f"  • Центроид: {np.round(centroid, 4)}")
        
        self.calculate_between_class_distances()
        
        print("\n" + "=" * 70)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО")
        print("=" * 70)
    
    def train_min_error(self, features, labels):
        """Обучение методом минимального числа ошибок"""
        print("\n" + "=" * 70)
        print("ОБУЧЕНИЕ МЕТОДОМ МИНИМАЛЬНОГО ЧИСЛА ОШИБОК")
        print("=" * 70)
        
        self.class_names = np.unique(labels)
        self.feature_count = features.shape[1]
        
        print(f"\n📊 Информация о данных:")
        print(f"  • Количество объектов: {len(features)}")
        print(f"  • Количество признаков: {self.feature_count}")
        print(f"  • Количество классов: {len(self.class_names)}")
        
        if len(self.class_names) > 2:
            print("\n⚠️ Для многоклассовой классификации используется подход 'один против всех'")
            
            self.weights = {}
            self.bias = {}
            
            for target_class in self.class_names:
                binary_labels = np.where(labels == target_class, 1, -1)
                w, b = self.train_binary_classifier(features.values, binary_labels)
                self.weights[target_class] = w
                self.bias[target_class] = b
                
                print(f"\n📌 Класс '{target_class}':")
                print(f"  • Веса: {np.round(w, 4)}")
                print(f"  • Смещение: {b:.4f}")
        else:
            class1, class2 = self.class_names
            binary_labels = np.where(labels == class1, 1, -1)
            
            self.weights, self.bias = self.train_binary_classifier(features.values, binary_labels)
            self.class_encoding = {class1: 1, class2: -1}
            
            print(f"\n📌 Результаты обучения:")
            print(f"  • Веса: {np.round(self.weights, 4)}")
            print(f"  • Смещение: {self.bias:.4f}")
        
        predictions = self.predict_batch(features.values)
        accuracy = np.mean(predictions == labels) * 100
        
        print(f"\n📊 Результаты на обучающей выборке:")
        print(f"  • Точность: {accuracy:.2f}%")
        
        print("\n" + "=" * 70)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО")
        print("=" * 70)
    
    def train_binary_classifier(self, X, y):
        """Обучение бинарного классификатора методом персептрона"""
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0
        
        for iteration in range(self.learning_params['max_iterations']):
            errors = 0
            for i in range(n_samples):
                linear_output = np.dot(w, X[i]) + b
                prediction = np.sign(linear_output)
                
                if prediction != y[i]:
                    w += self.learning_params['learning_rate'] * y[i] * X[i]
                    b += self.learning_params['learning_rate'] * y[i]
                    errors += 1
            
            if errors == 0:
                print(f"  • Сходимость достигнута на итерации {iteration + 1}")
                break
        
        return w, b
    
    def calculate_intraclass_distance(self, class_data):
        if len(class_data) < 2:
            return 0
        
        n = len(class_data)
        total_distance = 0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.calculate_distance(
                    class_data.iloc[i].values,
                    class_data.iloc[j].values
                )
                total_distance += dist
                count += 1
        
        return total_distance / count if count > 0 else 0
    
    def calculate_between_class_distances(self):
        if not self.class_centroids:
            return
        
        distances = {}
        class_list = list(self.class_centroids.keys())
        
        for i in range(len(class_list)):
            for j in range(i + 1, len(class_list)):
                class1 = class_list[i]
                class2 = class_list[j]
                
                euclidean_dist = self.calculate_euclidean_distance(
                    self.class_centroids[class1],
                    self.class_centroids[class2]
                )
                
                manhattan_dist = self.calculate_manhattan_distance(
                    self.class_centroids[class1],
                    self.class_centroids[class2]
                )
                
                distances[f"{class1} - {class2}"] = {
                    'euclidean': euclidean_dist,
                    'manhattan': manhattan_dist
                }
        
        self.class_distances = distances
        return distances
    
    def predict(self, features):
        features = np.array([float(x) for x in features])
        
        if self.method == 'min_distance':
            distances = {}
            for class_name, centroid in self.class_centroids.items():
                dist = self.calculate_distance(features, centroid)
                distances[class_name] = dist
            
            predicted_class = min(distances, key=distances.get)
            return predicted_class, distances
        
        elif self.method == 'min_error':
            if len(self.class_names) > 2:
                scores = {}
                for class_name in self.class_names:
                    linear_output = np.dot(self.weights[class_name], features) + self.bias[class_name]
                    scores[class_name] = linear_output
                
                predicted_class = max(scores, key=scores.get)
                return predicted_class, scores
            else:
                linear_output = np.dot(self.weights, features) + self.bias
                prediction = 1 if linear_output > 0 else -1
                
                for class_name, code in self.class_encoding.items():
                    if code == prediction:
                        predicted_class = class_name
                        break
                
                return predicted_class, {'score': linear_output}
    
    def predict_batch(self, X):
        predictions = []
        for i in range(len(X)):
            pred, _ = self.predict(X[i])
            predictions.append(pred)
        return np.array(predictions)
    
    def get_model_info(self):
        info = {
            'method': self.method,
            'metric': self.metric,
            'classes': self.class_names,
            'feature_count': self.feature_count,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if self.method == 'min_distance':
            info['centroids'] = self.class_centroids
        else:
            info['weights'] = self.weights
            info['bias'] = self.bias
        
        return info
    
    def get_class_stats_table(self):
        """Возвращает таблицу со статистикой классов"""
        if not self.class_stats:
            return None
        
        data = []
        for class_name in self.class_names:
            stats = self.class_stats[class_name]
            row = {
                'Класс': class_name,
                'Количество': stats['count'],
                'Центроид': ', '.join([f'{v:.4f}' for v in stats['mean']]),
                'Дисперсии': ', '.join([f'{v:.4f}' for v in stats['variance']]),
                'Внутриклассовое расстояние': f"{self.intraclass_distances.get(class_name, 0):.4f}"
            }
            data.append(row)
        return data
    
    def get_distances_table(self):
        """Возвращает таблицу с расстояниями между классами"""
        if not self.class_distances:
            return None
        
        data = []
        for pair, dists in self.class_distances.items():
            row = {
                'Пары классов': pair,
                'Евклидово расстояние': f"{dists['euclidean']:.4f}",
                'Манхэттенское расстояние': f"{dists['manhattan']:.4f}"
            }
            data.append(row)
        return data


def extract_number_from_string(text):
    """Извлечение числа из строки"""
    if pd.isna(text):
        return None
    
    text = str(text).strip()
    if text == '' or text == 'nan':
        return None
    
    try:
        return float(text)
    except:
        match = re.search(r'-?\d+\.?\d*', text)
        if match:
            try:
                return float(match.group())
            except:
                return None
    return None


def train_classifier(parent, data):
    """
    Функция обучения классификатора с выбором метода и параметров
    
    Аргументы:
        parent: родительское окно
        data: DataFrame с данными
    
    Возвращает:
        classifier: обученный классификатор
    """
    try:
        if data is None or len(data) == 0:
            messagebox.showerror("Ошибка", "Нет данных для обучения")
            return None
        
        # Создаем окно выбора параметров обучения
        param_window = Toplevel(parent)
        param_window.title("Параметры обучения")
        param_window.geometry("600x550")
        param_window.resizable(False, False)
        
        param_window.transient(parent)
        param_window.grab_set()
        
        # Переменные для хранения параметров
        method_var = tk.StringVar(value="min_distance")
        metric_var = tk.StringVar(value="euclidean")
        learning_rate_var = tk.StringVar(value="0.01")
        max_iter_var = tk.StringVar(value="1000")
        
        # Заголовок
        title_label = tk.Label(param_window, text="Настройка обучения классификатора",
                              font=('Arial', 14, 'bold'), pady=10)
        title_label.pack()
        
        # Основной фрейм
        main_frame = ttk.Frame(param_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Выбор метода обучения
        method_frame = ttk.LabelFrame(main_frame, text="Метод обучения", padding="10")
        method_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Radiobutton(method_frame, text="Метод минимальных расстояний (центроиды)",
                       variable=method_var, value="min_distance").pack(anchor=tk.W, pady=2)
        ttk.Label(method_frame, text="  • Вычисляются центроиды (средние образы) для каждого класса",
                 font=('Arial', 9), foreground='gray').pack(anchor=tk.W, padx=20)
        ttk.Label(method_frame, text="  • Новый объект относится к ближайшему центроиду",
                 font=('Arial', 9), foreground='gray').pack(anchor=tk.W, padx=20)
        
        ttk.Radiobutton(method_frame, text="Метод минимального числа ошибок (персептрон)",
                       variable=method_var, value="min_error").pack(anchor=tk.W, pady=(10, 2))
        ttk.Label(method_frame, text="  • Линейная дискриминантная функция f(x) = w·x + b",
                 font=('Arial', 9), foreground='gray').pack(anchor=tk.W, padx=20)
        ttk.Label(method_frame, text="  • Веса подбираются для минимизации числа ошибок",
                 font=('Arial', 9), foreground='gray').pack(anchor=tk.W, padx=20)
        
        # Выбор метрики расстояния
        metric_frame = ttk.LabelFrame(main_frame, text="Метрика расстояния", padding="10")
        metric_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Radiobutton(metric_frame, text="Евклидово расстояние",
                       variable=metric_var, value="euclidean").pack(anchor=tk.W, pady=2)
        ttk.Label(metric_frame, text="  • d = √(Σ(xi - yi)²)",
                 font=('Arial', 9), foreground='gray').pack(anchor=tk.W, padx=20)
        
        ttk.Radiobutton(metric_frame, text="Манхэттенское расстояние",
                       variable=metric_var, value="manhattan").pack(anchor=tk.W, pady=2)
        ttk.Label(metric_frame, text="  • d = Σ|xi - yi|",
                 font=('Arial', 9), foreground='gray').pack(anchor=tk.W, padx=20)
        
        # Параметры обучения
        params_frame = ttk.LabelFrame(main_frame, text="Параметры обучения (для персептрона)", padding="10")
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        rate_frame = ttk.Frame(params_frame)
        rate_frame.pack(fill=tk.X, pady=2)
        ttk.Label(rate_frame, text="Скорость обучения (learning rate):").pack(side=tk.LEFT, padx=5)
        ttk.Entry(rate_frame, textvariable=learning_rate_var, width=15).pack(side=tk.RIGHT, padx=5)
        
        iter_frame = ttk.Frame(params_frame)
        iter_frame.pack(fill=tk.X, pady=2)
        ttk.Label(iter_frame, text="Максимальное число итераций:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(iter_frame, textvariable=max_iter_var, width=15).pack(side=tk.RIGHT, padx=5)
        
        # Информация о данных
        info_frame = ttk.LabelFrame(main_frame, text="Информация о данных", padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        labels = data.iloc[:, 0]
        features = data.iloc[:, 1:]
        
        ttk.Label(info_frame, text=f"Объектов: {len(data)}").pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"Признаков: {features.shape[1]}").pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"Классов: {len(labels.unique())}").pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"Классы: {', '.join(map(str, labels.unique()))}").pack(anchor=tk.W)
        
        # Кнопки
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        def start_training():
            try:
                method = method_var.get()
                metric = metric_var.get()
                learning_rate = float(learning_rate_var.get())
                max_iterations = int(max_iter_var.get())
                
                # Обработка данных
                data_clean = data.copy()
                feature_columns = data_clean.columns[1:]
                for col in feature_columns:
                    data_clean[col] = data_clean[col].astype(str).apply(extract_number_from_string)
                
                data_clean = data_clean.dropna(subset=feature_columns)
                
                if len(data_clean) == 0:
                    messagebox.showerror("Ошибка", "Не удалось извлечь числовые данные")
                    param_window.destroy()
                    return
                
                labels = data_clean.iloc[:, 0]
                features = data_clean.iloc[:, 1:]
                
                classifier = Classifier(method=method, metric=metric)
                classifier.learning_params['learning_rate'] = learning_rate
                classifier.learning_params['max_iterations'] = max_iterations
                
                if method == 'min_distance':
                    classifier.train_min_distance(features, labels)
                else:
                    classifier.train_min_error(features, labels)
                
                param_window.destroy()
                
                # Показываем результаты
                show_training_results(parent, classifier, data_clean)
                
                messagebox.showinfo("Успех", f"Обучение завершено!\n\nМетод: {method}\nМетрика: {metric}\nКлассов: {len(classifier.class_names)}\nПризнаков: {classifier.feature_count}\nОбъектов: {len(data_clean)}")
                
                return classifier
                
            except ValueError as e:
                messagebox.showerror("Ошибка", f"Некорректные параметры:\n{str(e)}")
            except Exception as e:
                messagebox.showerror("Ошибка обучения", str(e))
        
        ttk.Button(button_frame, text="Начать обучение", command=start_training).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(button_frame, text="Отмена", command=param_window.destroy).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        param_window.wait_window()
        return None
        
    except Exception as e:
        messagebox.showerror("Ошибка обучения", str(e))
        return None


def show_training_results(parent, classifier, data):
    """Отображение результатов обучения"""
    window = Toplevel(parent)
    window.title("Результаты обучения")
    window.geometry("900x700")
    
    notebook = ttk.Notebook(window)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Вкладка 1: Параметры модели
    info_tab = ttk.Frame(notebook)
    notebook.add(info_tab, text="📊 Параметры модели")
    
    info_text = tk.Text(info_tab, wrap=tk.WORD, font=('Courier', 11))
    scrollbar = Scrollbar(info_tab, command=info_text.yview)
    info_text.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    model_info = "╔" + "=" * 68 + "╗\n"
    model_info += "║" + "РЕЗУЛЬТАТЫ ОБУЧЕНИЯ".center(68) + "║\n"
    model_info += "╚" + "=" * 68 + "╝\n\n"
    model_info += f"📅 Дата обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    model_info += f"🎯 Метод обучения: {classifier.method}\n"
    model_info += f"📏 Метрика расстояния: {classifier.metric}\n\n"
    model_info += "📊 Параметры:\n"
    model_info += f"  • Количество классов: {len(classifier.class_names)}\n"
    model_info += f"  • Количество признаков: {classifier.feature_count}\n"
    model_info += f"  • Количество объектов: {len(data)}\n"
    
    if classifier.method == 'min_distance':
        model_info += "\n🎯 ЦЕНТРОИДЫ КЛАССОВ:\n"
        for class_name, centroid in classifier.class_centroids.items():
            model_info += f"\n  {class_name}:\n"
            model_info += f"    Центроид: {np.round(centroid, 4)}\n"
            model_info += f"    Объектов: {classifier.class_stats[class_name]['count']}\n"
    else:
        model_info += "\n🎯 ДИСКРИМИНАНТНАЯ ФУНКЦИЯ:\n"
        if len(classifier.class_names) > 2:
            for class_name in classifier.class_names:
                model_info += f"\n  {class_name} (против всех):\n"
                model_info += f"    f(x) = {np.round(classifier.weights[class_name], 4)}·x + {classifier.bias[class_name]:.4f}\n"
        else:
            model_info += f"\n  f(x) = {np.round(classifier.weights, 4)}·x + {classifier.bias:.4f}\n"
            model_info += f"  Правило: если f(x) > 0 → класс {classifier.class_names[0]}\n"
            model_info += f"           если f(x) < 0 → класс {classifier.class_names[1]}\n"
    
    info_text.insert(tk.END, model_info)
    info_text.configure(state='disabled')
    
    # Вкладка 2: Статистика классов
    stats_tab = ttk.Frame(notebook)
    notebook.add(stats_tab, text="📈 Статистика классов")
    
    stats_text = tk.Text(stats_tab, wrap=tk.WORD, font=('Courier', 11))
    stats_scroll = Scrollbar(stats_tab, command=stats_text.yview)
    stats_text.configure(yscrollcommand=stats_scroll.set)
    stats_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    class_stats = "СТАТИСТИКА ПО КЛАССАМ\n"
    class_stats += "=" * 60 + "\n\n"
    
    for class_name in classifier.class_names:
        stats = classifier.class_stats[class_name]
        class_stats += f"Класс: {class_name}\n"
        class_stats += f"  Количество: {stats['count']}\n"
        class_stats += f"  Среднее: {np.round(stats['mean'], 4)}\n"
        class_stats += f"  Дисперсия: {np.round(stats['variance'], 4)}\n"
        class_stats += f"  STD: {np.round(stats['std'], 4)}\n"
        class_stats += f"  Мин: {np.round(stats['min'], 4)}\n"
        class_stats += f"  Макс: {np.round(stats['max'], 4)}\n\n"
    
    stats_text.insert(tk.END, class_stats)
    stats_text.configure(state='disabled')
    
    # Вкладка 3: Расстояния
    dist_tab = ttk.Frame(notebook)
    notebook.add(dist_tab, text="📏 Расстояния")
    
    dist_text = tk.Text(dist_tab, wrap=tk.WORD, font=('Courier', 11))
    dist_scroll = Scrollbar(dist_tab, command=dist_text.yview)
    dist_text.configure(yscrollcommand=dist_scroll.set)
    dist_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    dist_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    distances_info = "РАССТОЯНИЯ МЕЖДУ КЛАССАМИ\n"
    distances_info += "=" * 60 + "\n\n"
    
    if classifier.class_distances:
        for pair, dists in classifier.class_distances.items():
            distances_info += f"{pair}:\n"
            distances_info += f"  Евклидово: {dists['euclidean']:.4f}\n"
            distances_info += f"  Манхэттенское: {dists['manhattan']:.4f}\n\n"
    
    if classifier.intraclass_distances:
        distances_info += "ВНУТРИКЛАССОВЫЕ РАССТОЯНИЯ\n"
        distances_info += "-" * 40 + "\n"
        for class_name, distance in classifier.intraclass_distances.items():
            distances_info += f"{class_name}: {distance:.4f}\n"
    
    dist_text.insert(tk.END, distances_info)
    dist_text.configure(state='disabled')
    
    ttk.Button(window, text="Закрыть", command=window.destroy).pack(pady=10)


def train_classifier_simple(parent, data):
    """
    Упрощенная функция обучения классификатора (без диалогового окна)
    
    Аргументы:
        parent: родительское окно
        data: DataFrame с данными
    
    Возвращает:
        classifier: обученный классификатор
    """
    try:
        if data is None or len(data) == 0:
            messagebox.showerror("Ошибка", "Нет данных для обучения")
            return None
        
        # Обработка данных
        data_clean = data.copy()
        feature_columns = data_clean.columns[1:]
        
        # Извлекаем числа из всех признаков
        for col in feature_columns:
            data_clean[col] = data_clean[col].astype(str).apply(extract_number_from_string)
        
        # Удаляем строки с пропусками
        data_clean = data_clean.dropna(subset=feature_columns)
        
        if len(data_clean) == 0:
            messagebox.showerror("Ошибка", "Не удалось извлечь числовые данные")
            return None
        
        # Подготовка данных
        labels = data_clean.iloc[:, 0]
        features = data_clean.iloc[:, 1:]
        
        # Создаем и обучаем классификатор методом минимальных расстояний
        classifier = Classifier(method='min_distance', metric='euclidean')
        classifier.train_min_distance(features, labels)
        
        return classifier
        
    except Exception as e:
        messagebox.showerror("Ошибка обучения", str(e))
        return None