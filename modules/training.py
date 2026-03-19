import tkinter as tk
from tkinter import messagebox, Toplevel, Text, Scrollbar, ttk
import numpy as np
import pandas as pd
import re

class Classifier:
    def __init__(self):
        self.class_centroids = {}
        self.class_names = []
        self.feature_count = 0
        self.class_stats = {}  # Статистика по классам
        self.class_distances = {}  # Расстояния между классами
        self.intraclass_distances = {}  # Внутриклассовые расстояния
    
    def calculate_euclidean_distance(self, point1, point2):
        """Вычисление евклидова расстояния"""
        return np.linalg.norm(point1 - point2)
    
    def calculate_manhattan_distance(self, point1, point2):
        """Вычисление манхэттенского расстояния"""
        return np.sum(np.abs(point1 - point2))
    
    def calculate_class_stats(self, class_data):
        """Расчет статистики для класса"""
        stats = {
            'count': len(class_data),
            'mean': class_data.mean().values,
            'variance': class_data.var().values,  # Дисперсия
            'std': class_data.std().values,  # Среднеквадратическое отклонение
            'min': class_data.min().values,
            'max': class_data.max().values
        }
        return stats
    
    def calculate_intraclass_distance(self, class_data):
        """Расчет среднего расстояния между объектами внутри класса"""
        if len(class_data) < 2:
            return 0
        
        n = len(class_data)
        total_distance = 0
        count = 0
        
        for i in range(n):
            for j in range(i+1, n):
                dist = self.calculate_euclidean_distance(
                    class_data.iloc[i].values, 
                    class_data.iloc[j].values
                )
                total_distance += dist
                count += 1
        
        return total_distance / count if count > 0 else 0
    
    def calculate_between_class_distances(self):
        """Расчет расстояний между классами"""
        distances = {}
        class_list = list(self.class_centroids.keys())
        
        for i in range(len(class_list)):
            for j in range(i+1, len(class_list)):
                class1 = class_list[i]
                class2 = class_list[j]
                
                # Евклидово расстояние между центроидами
                euclidean_dist = self.calculate_euclidean_distance(
                    self.class_centroids[class1],
                    self.class_centroids[class2]
                )
                
                # Манхэттенское расстояние между центроидами
                manhattan_dist = self.calculate_manhattan_distance(
                    self.class_centroids[class1],
                    self.class_centroids[class2]
                )
                
                distances[f"{class1} - {class2}"] = {
                    'euclidean': euclidean_dist,
                    'manhattan': manhattan_dist
                }
        
        return distances
    
    def train(self, data):
        """Обучение классификатора на данных"""
        try:
            print("=" * 60)
            print("НАЧАЛО ОБУЧЕНИЯ")
            print("=" * 60)
            
            # Получаем метки классов (первый столбец)
            labels = data.iloc[:, 0]
            features = data.iloc[:, 1:5]
            
            print(f"Метки классов: {labels.unique()}")
            
            # Преобразуем признаки в числовой формат
            for col in features.columns:
                if features[col].dtype == 'object':
                    features[col] = pd.to_numeric(features[col], errors='coerce')
            
            # Заполняем пропуски средними
            features = features.fillna(features.mean())
            
            self.feature_count = features.shape[1]
            self.class_names = labels.unique()
            
            print(f"\nКоличество признаков: {self.feature_count}")
            print(f"Классы: {self.class_names}")
            
            # Рассчитываем статистику для каждого класса
            for class_name in self.class_names:
                class_data = features[labels == class_name]
                
                if len(class_data) == 0:
                    continue
                
                # Статистика класса
                self.class_stats[class_name] = self.calculate_class_stats(class_data)
                
                # Центроид
                centroid = class_data.mean().values
                self.class_centroids[class_name] = centroid
                
                # Внутриклассовое расстояние
                self.intraclass_distances[class_name] = self.calculate_intraclass_distance(class_data)
                
                print(f"\n📊 Класс '{class_name}':")
                print(f"  Объектов: {len(class_data)}")
                print(f"  Центроид: {centroid}")
                print(f"  Дисперсии: {self.class_stats[class_name]['variance']}")
                print(f"  Внутриклассовое расстояние: {self.intraclass_distances[class_name]:.4f}")
            
            # Расстояния между классами
            self.class_distances = self.calculate_between_class_distances()
            
            print("\n📏 Расстояния между классами (евклидово):")
            for pair, dists in self.class_distances.items():
                print(f"  {pair}: {dists['euclidean']:.4f}")
            
            print("\n" + "=" * 60)
            print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО")
            print("=" * 60)
            
            return self
            
        except Exception as e:
            print(f"\n!!! ОШИБКА ПРИ ОБУЧЕНИИ: {str(e)}")
            raise Exception(f"Ошибка при обучении: {str(e)}")
    
    def predict(self, features):
        """Предсказание класса для одного объекта"""
        if len(features) != self.feature_count:
            raise ValueError(f"Ожидалось {self.feature_count} признаков")
        
        features = np.array([float(x) for x in features])
        
        distances = {}
        for class_name, centroid in self.class_centroids.items():
            dist = self.calculate_euclidean_distance(features, centroid)
            distances[class_name] = dist
        
        predicted_class = min(distances, key=distances.get)
        return predicted_class, distances

def train_classifier(parent, data):
    """Функция обучения классификатора"""
    try:
        if data is None or len(data) == 0:
            messagebox.showerror("Ошибка", "Нет данных для обучения")
            return None
        
        # Обработка данных
        data_clean = data.copy()
        
        if data_clean.shape[1] > 5:
            data_clean = data_clean.iloc[:, :5]
            data_clean.columns = ['класс', 'признак1', 'признак2', 'признак3', 'признак4']
        
        # Извлечение чисел из признаков
        feature_columns = data_clean.columns[1:5]
        
        for col in feature_columns:
            data_clean[col] = data_clean[col].astype(str).apply(extract_number_from_string)
        
        # Удаляем строки с пропусками
        initial_count = len(data_clean)
        data_clean = data_clean.dropna(subset=feature_columns)
        
        if len(data_clean) == 0:
            messagebox.showerror("Ошибка", "Не удалось извлечь числовые данные")
            return None
        
        # Создаем и обучаем классификатор
        classifier = Classifier()
        classifier.train(data_clean)
        
        # Показываем результаты в отдельном окне
        show_class_statistics(parent, classifier, data_clean)
        
        messagebox.showinfo("Успех", 
                          f"Модель успешно обучена!\n\n"
                          f"• Классов: {len(classifier.class_names)}\n"
                          f"• Признаков: {classifier.feature_count}\n"
                          f"• Объектов: {len(data_clean)}")
        
        return classifier
        
    except Exception as e:
        messagebox.showerror("Ошибка обучения", str(e))
        return None

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

def show_class_statistics(parent, classifier, data):
    """Отображение статистики классов в отдельном окне"""
    window = Toplevel(parent)
    window.title("Характеристики классов")
    window.geometry("900x700")
    
    # Создаем вкладки
    notebook = ttk.Notebook(window)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Вкладка 1: Центроиды и статистика
    stats_frame = ttk.Frame(notebook)
    notebook.add(stats_frame, text="📊 Центроиды и статистика")
    
    # Вкладка 2: Дисперсии
    variance_frame = ttk.Frame(notebook)
    notebook.add(variance_frame, text="📈 Дисперсии")
    
    # Вкладка 3: Расстояния
    distances_frame = ttk.Frame(notebook)
    notebook.add(distances_frame, text="📏 Расстояния")
    
    # ============================================================
    # ВКЛАДКА 1: ЦЕНТРОИДЫ И СТАТИСТИКА
    # ============================================================
    
    # Таблица с центроидами
    centroids_label = tk.Label(stats_frame, text="ЦЕНТРОИДЫ КЛАССОВ", 
                                font=('Arial', 14, 'bold'))
    centroids_label.pack(pady=10)
    
    # Создаем таблицу
    columns = ['Класс'] + [f'Призн.{i}' for i in range(1, classifier.feature_count+1)] + ['Объектов']
    tree = ttk.Treeview(stats_frame, columns=columns, show='headings', height=8)
    
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=100, anchor='center')
    
    # Заполняем данными
    for class_name in classifier.class_names:
        row = [class_name]
        centroid = classifier.class_centroids[class_name]
        row.extend([f"{v:.4f}" for v in centroid])
        row.append(str(classifier.class_stats[class_name]['count']))
        tree.insert('', tk.END, values=row)
    
    tree.pack(pady=10, padx=10, fill=tk.X)
    
    # Текстовое описание
    text_frame = tk.Frame(stats_frame)
    text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    text_area = Text(text_frame, wrap=tk.WORD, font=('Arial', 11))
    scrollbar = Scrollbar(text_frame, command=text_area.yview)
    text_area.configure(yscrollcommand=scrollbar.set)
    
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Текстовый отчет
    report = "📊 ХАРАКТЕРИСТИКИ КЛАССОВ\n"
    report += "=" * 50 + "\n\n"
    
    for class_name in classifier.class_names:
        stats = classifier.class_stats[class_name]
        report += f"Класс: {class_name}\n"
        report += f"  Количество объектов: {stats['count']}\n"
        report += f"  Центроид: {[f'{v:.4f}' for v in stats['mean']]}\n"
        report += f"  Дисперсии: {[f'{v:.4f}' for v in stats['variance']]}\n"
        report += f"  STD: {[f'{v:.4f}' for v in stats['std']]}\n"
        report += f"  Мин: {[f'{v:.4f}' for v in stats['min']]}\n"
        report += f"  Макс: {[f'{v:.4f}' for v in stats['max']]}\n\n"
    
    text_area.insert(tk.END, report)
    text_area.configure(state='disabled')
    
    # ============================================================
    # ВКЛАДКА 2: ДИСПЕРСИИ
    # ============================================================
    
    # Таблица дисперсий
    var_label = tk.Label(variance_frame, text="ДИСПЕРСИИ ПРИЗНАКОВ ПО КЛАССАМ", 
                          font=('Arial', 14, 'bold'))
    var_label.pack(pady=10)
    
    var_columns = ['Класс'] + [f'Призн.{i}' for i in range(1, classifier.feature_count+1)]
    var_tree = ttk.Treeview(variance_frame, columns=var_columns, show='headings', height=8)
    
    for col in var_columns:
        var_tree.heading(col, text=col)
        var_tree.column(col, width=120, anchor='center')
    
    for class_name in classifier.class_names:
        row = [class_name]
        row.extend([f"{v:.4f}" for v in classifier.class_stats[class_name]['variance']])
        var_tree.insert('', tk.END, values=row)
    
    var_tree.pack(pady=10, padx=10, fill=tk.X)
    
    # ============================================================
    # ВКЛАДКА 3: РАССТОЯНИЯ
    # ============================================================
    
    dist_label = tk.Label(distances_frame, text="РАССТОЯНИЯ МЕЖДУ КЛАССАМИ", 
                           font=('Arial', 14, 'bold'))
    dist_label.pack(pady=10)
    
    # Информация о метрике
    metric_label = tk.Label(distances_frame, 
                           text="✓ Используется евклидова метрика",
                           font=('Arial', 12, 'bold'),
                           fg='green')
    metric_label.pack(pady=5)
    
    # Таблица расстояний между классами
    dist_columns = ['Пары классов', 'Евклидово', 'Манхэттенское']
    dist_tree = ttk.Treeview(distances_frame, columns=dist_columns, show='headings', height=10)
    
    for col in dist_columns:
        dist_tree.heading(col, text=col)
        dist_tree.column(col, width=200, anchor='center')
    
    for pair, dists in classifier.class_distances.items():
        dist_tree.insert('', tk.END, values=[
            pair,
            f"{dists['euclidean']:.4f}",
            f"{dists['manhattan']:.4f}"
        ])
    
    dist_tree.pack(pady=10, padx=10, fill=tk.X)
    
    # Таблица внутриклассовых расстояний
    intra_label = tk.Label(distances_frame, text="ВНУТРИКЛАССОВЫЕ РАССТОЯНИЯ", 
                            font=('Arial', 14, 'bold'))
    intra_label.pack(pady=(20, 10))
    
    intra_columns = ['Класс', 'Среднее расстояние между объектами']
    intra_tree = ttk.Treeview(distances_frame, columns=intra_columns, show='headings', height=6)
    
    for col in intra_columns:
        intra_tree.heading(col, text=col)
        intra_tree.column(col, width=250, anchor='center')
    
    for class_name, distance in classifier.intraclass_distances.items():
        intra_tree.insert('', tk.END, values=[class_name, f"{distance:.4f}"])
    
    intra_tree.pack(pady=10, padx=10, fill=tk.X)