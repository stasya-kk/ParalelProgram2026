"""
Верификация результатов умножения матриц с помощью NumPy
и построение графиков зависимости времени и производительности
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 

sizes = [200, 400, 800, 1200, 1600, 2000]

print("ВЕРИФИКАЦИЯ РЕЗУЛЬТАТОВ УМНОЖЕНИЯ МАТРИЦ")

all_passed = True
timing_data = []

timing_file = "results/timing_results.txt"
if os.path.exists(timing_file):
    with open(timing_file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 3:
                    size = int(parts[0])
                    time_sec = float(parts[1])
                    gflops = float(parts[2])
                    timing_data.append((size, time_sec, gflops))
    print(f"\nЗагружены данные о времени из {timing_file}")
else:
    print(f"\nФайл {timing_file} не найден!")
    timing_data = [(size, 0, 0) for size in sizes]

print("ВЕРИФИКАЦИЯ ВЫЧИСЛЕНИЙ")

for size in sizes:
    print(f"\nПроверка размера {size}×{size}...")

    expected_file = f"matrices/expected_C_{size}.txt"
    result_file = f"results/result_{size}.txt"
    
    if not os.path.exists(expected_file):
        print(f" Файл {expected_file} не найден!")
        all_passed = False
        continue
        
    if not os.path.exists(result_file):
        print(f" Файл {result_file} не найден!")
        all_passed = False
        continue

    try:
        expected = np.loadtxt(expected_file)
        result = np.loadtxt(result_file)
    except Exception as e:
        print(f" Ошибка загрузки: {e}")
        all_passed = False
        continue

    if expected.shape != result.shape:
        print(f" Несовпадение размеров: expected {expected.shape} vs result {result.shape}")
        all_passed = False
        continue
    
    abs_diff = np.abs(expected - result)
    max_abs_diff = np.max(abs_diff)

    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = np.abs((expected - result) / (np.abs(expected) + 1e-12))
        max_rel_diff = np.max(rel_diff)

    is_close = np.allclose(expected, result, rtol=1e-6, atol=1e-8)
    
    if is_close:
        print(f"   Верификация пройдена!")
        print(f"   Макс. абсолютная ошибка: {max_abs_diff:.2e}")
        print(f"   Макс. относительная ошибка: {max_rel_diff:.2e}")
    else:
        print(f"   Верификация НЕ пройдена!")
        print(f"   Макс. абсолютная ошибка: {max_abs_diff:.2e}")
        print(f"   Макс. относительная ошибка: {max_rel_diff:.2e}")
        print(f"   Количество несовпадающих элементов: {np.sum(~np.isclose(expected, result, rtol=1e-6, atol=1e-8))}")
        all_passed = False

if all_passed:
    print(" ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!")
else:
    print(" ОБНАРУЖЕНЫ ОШИБКИ ПРИ ВЕРИФИКАЦИИ!")

if timing_data:
    sizes_plot = [d[0] for d in timing_data]
    times = [d[1] for d in timing_data]
    gflops = [d[2] for d in timing_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # График 1: Время выполнения
    ax1.plot(sizes_plot, times, 'b-o', linewidth=2, markersize=8, label='Время выполнения')
    ax1.set_xlabel('Размер матрицы (n)', fontsize=12)
    ax1.set_ylabel('Время (секунды)', fontsize=12)
    ax1.set_title('Зависимость времени выполнения от размера матрицы', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    for i, (x, y) in enumerate(zip(sizes_plot, times)):
        ax1.annotate(f'{y:.3f}s', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    

    n3_fit = [times[-1] * (s / sizes_plot[-1])**3 for s in sizes_plot]
    ax1.plot(sizes_plot, n3_fit, 'r--', linewidth=2, label='Теоретическая O(n³)')
    ax1.legend()
    
    # График 2: Производительность
    ax2.plot(sizes_plot, gflops, 'g-s', linewidth=2, markersize=8, label='Производительность')
    ax2.set_xlabel('Размер матрицы (n)', fontsize=12)
    ax2.set_ylabel('Производительность (гигафлопс)', fontsize=12)
    ax2.set_title('Зависимость производительности от размера матрицы', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    for i, (x, y) in enumerate(zip(sizes_plot, gflops)):
        ax2.annotate(f'{y:.2f} GFLOPS', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()

    plt.savefig('results/performance_plot.png', dpi=150, bbox_inches='tight')
    print(" График сохранён как 'results/performance_plot.png'")

    try:
        plt.show()
    except:
        print("  (График сохранён в файл, для просмотра откройте его в браузере или просмотрщике изображений)")

    print("\n" + "-" * 70)
    print("ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("-" * 70)
    print(f"{'Размер матрицы':^15} | {'Время (сек)':^12} | {'Производительность (GFLOPS)':^25}")
    print("-" * 70)
    for size, time_sec, gflop in timing_data:
        print(f"{size:^15} | {time_sec:^12.6f} | {gflop:^25.2f}")
    print("-" * 70)

    with open('results/results_table.txt', 'w', encoding='utf-8') as f:
        f.write("Таблица результатов умножения квадратных матриц\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Размер матрицы':<15} {'Время (сек)':<12} {'Производительность (GFLOPS)':<20}\n")
        f.write("-" * 60 + "\n")
        for size, time_sec, gflop in timing_data:
            f.write(f"{size:<15} {time_sec:<12.6f} {gflop:<20.2f}\n")
        f.write("=" * 60 + "\n")
    print("Таблица сохранена как 'results/results_table.txt'")
    
else:
    print("Нет данных для построения графиков!")
    
print("ВСЕ ОПЕРАЦИИ ЗАВЕРШЕНЫ")