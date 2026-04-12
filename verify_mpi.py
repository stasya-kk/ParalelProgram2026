"""
Верификация результатов параллельного умножения матриц MPI
и построение графиков производительности
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sizes = [200, 400, 800, 1200, 1600, 2000]

possible_procs = [1, 2, 4, 8]

print("ВЕРИФИКАЦИЯ РЕЗУЛЬТАТОВ ПАРАЛЛЕЛЬНОГО УМНОЖЕНИЯ МАТРИЦ (MPI)")

if not os.path.exists("matrices"):
    print("\n Папка 'matrices' не найдена! Запустите generate_matrices.py")
    exit(1)

all_timing_data = []
available_procs = []

for np_procs in possible_procs:
    timing_file = f"results_mpi/timing_results_{np_procs}proc.txt"
    if os.path.exists(timing_file):
        available_procs.append(np_procs)
        with open(timing_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        size = int(parts[0])
                        procs = int(parts[1])
                        time_sec = float(parts[2])
                        gflops = float(parts[3])
                        all_timing_data.append((size, procs, time_sec, gflops))
        print(f"Загружены данные для {np_procs} процессов из {timing_file}")

if not available_procs:
    print("\nНет данных для анализа! Запустите MPI программу")
    exit(1)

print("ВЕРИФИКАЦИЯ ВЫЧИСЛЕНИЙ")

all_passed = True

for size in sizes:
    print(f"\nПроверка размера {size}×{size}...")
    
    expected_file = f"matrices/expected_C_{size}.txt"
    
    if not os.path.exists(expected_file):
        print(f" Файл {expected_file} не найден!")
        all_passed = False
        continue
    
    try:
        expected = np.loadtxt(expected_file)
    except Exception as e:
        print(f" Ошибка загрузки expected: {e}")
        all_passed = False
        continue
    
    for np_procs in available_procs:
        result_file = f"results_mpi/result_{size}_{np_procs}proc.txt"
        
        if not os.path.exists(result_file):
            print(f" Файл {result_file} не найден")
            continue
        
        try:
            result = np.loadtxt(result_file)
        except Exception as e:
            print(f"Ошибка загрузки result: {e}")
            all_passed = False
            continue
        
        abs_diff = np.abs(expected - result)
        max_abs_diff = np.max(abs_diff)
        
        is_close = np.allclose(expected, result, rtol=1e-6, atol=1e-8)
        
        if is_close:
            print(f" {np_procs} процесса: Верификация пройдена! (макс. ошибка: {max_abs_diff:.2e})")
        else:
            print(f" {np_procs} процесса: Ошибка! (макс. ошибка: {max_abs_diff:.2e})")
            all_passed = False

if all_passed:
    print(" ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!")
else:
    print(" ОБНАРУЖЕНЫ РАСХОЖДЕНИЯ В РЕЗУЛЬТАТАХ!")

if all_timing_data:
    colors = {1: 'blue', 2: 'green', 4: 'red', 8: 'purple'}
    markers = {1: 'o', 2: 's', 4: '^', 8: 'D'}

    #ГРАФИК 1: Время выполнения

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    for np_procs in available_procs:
        data = [(d[0], d[2]) for d in all_timing_data if d[1] == np_procs]
        sizes_plot = [d[0] for d in data]
        times = [d[1] for d in data]
        
        ax1.plot(sizes_plot, times, color=colors.get(np_procs, 'black'), 
                marker=markers.get(np_procs, 'o'), linewidth=2, markersize=8,
                label=f'{np_procs} процесс(ов)')
    
    ax1.set_xlabel('Размер матрицы (n)', fontsize=12)
    ax1.set_ylabel('Время (секунды)', fontsize=12)
    ax1.set_title('Зависимость времени выполнения от размера матрицы', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    plt.tight_layout()
    plt.savefig('results_mpi/time_plot.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("\n График времени сохранён как 'results_mpi/time_plot.png'")

    # ГРАФИК 2: Ускорение

    if 1 in available_procs:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        base_times = {d[0]: d[2] for d in all_timing_data if d[1] == 1}
        
        for np_procs in available_procs:
            if np_procs == 1:
                continue
            data = [(d[0], d[2]) for d in all_timing_data if d[1] == np_procs]
            sizes_plot = [d[0] for d in data]
            speedup = [base_times.get(s, 1) / t for s, t in zip(sizes_plot, [d[1] for d in data])]
            
            ax2.plot(sizes_plot, speedup, color=colors.get(np_procs, 'black'),
                    marker=markers.get(np_procs, 'o'), linewidth=2, markersize=8,
                    label=f'{np_procs} процесса')

            ax2.axhline(y=np_procs, color=colors.get(np_procs, 'black'), 
                       linestyle='--', alpha=0.3, linewidth=1)
        
        ax2.set_xlabel('Размер матрицы (n)', fontsize=12)
        ax2.set_ylabel('Ускорение', fontsize=12)
        ax2.set_title('Ускорение относительно последовательной версии', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('results_mpi/speedup_plot.png', dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(" График ускорения сохранён как 'results_mpi/speedup_plot.png'")

    # ГРАФИК 3: Эффективность
    if 1 in available_procs:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        base_times = {d[0]: d[2] for d in all_timing_data if d[1] == 1}
        
        for np_procs in available_procs:
            if np_procs == 1:
                continue
            data = [(d[0], d[2]) for d in all_timing_data if d[1] == np_procs]
            sizes_plot = [d[0] for d in data]
            efficiency = [(base_times.get(s, 1) / t / np_procs * 100) for s, t in zip(sizes_plot, [d[1] for d in data])]
            
            ax3.plot(sizes_plot, efficiency, color=colors.get(np_procs, 'black'),
                    marker=markers.get(np_procs, 'o'), linewidth=2, markersize=8,
                    label=f'{np_procs} процесса')
        
        ax3.set_xlabel('Размер матрицы (n)', fontsize=12)
        ax3.set_ylabel('Эффективность (%)', fontsize=12)
        ax3.set_title('Эффективность распараллеливания', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='100%')
        
        plt.tight_layout()
        plt.savefig('results_mpi/efficiency_plot.png', dpi=150, bbox_inches='tight')
        plt.close(fig3)
        print("График эффективности сохранён как 'results_mpi/efficiency_plot.png'")

    # ГРАФИК 4: Производительность в GFLOPS

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    for np_procs in available_procs:
        data = [(d[0], d[3]) for d in all_timing_data if d[1] == np_procs]
        sizes_plot = [d[0] for d in data]
        gflops = [d[1] for d in data]
        
        ax4.plot(sizes_plot, gflops, color=colors.get(np_procs, 'black'),
                marker=markers.get(np_procs, 'o'), linewidth=2, markersize=8,
                label=f'{np_procs} процесс(ов)')
    
    ax4.set_xlabel('Размер матрицы (n)', fontsize=12)
    ax4.set_ylabel('Производительность (GFLOPS)', fontsize=12)
    ax4.set_title('Производительность параллельного умножения матриц', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('results_mpi/gflops_plot.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print(" График производительности сохранён как 'results_mpi/gflops_plot.png'")

    # Сводная таблица

    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 100)
    print(f"{'Размер':<8} | {'Проц.':<6} | {'Время (сек)':<12} | {'GFLOPS':<10} | {'Ускор.':<8} | {'Эфф. %':<8}")
    print("-" * 100)
    
    base_times = {d[0]: d[2] for d in all_timing_data if d[1] == 1} if 1 in available_procs else {}
    
    for size in sizes:
        for np_procs in available_procs:
            data = [d for d in all_timing_data if d[0] == size and d[1] == np_procs]
            if data:
                time_sec = data[0][2]
                gflops = data[0][3]
                speedup = base_times.get(size, time_sec) / time_sec if base_times else 0
                efficiency = (speedup / np_procs * 100) if np_procs > 1 else 100
                
                print(f"{size:<8} | {np_procs:<6} | {time_sec:<12.6f} | {gflops:<10.2f} | {speedup:<8.2f} | {efficiency:<8.1f}")
    
    print("=" * 100)

    with open('results_mpi/summary_table.txt', 'w', encoding='utf-8') as f:
        f.write("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ПАРАЛЛЕЛЬНОГО УМНОЖЕНИЯ МАТРИЦ\n")
        f.write(f"{'Размер':<8} | {'Проц.':<6} | {'Время (сек)':<12} | {'GFLOPS':<10} | {'Ускор.':<8} | {'Эфф. %':<8}\n")
        f.write("-" * 100 + "\n")
        
        for size in sizes:
            for np_procs in available_procs:
                data = [d for d in all_timing_data if d[0] == size and d[1] == np_procs]
                if data:
                    time_sec = data[0][2]
                    gflops = data[0][3]
                    speedup = base_times.get(size, time_sec) / time_sec if base_times else 0
                    efficiency = (speedup / np_procs * 100) if np_procs > 1 else 100
                    
                    f.write(f"{size:<8} | {np_procs:<6} | {time_sec:<12.6f} | {gflops:<10.2f} | {speedup:<8.2f} | {efficiency:<8.1f}\n")
        
    
    print("\n Сводная таблица сохранена как 'results_mpi/summary_table.txt'")

print("\n ВСЕ ОПЕРАЦИИ ЗАВЕРШЕНЫ!")