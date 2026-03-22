import numpy as np
import matplotlib.pyplot as plt
import os

SIZES = [200, 400, 800, 1200, 1600, 2000]

data = []
with open("results/parallel_timing_results.txt", "r") as f:
    lines = f.readlines()
    for line in lines[1:]:
        if line.strip():
            parts = line.strip().split()
            size = int(parts[0])
            threads = int(parts[1])
            time = float(parts[2])
            gflops = float(parts[3])
            speedup = float(parts[4])
            efficiency = float(parts[5])
            data.append((size, threads, time, gflops, speedup, efficiency))

sizes = sorted(set([d[0] for d in data]))
threads = sorted(set([d[1] for d in data]))

print("=" * 70)
print("АНАЛИЗ ПАРАЛЛЕЛЬНЫХ ВЫЧИСЛЕНИЙ (OpenMP)")
print("=" * 70)
print(f"Размеры матриц: {sizes}")
print(f"Количество потоков: {threads}")


# График 1: Время выполнения
plt.figure(figsize=(10, 6))
for t in threads:
    times = [next(d[2] for d in data if d[0] == s and d[1] == t) for s in sizes]
    plt.plot(sizes, times, 'o-', label=f'{t} потоков', linewidth=2, markersize=8)
plt.xlabel('Размер матрицы', fontsize=12)
plt.ylabel('Время (сек)', fontsize=12)
plt.title('Зависимость времени выполнения от размера матрицы', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(sizes, [str(s) for s in sizes])
plt.tight_layout()
plt.savefig('results/parallel_time_plot.png', dpi=150, bbox_inches='tight')
print(" График времени: results/parallel_time_plot.png")
plt.close()

# График 2: Ускорение
plt.figure(figsize=(10, 6))
for t in threads:
    if t == 1:
        continue
    speedups = [next(d[4] for d in data if d[0] == s and d[1] == t) for s in sizes]
    plt.plot(sizes, speedups, 'o-', label=f'{t} потоков', linewidth=2, markersize=8)
plt.plot(sizes, [2]*len(sizes), 'k--', alpha=0.5, linewidth=1.5, label='Идеальное (2)')
plt.plot(sizes, [4]*len(sizes), 'k--', alpha=0.5, linewidth=1.5, label='Идеальное (4)')
plt.plot(sizes, [8]*len(sizes), 'k--', alpha=0.5, linewidth=1.5, label='Идеальное (8)')
plt.xlabel('Размер матрицы', fontsize=12)
plt.ylabel('Ускорение', fontsize=12)
plt.title('Ускорение при разном количестве потоков', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(sizes, [str(s) for s in sizes])
plt.tight_layout()
plt.savefig('results/parallel_speedup_plot.png', dpi=150, bbox_inches='tight')
print(" График ускорения: results/parallel_speedup_plot.png")
plt.close()

# График 3: Эффективность
plt.figure(figsize=(10, 6))
for t in threads:
    if t == 1:
        continue
    efficiencies = [next(d[5] for d in data if d[0] == s and d[1] == t) for s in sizes]
    plt.plot(sizes, efficiencies, 'o-', label=f'{t} потоков', linewidth=2, markersize=8)
plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Идеальная эффективность')
plt.xlabel('Размер матрицы', fontsize=12)
plt.ylabel('Эффективность', fontsize=12)
plt.title('Эффективность параллелизации', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(sizes, [str(s) for s in sizes])
plt.tight_layout()
plt.savefig('results/parallel_efficiency_plot.png', dpi=150, bbox_inches='tight')
print(" График эффективности: results/parallel_efficiency_plot.png")
plt.close()

# График 4: Производительность (GFLOPS)
plt.figure(figsize=(10, 6))
for t in threads:
    gflops = [next(d[3] for d in data if d[0] == s and d[1] == t) for s in sizes]
    plt.plot(sizes, gflops, 'o-', label=f'{t} потоков', linewidth=2, markersize=8)
plt.xlabel('Размер матрицы', fontsize=12)
plt.ylabel('Производительность (GFLOPS)', fontsize=12)
plt.title('Зависимость производительности от размера матрицы', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(sizes, [str(s) for s in sizes])
plt.tight_layout()
plt.savefig('results/parallel_gflops_plot.png', dpi=150, bbox_inches='tight')
print(" График производительности: results/parallel_gflops_plot.png")
plt.close()

with open('results/parallel_full_table.txt', 'w', encoding='utf-8') as f:
    f.write("РЕЗУЛЬТАТЫ ПАРАЛЛЕЛЬНОГО УМНОЖЕНИЯ МАТРИЦ (OpenMP)\n")
    
    for size in sizes:
        size_data = [d for d in data if d[0] == size]

        f.write(f"\n{'='*60}\n")
        f.write(f"РАЗМЕР МАТРИЦЫ: {size} × {size}\n")
        f.write(f"{'='*60}\n")
        f.write(f"{'Потоки':<10} {'Время (с)':<15} {'GFLOPS':<12} {'Ускорение':<12} {'Эффективность':<15}\n")
        f.write("-" * 70 + "\n")
        
        sequential_time = None
        for d in size_data:
            threads, time, gflops, speedup, efficiency = d[1], d[2], d[3], d[4], d[5]
            if threads == 1:
                sequential_time = time
            f.write(f"{threads:<10} {time:<15.6f} {gflops:<12.2f} {speedup:<12.2f} {efficiency:<15.2f}\n")
        
        f.write("-" * 70 + "\n")
        
        if sequential_time:
            f.write(f"\nАнализ:\n")
            f.write(f"  - Последовательное время: {sequential_time:.6f} с\n")
            for d in size_data[1:]:
                threads, time, speedup, efficiency = d[1], d[2], d[4], d[5]
                f.write(f"  - {threads} потоков: ускорение {speedup:.2f}x, эффективность {efficiency:.2f}\n")
        f.write("\n")

print(" Общая таблица сохранена: results/parallel_full_table.txt")

print("\n" + "=" * 80)
print("ТАБЛИЦА РЕЗУЛЬТАТОВ (сводная)")
print("=" * 80)
print(f"{'Размер':^8} | {'Потоки':^6} | {'Время (с)':^10} | {'GFLOPS':^8} | {'Ускорение':^9} | {'Эффективность':^12}")
print("-" * 80)

for d in data:
    print(f"{d[0]:^8} | {d[1]:^6} | {d[2]:^10.4f} | {d[3]:^8.2f} | {d[4]:^9.2f} | {d[5]:^12.2f}")

print("=" * 80)

print("ГЕНЕРАЦИЯ ЗАВЕРШЕНА!")
print("\nСозданные файлы:")
print("  - results/parallel_time_plot.png (время выполнения)")
print("  - results/parallel_speedup_plot.png (ускорение)")
print("  - results/parallel_efficiency_plot.png (эффективность)")
print("  - results/parallel_gflops_plot.png (производительность)")
print("  - results/parallel_full_table.txt (общая таблица со всеми размерами)")