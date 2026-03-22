import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sizes = [200, 400, 800, 1200, 1600, 2000]

print("ВЕРИФИКАЦИЯ РЕЗУЛЬТАТОВ УМНОЖЕНИЯ МАТРИЦ")

all_passed = True

print("ВЕРИФИКАЦИЯ ВЫЧИСЛЕНИЙ")

for size in sizes:
    print(f"\nПроверка размера {size}×{size}...")

    expected_file = f"matrices/expected_C_{size}.txt"
    result_file = f"results/result_{size}.txt"
    
    if not os.path.exists(expected_file):
        print(f"  Файл {expected_file} не найден!")
        all_passed = False
        continue
        
    if not os.path.exists(result_file):
        print(f"  Файл {result_file} не найден!")
        all_passed = False
        continue

    try:
        expected = np.loadtxt(expected_file)
        result = np.loadtxt(result_file)
    except Exception as e:
        print(f"  Ошибка загрузки: {e}")
        all_passed = False
        continue

    if expected.shape != result.shape:
        print(f"  Несовпадение размеров: expected {expected.shape} vs result {result.shape}")
        all_passed = False
        continue

    abs_diff = np.abs(expected - result)
    max_abs_diff = np.max(abs_diff)

    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = np.abs((expected - result) / (np.abs(expected) + 1e-12))
        max_rel_diff = np.max(rel_diff)

    is_close = np.allclose(expected, result, rtol=1e-6, atol=1e-8)
    
    if is_close:
        print(f"  Верификация пройдена!")
        print(f"  Макс. абсолютная ошибка: {max_abs_diff:.2e}")
        print(f"  Макс. относительная ошибка: {max_rel_diff:.2e}")
    else:
        print(f"  Верификация НЕ пройдена!")
        print(f"  Макс. абсолютная ошибка: {max_abs_diff:.2e}")
        print(f"  Макс. относительная ошибка: {max_rel_diff:.2e}")
        all_passed = False

if all_passed:
    print(" ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!")
else:
    print(" ОБНАРУЖЕНЫ ОШИБКИ ПРИ ВЕРИФИКАЦИИ!")