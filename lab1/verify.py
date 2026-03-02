import numpy as np


def read_matrix(filename):
    """Читает матрицу из файла."""
    with open(filename, 'r', encoding='utf-8') as f:
        n = int(f.readline().strip())
        matrix = []
        for _ in range(n):
            row = list(map(float, f.readline().strip().split()))
            matrix.append(row)
        return np.array(matrix), n


def main():
    """Основная функция программы."""
    print("ПРОВЕРКА РЕЗУЛЬТАТОВ УМНОЖЕНИЯ МАТРИЦ")
    
    try:
        A, n1 = read_matrix("matrix_A.txt")
        B, n2 = read_matrix("matrix_B.txt")
        C, n3 = read_matrix("matrix_C.txt")
        
        print(f"Размер матриц: {n1}x{n1}")
        
        if n1 == n2 == n3:
            correct = np.dot(A, B)
            
            if np.allclose(C, correct, rtol=1e-10, atol=1e-10):
                print("РЕЗУЛЬТАТ ВЕРНЫЙ!")
            else:
                print("ОШИБКА: результаты не совпадают!")
            
            print("\nРезультат умножения (Python):")
            print(correct)
        else:
            print("ОШИБКА: размеры матриц не совпадают!")
            print(f"  A: {n1}, B: {n2}, C: {n3}")
            
    except FileNotFoundError as e:
        print(f"Ошибка: файл не найден - {e.filename}")
    except ValueError as e:
        print(f"Ошибка в данных: {e}")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")


if __name__ == "__main__":
    main()