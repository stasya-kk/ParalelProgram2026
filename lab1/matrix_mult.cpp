#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <iomanip>
using namespace std;
using namespace chrono;

int main() {
    int n, n2, threads;
    double t1, t2;
    
    cout << "Потоков: ";
    cin >> threads;
    omp_set_num_threads(threads);

    ifstream fa("matrix_A.txt");
    if (!fa.is_open()) {
        cout << "Ошибка: не удалось открыть matrix_A.txt" << endl;
        return 1;
    }
    
    fa >> n;
    if (n <= 0) {
        cout << "Ошибка: неверный размер матрицы" << endl;
        return 1;
    }
    
    vector<vector<double>> A(n, vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (!(fa >> A[i][j])) {
                cout << "Ошибка: недостаточно данных в matrix_A.txt" << endl;
                return 1;
            }
        }
    }
    fa.close();

    ifstream fb("matrix_B.txt");
    if (!fb.is_open()) {
        cout << "Ошибка: не удалось открыть matrix_B.txt" << endl;
        return 1;
    }
    
    fb >> n2;
    if (n2 != n) {
        cout << "Ошибка: размеры матриц не совпадают! A=" << n << ", B=" << n2 << endl;
        return 1;
    }
    
    vector<vector<double>> B(n, vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (!(fb >> B[i][j])) {
                cout << "Ошибка: недостаточно данных в matrix_B.txt" << endl;
                return 1;
            }
        }
    }
    fb.close();
    
    vector<vector<double>> C(n, vector<double>(n, 0.0));
    
    cout << "Умножаем " << n << "x" << n << "..." << endl;
    
    t1 = omp_get_wtime();
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    t2 = omp_get_wtime();

    ofstream fout("matrix_C.txt");
    if (!fout.is_open()) {
        cout << "Ошибка: не удалось создать matrix_C.txt" << endl;
        return 1;
    }
    
    fout << n << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fout << C[i][j] << " ";
        }
        fout << endl;
    }
    fout.close();

    cout << "\n========== РЕЗУЛЬТАТЫ ==========" << endl;
    cout << "Размер матриц: " << n << "x" << n << endl;
    cout << "Потоков: " << threads << endl;
    cout << fixed << setprecision(10);
    cout << "Время: " << (t2 - t1) * 1000 << " мс" << endl;
    cout << "Операций: " << n * n * (2 * n - 1) << endl;
    cout << "=================================" << endl;
    
    return 0;
}