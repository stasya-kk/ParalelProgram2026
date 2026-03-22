#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <string>

using namespace std;
using namespace chrono;

vector<vector<double>> readMatrix(const string& filename, int size) {
    vector<vector<double>> matrix(size, vector<double>(size));
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Ошибка открытия файла: " << filename << endl;
        exit(1);
    }
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            file >> matrix[i][j];
        }
    }
    
    file.close();
    return matrix;
}

// Параллельное умножение матриц с использованием OpenMP
vector<vector<double>> multiplyMatricesParallel(const vector<vector<double>>& A, 
                                                 const vector<vector<double>>& B,
                                                 int num_threads) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));

    omp_set_num_threads(num_threads);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    
    return C;
}

void writeMatrix(const string& filename, const vector<vector<double>>& matrix) {
    ofstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Ошибка создания файла: " << filename << endl;
        exit(1);
    }
    
    for (const auto& row : matrix) {
        for (size_t j = 0; j < row.size(); j++) {
            file << fixed << setprecision(6) << row[j];
            if (j != row.size() - 1) file << " ";
        }
        file << endl;
    }
    
    file.close();
}

double computeGflops(int n, double timeSeconds) {
    double flops = 2.0 * n * n * n;
    return flops / (timeSeconds * 1e9);
}

int main() {
    vector<int> sizes = {200, 400, 800, 1200, 1600, 2000};

    vector<int> threads = {1, 2, 4, 8};

    system("mkdir results");
    
    ofstream resultsFile("results/parallel_timing_results.txt");
    resultsFile << "Размер\tПотоки\tВремя_сек\tGFLOPS\tУскорение\tЭффективность" << endl;
    
    cout << "ПАРАЛЛЕЛЬНОЕ УМНОЖЕНИЕ МАТРИЦ (OpenMP)" << endl;

    for (int size : sizes) {
        cout << "\nРазмер матрицы: " << size << "x" << size << endl;
        cout << "----------------------------------------------------------------------------------" << endl;

        string fileA = "matrices/matrix_A_" + to_string(size) + ".txt";
        string fileB = "matrices/matrix_B_" + to_string(size) + ".txt";
        
        vector<vector<double>> A = readMatrix(fileA, size);
        vector<vector<double>> B = readMatrix(fileB, size);
        
        double sequential_time = 0.0;

        for (int num_threads : threads) {
            auto start = high_resolution_clock::now();
            vector<vector<double>> C = multiplyMatricesParallel(A, B, num_threads);
            auto end = high_resolution_clock::now();
            
            double timeSec = duration<double>(end - start).count();
            double gflops = computeGflops(size, timeSec);
            if (num_threads == 1) {
                sequential_time = timeSec;
            }

            double speedup = sequential_time / timeSec;
            double efficiency = speedup / num_threads;

            cout << "Потоков: " << num_threads 
                 << " | Время: " << fixed << setprecision(6) << timeSec << " с"
                 << " | GFLOPS: " << setprecision(2) << gflops
                 << " | Ускорение: " << setprecision(2) << speedup
                 << " | Эффективность: " << setprecision(2) << efficiency << endl;

            resultsFile << size << "\t" << num_threads << "\t" 
                        << timeSec << "\t" << gflops << "\t"
                        << speedup << "\t" << efficiency << endl;

            if (num_threads == 1) {
                string resultFile = "results/result_" + to_string(size) + ".txt";
                writeMatrix(resultFile, C);
            }
        }
    }
    
    resultsFile.close();

    cout << "Все вычисления завершены. Результаты сохранены в папку results/" << endl;
    
    return 0;
}