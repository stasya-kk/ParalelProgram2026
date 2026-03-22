#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
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

vector<vector<double>> multiplyMatrices(const vector<vector<double>>& A, 
                                         const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));
    
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double aik = A[i][k];
            for (int j = 0; j < n; j++) {
                C[i][j] += aik * B[k][j];
            }
        }
    }
    
    return C;
}

double computeGflops(int n, double timeSeconds) {
    double flops = 2.0 * n * n * n;
    return flops / (timeSeconds * 1e9);
}

int main() {
    vector<int> sizes = {200, 400, 800, 1200, 1600, 2000};
    
    system("mkdir -p results");
    
    cout << "--------------------------------------------------------------" << endl;
    cout << "Размер матрицы\tВремя (сек)\tПроизводительность (гигафлопс)" << endl;
    cout << "--------------------------------------------------------------" << endl;

    ofstream resultsFile("results/timing_results.txt");
    resultsFile << "Размер\tВремя_сек\tГигафлопс" << endl;
    
    for (int size : sizes) {
        cout << size << "\t\t";

        string fileA = "matrices/matrix_A_" + to_string(size) + ".txt";
        string fileB = "matrices/matrix_B_" + to_string(size) + ".txt";
        
        vector<vector<double>> A = readMatrix(fileA, size);
        vector<vector<double>> B = readMatrix(fileB, size);

        auto start = high_resolution_clock::now();
        vector<vector<double>> C = multiplyMatrices(A, B);
        auto end = high_resolution_clock::now();
        
        double timeSec = duration<double>(end - start).count();
        double gflops = computeGflops(size, timeSec);

        cout << fixed << setprecision(6) << timeSec << "\t\t";
        cout << fixed << setprecision(2) << gflops << endl;

        string resultFile = "results/result_" + to_string(size) + ".txt";
        writeMatrix(resultFile, C);

        resultsFile << size << "\t" << timeSec << "\t" << gflops << endl;
    }
    
    resultsFile.close();
    
    cout << "--------------------------------------------------------------" << endl;
    cout << "Все вычисления завершены. Результаты сохранены в папку results/" << endl;
    
    return 0;
}