#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <string>
#include <mpi.h>

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

// Параллельное умножение матриц с использованием MPI
vector<vector<double>> multiplyMatricesMPI(const vector<vector<double>>& A, 
                                            const vector<vector<double>>& B,
                                            int rank, int size) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));

    int rows_per_process = n / size;
    int remainder = n % size;
    
    int start_row = rank * rows_per_process + min(rank, remainder);
    int end_row = start_row + rows_per_process + (rank < remainder ? 1 : 0);

    for (int i = start_row; i < end_row; i++) {
        for (int k = 0; k < n; k++) {
            double aik = A[i][k];
            for (int j = 0; j < n; j++) {
                C[i][j] += aik * B[k][j];
            }
        }
    }

    if (rank == 0) {
        for (int proc = 1; proc < size; proc++) {
            int proc_start = proc * rows_per_process + min(proc, remainder);
            int proc_end = proc_start + rows_per_process + (proc < remainder ? 1 : 0);
            
            for (int i = proc_start; i < proc_end; i++) {
                MPI_Recv(C[i].data(), n, MPI_DOUBLE, proc, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        for (int i = start_row; i < end_row; i++) {
            MPI_Send(C[i].data(), n, MPI_DOUBLE, 0, i, MPI_COMM_WORLD);
        }
    }
    
    return C;
}

double computeGflops(int n, double timeSeconds) {
    double flops = 2.0 * n * n * n;
    return flops / (timeSeconds * 1e9);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    vector<int> sizes = {200, 400, 800, 1200, 1600, 2000};
    
    if (rank == 0) {
        system("mkdir -p results_mpi");
        
        cout << "ПАРАЛЛЕЛЬНОЕ УМНОЖЕНИЕ МАТРИЦ С ИСПОЛЬЗОВАНИЕМ MPI" << endl;
        cout << "Количество процессов: " << num_procs << endl;
        cout << "Размер матрицы | Время (сек) | Производительность (GFLOPS)" << endl;
        cout << "------------------------------------------------------------------" << endl;
    }

    ofstream resultsFile;
    if (rank == 0) {
        string filename = "results_mpi/timing_results_" + to_string(num_procs) + "proc.txt";
        resultsFile.open(filename);
        resultsFile << "Размер\tПроцессов\tВремя_сек\tГигафлопс" << endl;
    }
    
    for (int size : sizes) {
        string fileA = "matrices/matrix_A_" + to_string(size) + ".txt";
        string fileB = "matrices/matrix_B_" + to_string(size) + ".txt";

        vector<vector<double>> A = readMatrix(fileA, size);
        vector<vector<double>> B = readMatrix(fileB, size);

        MPI_Barrier(MPI_COMM_WORLD);

        double start_time = MPI_Wtime();

        vector<vector<double>> C = multiplyMatricesMPI(A, B, rank, num_procs);

        MPI_Barrier(MPI_COMM_WORLD);
        
        double end_time = MPI_Wtime();
        
        if (rank == 0) {
            double timeSec = end_time - start_time;
            double gflops = computeGflops(size, timeSec);
            
            cout << setw(12) << size << " | " 
                 << fixed << setprecision(6) << setw(10) << timeSec << " | "
                 << fixed << setprecision(2) << setw(22) << gflops << endl;

            string resultFile = "results_mpi/result_" + to_string(size) + "_" + to_string(num_procs) + "proc.txt";
            writeMatrix(resultFile, C);

            resultsFile << size << "\t" << num_procs << "\t" << timeSec << "\t" << gflops << endl;
        }
    }
    
    if (rank == 0) {
        resultsFile.close();

        cout << "Вычисления завершены. Результаты сохранены в папку results_mpi/" << endl;
    }
    
    MPI_Finalize();
    return 0;
}