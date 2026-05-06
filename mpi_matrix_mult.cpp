#include <mpi.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <cmath>

using namespace std;

vector<vector<double> > generateMatrix(int size) {
    vector<vector<double> > matrix(size, vector<double>(size));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = rand() % 100;
        }
    }
    return matrix;
}

void multiplyMPI(const vector<double>& flat_A, const vector<double>& flat_B, 
                 vector<double>& flat_C, int n, int rank, int size) {
    int local_rows = n / size;
    int remainder = n % size;
    
    vector<int> send_counts(size);
    vector<int> displs(size);
    
    for (int i = 0; i < size; ++i) {
        send_counts[i] = (i < remainder) ? (local_rows + 1) * n : local_rows * n;
        displs[i] = (i == 0) ? 0 : displs[i-1] + send_counts[i-1];
    }
    
    vector<double> local_A(send_counts[rank]);
    
    MPI_Scatterv(const_cast<double*>(flat_A.data()), send_counts.data(), 
                 displs.data(), MPI_DOUBLE,
                 local_A.data(), send_counts[rank], MPI_DOUBLE, 
                 0, MPI_COMM_WORLD);
    
    int local_row_count = send_counts[rank] / n;
    vector<double> local_C(local_row_count * n, 0.0);
    
    for (int i = 0; i < local_row_count; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += local_A[i * n + k] * flat_B[k * n + j];
            }
            local_C[i * n + j] = sum;
        }
    }
    
    vector<int> recv_counts(size);
    vector<int> recv_displs(size);
    
    for (int i = 0; i < size; ++i) {
        recv_counts[i] = (i < remainder) ? (local_rows + 1) * n : local_rows * n;
        recv_displs[i] = (i == 0) ? 0 : recv_displs[i-1] + recv_counts[i-1];
    }
    
    MPI_Gatherv(local_C.data(), local_C.size(), MPI_DOUBLE,
                flat_C.data(), recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int n = 500;
    
    if (rank == 0) {
        cout << "Matrix size: " << n << " x " << n << endl;
        cout << "Number of processes: " << size << endl;
        
        vector<vector<double> > A_mat = generateMatrix(n);
        vector<vector<double> > B_mat = generateMatrix(n);
        
        vector<double> flat_A(n * n);
        vector<double> flat_B(n * n);
        vector<double> flat_C(n * n, 0.0);
        
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                flat_A[i * n + j] = A_mat[i][j];
                flat_B[i * n + j] = B_mat[i][j];
            }
        }
        
        MPI_Bcast(flat_B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        double start_time = MPI_Wtime();
        
        multiplyMPI(flat_A, flat_B, flat_C, n, rank, size);
        
        double end_time = MPI_Wtime();
        
        cout << fixed << setprecision(6);
        cout << "Time: " << (end_time - start_time) << " seconds" << endl;
        
        double check_sum = 0.0;
        for (int i = 0; i < n * n; ++i) {
            check_sum += flat_C[i];
        }
        cout << "Result checksum: " << check_sum << endl;
    } else {
        vector<double> flat_B(n * n);
        MPI_Bcast(flat_B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        vector<double> flat_A, flat_C;
        multiplyMPI(flat_A, flat_B, flat_C, n, rank, size);
    }
    
    MPI_Finalize();
    return 0;
}
