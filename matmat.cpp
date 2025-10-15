
// Implements C = C + A·B multiplication for row-major double matrices.
// Tracks Timing and flop-rate reporting across problem sizes from O(10) → O(10^4).
// Performs Loop order experiments (ijk, jik, kij) plus OpenMP and cache blocking.
// Makes Arithmetic intensity estimates and roofline ceilings (160 GB/s, 1 TF/s).
// Gives Straightforward CSV output ready for plotting with roofline.py.
// Build (serial):    g++ -O3 -std=c++17 matmat.cpp -o matmat
// Build (OpenMP):    g++ -O3 -std=c++17 -fopenmp matmat.cpp -o matmat

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using T = double;

namespace {

    constexpr double kPeakBandwidthGBs = 160.0;   // GB/s (from assignment)
    constexpr double kPeakTFLOPs = 1.0;           // TFLOP/s (from assignment)
    constexpr size_t kMaxVerifySize = 512;        // Verify exactly up to this size

    struct BlockSize {
        size_t n;
        size_t m;
        size_t k;
    };

// We experiment with the five kernel variants explicitly listed in the
// assignment prompt: three loop orders, a blocked kernel, and an OpenMP build.
    enum class Algorithm { IJK, JIK, KIJ, BLOCKED, OMP_IJK };

    const char* to_string(Algorithm algo) {
        switch (algo) {
            case Algorithm::IJK:     return "ijk";
            case Algorithm::JIK:     return "jik";
            case Algorithm::KIJ:     return "kij";
            case Algorithm::BLOCKED: return "blocked";
            case Algorithm::OMP_IJK: return "omp_ijk";
        }
        return "unknown";
    }

// Row-major helper: convert a logical (i, j) entry into a flat index using the
// number of columns (``leading dimension'') as the stride. Having this helper
// keeps every kernel readable.
    inline size_t idx(size_t i, size_t j, size_t ld) { return i * ld + j; }

/* Kernel implementations */

// Baseline triple loop with order i --> j --> k as shown in hw.tex.
void multiply_ijk(const T* A, const T* B, T* C,
                  size_t N, size_t M, size_t K) {
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < M; ++j) {
            T sum = C[idx(i, j, M)];
            for (size_t k = 0; k < K; ++k)
                sum += A[idx(i, k, K)] * B[idx(k, j, M)];
            C[idx(i, j, M)] = sum;
        }
}

// Loop re-ordering: swap the outer two loops (j --> i --> k).
void multiply_jik(const T* A, const T* B, T* C,
                  size_t N, size_t M, size_t K) {
    for (size_t j = 0; j < M; ++j)
        for (size_t i = 0; i < N; ++i) {
            T sum = C[idx(i, j, M)];
            for (size_t k = 0; k < K; ++k)
                sum += A[idx(i, k, K)] * B[idx(k, j, M)];
            C[idx(i, j, M)] = sum;
        }
}

// Loop re-ordering: accumulate along k first to highlight read/write reuse.
void multiply_kij(const T* A, const T* B, T* C,
                  size_t N, size_t M, size_t K) {
    for (size_t k = 0; k < K; ++k)
        for (size_t i = 0; i < N; ++i) {
            const T aik = A[idx(i, k, K)];
            for (size_t j = 0; j < M; ++j)
                C[idx(i, j, M)] += aik * B[idx(k, j, M)];
        }
}

// Cache blocking (tiling) over all three dimensions. For each tile we execute
// the familiar ijk pattern but on a much smaller working set that fits better
// in the cache hierarchy. Tile sizes are parameters so we can experiment.
void multiply_blocked(const T* A, const T* B, T* C,
                      size_t N, size_t M, size_t K,
                      const BlockSize& tile) {
    for (size_t ii = 0; ii < N; ii += tile.n) {
        const size_t i_end = std::min(N, ii + tile.n);
        for (size_t jj = 0; jj < M; jj += tile.m) {
            const size_t j_end = std::min(M, jj + tile.m);
            for (size_t kk = 0; kk < K; kk += tile.k) {
                const size_t k_end = std::min(K, kk + tile.k);
                for (size_t i = ii; i < i_end; ++i)
                    for (size_t j = jj; j < j_end; ++j) {
                        T sum = C[idx(i, j, M)];
                        for (size_t k = kk; k < k_end; ++k)
                            sum += A[idx(i, k, K)] * B[idx(k, j, M)];
                        C[idx(i, j, M)] = sum;
                    }
            }
        }
    }
}

// Shared-memory parallel implementation: identical arithmetic to ijk but
// split across OpenMP threads. Collapsing two loops improves load balance.
void multiply_omp_ijk(const T* A, const T* B, T* C,
                      size_t N, size_t M, size_t K, int threads) {
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
    #pragma omp parallel for collapse(2) schedule(static)
    for (long long i = 0; i < static_cast<long long>(N); ++i)
        for (long long j = 0; j < static_cast<long long>(M); ++j) {
            T sum = C[idx(static_cast<size_t>(i), static_cast<size_t>(j), M)];
            for (long long k = 0; k < static_cast<long long>(K); ++k)
                sum += A[idx(static_cast<size_t>(i), static_cast<size_t>(k), K)]
                     * B[idx(static_cast<size_t>(k), static_cast<size_t>(j), M)];
            C[idx(static_cast<size_t>(i), static_cast<size_t>(j), M)] = sum;
        }
#else
    (void)threads;
    multiply_ijk(A, B, C, N, M, K);
#endif
}

// Helper routines for metrics and correctness checks

double flop_count(size_t N, size_t M, size_t K) {
return 2.0 * static_cast<double>(N) * static_cast<double>(M) * static_cast<double>(K);
}

double byte_estimate(size_t N, size_t M, size_t K) {
const double elems = static_cast<double>(N) * K
                     + static_cast<double>(K) * M
                     + 2.0 * static_cast<double>(N) * M;
return elems * sizeof(T);
}

bool approx_equal(const std::vector<T>& X, const std::vector<T>& Y, double tol = 1e-9) {
    for (size_t i = 0; i < X.size(); ++i) {
        const double target = static_cast<double>(Y[i]);
        const double diff = std::abs(static_cast<double>(X[i]) - target);
        if (diff > tol * (1.0 + std::abs(target))) return false;
    }
    return true;
}

struct BenchmarkResult {
    size_t size;      // We use square matrices (N = M = K = size) for simplicity
    Algorithm algo;   // Which kernel variant ran
    int threads;      // Thread count (always 1 for serial kernels)
    double seconds;   // Wall-clock runtime
    double gflops;    // Achieved GFLOP/s
    double tflops;    // Achieved TFLOP/s
    double ai;        // Arithmetic intensity (flop/byte)
    double roof_gf;   // Roofline ceilings for quick comparison
    double roof_tf;
    bool verified;    // True if result matches reference (when available)
};

std::vector<T> make_random_matrix(size_t rows, size_t cols,
        std::mt19937& rng,
std::uniform_real_distribution<T>& dist) {
std::vector<T> matrix(rows * cols);
for (auto& x : matrix) x = dist(rng);
return matrix;
}

std::vector<T> make_reference(const std::vector<T>& A,
                              const std::vector<T>& B,
                              const std::vector<T>& C0,
                              size_t N, size_t M, size_t K) {
    std::vector<T> reference = C0;
    multiply_ijk(A.data(), B.data(), reference.data(), N, M, K);
    return reference;
}

BenchmarkResult run_benchmark(Algorithm algo,
                              size_t N, size_t M, size_t K,
                              const std::vector<T>& A,
                              const std::vector<T>& B,
                              const std::vector<T>& C0,
                              const std::vector<T>& reference,
                              const BlockSize& tile,
                              int threads) {
    std::vector<T> C = C0;

    // Time the selected kernel variant.
    const auto t0 = std::chrono::steady_clock::now();
    switch (algo) {
        case Algorithm::IJK:
            multiply_ijk(A.data(), B.data(), C.data(), N, M, K);
            break;
        case Algorithm::JIK:
            multiply_jik(A.data(), B.data(), C.data(), N, M, K);
            break;
        case Algorithm::KIJ:
            multiply_kij(A.data(), B.data(), C.data(), N, M, K);
            break;
        case Algorithm::BLOCKED:
            multiply_blocked(A.data(), B.data(), C.data(), N, M, K, tile);
            break;
        case Algorithm::OMP_IJK:
            multiply_omp_ijk(A.data(), B.data(), C.data(), N, M, K, threads);
            break;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();

    // Derive performance metrics directly from definitions in hw.tex.
    const double flops = flop_count(N, M, K);
    const double gflops = flops / seconds / 1e9;
    const double tflops = flops / seconds / 1e12;
    const double ai = flops / byte_estimate(N, M, K);
    const double roof_gf = std::min(kPeakBandwidthGBs * ai, kPeakTFLOPs * 1e3);
    const double roof_tf = roof_gf / 1e3;

    bool ok = true;
    if (!reference.empty()) ok = approx_equal(C, reference);

    return BenchmarkResult{N, algo, threads, seconds, gflops, tflops, ai, roof_gf, roof_tf, ok};
}

std::vector<size_t> default_sizes() {
    return {32, 64, 128, 256, 512, 1024, 1536, 2048};
}

std::vector<Algorithm> default_algorithms() {
    return {Algorithm::IJK, Algorithm::JIK, Algorithm::KIJ, Algorithm::BLOCKED, Algorithm::OMP_IJK};
}

std::vector<int> omp_thread_counts() {
#ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
    std::vector<int> counts = {1, 2, 4, 8};
    counts.erase(std::remove_if(counts.begin(), counts.end(),
                                [max_threads](int t) { return t > max_threads; }),
                 counts.end());
    if (counts.empty() || counts.front() != 1) counts.insert(counts.begin(), 1);
    counts.erase(std::unique(counts.begin(), counts.end()), counts.end());
    return counts;
#else
    return {1};
#endif
}

void print_header() {
    std::cout << "size,algo,threads,seconds,GF/s,TF/s,AI,RoofGF/s,RoofTF/s\n";
}

void emit_result(const BenchmarkResult& r) {
    std::cout << std::fixed << std::setprecision(6)
              << r.size << ',' << to_string(r.algo) << ',' << r.threads << ','
              << r.seconds << ',' << r.gflops << ',' << r.tflops << ','
              << r.ai << ',' << r.roof_gf << ',' << r.roof_tf << '\n';
}

// Friendly reminder if the binary was compiled without OpenMP support.
void ensure_openmp_notice() {
#ifndef _OPENMP
    static bool warned = false;
    if (!warned) {
        std::cerr << "[note] Program compiled without OpenMP; omp_ijk runs serially.\n";
        warned = true;
    }
#endif
}

} // namespace

int main() {
    const auto sizes = default_sizes();
    const auto algos = default_algorithms();
    const BlockSize tile{128, 128, 128};
    const auto threads_list = omp_thread_counts();

    std::cerr << "Matrix–Matrix Multiplication Benchmarks\n"
              << "  • Matrices use row-major doubles with C ← C + A·B\n"
              << "  • Random data seeded for reproducibility\n"
              << "  • Results printed as CSV for roofline.py\n"
              << "  • Reference check performed up to size " << kMaxVerifySize << '\n';

    print_header();

    std::mt19937 rng(12345);
    std::uniform_real_distribution<T> dist(-1.0, 1.0);

    for (size_t N : sizes) {
        const size_t M = N;
        const size_t K = N;

        // Generate fresh random matrices for the current problem size.
        const std::vector<T> A = make_random_matrix(N, K, rng, dist);
        const std::vector<T> B = make_random_matrix(K, M, rng, dist);
        const std::vector<T> C0 = make_random_matrix(N, M, rng, dist);

        std::vector<T> reference;
        if (N <= kMaxVerifySize) {
            reference = make_reference(A, B, C0, N, M, K);
        }

        for (Algorithm algo : algos) {
            if (algo == Algorithm::OMP_IJK) ensure_openmp_notice();

            const std::vector<int> threads_to_use =
                    (algo == Algorithm::OMP_IJK) ? threads_list : std::vector<int>{1};

            for (int threads : threads_to_use) {
                const auto result = run_benchmark(algo, N, M, K, A, B, C0, reference, tile, threads);
                emit_result(result);

                if (!result.verified && !reference.empty()) {
                    std::cerr << "[warn] Verification failed for "
                              << to_string(algo) << " at size " << N
                              << " (threads=" << threads << ")\n";
                }
            }
        }
    }

    return 0;
}
