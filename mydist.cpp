#include <immintrin.h>
#include <iostream>
#include <string.h>
#include <sys/time.h>
#include <chrono>
#pragma pack(16)

using namespace std;

class timer{
public:
    timer(string name){
        name_ = name;
        last_time_ =  std::chrono::high_resolution_clock::now();
    }

    ~timer(){
        long det_time = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - last_time_).count();
        printf("name: %s, consumer times: %f ms\n", name_.c_str(), (float)det_time/1000);
    }

private:
    chrono::high_resolution_clock::time_point last_time_;
    string name_;
};

float estimate(string name, const float *A, const float *B, int Dim, float (*func)(const float *a, const float *b, int dim)){
    timer time = timer(name);
    float temp;
    for(int i = 0; i < 100; ++i)
        temp = func(A, B, Dim);
    return temp;
}

float getDistanceCPUNaive(const float *a, const float *b, int dim){
    float dis = 0.0f;
    float temp;
    for(int i = 0; i < dim; ++i){
        temp = a[i] - b[i];
        dis += temp * temp;
    }
    return sqrt(dis);
}

float getDistanceCPULoopExpansion(const float *a, const float *b, int dim){
    float result[4] = {0.0f};
    float temp[4];

    for(int i = 0; i < dim; i += 4){
        for(int ii = 0; ii < 4; ++ii){
            temp[ii] = a[i + ii] - b[i + ii];
            result[ii] += temp[ii] * temp[ii];
        }
    }
    for(int ii = 1; ii < 4; ++ii){
        result[0] += result[ii];
    }
    return sqrt(result[0]);
}

float inline reduceM128(__m128 r){
    float f[4] __attribute__((aligned(16)));
    _mm_store_ps(f, r);
    return (f[0] + f[1]) + (f[2] + f[3]);
}

float inline reduceM256(__m256 r){
    __m128 h = _mm256_extractf128_ps(r, 1);
    __m128 l = _mm256_extractf128_ps(r, 0);
    h = _mm_add_ps(h, l);
    return reduceM128(h);
}

float getDistanceCPUAVX(const float *a, const float *b, int dim){
    int step = dim / 8;
    __m256* one = (__m256*)a;
    __m256* two = (__m256*)b;

    __m256 result = _mm256_setzero_ps();
    __m256 temp;

    for(int i = 0; i < step; ++i){
        temp = _mm256_sub_ps(one[i], two[i]);
        temp = _mm256_mul_ps(temp, temp);
        result = _mm256_add_ps(temp, result);
    }
    float r = reduceM256(result);
    for(int i = 8 * step; i < dim; ++i){
        r += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(r);
}

float getDistanceCPUAVXUnroll(const float *a, const float *b, int dim){
    int step = dim / (8 * 4);
    __m256* one = (__m256*)a;
    __m256* two = (__m256*)b;

    __m256 result[4] = {_mm256_setzero_ps()};
    for(int i = 0; i < step; ++i){
        for(int j = 0; j < 4; ++j){
            __m256 temp = _mm256_sub_ps(one[j + 4 * i], two[j + 4 * i]);
            temp = _mm256_mul_ps(temp, temp);
            result[j] = _mm256_add_ps(temp, result[j]);
        }
    }
    for(int i = 1; i < 4; ++i){
        result[0] = _mm256_add_ps(result[0], result[i]);
    }
    float r = reduceM256(result[0]);
    for(int i = (8 * 4) * step; i < dim; ++i){
        r += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(r);
}

int main() {

    int dim = 4000;
    float *a = new float[dim];
    float *b = new float[dim];

    for(int i = 0; i < dim; ++i){
        a[i] = 2.0f;
        b[i] = 0.0f;
    }

    float res = estimate("naive", a, b, dim, getDistanceCPUNaive);
    cout << res << endl;

    float res1 = estimate("LoopExpansion", a, b, dim, getDistanceCPULoopExpansion);
    cout << res1 << endl;

    float res2 = estimate("avx", a, b, dim, getDistanceCPUAVX);
    cout << res2 << endl;

    float res3 = estimate("avxLoopExpansion", a, b, dim, getDistanceCPUAVXUnroll);
    cout << res3 << endl;


    delete[] a; a = nullptr;
    delete[] b; b = nullptr;

    cout << "Hello World!" << endl;
    return 0;
}

