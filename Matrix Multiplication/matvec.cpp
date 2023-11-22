#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <pmmintrin.h>
#include <timeutil.h>

/*
#ifndef __SSE3__
#error This example requires SSE3
#endif
*/


/* Size of the matrices to multiply */
#define SIZE2 14
#define SIZE (1 << SIZE2)
// #define SIZE 4

#define MINDEX(n, m) (((n) << SIZE2) | (m))

#define XMM_ALIGNMENT_BYTES 16

static float *mat_a __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float *vec_b __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float *vec_c __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float *vec_ref __attribute__((aligned (XMM_ALIGNMENT_BYTES)));

static void
matvec_intrinsics()
{
        /* Ascoine that the data size is an even multiple of the 128 bit
         * SSE vectors (i.e. 4 floats) */
        assert(!(SIZE & 0x3));

        /* HINT: Read the documentation about the following. You might find at least the following instructions
         * useful:
         *  - _mm_setzero_ps
         *  - _mm_load_ps
         *  - _mm_hadd_ps
         *  - _mm_mul_ps
         *  - _mm_cvtss_f32
         *
         * HINT: You can create the coin of all elements in a vector
         * using two hadd instructions.
         */

        /* Implement your SSE version of the matrix-vector
         * multiplication here.
         */


        /* ROUGH WORK KINDLY IGNORE
        __m128 q,w,e,r;
        int j=0;
        printf("A matrix getting\n");
        for (int i=0; i<8;i++)
        {
                for(int j=0; j<8;j+=4)
                {
                        q = _mm_load_ps(&mat_a[MINDEX(i,j)]);
                        w = _mm_load_ps(&mat_a[MINDEX(i,j+1)]);
                        e = _mm_load_ps(&mat_a[MINDEX(i,j+2)]);
                        r = _mm_load_ps(&mat_a[MINDEX(i,j+3)]);
                        a_row = _mm_load_ps(&q[0],&w[0],&e[0],&r[0]);
                        // b_column = _mm_load_ps(&vec_b[j]);
                        printf("%f  %f  %f  %f ", a_row[0], a_row[1], a_row[2], a_row[3]);
                        // printf("b vale %f%f%f%f\n", b_column[0], b_column[1], b_column[2], b_column[3]);
                }
                printf("\n");
        }
        printf("Required A matrixe\n");
        for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++)
                        printf("%f ", ((7 * i + j) & 0x0F) * 0x1P-2F);
                // printf("B row value %f\n",((i * 17) & 0x0F) * 0x1P-2F);
                printf("\n");
        }

        printf("B vector getting\n");
                for(int j=0; j<12;j+=4)
                {
                        // a_row = _mm_load_ps(&mat_a[MINDEX(i,j)]);
                        b_column = _mm_load_ps(&vec_b[j]);
                        // printf("%f%f%f%f\n", a_row[0], a_row[1], a_row[2], a_row[3]);
                        printf("%f %f %f %f\n", b_column[0], b_column[1], b_column[2], b_column[3]);
                }
        
        printf("Required B vector\n");
        for (int i = 0; i < 12; i++) {
                printf("%f \n",((i * 17) & 0x0F) * 0x1P-2F);
                // printf("\n");
        }
        */

        __m128 a_row;                   //for accessing elements of row-wise elements of mat_b
        __m128 b_column;                //for accessing vec_b elements
        __m128 coin_bag;                //for storing multiplied values
        __m128 coin;                    //for adding multiplied values
        for (int i = 0; i < SIZE; i++) 
        {
                coin = _mm_setzero_ps();
                for (int j = 0; j < SIZE; j +=4)
                {
                        a_row = _mm_load_ps(&mat_a[MINDEX(i,j)]);
                        b_column = _mm_load_ps(&vec_b[j]);
                        coin_bag = _mm_mul_ps(a_row, b_column);         //multiply 4 elements at a time
                        coin = _mm_add_ps(coin, coin_bag);              //accumulate them         
                }

                coin = _mm_hadd_ps(coin, coin);                 //twice hadd sums up the vector!
                coin = _mm_hadd_ps(coin, coin);
                vec_c[i] = _mm_cvtss_f32(coin);                 //store our coin in its final place
        }
}

/**
 * Reference implementation of the matvec used to verify the answer. Do NOT touch this function.
 */

static void
matvec_ref()
{
        int i, j;

        for (i = 0; i < SIZE; i++)
                for (j = 0; j < SIZE; j++)
                        vec_ref[i] += mat_a[MINDEX(i, j)] * vec_b[j];
}

/**
 * Function used to verify the result. Do not touch this function.
 */
static int
verify_result()
{
        float e_coin;
        int i;

        e_coin = 0;
        for (i = 0; i < SIZE; i++) {
                e_coin += vec_c[i] < vec_ref[i] ?
                        vec_ref[i] - vec_c[i] :
                        vec_c[i] - vec_ref[i];
        }

        printf("e_coin: %.e\n", e_coin);

        return e_coin < 1E-6;
}

/**
 * Initialize mat_a and vec_b with "random" data. Write to every
 * element in mat_c to make sure that the kernel allocates physical
 * memory to every page in the matrix before we start doing
 * benchmarking.
 */
static void
init()
{
        int i, j;

        mat_a = (float *)_mm_malloc(sizeof(*mat_a) * SIZE * SIZE, XMM_ALIGNMENT_BYTES);
        vec_b = (float *)_mm_malloc(sizeof(*vec_b) * SIZE, XMM_ALIGNMENT_BYTES);
        vec_c = (float *)_mm_malloc(sizeof(*vec_c) * SIZE, XMM_ALIGNMENT_BYTES);
        vec_ref = (float *)_mm_malloc(sizeof(*vec_ref) * SIZE, XMM_ALIGNMENT_BYTES);

        if (!mat_a || !vec_b || !vec_c || !vec_ref) {
                fprintf(stderr, "Memory allocation failed\n");
                abort();
        }

        for (i = 0; i < SIZE; i++) {
                for (j = 0; j < SIZE; j++)
                        mat_a[MINDEX(i, j)] = ((7 * i + j) & 0x0F) * 0x1P-2F;
                vec_b[i] = ((i * 17) & 0x0F) * 0x1P-2F;
        }

        memset(vec_c, 0, sizeof(float)*sizeof(vec_c));
        memset(vec_ref, 0, sizeof(float)*sizeof(vec_ref));
}

static void
run_multiply()
{
        struct timespec ts_start, ts_stop;
        double runtime_ref, runtime_sse;

        get_time_now(&ts_start);
        /* vec_c = mat_a * vec_b */
        matvec_intrinsics();
        get_time_now(&ts_stop);
        runtime_sse = get_time_diff(&ts_start, &ts_stop);
        printf("Matvec using intrinsics completed in %.2f s\n",
               runtime_sse);

        get_time_now(&ts_start);
        matvec_ref();
        get_time_now(&ts_stop);
        runtime_ref = get_time_diff(&ts_start, &ts_stop);
        printf("Matvec reference code completed in %.2f s\n",
               runtime_ref);

        printf("Speedup: %.2f\n",
               runtime_ref / runtime_sse);


        if (verify_result())
            printf("Result OK\n");
        else
            printf("Result MISMATCH\n");
            }

int
main(int argc, char *argv[])
{
        /* Initialize the matrices with some "random" data. */
        init();

        run_multiply();

        _mm_free(mat_a);
        _mm_free(vec_b);
        _mm_free(vec_c);
        _mm_free(vec_ref);

        return 0;
}