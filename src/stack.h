#pragma once

#define STACK_MAX 50

struct Stack {
    int data[STACK_MAX];
    char curr;

    
    __host__ __device__ void init() {
        curr = -1;
    }

    __host__ __device__ bool isEmpty();
    __host__ __device__ void push(int v);
    __host__ __device__ int pop();
};

