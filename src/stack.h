#pragma once

#define STACK_MAX 25

struct Stack {
    int data[STACK_MAX];

    int curr = -1;

    __host__ __device__ bool isEmpty();
    __host__ __device__ void push(int v);
    __host__ __device__ int pop();
};

