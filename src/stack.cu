#include "stack.h"

#include <assert.h>

__host__ __device__
bool Stack::isEmpty() {
    return curr == -1;
}

__host__ __device__
void Stack::push(int v) {
    curr++;
    data[curr] = v;

    assert(curr < STACK_MAX);
    assert(curr >= 0);
}

__host__ __device__
int Stack::pop() {
    assert(!isEmpty());
    
    int d = data[curr];
    curr--;
    return d;
}


