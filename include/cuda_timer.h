#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

struct Event {
    std::string name;
    cudaEvent_t event;
};

struct CudaTimer {
    std::vector<Event> events;

    CudaTimer() = default;

    ~CudaTimer();
    void record(const std::string& name);
    void report();
    void clean();
    float get_elapsed(const std::string& from, const std::string& to);


    auto findEvent(const std::string& name) {
        return std::find_if(events.begin(), events.end(),
            [&](const Event& e) { return e.name == name; });
    }

};