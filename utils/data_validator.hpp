#ifndef DATA_VALIDATOR_IMPL_HPP
#define DATA_VALIDATOR_IMPL_HPP

#include <iostream>
#include <cstdlib>
#include <ctime>

template <typename T>
class DataValidator {
public:
    DataValidator() {
        srand(time(0));  // Seed the random number generator
    }

    void init_buffer(T* buffer, unsigned long int numberOfElems);
    void validate_data(T* buffer_host, T* buffer_host_out, unsigned long int numberOfElems);
};


template <typename T>
void DataValidator<T>::init_buffer(T* buffer, unsigned long int numberOfElems) {
    for (unsigned long int i = 0; i < numberOfElems; ++i) {
        buffer[i] = static_cast<T>(rand()) / RAND_MAX;
    }
}

template <typename T>
void DataValidator<T>::validate_data(T* buffer_host, T* buffer_host_out, unsigned long int numberOfElems) {   
    for (unsigned long int i = 0; i < numberOfElems; ++i) {
        if (buffer_host[i] != buffer_host_out[i]) {
            std::cout << "Data mismatch at index " << i << ": host = " << buffer_host[i] << ", host_out = " << buffer_host_out[i] << std::endl;
            break;
        }
    }
}

#endif