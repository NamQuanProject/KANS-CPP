#pragma once

#include <torch/torch.h>
#include <iostream>
#include <string>
#include "KAN.h"
#include <cmath>
        
class KAN_Convolutional_Layer : public torch::nn::Module {
public:
    KAN_Convolutional_Layer(
        int64_t n_convs= 1, 
        int64_t kernel_size= 2, 
        int64_t stride= 1, 
        int64_t padding= 0, 
        int64_t dilation= 1, 
        int64_t grid_size= 5,
        int64_t spline_order= 3,
        double scale_noise= 0.1,
        double scale_base= 1.0,
        double scale_spline= 1.0,
        torch::nn::SiLU base_activation= torch::nn::SiLU(),
        double grid_eps= 0.02,
        std::pair<double, double> grid_range = {-1.0, 1.0},
        std::string device = "cpu"
    );
    int64_t n_convs= n_convs;
    int64_t kernel_size= kernel_size;
    int64_t stride= stride;
    int64_t padding= padding;
    int64_t dilation= dilation;
    int64_t grid_size= grid_size;
    int64_t spline_order= spline_order;
    double scale_noise= scale_noise;
    double scale_base= scale_base;
    double scale_spline;
    
    torch::Tensor forward();
};