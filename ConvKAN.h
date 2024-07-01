#pragma once


#include <torch/torch.h>
#include <utility>
#include <vector>
#include <string>
#include <cmath>
#include "KANsLinear.h"  

class KAN_Convolution : public torch::nn::Module {
public:
    KAN_Convolution(
        std::pair<int64_t, int64_t> kernel_size = {2, 2},
        std::pair<int64_t, int64_t> stride = {1, 1},
        std::pair<int64_t, int64_t> padding = {0, 0},
        std::pair<int64_t, int64_t> dilation = {1, 1},
        int64_t grid_size = 5,
        int64_t spline_order = 3,
        double scale_noise = 0.1,
        double scale_base = 1.0,
        double scale_spline = 1.0,
        torch::nn::SiLU base_activation = torch::nn::SiLU(),
        double grid_eps = 0.02,
        std::pair<double, double> grid_range = {-1.0, 1.0},
        std::string device = "cpu"
    );

    
    // PARAMETERS:
    int64_t grid_size;
    int64_t spline_order;
    std::pair<int64_t, int64_t> kernel_size;
    std::pair<int64_t, int64_t> stride;
    std::pair<int64_t, int64_t> padding;
    std::pair<int64_t, int64_t> dilation;
    std::string device;
    KANLinear conv;


    // FUNCTIONS:
    torch::Tensor forward(torch::Tensor x);
    torch::Tensor regularization_loss(double regularize_activation = 1.0, double regularize_entropy = 1.0)
    
};

   