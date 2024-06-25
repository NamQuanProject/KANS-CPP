#pragma once

#include <iostream>
#include <torch/torch.h>

class KANLinearImpl : public torch::nn::Module {
public:
    KANLinearImpl(
        int64_t in_features,
        int64_t out_features,
        int64_t grid_size = 5,
        int64_t spline_order = 3,
        double scale_noise = 0.1,
        double scale_base = 1.0,
        double scale_spline = 1.0,
        bool enable_standalone_scale_spline = true,
        torch::nn::SiLU base_activation = torch::nn::SiLU(),
        double grid_eps = 0.02,
        std::pair<double, double> grid_range = {-1.0, 1.0}
    );

    void reset_parameters();
    torch::Tensor b_splines(torch::Tensor x);
    torch::Tensor curve2coeff(torch::Tensor x, torch::Tensor y);
    torch::Tensor forward(torch::Tensor x);
    void update_grid(torch::Tensor x, double margin = 0.01);
    torch::Tensor regularization_loss(double regularize_activation = 1.0, double regularize_entropy = 1.0);
    torch::Tensor scaled_spline_weight();

    int64_t in_features;
    int64_t out_features;
    int64_t grid_size;
    int64_t spline_order;
    double scale_noise;
    double scale_base;
    double scale_spline;
    bool enable_standalone_scale_spline;
    torch::nn::SiLU base_activation;
    double grid_eps;

    torch::Tensor grid;
    torch::Tensor base_weight;
    torch::Tensor spline_weight;
    torch::Tensor spline_scaler;
};

TORCH_MODULE(KANLinear);
