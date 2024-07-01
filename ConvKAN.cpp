#include "ConvKAN.h"
#include "convolution.cpp"
KAN_Convolution::KAN_Convolution(
    std::pair<int64_t, int64_t> kernel_size,
    std::pair<int64_t, int64_t> stride,
    std::pair<int64_t, int64_t> padding,
    std::pair<int64_t, int64_t> dilation,
    int64_t grid_size,
    int64_t spline_order,
    double scale_noise,
    double scale_base,
    double scale_spline,
    torch::nn::SiLU base_activation,
    double grid_eps,
    std::pair<double, double> grid_range,
    std::string device
) :
    grid_size(grid_size),
    spline_order(spline_order),
    kernel_size(kernel_size),
    stride(stride),
    padding(padding),
    dilation(dilation),
    device(device),
    conv(
        register_module(
            "conv",
            KANsLinear(
                kernel_size.first * kernel_size.second,
                1,
                grid_size,
                spline_order,
                scale_noise,
                scale_base,
                scale_spline,
                true,
                base_activation,
                grid_eps,
                grid_range 
            )
        )
    )
{
    
};

torch::Tensor KAN_Convolution::forward(torch::Tensor x) {
   return kan_conv2d(x, conv, kernel_size[0], stride, dilation, padding, device);
}

 def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
    return torch::sum( layer.regularization_loss(regularize_activation, regularize_entropy) for layer in self.layers)
