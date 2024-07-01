#include "ConvKANLayer.h"



KAN_Convolutional_Layer::KAN_Convolutional_Layer(
    int64_t n_convs, 
    int64_t kernel_size, 
    int64_t stride, 
    int64_t padding, 
    int64_t dilation, 
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
    n_convs(n_convs),
    
{

}