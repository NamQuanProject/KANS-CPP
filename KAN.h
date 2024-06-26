#include <iostream>
#include <torch/torch.h>
#include "KANsLinear.cpp"
#include <vector>


class KANImpl : public torch::nn::Module {
public:
   KANImpl(
       std::vector<int64_t> layers_hidden,
       int64_t grid_size=5,
       int64_t spline_order=3,
       double scale_noise=0.1,
       double scale_base=1.0,
       double scale_spline=1.0,
       torch::nn::SiLU base_activation = torch::nn::SiLU(),
       double grid_eps=0.02,
       std::pair<double, double> grid_range = {-1.0, 1.0}
   );
   torch::Tensor forward(torch::Tensor x, bool update_grid = false);
   torch::Tensor regularization_loss(double regularize_activation = 1.0, double regularize_entropy = 1.0);
   std::vector<KANLinear> layers;
   int64_t grid_size;
   int64_t spline_order;
};


TORCH_MODULE(KAN);
