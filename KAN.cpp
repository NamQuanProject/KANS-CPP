#include "KAN.h"


KANImpl::KANImpl(
   std::vector<int64_t> layers_hidden,
   int64_t grid_size,
   int64_t spline_order,
   double scale_noise,
   double scale_base,
   double scale_spline,
   torch::nn::SiLU base_activation,
   double grid_eps,
   std::pair<double, double> grid_range)
:
   grid_size(grid_size),
   spline_order(spline_order) {


   for (size_t i = 0; i < layers_hidden.size() - 1; ++i) {
       layers.push_back(
           register_module( "layer" + std::to_string(i),
               KANLinear(
                   layers_hidden[i],
                   layers_hidden[i + 1],
                   grid_size,
                   spline_order,
                   scale_noise,
                   scale_base,
                   scale_spline,
                   true,
                   base_activation,
                   grid_eps,
                   grid_range)
           )
       );
   }
}


torch::Tensor KANImpl::forward(torch::Tensor x, bool update_grid) {
   for (auto& layer : layers) {
       if (update_grid) {
           layer->update_grid(x);
       }
       x = layer->forward(x);
   }
   return x;
}


torch::Tensor KANImpl::regularization_loss(double regularize_activation, double regularize_entropy) {
   torch::Tensor loss = torch::zeros({1});
   for (auto layer : layers) {
       loss += layer->regularization_loss(regularize_activation, regularize_entropy);
   }
   return loss;
}
