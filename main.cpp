#include <torch/torch.h>
#include <iostream>
#include "KANsLinear.cpp" // Ensure this includes the definitions for KANLinear and KANLinearImpl

int main() {
   // Define input and output features
   int64_t in_features = 2;
   int64_t out_features = 1;

   std::cout << "CHECKING: ..." << std::endl;

   KANLinear kan_linear(in_features, out_features);

   torch::Tensor input = torch::randn({5, in_features});

   torch::Tensor output = kan_linear->forward(input);

   std::cout << "Input: " << input << std::endl;
   std::cout << "Output: " << output << std::endl;

   kan_linear->update_grid(input);

   torch::Tensor reg_loss = kan_linear->regularization_loss(1, 0);

   auto u = input.index({torch::indexing::Slice(), 0});
   auto v = input.index({torch::indexing::Slice(), 1});

   torch::Tensor loss = torch::nn::functional::mse_loss(output.squeeze(-1), (u + v) / (1 + u * v));

   auto total_loss = loss + 1e-5 * reg_loss;
   std::cout << "Total Loss: " << total_loss << std::endl;

   return 0;
}
