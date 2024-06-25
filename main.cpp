#include <torch/torch.h>
#include <iostream>
#include "KANsLinear.cpp" // Ensure this includes the definitions for KANLinear and KANLinearImpl


int main() {
   // Define input and output features
   int64_t in_features = 10;
   int64_t out_features = 1;


   // Create a KANLinear module
   KANLinear kan_linear(in_features, out_features, 5, 3, 0.1, 1.0, 1.0, true, torch::nn::SiLU(), 0.02, std::make_pair(-1.0, 1.0));


   // Generate some random input data
   torch::Tensor input = torch::randn({3, in_features});


   // Forward pass
   torch::Tensor output = kan_linear->forward(input);


   // Print the input and output sizes
   std::cout << "Input size: " << input.sizes() << std::endl;
   std::cout << "Output size: " << output.sizes() << std::endl;


   // Update the grid based on new input data
   kan_linear->update_grid(input);


   return 0;
}
