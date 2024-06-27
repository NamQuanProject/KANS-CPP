#include <torch/torch.h>
#include <iostream>
#include "KANsLinear.h"



void test_simple_math(){
    int64_t in_features = 2;
    int64_t out_features = 1;

    std::cout << "CHECKING: ..." << std::endl;

    // Create an instance of the KANLinear module
    KANLinear kan_linear(in_features, out_features);

    int64_t epoch = 20;
    auto optimizer = torch::optim::LBFGS(kan_linear->parameters(), torch::optim::LBFGSOptions().max_iter(100));
    torch::Tensor input = torch::randn({4, in_features});
    
    for (int i = 0; i < epoch; ++i) {
        auto closure = [&]() -> torch::Tensor {
            optimizer.zero_grad();
            // Generate random input tensor
            // std::cout << "Input: " << input << std::endl;
            // std::cout << "Output: " << output << std::endl;
            // Compute regularization loss
            torch::Tensor output = kan_linear->forward(input);
            torch::Tensor reg_loss = kan_linear->regularization_loss(1, 0);

            // Extract columns and compute loss
            auto u = input.index({torch::indexing::Slice(), 0});
            auto v = input.index({torch::indexing::Slice(), 1});
            torch::Tensor target = (u + v) / (1 + u * v);
            torch::Tensor loss = torch::nn::functional::mse_loss(output.squeeze(-1), target);

            // Compute total loss and perform backward pass
            torch::Tensor total_loss = loss + 1e-5 * reg_loss;
            std::cout << "epoch: " << i <<  "Total Loss: " << total_loss.item<double>() << std::endl;

            total_loss.backward();
            return total_loss;
        };

        optimizer.step(closure);
    }
    std::cout << "Input: " << input << std::endl;
    std::cout << "Output: " << kan_linear->forward(input) << std::endl;
}