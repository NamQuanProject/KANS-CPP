#include "testKAN.h"

void test_KANLinear() {
    int64_t in_features = 2;
    int64_t out_features = 1;
    std::cout << "CHECKING: ..." << std::endl;
    KANLinear kan_linear(in_features, out_features);

    int64_t epoch = 30;
    auto optimizer = torch::optim::LBFGS(kan_linear->parameters(), torch::optim::LBFGSOptions(1).max_iter(100));
    torch::Tensor input = torch::randn({2024, in_features});
    
    for (int i = 0; i < epoch; ++i) {
        auto closure = [&]() -> torch::Tensor {
            optimizer.zero_grad();
            torch::Tensor output = kan_linear->forward(input);
            torch::Tensor reg_loss = kan_linear->regularization_loss(1, 0);
            auto u = input.index({torch::indexing::Slice(), 0});
            auto v = input.index({torch::indexing::Slice(), 1});
            torch::Tensor target = (u + v) / (1 + u * v);
            torch::Tensor loss = torch::nn::functional::mse_loss(output.squeeze(-1), target);
            torch::Tensor total_loss = loss + 1e-5 * reg_loss;
            std::cout << "epoch: " << i <<  "Total Loss: " << total_loss.item<double>() << std::endl;

            total_loss.backward();
            return total_loss;
        };

        optimizer.step(closure);
    }
    torch::Tensor new_input = torch::randn({3, in_features});
    std::cout << "Input: " << new_input << std::endl;
    std::cout << "Output: " << kan_linear->forward(new_input) << std::endl;
}


void test_KAN() {
    std::vector<int64_t> layers_hidden = {2, 5, 1};
    KAN kan(layers_hidden);
    auto optimizer = torch::optim::LBFGS(kan->parameters(), torch::optim::LBFGSOptions(1).max_iter(100));
    int64_t epoch = 20;
    auto input = torch::rand({1024, 2});
    for (int i = 0; i < epoch; ++i) {
        auto closure = [&]() -> torch::Tensor {
            optimizer.zero_grad();
            torch::Tensor output = kan->forward(input);
            torch::Tensor reg_loss = kan->regularization_loss(1, 0);
        
            auto x1 = input.index({torch::indexing::Slice(), 0});
            auto x2 = input.index({torch::indexing::Slice(), 1});

            const double pi = M_PI; 
            torch::Tensor target = torch::exp(torch::sin(pi * x1) + x2 * x2);

            torch::Tensor loss = torch::nn::functional::mse_loss(output.squeeze(-1), target);

            torch::Tensor total_loss = loss + 1e-5 * reg_loss;
            std::cout << "epoch: " << i+1 <<  "Total Loss: " << total_loss.item<double>() << std::endl;

            total_loss.backward();
            return total_loss;
        };
        optimizer.step(closure);
    }
    auto new_input = torch::rand({5, 2});
    std::cout << "Input: " << new_input << std::endl;
    std::cout << "Output: " << kan->forward(new_input) << std::endl;
}