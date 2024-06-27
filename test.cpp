// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
// #include "KAN.cpp"

// void test_mul() {
//     std::vector<int64_t> layers_hidden = {2, 2, 1};
//     KAN kan(layers_hidden);
//     auto optimizer = torch::optim::LBFGS(kan->parameters(), torch::optim::LBFGSOptions(1).max_iter(100));

//     for (int i = 0; i < 10; ++i) {
//         torch::Tensor loss, reg_loss;
        
//         auto closure = [&]() -> torch::Tensor {
//             optimizer.zero_grad();
//             auto x = torch::rand({1024, 2});
//             auto y = kan->forward(x, (i % 20 == 0));
            
//             assert(y.sizes() == std::vector<int64_t>({1024, 1}));
//             auto u = x.index({torch::indexing::Slice(), 0});
//             auto v = x.index({torch::indexing::Slice(), 1});
//             loss = torch::nn::functional::mse_loss(y.squeeze(-1), (u + v) / (1 + u * v));
//             reg_loss = kan->regularization_loss(1, 0);
//             (loss + 1e-5 * reg_loss).backward(/* retain_graph= */ true);
//             return loss + reg_loss;
//         };

//         optimizer.step(closure);
//         std::cout << "Epoch: " << i << ", mse_loss: " << loss << ", reg_loss: " << reg_loss << std::endl;
//     }
    
//     for (auto& layer : kan->layers) {
//       std::cout << layer->spline_weight << std::endl;
//     }
// }

// int main() {
//     test_mul();
//     return 0;
// }


// #include <torch/torch.h>
// #include <iostream>
// #include "KANsLinear.cpp" 

// int main() {
//    int64_t in_features = 2;
//    int64_t out_features = 1;

//    std::cout << "CHECKING: ..." << std::endl;

//    KANLinear kan_linear(in_features, out_features);

//    torch::Tensor input = torch::randn({5, in_features});

//    torch::Tensor output = kan_linear->forward(input);

//    std::cout << "Input: " << input << std::endl;
//    std::cout << "Output: " << output << std::endl;

//    kan_linear->update_grid(input);

//    torch::Tensor reg_loss = kan_linear->regularization_loss(1, 0);

//    auto u = input.index({torch::indexing::Slice(), 0});
//    auto v = input.index({torch::indexing::Slice(), 1});

//    torch::Tensor loss = torch::nn::functional::mse_loss(output.squeeze(-1), (u + v) / (1 + u * v));

//    auto total_loss = loss + 1e-5 * reg_loss;
//    std::cout << "Total Loss: " << total_loss.item<double>() << std::endl;

//    return 0;
// }
