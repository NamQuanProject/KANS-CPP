#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "KAN.cpp"

void test_mul() {
    std::vector<int64_t> layers_hidden = {2, 2, 1};
    KAN kan(layers_hidden);
    auto x = torch::rand({3, 2});
    auto y = kan->forward(x, (1 % 20 == 0));

    std::cout << kan->regularization_loss(1,0) <<  std::endl;

    std::cout << x << std::endl;
    std::cout << y << std::endl;
    
    
}

int main() {
    test_mul();
    return 0;
}