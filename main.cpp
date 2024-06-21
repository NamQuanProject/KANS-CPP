#include "network.h"
#include <iostream>
#include "torch/torch.h"


using namespace torch;


int main(){
  Net network(50, 10);
  std::cout << network ;
}
