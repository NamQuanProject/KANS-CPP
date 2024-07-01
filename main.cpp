#include "dataset.h"


int main() {
    int batch_size = 32;
    MNISTDataset dataset(batch_size);

    int batch_num = 50; 
    auto [data, labels] = dataset.loadData(batch_num);

    // Extract the first image in the batch and convert it to a cv::Mat
    torch::Tensor img_tensor = data[31].squeeze(); // Remove the channel dimension
    std::cout << img_tensor << std::endl;
    torch::Tensor label_tensor = labels[31]; // Remove the channel dimension
    std::cout << label_tensor << std::endl;
    return 0;
}
