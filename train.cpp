#include <iostream>
#include <torch/torch.h>


int main() {
    auto model = std::make_shared<KANImpl>(std::vector<int64_t>{28 * 28, 64, 10});
    torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(1e-3).weight_decay(1e-4));
    torch::optim::ExponentialLR scheduler(optimizer, 0.8);

    torch::nn::CrossEntropyLoss criterion;
    for (size_t epoch = 0; epoch < 10; ++epoch) {
            model->train();
            float train_loss = 0;
            float train_accuracy = 0;

            for (auto& batch : *train_loader) {
                auto data = batch.data.view({-1, 28 * 28}).to(torch::kCUDA);
                auto targets = batch.target.to(torch::kCUDA);

                optimizer.zero_grad();
                auto output = model->forward(data);
                auto loss = criterion(output, targets);
                loss.backward();
                optimizer.step();

                train_loss += loss.item<float>();
                train_accuracy += output.argmax(1).eq(targets).sum().item<float>() / targets.size(0);
            }

            train_loss /= train_loader->size().value();
            train_accuracy /= train_loader->size().value();

            model->eval();
            float test_loss = 0;
            float test_accuracy = 0;
            torch::NoGradGuard no_grad;

            for (const auto& batch : *test_loader) {
                auto data = batch.data.view({-1, 28 * 28}).to(torch::kCUDA);
                auto targets = batch.target.to(torch::kCUDA);
                auto output = model->forward(data);

                test_loss += criterion(output, targets).item<float>();
                test_accuracy += output.argmax(1).eq(targets).sum().item<float>() / targets.size(0);
            }

            test_loss /= test_loader->size().value();
            test_accuracy /= test_loader->size().value();

            scheduler.step();

            std::cout << "Epoch: " << epoch + 1 << ", Train Loss: " << train_loss
                    << ", Train Accuracy: " << train_accuracy
                    << ", Test Loss: " << test_loss
                    << ", Test Accuracy: " << test_accuracy << std::endl;
        }

        return 0;
}