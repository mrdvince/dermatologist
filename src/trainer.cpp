#include "trainer.h"

template <typename Dataloader>
void train(torch::jit::script::Module module,
           torch::nn::Linear linear,
           Dataloader &train_loader,
           Dataloader &valid_loader,
           torch::optim::Optimizer &optimizer,
           size_t dataset_size) {

    float valid_loss = 0.0;
    float valid_loss_min = std::numeric_limits<float>::infinity();
    float train_loss = 0.0, acc = 0.0;
    for (int i = 0; i < 25; i++) {
        for (auto &batch : *train_loader) {
            auto data = batch.data;
            auto targets = batch.target.squeeze();

            data = data.to(torch::kF32);
            targets = data.to(torch::kInt64);
            std::vector<torch::jit::IValue> input;
            input.push_back(data);
            optimizer.zero_grad();
            auto output = module.forward(input).toTensor();
            output = output.view({output.size(0), -1});
            output = linear(output);

            auto loss = torch::cross_entropy_loss(torch::softmax(output, 1), targets);
            loss.backward();
            optimizer.step();

            // auto acc = output.argmax(1).eq(targets).sum();
            // acc += acc.template item<float>();
            train_loss += loss.template item<float>();
        }
        module.eval();
        for (auto &batch : *valid_loader) {
            auto data = batch.data;
            auto targets = batch.target.squeeze();
            data = data.to(torch::kF32);
            targets = data.to(torch::kInt64);
            std::vector<torch::jit::IValue> input;
            input.push_back(data);
            auto output = module.forward(input).toTensor();
            output = output.view({output.size(0), -1});
            output = linear(output);
            auto loss = torch::cross_entropy_loss(torch::softmax(output, 1), targets);
            auto acc = output.argmax(1).eq(targets).sum();
            valid_loss += loss.template item<float>();
        }
        train_loss = train_loss / dataset_size;
        valid_loss = valid_loss / dataset_size;
        std::cout << "Epoch: " << i << ", "
                  << "Training Loss: " << train_loss << std::endl;
        // save model if validation loss has decreased
        if (valid_loss <= valid_loss_min) {
            std::cout << "Validation loss decreased from: " << valid_loss_min << "-->" << valid_loss << "\nSaving model ...";
            valid_loss_min = valid_loss;
            module.save("module_model.pt");
            torch::save(linear, "model_linear.pt");
        }
    }
}