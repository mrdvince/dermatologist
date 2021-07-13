#include <torch/torch.h>

template <typename Dataloader>
void train(torch::jit::script::Module module,
           torch::nn::Linear linear,
           Dataloader &train_loader,
           Dataloader &valid_loader,
           torch::optim::Optimizer &optimizer,
           size_t dataset_size);