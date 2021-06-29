// #include <dirent.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <opencv2/opencv.hpp>

// rezize image to -> (224, 224, 3)
torch::Tensor read_data(std::string path);

// return label as integer
torch::Tensor read_label(int label);

// return a vector(list) of tensor images from folder
std::vector<torch::Tensor> process_images(std::vector<std::string> list_images);

// same for labels as above
std::vector<torch::Tensor> process_labels(std::vector<int> list_labels);

// load data from folder -> return a pair -> image and respective label
std::pair<std::vector<std::string>, std::vector<int>> load_data_from_folder(std::vector<std::string> folder_name);

// train network
template <typename Dataloader>
void train(
    torch::jit::script::Module net,
    torch::nn::Linear lin,
    Dataloader &dataloader,
    torch::optim::Optimizer &optimizer,
    size_t dataset_size);

// test
template <typename Dataloader>
void test(
    torch::jit::script::Module net,
    torch::nn::Linear lin,
    Dataloader &dataloader,
    size_t dataset_size);

// custom dataset class
class DermDataset : public torch::data::Dataset<DermDataset> {
   private:
    std::vector<torch::Tensor> states, labels;
    size_t ds_size;

   public:
    DermDataset(std::vector<std::string> list_images, std::vector<int> list_labels) {
        states = process_images(list_images);
        labels = process_labels(list_labels);
        ds_size = states.size();
    };
    torch::data::Example<> get(size_t index) override {
        torch::Tensor sample_img = states.at(index);
        torch::Tensor sample_label = labels.at(index);
        return {sample_img.clone(), sample_label.clone()};
    };
    torch::optional<size_t> size() const override {
       return ds_size; 
    }
};