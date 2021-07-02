#include "dermatologist.h"

torch::Tensor read_data(std::string path) {
    /**
* Returns an image tensor (shape 224,224,3) -> takes in an image path
*
* @param Image path
* @return Image tensor (shape -> 224, 224, 3)
*/
    cv::Mat img = cv::imread(path);
    cv::resize(img, img, cv::Size(224, 224), cv::INTER_CUBIC);
    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
    img_tensor = img_tensor.permute({2, 0, 1});
    return img_tensor.clone();
}

torch::Tensor read_label(int label) {
    /**
* Returns label tensor
*/
    torch::Tensor label_tensor = torch::full({1}, label);
    return label_tensor.clone();
}

std::vector<torch::Tensor> process_images(std::vector<std::string> list_images) {
    /**
* Takes in a list of image paths and returns a list of tensor images
*
* @param <list_images> List of images being 
* @return Return a list of tensor images
*/
    std::vector<torch::Tensor> images;
    for (std::vector<std::string>::iterator it = list_images.begin();
         it != list_images.end();
         ++it) {
        torch::Tensor img = read_data(*it);
        images.push_back(img);
    };
    return images;
}

std::vector<torch::Tensor> process_labels(std::vector<int> list_labels) {
    /**
* @return Return a list of tensor images
 */
    std::vector<torch::Tensor> labels;
    for (std::vector<int>::iterator it = list_labels.begin(); it != list_labels.end(); ++it) {
        torch::Tensor label = read_label(*it);
        labels.push_back(label);
    };
    return labels;
}

std::pair<std::vector<std::string>, std::vector<int>> load_data_from_folder(
    std::vector<std::string> folder_names) {
    std::vector<std::string> list_images;
    std::vector<int> list_labels;
    int label = 0;
    for (auto const &value : folder_names) {
        std::string base_name = value + "/";
        printf("base_name");
        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir(base_name.c_str())) != NULL) {
            while ((ent = readdir(dir)) != NULL) {
                std::string filename = ent->d_name;
                if (filename.length() > 4 && filename.substr(filename.length() - 3) == "jpg") {
                    list_images.push_back(base_name + ent->d_name);
                    list_labels.push_back(label);
                }
            }
            closedir(dir);
        } else {
            printf("Could not open directory");
        }
        label += 1;
    }
    return std::make_pair(list_images, list_labels);
}

template <typename Dataloader>
void train(torch::jit::script::Module model,
           torch::nn::Linear linear,
           Dataloader &dataloader,
           torch::optim::Optimizer &optimizer,
           size_t dataset_size) {
    float best_accuracy = 0.0;
    int batch_index = 0;

    for (int i = 0; i < 25; i++) {
        float mse = 0.0, acc = 0.0;
        for (auto &batch : *dataloader) {
            auto data = batch.data;
            auto target = batch.target.squeeze();
            // TODO: check the docs
            data = data.to(torch::kF32);
            target = target.to(torch::kInt64);

            std::vector<torch::jit::IValue> input;
            input.push_back(data);
            optimizer.zero_grad();
            auto output = model.forward(input).toTensor();
            output = output.view({output.size(0), -1});
            output = linear(output);
            auto loss = torch::cross_entropy_loss(torch::softmax(output, 1), target);
            loss.backward();
            optimizer.step();
            auto accuracy = output.argmax(1).eq(target).sum;
            acc += accuracy.template item<float>();
            mse += mse.template item<float>();
            batch_index += 1;
        }
        mse = mse / float(batch_index);  // mean of the loss
        std::cout << "Epoch: " << i << ", "
                  << "Accuracy: " << acc / dataset_size << ", "
                  << "MSE: " << mse << std::endl;

        test(model, linear, dataloader, dataset_size);

        if (acc / dataset_size > best_accuracy) {
            best_accuracy = acc / dataset_size;
            printf("Saving model");
            model.save("model.pt");
            torch::save(linear, "model_linear.pt");
        }
    }
}

template <typename Dataloader>
void test(torch::jit::script::Module model, torch::nn::Linear linear, Dataloader &dataloader, size_t dataset_size) {
    model.eval();
    float loss = 0.0, accuracy = 0.0;
    for (const auto &batch : *dataloader) {
        auto data = batch.data;
        auto targets = batch.targets.squeeze();
        data = data.to(torch::kF32);
        targets = targets.to(torch::kInt64);
        std::vector<torch::jit::IValue> input;
        input.push_back(data);
        auto output = model.forward(input).toTensor();
        output = output.view({output.size(0), -1});
        output = linear(output);
        auto loss = torch::cross_entropy_loss(torch::softmax(output, 1), targets);
        auto acc = output.argmax(1).eq(targets).sum();
        loss += loss.template item<float>();
        accuracy += acc.template item<float>();
    }
    std::cout << "Test Loss: " << loss / dataset_size << ", Acc:" << accuracy / dataset_size << std::endl;
}