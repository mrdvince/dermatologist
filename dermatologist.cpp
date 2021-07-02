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
        float mse = 0.0, accuracy = 0.0;
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
            auto acc = output.argmax(1).eq(target).sum();
            accuracy += acc.template item<float>();
            mse += loss.template item<float>();
            batch_index += 1;
        }
        mse = mse / float(batch_index);  // mean of the loss
        std::cout << "Epoch: " << i << ", "
                  << "Accuracy: " << accuracy / dataset_size << ", "
                  << "MSE: " << mse << std::endl;

        test(model, linear, dataloader, dataset_size);

        if (accuracy / dataset_size > best_accuracy) {
            best_accuracy = accuracy / dataset_size;
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
        auto targets = batch.target.squeeze();
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


int main(int argc, const char * argv[]) {
    //folder names for the image classes
    std::string melanoma = "data/melanoma";
    std::string nevus = "data/nevus";
    std::string seborrheic_keratosis = "data/seborrheic_keratosis";
    std::vector<std::string> folder_names;
    folder_names.push_back(melanoma);
    folder_names.push_back(nevus);
    folder_names.push_back(seborrheic_keratosis);

    // paths of images and integer labels -> still no idea how this works TODO
    std::pair<std::vector<std::string>, std::vector<int>> pair_images_labels = load_data_from_folder(folder_names);

    std::vector<std::string> list_images = pair_images_labels.first;
    std::vector<int> list_labels = pair_images_labels.second;
    std::cout << list_labels << std::endl;

    // init custom dataset class and read data
    auto cs_dataset = DermDataset(list_images, list_labels).map(torch::data::transforms::Stack<>());

    // load pretrained model
    torch::jit::script::Module module = torch::jit::load(argv[1]);

    torch::nn::Linear linear(512, 3);
    torch::optim::Adam opt(linear->parameters(), torch::optim::AdamOptions(0.001));
    auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(cs_dataset));

    train(module, linear, dataloader, opt, cs_dataset.size().value());
    return 0;
}