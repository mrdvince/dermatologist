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
