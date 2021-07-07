
# Skin Cancer Detection

This is an hobby project to design an algorithm 
that can visually diagnose 3 classes of skin cancer using PyTorch's C++ frontend.

> Disclaimer: Melanoma is one of the most deadliest forms of skin cancer, so definetly don't use anything on this repo to diagnose yourself.

These classes include:

- Melanoma 
- Nevus
- Seborrheic keratosis

The algorithm will distinguish this malignant skin tumor from two types of benign lesions (nevi and seborrheic keratoses).


## Requirements

1. [C++](http://www.cplusplus.com/doc/tutorial/introduction/)
2. [CMake](https://cmake.org/download/) (minimum version 3.14)
3. [LibTorch v1.8.0](https://pytorch.org/cppdocs/installing.html)
4. [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html)

## Getting started
1. Clone this repo and cd into the cloned directory
```bash
  https://github.com/mrdvince/dermatologist.git
  cd dermatologist
```
2. Create a build and cd into it. Then build the project using cmake.

The build process will download the training, testing and validation datasets (it's a large dataset)
```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=~/libtorch ..
```
3. Build
```bash
cmake --build . --config Release
```
__Note:__ if the a data folder is not created and the dataset downloaded modify the CMakelists.txt file
and set the download option to `ON`

4. Run the python convert file included in the cloned folder.
This file will download the resnet18 pretrained model and "trace" it and save on disc without the final fully connected layer.
```bash
python ../convert.py
```
5. Finally train the model
```bash
./dermatologist resnet18_without_last_layer.pt
```

That's pretty much it.
## Acknowledgements

 - [PyTorch C++ examples](https://github.com/pytorch/examples/tree/master/cpp)
