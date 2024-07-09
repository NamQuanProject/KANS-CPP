# KANS Implementation with C++

## How to Set Up the Program
### Build OpenCV Library

To build the OpenCV library, follow these steps:
1. **Clone the OpenCV repository:**

    ```bash
    git clone https://github.com/opencv/opencv.git
    cd opencv
    ```

2. **Create a build directory and navigate into it:**

    ```bash
    mkdir build
    cd build
    ```

3. **Configure the build using CMake:**

    ```bash
    cmake ..
    ```

4. **Compile the library:**

    ```bash
    make -j7
    ```


### Set Up Your C++ Project
Assuming you have your project organized with the following structure:
1. **Configure the build using CMake:**

    Navigate to your build directory and configure the build. Make sure you have the torch library installed on your computer.

    ```bash
    mkdir build
    cd build
    cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
    ```

2. **Run the program:**
    ```bash
    ./kans
    ```





