#!/bin/bash
if [ -d "opencv" ]; then
    echo "OpenCV directory already exists. Skipping clone."
else
    git clone https://github.com/opencv/opencv.git
    if [ $? -ne 0 ]; then
        echo "Failed to clone OpenCV repository."
        exit 1
    fi
fi

cd opencv

mkdir -p build
cd build
cmake ..
if [ $? -ne 0 ]; then
    echo "CMake configuration failed."
    exit 1
fi

make -j7
if [ $? -ne 0 ]; then
    echo "Failed to build OpenCV."
    exit 1
fi
echo "OpenCV built successfully."
