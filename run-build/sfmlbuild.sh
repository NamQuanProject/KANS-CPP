if [ -d "SFML-2.6.1" ]; then
    echo "SFML directory already exists. Skipping clone."
else
    wget https://www.sfml-dev.org/files/SFML-2.6.1-sources.zip
    if [ $? -ne 0 ]; then
        echo "Failed to download SFML source archive."
        exit 1
    fi

    unzip SFML-2.6.1-sources.zip
    if [ $? -ne 0 ]; then
        echo "Failed to unzip SFML source archive."
        exit 1
    fi

    rm SFML-2.6.1-sources.zip
fi

cd SFML-2.6.1

mkdir -p build
cd build

cmake ..
if [ $? -ne 0 ]; then
    echo "CMake configuration failed."
    exit 1
fi

make -j7
if [ $? -ne 0 ]; then
    echo "Failed to build SFML."
    exit 1
fi

echo "SFML built successfully."
