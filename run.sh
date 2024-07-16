sh run-build/opencvbuild.sh
sh run-build/sfmlbuild.sh
cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
