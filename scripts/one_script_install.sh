#!/bin/sh
pip install -r requirements.txt

mkdir -p external
git clone --branch v0.2.1 https://github.com/facebookresearch/detectron2.git external/detectron2
cd external/detectron2 && git checkout 4aca4bd && cd ../..
pip install external/detectron2

git clone https://github.com/hassony2/multiperson.git external/multiperson
pip install external/multiperson/neural_renderer
pip install external/multiperson/sdf

git clone https://github.com/hassony2/frankmocap.git external/frankmocap
sh scripts/install_frankmocap.sh