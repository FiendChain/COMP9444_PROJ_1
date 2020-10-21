python kuzu_main.py --override --save --net conv_mini --activation selu
python kuzu_main.py --override --save --net conv_mini --activation elu
python kuzu_main.py --override --save --net conv_mini --activation lelu
python kuzu_main.py --override --save --net conv_mini --activation tanh
python kuzu_main.py --override --save --net conv_mini --activation selu --dense 1024
python kuzu_main.py --override --save --net conv_mini --activation selu --dense 32
python kuzu_main.py --override --save --net conv_mini --activation selu --dense 64