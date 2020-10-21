python kuzu_main.py --override --save --net conv_mini --activation selu --dropout
python kuzu_main.py --override --save --net conv_mini --activation elu --dropout
python kuzu_main.py --override --save --net conv_mini --activation lelu --dropout
python kuzu_main.py --override --save --net conv_mini --activation tanh --dropout
python kuzu_main.py --override --save --net conv_mini --activation selu --dense 1024 --dropout
python kuzu_main.py --override --save --net conv_mini --activation selu --dense 32 --dropout
python kuzu_main.py --override --save --net conv_mini --activation selu --dense 64 --dropout