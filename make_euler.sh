cd ./tf-lstm
jupyter nbconvert --to python tf_lstm.ipynb
cd ../feedforward
jupyter nbconvert --to python lstm.ipynb
jupyter nbconvert --to python feedforward.ipynb
cd ..
