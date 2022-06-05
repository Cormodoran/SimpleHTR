# Handwritten Text Recognition with TensorFlow

An adapted version of: https://github.com/githubharald/SimpleHTR

---

Handwritten Text Recognition (HTR) system implemented with TensorFlow (TF) and trained on the IAM HTR dataset provided by RUG.
The model takes **images of text lines (multiple words) as input** and **outputs the recognized text**.

![htr](./doc/htr.png)


## Setup

Install the dependencies using the environment file

    pip install -r requirements.txt

## Run demo

**NOTE**: Make sure that you are in the `src` directory before attempting to run the model.

* Run inference code:
  * Change the directory to `src`
  * Execute `python main.py` to run the model on an image of a word
  * Execute `python main.py --data_dir `../your-image-folder` to run the model on images of sentences. (This path is relative to the `src` directory, or can be absolute to your system)
  * Output appears in `results.txt`

The input images, and the expected outputs are shown below when the text line model is used.

![test](./data/line.png)

```
# from src
> python main.py --data_dir ../data
Init with stored values from ../model/snapshot-13
```

## Other flags (Useful for training)
* `--mode`: select between "train", "validate" and "infer". Defaults to "infer".
* `--decoder`: select from CTC decoders "bestpath" and "beamsearch". Defaults to "bestpath". For option "wordbeamsearch" see details below.
* `--batch_size`: batch size.
* `--data_dir`: directory containing IAM dataset (with subdirectories `img` and `gt`).
* `--dump`: dumps the output of the NN to CSV file(s) saved in the `dump` folder. Can be used as input for the [CTCDecoder](https://github.com/githubharald/CTCDecoder).



### Run training

* Delete files from `model` directory if you want to train from scratch
* Go to the `src` directory and execute `python main.py --mode train --data_dir path/to/IAM`
* The IAM dataset is split into 80% training data and 20% validation data  
* Training stops after a fixed number of epochs without improvement

The pretrained word model was trained with this command on a GTX 1050 Ti:
```
python main.py --mode train --data_dir path/to/iam  --batch_size 250 --early_stopping 15
```


## References
* [Build a Handwritten Text Recognition System using TensorFlow](https://towardsdatascience.com/2326a3487cd5)
* [Scheidl - Handwritten Text Recognition in Historical Documents](https://repositum.tuwien.ac.at/obvutwhs/download/pdf/2874742)
* [Scheidl - Word Beam Search: A Connectionist Temporal Classification Decoding Algorithm](https://repositum.tuwien.ac.at/obvutwoa/download/pdf/2774578)

