# Machine Learning Based Text Generation Tutorial

For [Population Infinite Fall 2019](https://github.com/agermanidis/population-infinite)

## Overview

This tutorial will walk you through how to use text based machine-learning models to generate text in the style of some existing source text.

In this tutorial, we will do the following:
1. Run an existing pre-trained [RNN](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) model to generate some text, and see the output.  
2. Learn how to train our own models based on a custom text corpus.  
3. Learn how to gather text for that corpus to train. 
4. We will see how to use a state of the art text generation model, GPT-2, with Runway.

This example illustrates how to use [TensorFlow.js](https://www.tensorflow.org/js/) to train a LSTM model to
generate random text based on the patterns in a text corpus such as
Shakepear or your own Facebook Posts.

[Let's see an example live!](https://storage.googleapis.com/tfjs-examples/lstm-text-generation/dist/index.html)

## Let's use a model trained on Shakespear text to generate some text

First, we will run a model trained on Shakespeare text to generate some text that sounds like Shakespeare

### Setting up the Code

Download the code from this repository:

    git clone https://github.com/oveddan/ml-text-gen-tutorial.git

Install yarn.  On **mac,** this can be done with:

    brew install yarn

On windows, [follow the instructions here](https://yarnpkg.com/lang/en/docs/install/#windows-stable)

Install the dependencies:

    yarn

Now, let's generate some text:

    yarn gen shakespeare.txt shakespeare \
      --genLength 250 \
      --temperature 0.6

What the above command did, was use an LSTM model trained on the text in [corpus/shakespeare.txt](corpus/shakespeare.txt) to generate some text of length 250, that the model predicted should come after some random text. Everything is run on your computer using Tensorflow.js in Node.

## The Model

### What is an LSTM RNN

An LSTM is a type of [Recurrrent Neural Network,](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) which is a type of machine learning model designed to work with **sequences of information.**

![RNN Diagrams](https://karpathy.github.io/assets/rnn/diags.jpeg)

> What makes Recurrent Networks so special? A glaring limitation of Vanilla Neural Networks (and also Convolutional Networks) is that their API is too constrained: they accept a fixed-sized vector as input (e.g. an image) and produce a fixed-sized vector as output (e.g. probabilities of different classes). 

Some examples of how types of RNNs:

[Text translation,](https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention) where a sequence of words from one language is translated into a sequence of words in another language:

![translation](images/attention_mechanism.jpg)

[Human Activity Recognition from a sequence of poses](https://github.com/stuarteiffert/RNN-for-Human-Activity-Recognition-using-2D-Pose-Input)

![Animated Gif of Poses](https://github.com/stuarteiffert/RNN-for-Human-Activity-Recognition-using-2D-Pose-Input/raw/master/images/boxing_all_views.gif.png)

Sketch classification, where a sequence of stroked are classified:
![Google Quick Draw](https://www.seoclerks.com/files/user/images/google%20quick%20draw.jpg)

[Behavior Prediction](https://jobs.zalando.com/tech/blog/deep-learning-for-understanding-consumer-histories/?gh_src=4n3gxh1), where a sequence of behaviors can be used to predict the next behavior:

![BehaviorPredictionImage](https://zalando-jobsite.cdn.prismic.io/zalando-jobsite/2ac1f20f2c8beeb95a9d28988aa1966a077dd3ba_consumer_history.png)

Some nice things about Recurrent Neural Networks:

1. They can be trained with a lot less resources than traditional convolutional neural networks, allowing for them to be trained on our own computers.
2. When training, it is easy to see the progress.

### How this model works

The LSTM model operates at the character level. It takes an input of a character of strings, each character encoded as a one-hot value. With the input, the model
outputs a list of values, which represents the
model's predicted probabilites of a character that follows the input sequence.
The application then draws a random sample based on the predicted
probabilities to get the next character. Once the next character is obtained,
its one-hot encoding is concatenated with the previous input sequence to form
the input for the next time step. This process is repeated in order to generate
a character sequence of a given length. The randomness (diversity) is controlled
by a temperature parameter.

The UI allows creation of models consisting of a single
[LSTM layer](https://js.tensorflow.org/api/latest/#layers.lstm) or multiple,
stacked LSTM layers.

## Usage

### Training Models in Node.js

Training a model in Node.js should give you a faster performance than the browser
environment.

To start a training job, enter command lines such as:

```sh
yarn
yarn train shakespeare.txt \
    --lstmLayerSize 128,128 \
    --epochs 120 \
    --savePath ./my-shakespeare-model
```

- The first argument to `yarn train` (`shakespeare`) specifies what text corpus
  to train the model on. See the console output of `yarn train --help` for a set
  of supported text data.
- The argument `--lstmLayerSize 128,128` specifies that the next-character
  prediction model should contain two LSTM layers stacked on top of each other,
  each with 128 units.
- The flag `--epochs` is used to specify the number of training epochs.
- The argument `--savePath ...` lets the training script save the model at the
  specified path once the training completes

If you have a CUDA-enabled GPU set up properly on your system, you can
add the `--gpu` flag to the command line to train the model on the GPU, which
should give you a further performance boost.

### Generating Text in Node.js using Saved Model Files

The example command line above generates a set of model files in the
`./my-shakespeare-model` folder after the completion of the training. You can
load the model and use it to generate text. For example:

```sh
yarn gen shakespeare ./my-shakespeare-model/model.json \
    --genLength 250 \
    --temperature 0.6
```

The command will randomly sample a snippet of text from the shakespeare
text corpus and use it as the seed to generate text.

- The first argument (`shakespeare`) specifies the text corpus.
- The second argument specifies the path to the saved JSON file for the
  model, which has been generated in the previous section.
- The `--genLength` flag allows you to speicify how many characters
  to generate.
- The `--temperature` flag allows you to specify the stochacity (randomness)
  of the generation processs. It should be a number greater than or equal to
  zero. The higher the value is, the more random the generated text will be.
