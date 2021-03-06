# Sequence Models

[![Build Status](https://travis-ci.com/dmitryaleks/sequence-models.svg?branch=master)](https://travis-ci.com/dmitryaleks/sequence-models)

Examples of sequences: text, speech, financial time series.

## Notation

<https://www.coursera.org/learn/nlp-sequence-models/lecture/aJT8i/notation>

Eleents of the input sequence:
```
x<1>, x<2>, ..., x<t>,...,x<l>
```

Output (feature) sequence based on the input sequence:
```
y<1>, y<2>, ..., y<t>,...,y<l>
```

Let's also define:
```
Tx: length of the input sequence;
Ty: length of the output sequence.
```

A particular training example (number #i) is denoted as:
```
X(i)
```

Then a particular element of a given training example is denoted as:
```
X(i)<t>
```

Input sequence length is:
```
Tx(i)
```

## Natural Language Processing

### Vocabulary and training set representation

A set of unique words (as in a dictionary) encoded as a vector and addressed by index of a given element in the vector.

Reasonable size of a vocabulary is around 30K, 50K, 100K or more.

Each element (word) of the input example (sequence of words) is than represented as a one-hot vector of a lenght of vocabulary size, where the element corresponding to the word is set to one while all other elements are set to zero.

## Recurrent Neural Networks

<https://www.coursera.org/learn/nlp-sequence-models/lecture/ftkzt/recurrent-neural-network-model>

Applying a standard neural network to a sequence data is problematic due to:
  - varying lenght of input sequences (different examples have different lenghts);
  - it doesn't take element positions into account, so there is a loss of information.

RNN addresses those problems by consuming input sequence element by element, and passing activation values from previous to the next step each time it advances forward. It allows to consume an input sequence of any lenght in a uniform way and also takes sequence features into account (i.e. relative position of elements in the input sequence).

### Architecture

RNN consumes input sequence element by element, and passes activation values from previous to the next step each time it advances forward.

![RNN Architecure](docs/img/RNN-architecture.png)

### Forward propagation

![RNN Forward Propagation](docs/img/RNN-forward-propagation.png)

Simplified notation:

![RNN Forward Propagation: Simplified Notation](docs/img/RNN-simplified-notation.png)

### Backpropagation through time

<https://www.coursera.org/learn/nlp-sequence-models/lecture/bc7ED/backpropagation-through-time>

Backpropagation pushes losses backwards through the network, calculating derivatives of training parameters and then applying gradient descent to optimize parameters to find the optimal fit.

Defining a loss function, E.g. a cross-entropy loss as in a logistic regression:
```
L<t>(^y<t>, y<t>) = -y(t) * log(^y(t)) - (1 - y(t)) * log(1 - ^y(t))

# a total loss:

L(^y<t>, y<t>) = Sum(L<t>(^y<t>, y<t>))
```

![RNN: Backpropagation through time](docs/img/RNN-backpropagation-through-time.png)

### Different types of RNN architectures

<https://www.coursera.org/learn/nlp-sequence-models/lecture/BO8PS/different-types-of-rnns>

In general case, the length of the input sequence may be different from the length of the ouput sequence.

Examples are as follows:

  * many-to-one: in a Sentiment Classification problem the input is the text of an arbitrary length, while the output is a single integer value indicating the class (E.g. binary 0/1, or more classes).

  * one-to-one: temperature in degerees Celsius mapped to a human readable classification (cold, warm, hot);

  * one-to-many: music generation with the input of a genre and the output is the sequence of notes of the musical piece;

  * many-to-many: machine translation, where RNN starts with "encoder" part that consumes the input sequence of words, followed by a "decoder" part of lenght "b", where "b" is the amount of the words in the output sequence;

One-to-many and many-to-many architecture are exemplified in the following diagram:

![RNN: architecture types](docs/img/RNN-architecure-types.png)

Below is the full summary of RNN types:

![RNN: summary of types](docs/img/RNN-summary-of-types.png)

### Language modeling and sequence generation

<https://www.coursera.org/learn/nlp-sequence-models/lecture/gw1Xw/language-model-and-sequence-generation>

Language modeling entails:

  * using a large corpus of text as a training set:
  * tokenizing sentences and adding <EOS> character (End-of-Sentence) at the end of each sentence;
  * unknown words can be encoded with a special <UNK> token.

#### Speech recognition

Speech recognition model estimates probability of each input sentence in order to pick the most likely one.

In order to do so it consumes words from the input sequence one by one and estimates a probability of each word being in a given position in a sentence.

RNN would be architectured to carry over all previously recognized words to the next unit, where each unit is trying to pick the most probable next word in a sentence given a preceeding sequence of words.

Softmax is a suitable loss function for this task:
```
L(^y<t>, y<t>) = -sum<i>(y(i)<t> * log(^y(i)<t>))

# overall loss:

L = sum<t>(L(^y<t>, y<t>))
```

After training such model on a large corpus of text, it will be able to give probabilities of a given word coming in an input sequence.

This enables calculation of sentence probabilities as follows:
```
P(y<1>, y<2>, y<3>) = P(y<1>) * P(y<2> | y<1>) * P(y<3> | y<1>, y<2>)
```

![RNN: language model](docs/img/RNN-language-model.png)

### Sampling novel sequences from trained models

<https://www.coursera.org/learn/nlp-sequence-models/lecture/MACos/sampling-novel-sequences>

First we train a network as usual, and as the result obtain a "language model" which is essentially is a set of probabilities for a certain word to appear within a certain context built based on the input text corpus (E.g. as the first word, or after a certain sequence of words).

This allows us to generate new sentences as follows by means of the process called "sampling". It works as follows:

  * first, generate the first word by choosing an element from the probability distribution that has resulted from the model training (E.g. by using "numpy.random.choice";

  * for each subseqent step, carry over the sequence from the previous step to be used as an input and again generate a random choice according to the probability distribution learned by the model.

Sampling process is illustrated in the diagram below:

![RNN: Sampling a sequence from a trained RNN](docs/img/RNN-sampling-sequence.png)

In general, there are two language models:
  * word-level (conventional);
  * character-level.

Word-level models are prevalent.

Character-level models are more demanding in terms of training costs, but has an advantage that it can handle unknown words. Those are sometimes more suitable for some specialized applications, E.g. where unknown words come up often.

### Vanishing gradient problem when training RNN

<https://www.coursera.org/learn/nlp-sequence-models/lecture/PKMRR/vanishing-gradients-with-rnns>

Vanishing gradient problem is the phenomenon in which during backpropagation errors coming from later elements in the model (E.g. last layers) are not affecting earlier layers enough (E.g. first layer), i.e. the influence is vanishing.

As an example, in Natural Language Processing it is possible to have a sentence that has grammatically connected elements that are far apart (E.g. a noun and a corresponding pronous that are both either singular or plural). Vanishing gradient then would result in little regard for those related elements of the sequence as they simply happened to be too far apart.

Unless addressed, vanishing gradient problem leads to RNNs not being able to capture long range dependencies in input sequences.

This problem is different from "Exploding Gradients" in conventional neural networks, where trained parameters blow up, often to NaN values due to numerical overflow. "Gradient Clipping" technique is a relatively robust solution that helps for it (gradient vector rescaling according to some maximum values).

Vanishing gradient problem is more intricate and requires more elaborate solution.

![RNN: Vanishing gradient](docs/img/RNN-vanishing-gradient.png)

### Gated Recurrent Unit (GRU)

<https://www.coursera.org/learn/nlp-sequence-models/lecture/agZiL/gated-recurrent-unit-gru>

GRU is a modification to the RNN hidden layer, which makes it better to capture long-range connections and helps a lot with the vanishing gradient problem.

A simplified GRU has a notion of a memory cell (C) that is used to carry some important information throughout steps (E.g. whether the topic noun is singular or plural in an NLP task). Memory cell is subject to a "gating" in which a gate operation is applied to the memory cell at every step. Gate operation either takes a newly candidate for the memory cell value or takes pick a memory cell value from the previous step.

Both memory cell value calculation and gate calculation contain trainale parameters and therefore are fit to the training set during the training phase. Training of those parameters results in picking up the most useful values for the memory cell (C) and caryying them for as long as it is required to get the best fit for the training set.

A simplified GRU can be depicted as follows:

![RNN: Simplified GRU](docs/img/RNN-simplified-GRU.png)

Full GRU can be depicted as follows (note a new Gamma_r term that captures relevance of a memory cell value from the previous step to a subsequent step):

![RNN: Simplified GRU](docs/img/RNN-full-GRU.png)

### LSTM

<https://www.coursera.org/learn/nlp-sequence-models/lecture/KXoay/long-short-term-memory-lstm>

LSTM (Long Short Term Memory) is a more powerfull alternative to GRU that allows learning long range connections in sequences.

LSTM has three gates instead of two in GRU (Update Gate and Reelvance Gate). Those are:
  * update gate (G_u);
  * forget gate (G_f);
  * output gate (G_o).

Notes that gates "G" as well as memory cell "C" can be multi-dimensional (i.e. a memory cell can carry lots of information).


Equasions governing LSTM are on the right in the figure below:

![LSTM: basic equasions](docs/img/LSTM-equasions.png)

It is worth noting that LSTMs have been invented before GRU, where the latter came as a simplified model. There is no universal rule on when to pick up LSTM over GRU. GRU are cheaper to train and therefore can be used in larger networks. By default LSTM is a proven choice.

LSTM can be expressed in a form of a diagram as follows:

![LSTM: block diagram](docs/img/LSTM-block-diagram.png)

Note that when a series of LSTM units are connected lineraly, it is clear that the memory cell can carry some usefull information across wide ranges therefore enabling learning long range connections in sequences.

### Bidirectional RNN (BRNN)

<https://www.coursera.org/learn/nlp-sequence-models/lecture/fyXnn/bidirectional-rnn>

Bidirectional RNN allows at any point in time (in sequence) taking information from both earlier and later elements in the sequence.

Motivation for BRNN is to use the information from future points in the sequence to draw conclusion with regard to earlier elements.

For example, BRNN is helpful when trying to figure out whether the third word in the following example is a person's name or not:

![RNN: motivation for BRNN](docs/img/RNN-motivation-for-BRNN.png)

Bidirectional RNN create an acyclic graph by adding extra blocks and connections that allow data to flow backwards (note the elements highlighted in green in the diagrem below):

![RNN: bidirectional](docs/img/RNN-bidirectional.png)

Also note that prediction at point "t" in BRNN is calculated as follows:
```
^y<t> = g(W_y(a_forward<t>, a_backward<t>) + b_y)
```

Note the "a_backwards" activations in the formula.

NLP problems are commonly solved with RNN that use LSTM as building blocks.

One downside of BRNN is that the whole sequence needs to be consumed in order to make predictions (i.e. future data elements matter).

### Deep RNNs

<https://www.coursera.org/learn/nlp-sequence-models/lecture/ehs0S/deep-rnns>

For some problems it is beneficial to stack multiple layers of RNNs together to get an even deeper model.

Even three layers of RNN would be considered a deep network due to a temporal component of each individual layer.


Notes:
  * individual blocks in RNNs could be of a standard type, as well as of GRU or LSTM type;
  * a deep RNN network can also be bidirectional;
  * RNN are computationally expensive by themselves, and adding layers makes them even more expensive.

Deep RNN example is as follows:

![RNN: deep network example](docs/img/RNN-deep-network-example.png)

### Gradient Clipping

GC is a technique that helps to avoid the "Exploding Gradients" problem.

Exploding Gradients result in gradient values taking extremely large numbers.

To avoid this gradient clipping is simply truncating gradient value to be within a given [-N, N] range.

### Sample implementation

Implementations below are based on the following Coursera Python Notebook:

<http://htmlpreview.github.com/?https://github.com/dmitryaleks/sequence-models/blob/master/notebooks/building-rnn/building-rnn.html>

Online version of the original Coursera Python Notebook:

<https://tjaxlggwxiafkpcnxzjsmm.coursera-apps.org/notebooks/Week%201/Building%20a%20Recurrent%20Neural%20Network%20-%20Step%20by%20Step/Building%20a%20Recurrent%20Neural%20Network%20-%20Step%20by%20Step%20-%20v3.ipynb#>

#### Concrete implementations of RNN with Python and numpy

[Basic RNN cell](rnn/rnn_cell_forward.py)

[Forward propagation with RNN](rnn/rnn_forward.py)

[Basic LSTM cell](rnn/lstm_cell_forward.py)

[Forward propagation with LSTM](rnn/lstm_forward.py)

[Language model and sequence sampling](languagemodel/model.py)


### NLP and Word Embeddings

>https://www.coursera.org/learn/nlp-sequence-models/lecture/6Oq70/word-representation>

#### Word representation

The standard way of representing words is a one-hot verctor over a vocabuary of words.

The down side of this approach is that algorithm cannot generalize beyond a single word and learn similarities between words as each word is separated from the rest of them by representation, as by definition there are no common elements to two vectors representing any two different words.

Instead it is possible to represent words usian an alternative featurized representations.

![Words representation: featurized representation](docs/img/words-representation-featurized.png)

Based on the above, a single word can be represented as a vector of features, where each value denotes the degree to which a given feature can be attributed to a given word.

This appreach is similar to that of a Word2Vec:
<https://en.wikipedia.org/wiki/Word2vec>

Introduction to Word2Vec:
<https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa>

Word2Vec white paper:
<https://arxiv.org/pdf/1310.4546.pdf>

#### Word embeddings


We say we embed a word when we get a word and map (embed) it into an N-dimensional space of features (governed by N unique features).

Note that it is sometimes convenient to reduce dimensionality of the feature space to visualize it. E.g. t-SNE (t-Distributed Stochastic Neighbor Embedding) algorithm allows reducing a 300 dimensional feature space to a 2D space for visualization purposes. It often happens that related words tend to appear close to each other spatially.

<https://lvdmaaten.github.io/tsne>

![Words representation: word embedding](docs/img/word-embedding.png)

Formal definition is as follows:

<https://en.wikipedia.org/wiki/Word_embedding>

"Word embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers. Conceptually it involves a mathematical embedding from a space with one dimension per word to a continuous vector space with a much lower dimension.

Methods to generate this mapping include neural networks, dimensionality reduction on the word co-occurrence matrix, probabilistic models, explainable knowledge base method, and explicit representation in terms of the context in which words appear.

Word and phrase embeddings, when used as the underlying input representation, have been shown to boost the performance in NLP tasks such as syntactic parsing and sentiment analysis."

#### Transfer Learning and word embeddings

"Transfer learning is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem."

Word embeddings learned on a large corpus of text can be used on a new smaller training set of the new problem.

#### Analogies Reasoning

With word embeddings we can answer questions like:

```
"man is to woman is what king is to X"
```

![Analogy Reasoning](docs/img/word-embeddings-analogies-reasoning.png)

The algorithm for the analogic reasoning is based on finding the solution to the following equasion:


```
find word w(i) such that: argmax( similarity(e(w), e(king) - e(man) + e(woman)) )
```

![Analogy Reasoning Algorithm](docs/img/word-embeddings-analogies-reasoning-algorithm.png)

#### Similarity Functions

Most common similarity function is Cosine Similarity:

```
sim(u,v) = (transpose(u)*v)/(eucld_len(u)*euclid_len(v))
```

This is effectively a cosine of the anlge between two vectors. If the anle is zero - the cosine is one (i.e. maximum).

![Cosine Similarity](docs/img/word-embeddings-cosine-similarity.png)

#### Embedding Matrix

Embedding Martix is a learned matrix that has features of words as rows and contrete words as columns, where values in the matrix are degrees to which a given feature pertains to a given word.

![Embedding Matrix](docs/img/word-embeddings-embedding-matrix.png)

#### Learning Word Embeddings

One way to lear word embeddings from the Language Model and a large corpus of text:

  - start with a randomly initialised embedding matrix E that transforms a given word into an embedding (vector of features);
  - use the same embedding matrix E to transform all words in a window (the context) into embeddings;
  - use the word that follows the window as the label (the target);
  - train ANN using a gradient descent to find the right coefficients in the Embedding Matrix E.

This algorithm will be incentivised to come up with similar embeddings for similar words.

![Learning Word Embeddings with ANN](docs/img/word-embeddings-learning-ann.png)

#### Skip-gram

For a real Language Model it is beneficial to learn the target word from a window of context words.

However, research has found that to learn Word Embeddings it is enough to use the a nearby one word. This approach is called Skip-gram.

Here is how we can learn word embeddings by training a Skip-gram model:

![Skip-gram model](docs/img/word-embeddings-skipgram-model.png)

Softmax classifier is computationally expensive to train as it requires taking a sum over all categories each time denominator is computed. Alternative is a so called Hierarchical Softmax, which splits the set of categories as a binary tree (scales as a log function of the number of categories). In a Hierarchical Softmax it is beneficial to have popular words to be at the top of the tree.

##### How to sample the context to select the context word?

We want to avoid our training set being dominated by frequently occuring words like: "the", "a", "and", etc.

There are heuristics that allow us to sample the context in a way where we select less common words as often as those that occur frequently.

#### Negative Sampling

General definition:

```
Negative sampling. Negative sampling idea is based on the concept of noise contrastive estimation (similarly, as generative adversarial networks), which persists, that a good model should differentiate fake signal from the real one by the means of logistic regression.
```

Algorithm:
  - select a context word and a real target word (E.g. the next one) from some sentence and give it a label "1";
  - pick (k-1) random target words for the same context word and give them lable "0" (E.g. where k = 5).

![Negative Sampling](docs/img/word-embeddings-negative-sampling.png)

Then we solve a Logistic Regression problem for each target word.

So essentially instead of training a computationally expensive Softmax that has N elements to it (N is the size of the vocabulary), we are training N binary classifiers, which are faster to train.

![Negative Sampling: Training a Model](docs/img/word-embeddings-negative-sampling-algorithm.png)

##### Sampling Negative Examples

We pick a given target words using a probabilistic heuristic:

```
P(w) = (frequency(w)^(3/4))/SUM(frequency(words)^(3/4))
```

#### GloVe (Global vectors for word representation) algorithm

We first go through the training set corpus and for each pair of words calculate how often word A appears in the context (vicinity) of word B.

Then GloVe algorithm finds word embeddings by optimizing the following objective:

![GloVe algorithm](docs/img/word-embeddings-glove-algorithm.png)

#### Sentiment Classification

This is a task of looking at a text and figuring out whether whoever wrote it likes or dislikes the thing they are talking about.

Sentiment can be classified in two or more categories, E.g. like/dislike or a five-star rating scale respectively.

##### Simple Model for Sentiment Classification

We can train a simple model on a labeled training set as follows:

  - take an embedding on each word in the text based on previously learned word embeddings matrix;
  - calculate an averaged embedding vector of all words in the text;
  - pass the averaged embedding vector to a Softmax to produce the predicted sentiment category.

![Sentiment Classification: Simple Model](docs/img/word-embeddings-sentiment-classification-simple-model.png)

The problem of this simplistic algoirthm is that it completely disregards the order of words, so something like "lacks a good taste and good atmosphere" would gravitate towards a positive sentiment due to the word "good" occuring twice.

##### RNN for Sentiment Classification

Applying RNN allows us to respect the order of words in a sentence.

![Sentiment Classification: RNN](docs/img/word-embeddings-sentiment-classification-rnn.png)

#### Debiasing Word Embeddings

Bias may get introduced into learned word embeddings in case it is present in the underlying text corpus.

Addressing bias in word embedings entails:

  - identify bias direction in embeddings space by examining differences between pairs of words (E.g. "he" and "she");
  - calculate the direction of the no-bias axis for a given type of bias by taking a difference between two words from step #1;
  - neutralize: for every word that is not definitional (those that do not have inherent gender component to them), project it to a no-bias axis to get rid of bias.
  - equalize pairs of words: make sure that words like grandmother and grandfather are at equal distance from non-definitional words that are supposed to be gender neutral, E.g. a babysitter.

![Debiasing Word Embeddings](docs/img/word-embeddings-eleminating-bias.png)

## Sequence to Sequence models

### Basic models

#### Machine translation

Let's consider an example in which we translate a sentence from French language to English.

We can build a model as follows:

  - start with an encoder network that takes French words as an input - it will be trained to output the vector that represents the input sentence;
  - the vector output by the encoder is then fed into the decoder network, wich outputs the translated sentence token-by-token, feeding forward translated tokens to next steps.

![Sequence to Sequence: Machine Translation](docs/img/sequence-to-sequence-machine-translation.png)

#### Image captioning

A CNN can be trained to output a vector that can, instead of a Softmax, then be fed into a decoder that would produce a textual caption for a picture, E.g. as in the following architecture:

![Sequence to Sequence: Image Captioning](docs/img/sequence-to-sequence-image-captioning.png)

#### Finding the most likely sentence

In machine translation we want to find the most likely translation by maximising the probability function that tells that a sentence in a target language comes from the given sentence in a source language.

Machine translation can be posed as a Conditional Language Model problem. It has an encoder in front of decoder, while Language Model starts with a decoder. Instead of generating words based on an input vector of zeros, it takes an input sentence, pipes it though the encoder to get a representation vector, and then feeds this vector to a decoder.

![Machine Translation as a Conditional Language Model](docs/img/sequence-to-sequence-machine-translation-conditional-language-model.png)

### Beam Search

Beam Search is an algorithm that allows finding the most likely sentence. In a more general sense beam search is a heuristic search algorithm that explores a graph by expanding the most promising node in a limited set.

It has a beam width parameter (E.g. B = 3). In each step we examing each input word and find three most likely words that come after it given the underlying input sentence in the source language.

![Machine Translation: Beam Search](docs/img/sequence-to-sequence-machine-translation-beam-search.png)

With Beam Search we always consider and evaluate B (E.g. B = 3) most likely possibilities for each node (word):

![Machine Translation: Beam Search Iterations](docs/img/sequence-to-sequence-machine-translation-beam-search-iterations.png)

Performance of Beam search depends on the hyper-parameter B (beam width). Typically we would expect to see a significant performance improvement when going from B = 1 to B = 3 and B = 10, however, going from B = 1000 to B = 3000, though it is computationally expensive, leads to a small performance improvement.

#### Error Analysis for Beam Search

It is important to be able to tell whether it is the Beam Search that is at fault with regard to a given mis-translation, or it is the RNN.
In order to do so we can take a mistraslated sentence and check whether the probability of the correct translation is higher than that of a mistranslation as seen by RNN:

  - if it is higher, thatn Beam Search has selected a wrong translation;
  - otherwise, we have a case where RNN predicted a lower probability for the correct translation.

![Beam Search: Error Analysis](docs/img/sequence-to-sequence-machine-translation-beam-search-error-analysis.png)

By analysing errors in the Development Set, we can figure out where the majority of errors is coming from, and therefore focus on optimizing corresponding componenet.

### Bleu score (Bilingual Evaluation Understudy)

This is a method for automatic evaluation of machine translation.

In instance of a machine translation is evaluate based on its Bleu Score which is a distance to human-provided translations for the same source sentence.

We could use a trivial metric called Modified Precision to evaluate the quality of Machine Translation. We examine each word in the MT sentence and give ourselves credit if it appears in a human-translated sentence, but make sure to limit the credit to the amount of occurences of a given word in the human-translated sentence, E.g. as follows:

![Evaluating Machine Translation: Modified Precision](docs/img/sequence-to-sequence-machine-translation-modified-precision.png)

Belu Score, on the other hand, operates on bigrams (pairs of words) - we give MT translation a credit only if a bigram appears in one of the reference translations done by a human:

![Evaluating Machine Translation: Bleu Score](docs/img/sequence-to-sequence-machine-translation-bleu-score-bigrams.png)

Going further, Blue Score is evaluated on n-grams.

Combined Bleu Score is calculated as follows. Note that BP (Brevity Penalty) that penalizes translations that are too short (as they would otherwise end up with unfairly high scores):

![Evaluating Machine Translation: Combined Bleu Score](docs/img/sequence-to-sequence-machine-translation-combined-bleu-score.png)

### Attention Model

"The seq2seq models is normally composed of an encoder-decoder architecture, where the encoder processes the input sequence and encodes/compresses/summarizes the information into a context vector (also called as the “thought vector”) of a fixed length. This representation is expected to be a good summary of the entire input sequence. The decoder is then initialized with this context vector, using which it starts generating the transformed output.

A critical and apparent disadvantage of this fixed-length context vector design is the incapability of the system to remember longer sequences."

With Attention Model we are training a bi-directional RNN to learn Attention Weights which indicate how much attention the model should pay to certain input words when producing a target word.

![Attention Model Intuition](docs/img/sequence-to-sequence-machine-translation-attention-model-intuition.png)

### Speech Recognition

Problem: given an audio clip x produce a transcript y.

CTC cost for speech recognition (Connectionist temporal classification): we get a time series of frequencies as an input and train an RNN to produce a time series of characters. Since sampling rate of audio frequencies is quite high - not every input frequency will correspond to a singe character, so we will often have strides of the same character that we can collapse to obtain the resulting transcript.


![Speech Recognition: CTC Cost](docs/img/sequence-to-sequence-speech-recognition-ctc-cost.png)
