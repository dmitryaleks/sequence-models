# Sequence Models

Examples of sequences: text, speech, video.

## Notation

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

Backpropagation pushes losses backwards through the network, calculating derivatives of training parameters and then applying gradient descent to optimize parameters to find the optimal fit.

Defining a loss function, E.g. a cross-entropy loss as in a logistic regression:
```
L<t>(^y<t>, y<t>) = -y(t) * log(^y(t)) - (1 - y(t)) * log(1 - ^y(t))

# a total loss:

L(^y<t>, y<t>) = Sum(L<t>(^y<t>, y<t>))
```

![RNN: Backpropagation through time](docs/img/RNN-backpropagation-through-time.png)

### Different types of RNN architectures

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
