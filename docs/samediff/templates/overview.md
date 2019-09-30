---
title: SameDiff - an autodifferentiation engine in Deeplearning4j
short_title: Overview
description: `SameDiff` - its purpose, advantages and limitations.
category: SameDiff
weight: 1
---


# DL4J `SameDiff` computation graph engine

`SameDiff` is a newer module of Deeplearning4j that allows you to increase flexibility of your DL4J-based algorithms 
by creating custom networks or their elements using the general logic of automatic differentiation.

Ok, now that was quite general, let us see what `SameDiff` is designed for and why.

## The challenge
Our aim in Deeplearning4j is to deliver practical solutions: our layers are designed to cope with the most common 
real-life tasks, and indeed you may implement virtually any popular network topology using layers and vertices that 
already exist in our framework. 

But what if you come up with, or stumble upon an entirely fresh idea - of an activation function, a layer, or a whole 
network - and wish to give a it try with Deeplearninig4j, but see that it can not yet be fully covered with the existing
standard DL4J functionality because of the novelty of this idea? One way is to extend the existing Deeplearning4j, but
it requires a lot of work, and the learning curve is steep. Recognizing that this may hinder creative potential of our 
framework, we strove to deliver you a greater flexibility. And `SameDiff` is out catch on that. 

## What is SameDiff?

So, strictly speaking, `SameDiff` is a Deeplearning4j approach to [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation). 
There is no real need for you to obtain an in-depth knowledge on automatic differentiation in order to use `SameDiff` 
methods in your project (if you wish to have them, you may check out e.g. [this recent overview](https://arxiv.org/pdf/1811.05031.pdf))I
In fact, gradient descent through neural network layers via back-propagation is particular kind of automatic 
differentiation. Vaguely speaking, the essential difference of `SameDiff` from the native methods is that, while in the 
classical Deeplearning4j a minimal building block of a network topology is an individual layer or vertex, in `SameDiff` 
it is individual mathematical [operation](./samediff/ops).

With `SameDiff`, we thus address several issues at once:

### Flexibility with productivity
In order to create new layers, or vertices, you need not dig deep into the inner workings of our framework. With
`SameDiff`, you may create new, custom layers, vertices, activations or whole networks within your own project using a 
large pool of [operations](./samediff/ops) and then use them immediately in your network. The pool of contains as simple 
ones as addition or taking a sine function, all the way up to creating an LSTM layer segment with a few lines of code. 
Thus, you may promptly build complex networks while retaining the ability to fine-tune them.

### Importing models from other toolkits
Greater flexibility also allows `SameDiff` to import a larger variety of networks from other frameworks. So, if you have
a network with custom layers in your TensorFlow prototype, which for now refuses be completely imported into the 
standard Deeplearning4j, chances are you'll be able to able to do it with `SameDiff`. ONNX import is of course supported 
as well. We discuss model import in detail [here](./samedif/model_import).

### Compatibility with DL4J
Although `SameDiff` alone suffices to create and train a network of any complexity, it is not thought as a replacement 
of, or a parallel to the native methods. Rather, its primary aim is to enhance the native methods when required. The 
proposed to-go usage of `SameDiff` from within Java is to create **custom layers and vertices**. Those, in turn, may be 
built into your network the same way it is done with standard layers and vertices, in literally the same way. We 
elaborate this in the [integration](./samediff/dl4j-integration) section, which also provides concrete examples of 
creating custom `SameDiff` layers. 

It should be noted, that the compatibility of DL4J and SameDiff is one-way, though. While you may alter values of 
parameters within an instance of `SameDiff` using non-`SameDiff` methods, it will in general produce undesired results 
when training your `SameDiff` (part of the) network. Luckily, all native operations used in deeplearning4j (and perhaps 
even more) now have their `SameDiff` counterparts, so this issue shall not hinder your creativity. 

## Example: dense MNIST classifier with `SameDiff`

But enough talk, let's do a "Hello World" example, which in the realm of neural networks means that we will create a 
simple MNIST classifier unsing entirely `SameDiff` for that. 
We shall follow, though not quite literally, this [example on GitHub](https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/samediff/training/SameDiffTrainingExample.java) 
In process we shall have a first glance on how to employ methods that are discussed in more detail in 
the [variables](./samediff/variables) and [operations](./samediff/ops) sections.  

###Defining topology

In our file, in the `main(...)`, we first create an instance of `Samediff`.
```java
SameDiff samediff = SameDiff.create();
```
`samediff` will be our workhorse, in which we shall keep all our variables and eventually perform all operators - more
strictly, it is the place where our ([computation graph](.samediff/graphs) wil be built.

So, let us set our first variable - an input:
```java
SDVariable input = samediff.placeHolder("input", DataType.FLOAT, -1, 784);
```
Here, we've derived a variable from our `SameDiff` instance. This variable is a [placeholder](./samediff/variables), 
which basically tells the system that we shall supply its values from the outside at each iteration. We have also 
specified its shape: our MNIST pictures will be considered as simple 1d tensors with 784 single precision values, coming 
in batches of varying sizes (the `-1` shape argument).  

Next step is to add a trainable hidden layer. For that, we first introduce another two variables that will contain 
weights and bias:
```java
SDVAriable weights_1 = samediff.var(new XavierInitScheme('c', 784, 100), Datatype.FlOAT, 784, 100);
SDVariable bias_1 = samediff.var(new XavierInitScheme('c', 100), DataType.FLOAT, 100);
```
The variables created using the `.var(...)` function of `SameDiff` retain their values between iterations, and their
values are affected by back-propagation. In other words, we've told the system that we want to *train* weights and bias. 
We have also specified that we would like initial values to be randomly set using Xavier initialization scheme.

Let us now use these variables to create a dense layer with a ReLU activation. To do this, we set:
```java
SDVariable prod_1 = weighs_1.mmul(input);      //matrix multiplication
SDVariabel linear_1 = prod_1.add(bias_1);      //adding bias
SDVariable out_1 = samediff.nn.relu(linear_1); //activation
```
We may also jam this all into just one line:
```java
SDVariable out_1 = samediff.nn.relu(weights_1.mmul(input).add(bias_1));
```
Here, we use [operations](./samediff/ops) to create new variables from the already defined ones. The output layer
is created analogously, though now we use `softmax` activation:
```java
SDVAriable weights_2 = samediff.var(new XavierInitScheme('c', 100, 10), Datatype.FlOAT, 100, 10);
SDVariable bias_2 = samediff.var(new XavierInitScheme('c', 10), DataType.FLOAT, 10);
SDVariable output = samediff.nn.softmax("output", weights_1.mmul(input).add(bias_1));
```
Note that the output variable now has a name. Names are necessary for interaction with network outside of a given 
`SameDiff` instance - first of all for inputs and labels, but also any other variables whose values you wish to monitor 
or re-employ.

So, now that we have an output, we need labels for training. For that, we first create a corresponding variable.
As you may have already guessed, this will again be a placeholder as the labels are supplied from the outside:
```java
SDVariable lables = samediff.placeHolder("lables", 10);
```
To compute gradients, we just need to add a loss function. We do it like that
```java
SDVariable loss = samediff.loss.crossEntropy(output, labels);
```
And that is it with defining the topology. 

### Training and evaluating

To train the network, we first are to define a training configuration. This is done e.g. as follows:
```java
TrainingConfig config = TrainingConfig.Builder()
    .l2(1e-4)
    .updater(new Adam(1e-3))
    .dataSetFeatureMapping("input")
    .dataSetLabelsMapping("lables")
    .build();
samediff.setTrainingConfig(config);
```
In the configuration we tell, which vertices are to be used as inputs and labels during the training, by giving `String` 
names of corresponding variables. Observe that we didn't need to specify the loss function: every variable introduced 
via `.loss` is automatically considered a loss function to minimize. 

Now everything is set up for training and testing, which are almost indistinguishable from those in native DL4J networks.
After setting two `DatasetIterator`, say `trainData` and `testData` we just use
```java
samediff.fit(trainData, numEpochs);
``` 
to train our network for a desired number of epochs. For evaluation, we also need to tell which variable will contain 
the outputs of our network, like so:
```java
samediff.evaluate(testData, "output", new Evaluation());
```
Note that a variable named `output` is to exist within the instance of `SameDiff`. While it may look somewhat 
counterintuitive that you need to tell the output when we've already told the system that exactly these values are to be
taken in the loss function. However, it may not be the case for some advanced networks with e.g. intermediate results
used during training or labels preprocessing, and we reserve this freedom for the user.

Finally, to save and load your model, you simply use
```java
String modelFileName = "sameDiffModel.fd";
samediff.asFlatFile(new File(modelFileName));
samediff samediffFromFile = fromFlatFile(modelFileName);
```

### The full code

So. let us bring all together now:
 
```java
//Define terwork topology
//samediff instance
SameDiff samediff = SameDiff.create();

//Inputs
SDVariable input = samediff.placeHolder("input", DataType.FLOAT, -1, 784);

//Hidden ReLU layer
SDVAriable weights_1 = samediff.var(new XavierInitScheme('c', 784, 100), Datatype.FlOAT, 784, 100);
SDVariable bias_1 = samediff.var(new XavierInitScheme('c', 100), DataType.FLOAT, 100);
SDVariable out_1 = samediff.nn.relu(weights_1.mmul(input).add(bias_1));

//Output SomftMax layer
SDVAriable weights_2 = samediff.var(new XavierInitScheme('c', 100, 10), Datatype.FlOAT, 100, 10);
SDVariable bias_2 = samediff.var(new XavierInitScheme('c', 10), DataType.FLOAT, 10);
SDVariable output = samediff.nn.relu("output", weights_1.mmul(input).add(bias_1));

//Lables and loss
SDVariable lables = samediff.placeHolder("lables", 10);
SDVariable loss = samediff.loss.crossEntropy(output, labels);

//Setting training configuration
TrainingConfig config = TrainingConfig.Builder()
    .l2(1e-4)
    .updater(new Adam(1e-3))
    .dataSetFeatureMapping("input")
    .dataSetLabelsMapping("lables")
    .build();
samediff.setTrainingConfig(config);

//Introduce dataset iterators for train and test
int batchSize = 32;
DataSetIterator trainData = new MnistDataSetIterator(batchSize, true, 12345);
DataSetIterator testData = new MnistDataSetIterator(batchSize, false, 12345);

//Perform training - set your own number of epochs.
int numEpochs = 2;
sameddiff.fit(trainData, numEpochs);

//Evaluate on test set:
String outputVariable = "output";
Evaluation evaluation = new Evaluation();
sd.evaluate(testData, outputVariable, evaluation);

//Print evaluation statistics:
System.out.println(evaluation.stats());

```
In the section about [integration](./samedif/dl4j-integration) we will show how in a similar - and even more simple way
we create not a whole network, but layers, vertices and custom activations.

##What to be aware of?

Well, each freedom comes at price. So, whereas for computing output or performing back-propagation in a single native 
layer a single inference to the `C++`-backend is performed, `SameDiff` has to do it for each elementary operation 
involved in its creation. For instance, applying weights, adding bias and then computing activations results in three 
backend calls with `SameDiff`, whereas a native layer does it in a single run. This increases both running time and 
memory demand. Notice, however, that many popular frameworks are built on the same idea of automatic differentiation, 
and therefore may be prone to similar drawbacks.

Also please bear in mind that `SameDiff` is still in its alpha, some useful functions may still perhaps be lacking, and 
GPU-support comes with the release version. 

If you experience issues or have suggestion, don't hesitate to let us know.