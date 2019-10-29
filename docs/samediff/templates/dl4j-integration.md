---
title: How `SameDiff` fits into Deeplearning4j
short_title: DL4J-Integration
description: How to create custom `SameDiff` layers and vertices to employ in standard DL4J networks
category: SameDiff
weight: 5
---


# How `SameDiff` fits into Deeplearning4j
We have already seen [how to use `SameDiff` in a standalone way](./samediff/overview) to create a trainable network. 
However, as we've mentioned there, in order to boost productivity it is advantageous to employ `SameDiff` as a means 
to make custom building blocks within a standard DL4J network. With `SameDiff` you can easily create layers and 
vertices that, once defined, act indistinguishably from the ones in the standard DL4J. We follow the examples provided 
[here](https://github.com/eclipse/deeplearning4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/samediff/dl4j), giving direct code references for each.

So, let's begin.

## Example 1: A simple `SameDiff` layer
[`SameDiff` layer definition code on GitHub](https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/samediff/dl4j/layers/MinimalSameDiffDense.java)
[`SameDiff` layer usage code on GitHub](https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/samediff/dl4j/Ex1BasicSameDiffLayerExample.java)

We start by showing how to make a simple dense layer and then build it into a standard DL4J network. 

### Creating layer class 
We make our layer as a separate class. We name it `MinimalSameDiffDense` here, and it will extend the 
abstract class `SameDiffLayer`
```java
public class MinimalSameDiffDense extends SameDiffLayer 
```
The class will have three fields that we shall serilize - number of inputs, number of outputs and activation:
```java
private int nIn;
private int nOut;
private Activation activation;
```
as well as a weight initialization scheme, that needs not be serialized:
```java
private WeightInit weightInit;
```
We create an all-args constructor, which will be our main entry point for the new layer:
```java
public MinimalSameDiffDense(int nIn, int nOut, Activation activation, WeightInit weightInit){
    this.nIn = nIn;
    this.nOut = nOut;
    this.activation = activation;
    this.weightInit = weightInit;
}
```

There are four abstract methods in the parent class `SameDiffLayer` we need to implement explicitly in any of its 
concrete child classes: these are `defineLayer(...)`, `defineParameters(...)`, `initializeParameters(...)` and 
`getOutputType(...)`. 

### Overriding parameter creation methods

We start with `defineParameters(...)`. In this method we set, what **trainable** parameters will be used in the network. 
For a simple dense layer, those are weight matrix and bias vector, the shape of which will be determined by the fields 
`nIn` and `nOut`:   
```java
@Override 
public void defineParameters(SDlayerParams params) {
    params.addWeightParam(DefaultParamInitializer.WEIGHT_KEY, nIn, nOut);
    params.addBiasParam(DefaultParamInitializer.BIAS_KEY, 1, nOut);
}
```
The keys within `DefaultParamInitializer` are nothing but `String`'s; you thus may thus easily add parameters named with 
keys of your choice. They will be re-employed in the two upcoming methods.  

Next, in `initializeParameters(...)` we tell how the variables introduced in `defineParameters(...)` are to be filled 
with initial values. Here we just use the `weightInit` field we've provided in constructor, as our initialization scheme 
for weights, and fill bias vector with zeros:
```java
@Override
public void initializeParameters(Map<String, INDArray> params) {
    params.get(DefaultParamInitializer.BIAS_KEY).assign(0);
    initWeights(nIn, nOut, weightInit, params.get(DefaultParamInitializer.WEIGHT_KEY));
}
```
Note that we've used two different ways of array initialization: filling with randomized and pre-defined values.

The method `getOutputType(...)` is inherited from the very abstract `Layer` class, and we implement it in a more or less
default way:
```java
@Override
public InputType getOutputType(int layerIndex, InputType inputType) {return InputType.feedForward(nOut);}
```

### Overriding `defineLayer` method

The main logic of you layer's functioning - that is, forward pass, and hence by virtue of working in `SameDiff` also the 
backward pass - is contained in the `defineLayer(...)` method. In our case it is declared as follows:
```java
public SDVariable defineLayer(SameDiff sd, SDVariable layerInput, Map<String, SDVariable> paramTable, SDVariable mask) {
    
    //Pick the predefined weights and bias from the parameter table
    SDVariable weights = paramTable.get(DefaultParamInitializer.WEIGHT_KEY);
    SDVariable bias = paramTable.get(DefaultParamInitializer.BIAS_KEY);
    
    //apply them to the input and use activation function
    SDVariable mmul = sd.mmul("mmul", layerInput, weights);
    SDVariable z = mmul.add("z", bias);
    SDVariabble output = activation.asSameDiff("out", sd, z);
    
    //set the resulting variable as the output
    return output;
}
```
Let us briefly discuss the code. As you may see, the parameter `mask` is never employed here - it is needed for a more 
advanced use. Now, `paramTable` will at execution time contain the variables created in `defineParameters(...)` and 
`initializeParameters(...)`, and so in the first two lines we take them from there. In the next three lines, we apply
the main logic of our layer - the usual guidelines for [operations](./samediff/ops) apply here. What we return is an
`SDVariable` that will contain the output of the layer.

There are several points to observe here, especially if you are familiar with the logic of using `SDVariable`'s 
described in the [variables](./samediff/variables) section. 
- Standard activations normally used for the DL4J layers also have their `SameDiff` counterparts invoked 
via `asSameDiff(...)` function that makes [operations](./samediff/ops) out of them.
- We could have left the methods `defineParameters` and `initializeParameters` empty, creating weights
and bias by using `sd.var(...)` as described in [variables section](./samediff/variables). Thus, the two methods do not
bear any critical structural function, but rather help organizing trainable parameters in a neat way.

### Methods for serialization

In order to make our layer JSON-serializable, we need to add some boilerplate functions. This includes the no-arguments
constructor for your layer - in our case it remains empty - as well as getters and setters 
for each of the fields that matter during the application - here it is `nIn`, `nOut` and `activation`, but not
necessarily `weithInit` as we use it only once. If you employ [lombok](????) in your project, you may just add 
`@Getter`, `@Setter` and `NoArgsConstructor` before your class declaration.

### Building the layer into the network

The layer is now ready to use, and all we need is to build it into the network. For that, we simply add the layer within
the usual layer configuration object, like so:
```java
MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    .updater(new Adam(1e-1))
    .seed(12345)
    .list()
    //Add two custom layers:
    .layer(new MinimalSameDiffDense(networkNumInputs, layerSize, Activation.TANH, WeightInit.XAVIER))
    .layer(new MinimalSameDiffDense(layerSize, layerSize, Activation.TANH, WeightInit.XAVIER))
    //Combine with a standard DL4J output layer
    .layer(new OutputLayer.Builder().nIn(layerSize).nOut(networkNumOutputs).activation(Activation.SOFTMAX)
    .weightInit(WeightInit.XAVIER)
    .lossFunction(LossFunctions.LossFunction.MCXENT).build())
    .build();
```
Note that theoretically we could have
- added more standard DL4J layers at any point of the network;
- put our own `SameDiff` output layer, should we have created one.

And that is it. We have our layers built into the computation graph, and can now train it as usual.  

### Optional methods to override
There are several methods in `SameDifflayer` that may, but should not necessarily be overridden in the concrete 
implementation. Their use goes beyond the scope of this simple example - for advanced use, see our [javadoc](???).

## Example 2: Lambda layer

[`SameDiff` lambda layer definition code on GitHub](https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/samediff/dl4j/layers/L2NormalizeLambdaLayer.java)
[`SameDiff` lambda layer usage code on GitHub](https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/samediff/dl4j/Ex2LambdaLayer.java)

Quite often it is a custom activation for a layer that one wishes to try, and which is not yet present in the pool. The 
cleanest way to create such a layer is to use a combination of a standard with identity activation followed by a 
**lambda layer**, which realizes the custom activation itself.

Lambda layers are even easier to create than the usual `SameDiff` layers: the only method to override there is 
`defineLayer(...)`. In the following example, we define a lambda layer that normalizes the input along specified 
dimensions.
```java
public class L2NormalizeLambdaLayer extends SameDiffLambdaLayer {

    private int[] dimensions;

    public L2NormalizeLambdaLayer(int... dimensions){
        this.dimensions = dimensions;
    }


    @Override
    public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput) {
        SDVariable norm2 = layerInput.norm2(true, dimensions);
        return layerInput.div(norm2);
    }
    
    /* 
    ...
    Add no-args constructor, getters and setters for JSON-serialization here 
    ...
     */
}
```
Observe that our layer class now extends `SameDiffLambdaLayer` that itself is an extension of `SameDiff` layer (though 
in fact it is rather a narrowing). The fields and the constructor require no further comments. Note that `defineLayer` 
now has only two arguments - the `SameDiff` instance and the variable containing input - as activations (normally) have 
no trainable parameters (also, see the  full example code what the `true` token is used for in `norm2` function).

And we are done with creating our lambda layer. Now you may simply build the lambda layer into your network like so:

```java
MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    //...
    // Define global parameters, add some layers
    //... 
    .layer(new ConvolutionLayer.Builder().nIn(1).nOut(16).kernelSize(2,2).stride(1,1).build())
    .layer(new L2NormalizeLambdaLayer(1,2,3))
    //...
    //add more layers, output, loss, etc..
    //...
    .build();
```
And that's it - now these two layers effectively work as a 2d-convolutional layer with a custom activation.

## Example: lambda vertices

[`SameDiff` lambda vertex definition code on GitHub](https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/samediff/dl4j/layers/MergeLambdaVertex.java)
[`SameDiff` lambda vertex usage code on GitHub](https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/samediff/dl4j/Ex3LambdaVertex.java)

**Lambda vertices** in `SameDiff` do pretty much the same thing as lambda layers; the difference is that they can have 
multiple inputs. Lambda vertices extend `SameDiffLambdavertex` abstract class, and the only method to be overridden 
there is `defineVertex(...)`. It works as simple as that:
```java
public class MergeLambdaVertex extends SameDiffLambdaVertex {
    @Override
    public SDVariable defineVertex(SameDiff sameDiff, VertexInputs inputs) {
        //2 inputs to the vertex. The VertexInputs class will dynamically add as many variables as we request from it!
        SDVariable input1 = inputs.getInput(0);
        SDVariable input2 = inputs.getInput(1);
        SDVariable average = sameDiff.math().mergeAvg(input1, input2);
        return average;
    }
}
```
In the network definition, you then just put something like
```java
ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
    
    //... global configuration, some layers
    
    .addLayer("0", new DenseLayer.Builder().nIn(784).nOut(128).activation(Activation.TANH).build(), "in")
    .addLayer("1", new DenseLayer.Builder().nIn(784).nOut(128).activation(Activation.TANH).build(), "in")
    //Add custom lambda merge vertex:
    //Note that the vertex definition expects 2 inputs - and we are providing 2 inputs here
    .addVertex("merge", new MergeLambdaVertex(), "0", "1")
    
    //... some more layers, output
```
where `"in"` is the name of the input layer for the dense layers.

>While `SameDiffLambdaVertex` is in general non-parametric, you may also create parametric vertices by extending the 
superclass `SameDiffVertex`. However, in practice layers, lambda layers and lambda vertices usually suffice for a task
of any complexity. To learn more about `SameDiffVertex`, check our [`SameDiffVertex` source code on GitHub](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/layers/samediff/SameDiffVertex.java).