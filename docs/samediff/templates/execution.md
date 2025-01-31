---
title: Executing standalone SameDiff models
short_title:Execution
description: Describing how to set inputs and resulting outputs for trained or imported SameDiff models.
category: SameDiff
weight: 2
---


# SameDiff graph execution

In the [overview](.samediff/overview) section, we've shown how to create, train and evaluate standalone SameDiff models
i.e. **not necessarily** integrated into a DL4J network (as we discuss [here](.samediff/dl4j-integration)). Here, we 
shall discuss how to make predictions with those models, which may at times be a bit involved. 

## Setting the scene

In what follows, we shall have a `SameDiff` instance named `samediff`, which contains a trained or 
[imported](./samediff/model-import) computation graph. Recall that individual [variable](./samediff/variables) in 
`samediff` has a `String` name that was either explicitly given elsewhere in the code or autogenerated during network 
creation or import. For each `samediff`, you may, if needed, inquire a `List<String>` of names of all the graph inputs 
and outputs by calling
```java
List<String> inputs = samediff.inputs();
List<String> outputs = samediff.outputs();
```
Note that:
- to execute the graph, you need to supply values in form of `INDArray`'s to **each** of the inputs; **however**
- for a given input, you may inquire the value of **any** variable in the graph, not necessarily one of the outputs; all 
you need to tell is its/their `String` name(s).

## An elegant way

### Starting it easy

To make things simple, let us first assume that out network has just one input called `"in"`, and we wish to compute
the value at the vertex named `"out"` for a single input `INDArray` named `features`. It may be done like that:
```java
INDArray prediction = samediff.batchOutput().input("in", features).output("out").execSingle();
```
We have already seen this line in the [overview](./samediff/overview), so let's briefly discuss what it does. The first
`batchOutput` creates an instance of `BatchOutputConfig` class, in which... the output for the graph is being 
configured. In this configuration, we first set the value of the input, the name of the variable we wish to know, and 
then ask for a single output value.

### Multiple outputs

Let's now say, that the network has some intermediate variable named `"vertex"`, which value you also wish to learn. 
For that, you just need to put one more parameter into the `.output` section, like so:
```java
Map<String, INDArray> predictions = samediff.batchOutput().input("in", features).output("out", "vertex").exec();
```
Note that, the computation is performed by the method `exec` instead of `execSingle`, which is reserved for the call 
from a single output. As you see, the result is a map, its keys being the output string names. As the argument type of 
the `output` method is `String ...`, so you may set any number of any vertex names from `samediff` there.

Alternatively, you may add each output individually:
```java
Map<String, INDArray> predictions = samediff.batchOutput()
    .input("in", features)
    .output("out")
    .output("vertex")
    .exec();
``` 
You may also inquire the value of each an every vertex in your graph by setting
```java
Map<String, INDArray> predictions =
samediff.batchOutput()
    .input("in", features)
    .output("out")
    .outputAll()
    .exec();
```

### Multiple inputs

If you have multiple inputs, you will need to specify all of them. One way to do this is to set them as follows:
```java
INDArray prediction = samediff.output()
    .input("in_1", features_1)
    .input("in_2", features_2)
    .output("out")
    .execSingle();        
```
Otherwise, you may pack the input into a `Map<String, INDArray`, with keys being the input names, using the method 
`inputs`, like that:
```java
INDArray prediction = samediff.output()
    .inputs(inputsMap)
    .output("out")
    .execSingle();
```

### Input in batches

Computing output(s) for inputs coming in batches is performed is exactly the same way as for single samples: you just
need to supply all the inputs with `INDArray`'s containing batches of equal length; the output(s) will accordingly 
contain batches of the same length.

## A shortcut way

If you already feel comfortable with the framework, you may bypass creating a `BatchOutputConfig`, and employ methods
from `SameDiff` directly. For that, you will need to specify your inputs as `Map<String, INDArray>`, even if you have
a single input. To get a single output, you may then write

```java
INDArray prediction = samediff.outputSingle(inputs, "output");
```
Similarly, the `samediff.output` will return `Map<String, INDArray>` for multiple outputs. The latter method can take
`List<String>` to specify the list of outputs, so, e.g. by executing
```java
Map<String, INDArray> predictions = samediff.output(inputs, samediff.outputs());
```
you will get the values only for the true output vertices of your network rather than for all the variables at once.