---
title: Importing TensorFlow and ONNX models into SameDiff
short_title: Model Import
description: Describes how to import pretrained models from other networks, and limitations of imported models.
category: SameDiff
weight: 7
---
# Getting started: importing models from other frameworks using SameDiff

As we have mentioned in the [overview](.samediff/overview) one of the main reasons to introduce `SameDiff` was to 
make possible importing more  networks from other frameworks. In this part, we briefly discuss how you import models
from other frameworks.

## Importing TensorFlow models

Loading a TensorFlow model from a file may be done in just one line
```java
SameDiff samediff = TFGraphMapper.getInstance().importGraph(new File("C:\\MyTFModles\\myTFModel.pb"));
```
Any valid URL may be used for the file. Instead of calling `TFGraphMapper`, one may also use 
`SameDiff.importFrozenTF(File file)` to load TF models, with the same effect. 

As you can see, the imported model creates an instance of `SameDiff` (and not e.g. of a native DL4J network with 
`SameDiff` elements). The imported graph will be frozen, that is, it will not be trained from within Deeplearning4j 
without further adjustments. The execution of the model is performed as described in the [execution section](./samediff/execution). 
Recall that you will need to know the names given to the input and output vertices, which may be inquired using 
`sd.input()` or `sd.output()`, or alternatively get a printout of the network's summary using `sd.summary()`. 

## Importing ONNX models
**under construction**