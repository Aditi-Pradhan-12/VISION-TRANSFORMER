***VISION TRANSFORMER PROJECT (ViT)***
***Dataset used: CIFAR (Canadian Institute of Advanced Research)

* It is a deep learning model arcgitecture that applies the TRANSFORMER architecture to ***IMAGE RECOGNITION***     tasks.
** It was initially developed for NLP tasks.
*** It introduces a new mechanism called as #SELF ATTENTION MECHANISM# to process images.
**** This mechanism allows the model to focus on different paerts of the I/P sequence while making the predictions and also it helps in capturing the relationship between the elements of the sequence and enables the parallel working of the model.
***** The input image is broken down into smaller patches that collectively represents the image.

***KEY COMPONENTS***
1. Patch Embedding (input image is divided into ***NON-OVERLAPPING PATCHES*** i.e. separated pixel-to-pixel. Each patch acts as a basic building block of the model)
2. Positional Embedding (it is introduced to retain the spatial info. It encodes the ***relative and absolute position*** of each patch within the image)
3. Transformer Encoder (it is the ***code building block*** of the transformer. It will be having ***multiple identical layers*** using the ***ITERATIVE LOOP***. Each layer will consist of ***2 sub-layers***: 1. self attention mechanism & 2. Feed forward neural network)
    1. SELF-ATTENTION MECHANISM: It allows the model to capture global dependencies and find the relationship between different patches. It calculates the attention score between patches and generates weights representation.
    2. FEED FORWARD NEURAL NETWORK: It is applied for the patch representation. It consists of multiple layer of linear transformation and non-linear activations. It also allows the tranformer to capture complex and non-linear relationship between patches. 
4. Classification Head (the output of the transformer encoder is the sequence of patch represenattion)

***This model uses LABELLED DATASET AND A SUPERVISED LEARNING APPROACH***

*Limitation of ViT: 
***ViT doesnot capture local spatial info as effectively as traditional CNN does***

