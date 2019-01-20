# AdamW

Implemenation of DECOUPLED WEIGHT DECAY REGULARIZATION [https://arxiv.org/pdf/1711.05101.pdf] for Keras

You can easily import AdamW and use it as a Keras optimizer or you can use create_decouple_optimizer to decouple weight decay for any keras optimizer.

Because we need to change weight decay value based on the learning rate scheduler, don't forget to add WeightDecayScheduler to the list of callbacks. You can take a look at resnet example for more information.
