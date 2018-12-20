#### Prerequisites

[AWS Sagemaker](https://github.com/aws/sagemaker-python-sdk)  
[Comet.ml](https://www.comet.ml/)  
[TensorFlow](https://www.tensorflow.org/)


#### To Run
Clone this repository
```
git clone https://github.com/comet-ml/comet-sagemaker; cd comet-sagemaker
python main.py
```

#### Hyperparameters

Hyperparameters for this experiment include,

**Momentum**: Momentum term for the optimizer that helps dampen oscillations in the loss. Used to achieve a faster convergence of the loss curve.   
**Resnet Size**: Parameter that controls the number of Resnet blocks in the model. This value must satisfy the condition

```
size % 6 = 2 
```

**Initial Learning Rate**: Initial learning rate for the optimizer  
**Weight Decay**: Regularization parameter that is applied to the loss in order to penalize large weights.  
**Batch Size**: The size of an individual batch of training examples
**Number of Data Batches**: Number of training batches to process  