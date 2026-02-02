# A survey of uncertainty in deep neural networks
## Link
https://link.springer.com/article/10.1007/s10462-023-10562-9
## Comments

### Factors of unsertainty
#### Variability in real world situations
The real world changes constantly, resulting in different settings where objects can behave / look differently. 
All situations should be covered sufficiently by the training set. 
If not, the network trained on the data can not garentee a good output for every new real world situation. 
A distribition shift occurs when real world situations differ from the training set. 
These shifts are very hard for a NN to grasp, causing its performence to change significantly.

Presume that there is a dataset created for detecting a sertain object in a factory setting at a set machine during the day. 
When a NN is trained on this dataset, it will be very good at detecting this object in that specific setting. 
However, it might happen that, because of a sudden increase in orders, the machine needs to run overtime and work during the night, moreover more machines at different locations need to help out. 
Because the dataset does not cover the objects at night, nor the settings of the other machines (lighting is different, shadows are cast differently, etc.), the trained NN fails to addapt to these new conditions. 
Therefore, its performence drops, which can lead to all kinds of problems.

#### Error and noise in measurement systems
This comes down to the acquired data having some mistakes in them. 
These can come from limitations of the measurement devices or incorrect / inprecise labeling of the data. 
These mistakes can however be a way to regulize a network, however it needs to be subtle.

To continue the example of the dataset used for object detection on a machine, it could be that the camera used for the capturing of data has a low resolution. 
This could lead to an object, which needed to be detected, getting lost because of the lack of pixels in the image. 
Even more, it could lead to incorrect labeling of an object, as it is labeled as object a, when it is in fact object b. 
Finally it can lead to the bounding box not precisely being placed around the object.

#### Errors in the model structure
The structure of a network differs from model to model, with each model having its advantages and disadvantages.
Each model will be different and thus its output will have a sertain level of unsertainty.

Suppose the factory requires an object detection model to best fit their need. 
There are a lot of models available, think of any CNN network, U-net, etc. 
Each model will have its own capabilities and thus unsertainty.
#### Errors in the training procedure
During the training of a model, there are much decisions that need to be made. 
The learning rate, the number of epochs, the way of using the data, the optimizer used, the weight initialization, etc. 
Each combination of parameters will result in a different model with its own errors and unsertainty.

The factory dataset can be split in different batch sizes, ordered in different ways, etc. 
The chosen model can be trained with these different ways of handeling data, using different parameters to optimize the chosen model. 
Again leading to different models who are better in sertain things and worse in other.
#### Errors caused by unknown data
When the model is fully trained, it can happen that, during inference, the model encounters a situation it has not seen before. 
This situation was not expected in the original scope of the problem, but the model will need to deal with it, causing unsertainty.

In the factory example, the model is trained for a sertain enviorment and objective: detecting sertain objects in a factory setting with variable lighting conditions and different machines.
If an object that was not expected to be in the enviorment, a bird for example, enters the camera's viewpoint, it can cause the model to be unstertain on what to do, since it has never seen a bird before. 
This is not because the enviorment was insufficiently specified, but because of the bird being there unexpectedly.

### Unsertainty estimation
from page 12