# ICGAN 
## AN IMPLICIT CONDITIONING METHOD FOR INTERPRETABLE FEATURE CONTROL OF NEURAL AUDIO SYNTHESIS

Generative models are typically conditioned on discrete labels, especially one-hot vectors when conditioned on different classes. However, such labels have drawbacks in limited expressiveness, lack of continuity, and failure to capture the hierarchical semantic relationships between classes. We propose to condition neural audio synthesis models on continuous vectors sampled from Gaussian, which is parameterized by the learned mean and variance from an encoder classifier. Once the model is completely trained, users can interpolate the conditioning vector to morph the sounds among different categories even though the class labels are binary.

## Model architecture
![Architecture](https://github.com/Reinliu/ICGAN/assets/50271800/77481b98-7ead-4ec1-ab51-0b566e2868a0)


### Environment configuration:
~~~
pip install -r requirements.txt
~~~

### Preprocess:
Configure appropriate parameters for the preprocessing and training using config.json.
You should ensure that your dataset folder contains subcategories, because our model integrates class labels as conditioning vectors.
Run 'python preprocess.py' and it will save the preprocessed features and data into a folder called 'preprocessed'

### Train:
Run 
~~~
python train.py
~~~
to train your model. 

### Generate:
Run 
~~~
python generate.py
~~~
