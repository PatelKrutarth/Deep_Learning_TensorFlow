
#### Image_Classification_Parameter_Tuning
- Using TensorFlow with abstraction.
 - Part I, Preparation: load the CIFAR-10 dataset.
 - Part II, Barebone TensorFlow: Abstraction Level 1, we will work directly with low-level TensorFlow graphs.
 - Part III, Keras Model API: Abstraction Level 2, we will use tf.keras.Model to define arbitrary neural network architecture.
 - Part IV, Keras Sequential and Functional API: Abstraction Level 3, we will use tf.keras.Sequential to define a linear feed-forward network very conveniently, and then explore the functional libraries for building unique and uncommon models that require more flexibility.
 - Part V, Tuning: Experiment with different architectures, activation functions, weight initializations, optimizers, hyperparameters, regularizations or other advanced features. 

#### Imojifier
- Classification of short sentences with respect to five emojis.
- In the model, each string is represented as the average, min, and max (element-wise) of the embeddings of the words in the sentence. Then after fully connected layer is used for the classification task. train/validation/test datasets are provided in the directory.
- Pretrained Word2vec and GloVe word embeddings are used in the context of a text classification problem.
	
#### Sentiment_Classification
- Implementation of RNN, LSTM, and GRU based sentiment classifiers using character level input. The element-wise max of all hidden states is used as a sentense respresentation for the classification task.
- The Stanford Sentiment Treebank (SST-5) dataset is used for train/validation/test. Provided in the directory.
	
#### Language_Modeling
- Implementation of GRU based language modeling for character level text prediction task.
- The Stanford Sentiment Treebank (SST-5) dataset is used for train/validation/test. Provided in the directory.
	
#### Disaster_Image_Classification_Finetuning_pretrained_CNN
- Used pretrained VGG16 Model, and also implemented noisy student-teacher model for exploiting unlabeled data for further training the classifier.
- The dataset is available at https://crisisnlp.qcri.org/#resource9. Train/validation/test subsets are provided for each event in the dataset. In the original dataset, there are three damage classes: severe, mild, and none. However, Nguyen et al. (2017) suggested that the task of discriminating between mild and severe damage is very subjective, and there is significant overlap in the dataset between the two classes. Therefore, severe and mild classes are combined into one class called damage.
	
	
