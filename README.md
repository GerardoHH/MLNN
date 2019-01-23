# Morphological-Linear Neural Network


This repository contains the source code of the hybrid neural network model Morphological-Linear Neural Network. For more detail on the implementation and experimental results consult the paper "Hybrid neural networks with morphological neurons and perceptrons".


## Requirements:

	1.- Python 2.6 or greater
	2.- tensorflow with CPU/GPU
	3.- Keras framework 2.2 


It is recommended to use GPU to speed up the training.


## Instructions

	1.- Copy or download the entire repository.
	2.- In a terminal/cmd change to the directory %download_directory%/MLNN/
	3.- Once in the directory %download_directory%/MLNN/ execute the following command:
		
		python -W ignore Hybrid_MNN.py
		
	4.- The classification results of the selected dataset will be shown on the screen.
  
  ## Datasets

The repository has several datasets ready for its classification:

 
##### Dataset
		1.- A  
		2.- B
		3.- XOR
		4.- Iris
		5.- 2C_5L_Spiral   
		6.- 3D_2C_1L_Spiral
		7.- MNIST
	
If you want to classify your own dataset you will have to pre-process it. For pre-processing examples you can consult the Datasource/Datasource.py module. This is the module where the dataset to be classified is loaded into memory.

## Change the dataset classification

To modify the dataset that you want to classify, you will have to modify the file Hybrid_MNN.py. In the main method, the calls for each dataset are already listed and commented. For example, if you want to classify the MNIST dataset you will have to uncomment the line:

	#classify_MNIST()   
	
	to
	
	classify_MNIST()
	
and run the program again.

## Of the reproducibility of results

Like any algorithm whose training is based on stochastic gradient descent. The reproducibility of the experiments can be a problem that depends on the platform (CPU / GPU), software versions, etc.

If after executing the training of the desired dataset the results are not what was expected, the following parameters can be modified with the purpose of obtaining different results.


#### seed for random numbers
The program has a fixed seed for random number generation on the main() method, to obtain a different behavior/results it will be enough to modify or delete this seed.

	def main():
		np.random.seed(12345)   #  modify or eliminate
    
#### hyperparameter configuration for MLNN  model
The MLNN model has two hyperparameter configuration, the number of morphological neurons to be used in the intermediate layer and the learning rate. Because these parameters highly depend on the dataset to be classified, these values are defined for each dataset in their respective method.

For example, for classify  the XOR dataset, these values are found in the method:

	classify_XOR()
  
with values:

    ### Build Model
    dendral_neurons =  6
    lr = 0.08971484393708822   
    
If you wish to add more morphological neurons to the model that classifies the XOR dataset, it will be necesary to increase the value of the variable:

	from 
	
	dendral_neurons =  6
	
	to
	
	dendral_neurons =  10 
     
with which now the model will use 10 morphological neurons. Similarly, if you want to change the value of the learning rate, just modify the line of code:
    
    lr = 0.08971484393708822
     
to some other value.

The last adjustment hyperparameter is the number of epochs, this is found within each method that classifies certain dataset. Therefore, to increase or decrease the number of epochs the model is trained, it is necessary to change the value of the parameter:

    nb_epoch = 100

    of line
    
    [hist, train_time] = bm.train_HybridModel_MLNN( model, lr, P, T, Ptest, Ttest, batch_size = 512, nb_epoch = 100, v_verbose= False )
