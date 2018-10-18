from scipy.io import loadmat

from keras.utils import np_utils
from keras.datasets import mnist 
from sklearn.cross_validation import train_test_split

from sklearn import preprocessing as preproc

import numpy as np
import os
 
     
def loadDataset_XOR( to_categorical = True ):
           
    abs_path = str(os.path.abspath(__file__) ) 
    abs_path =  abs_path[ 0 :    abs_path.rfind("MLNN") ]
    abs_path = abs_path + 'MLNN/Datasets/XOR/X_OR_Gaussian2_dim.mat'
    
    dict = loadmat( abs_path )
    
    t_p = dict['P']
    t_t = dict['T']
    t_ptest = dict ['Ptest']
    t_ttest = dict ['Ttest']

    t_P = np.array( t_p, dtype = np.float32)    
    t_T = np.array( t_t, dtype = np.int )
    
    t_Ptest = np.array( t_ptest, dtype = np.float32 )
    t_Ttest = np.array( t_ttest, dtype = np.int )
    
    del t_p
    del t_t
    del t_ptest
    del t_ttest


    P = np.zeros( [t_P.shape[1], t_P.shape[0]], np.float32 )
    T = np.zeros( [t_T.shape[1], t_T.shape[0]] , np.int )
    
    Ptest = np.zeros( [t_Ptest.shape[1],t_Ptest.shape[0]], np.float32 )
    Ttest = np.zeros( [t_Ttest.shape[1],t_Ttest.shape[0]], np.int )
    
    for idx in range( P.shape[0] ):
        P[idx ] = t_P[ :, idx ] 
        T[idx] = t_T[:, idx ]

    for idx in range ( Ptest.shape[0] ):
        Ptest[ idx ] = t_Ptest[ :, idx ]
        Ttest[ idx ] = t_Ttest[ :, idx ]

    del t_P
    del t_T
    del t_Ptest
    del t_Ttest

    if ( to_categorical ):
        T = T -1
        Ttest = Ttest -1

        T = np_utils.to_categorical( T , 2)
        Ttest = np_utils.to_categorical(Ttest, 2)
    
    print("\t Dataset XOR Loaded ")
    print("\n\t Training: ")
    print("\t\t ---> P: " + str ( P.shape) )  
    print("\t\t ---> T: " + str ( T.shape) )
    
    print("\n\t Testing ")
    print("\t\t ---> Ptest: " + str( Ptest.shape ))
    print("\t\t ---> Ttest: " + str( Ttest.shape ))

    return  P, T, Ptest, Ttest

def loadDataset_A( to_categorical = True):
      
    abs_path = str(os.path.abspath(__file__) ) 
    abs_path =  abs_path[ 0 :    abs_path.rfind("MLNN") ]
    abs_path = abs_path + 'MLNN/Datasets/A/A.mat'
    
    dict = loadmat( abs_path )
        
    t_p = dict['P']
    t_t = dict['T']
    t_ptest = dict ['Ptest']
    t_ttest = dict ['Ttest']

    t_P = np.array( t_p, dtype = np.float32)    
    t_T = np.array( t_t, dtype = np.int )
    
    t_Ptest = np.array( t_ptest, dtype = np.float32 )
    t_Ttest = np.array( t_ttest, dtype = np.int )
    
    del t_p
    del t_t
    del t_ptest
    del t_ttest


    P = np.zeros( [t_P.shape[1], t_P.shape[0]], np.float32 )
    T = np.zeros( [t_T.shape[1], t_T.shape[0]] , np.int )
    
    Ptest = np.zeros( [t_Ptest.shape[1],t_Ptest.shape[0]], np.float32 )
    Ttest = np.zeros( [t_Ttest.shape[1],t_Ttest.shape[0]], np.int )
    
    for idx in range( P.shape[0] ):
        P[idx ] = t_P[ :, idx ] 
        T[idx] = t_T[:, idx ]

    for idx in range ( Ptest.shape[0] ):
        Ptest[ idx ] = t_Ptest[ :, idx ]
        Ttest[ idx ] = t_Ttest[ :, idx ]

    del t_P
    del t_T
    del t_Ptest
    del t_Ttest

    if ( to_categorical ):
        T = T -1
        Ttest = Ttest -1

        T = np_utils.to_categorical( T , 2)
        Ttest = np_utils.to_categorical(Ttest, 2)

    print("\t Dataset A Loaded ")
    print("\n\t Training: ")
    print("\t\t ---> P: " + str ( P.shape) )  
    print("\t\t ---> T: " + str ( T.shape) )
    
    print("\n\t Testing ")
    print("\t\t ---> Ptest: " + str( Ptest.shape ))
    print("\t\t ---> Ttest: " + str( Ttest.shape ))

    return  P, T, Ptest, Ttest

def loadDataset_B( to_categorical = True):
    
    abs_path = str(os.path.abspath(__file__) ) 
    abs_path =  abs_path[ 0 :    abs_path.rfind("MLNN") ]
    abs_path = abs_path + 'MLNN/Datasets/B/B.mat'
    
    dict = loadmat( abs_path )
            
    t_p = dict['P']
    t_t = dict['T']
    t_ptest = dict ['Ptest']
    t_ttest = dict ['Ttest']

    t_P = np.array( t_p, dtype = np.float32)    
    t_T = np.array( t_t, dtype = np.int )
    
    t_Ptest = np.array( t_ptest, dtype = np.float32 )
    t_Ttest = np.array( t_ttest, dtype = np.int )
    
    del t_p
    del t_t
    del t_ptest
    del t_ttest


    P = np.zeros( [t_P.shape[1], t_P.shape[0]], np.float32 )
    T = np.zeros( [t_T.shape[1], t_T.shape[0]] , np.int )
    
    Ptest = np.zeros( [t_Ptest.shape[1],t_Ptest.shape[0]], np.float32 )
    Ttest = np.zeros( [t_Ttest.shape[1],t_Ttest.shape[0]], np.int )
    
    for idx in range( P.shape[0] ):
        P[idx ] = t_P[ :, idx ] 
        T[idx] = t_T[:, idx ]

    for idx in range ( Ptest.shape[0] ):
        Ptest[ idx ] = t_Ptest[ :, idx ]
        Ttest[ idx ] = t_Ttest[ :, idx ]

    del t_P
    del t_T
    del t_Ptest
    del t_Ttest

    if ( to_categorical ):
        T = T -1
        Ttest = Ttest -1

        T = np_utils.to_categorical( T , 3)
        Ttest = np_utils.to_categorical(Ttest, 3)
    
    print("\t Dataset B Loaded ")
    print("\n\t Training: ")
    print("\t\t ---> P: " + str ( P.shape) )  
    print("\t\t ---> T: " + str ( T.shape) )
    
    print("\n\t Testing ")
    print("\t\t ---> Ptest: " + str( Ptest.shape ))
    print("\t\t ---> Ttest: " + str( Ttest.shape ))

    return  P, T, Ptest, Ttest

def loadDataset_Iris ():
    
    abs_path = str(os.path.abspath(__file__) ) 
    abs_path =  abs_path[ 0 :    abs_path.rfind("MLNN") ]
    abs_path = abs_path + 'MLNN/Datasets/Iris/iris.mat'
    
    dict = loadmat( abs_path )
    
    t_p = dict['P']
    t_t = dict['T']
    t_ptest = dict ['Ptest']
    t_ttest = dict ['Ttest']

    t_P = np.array( t_p, dtype = np.float32)    
    t_T = np.array( t_t, dtype = np.int )
    
    t_Ptest = np.array( t_ptest, dtype = np.float32 )
    t_Ttest = np.array( t_ttest, dtype = np.int )
    
    del t_p
    del t_t
    del t_ptest
    del t_ttest


    P = np.zeros( [t_P.shape[1], t_P.shape[0]], np.float32 )
    T = np.zeros( [t_T.shape[1], t_T.shape[0]] , np.int )
    
    Ptest = np.zeros( [t_Ptest.shape[1],t_Ptest.shape[0]], np.float32 )
    Ttest = np.zeros( [t_Ttest.shape[1],t_Ttest.shape[0]], np.int )
    
    for idx in range( P.shape[0] ):
        P[idx ] = t_P[ :, idx ] 
        T[idx] = t_T[:, idx ]

    for idx in range ( Ptest.shape[0] ):
        Ptest[ idx ] = t_Ptest[ :, idx ]
        Ttest[ idx ] = t_Ttest[ :, idx ]

    del t_P
    del t_T
    del t_Ptest
    del t_Ttest

    T = T -1
    Ttest = Ttest -1

    T = np_utils.to_categorical( T , 3)
    Ttest = np_utils.to_categorical(Ttest, 3)
    
    print("\t Dataset Iris Loaded ")
    print("\n\t Training: ")
    print("\t\t ---> P: " + str ( P.shape) )  
    print("\t\t ---> T: " + str ( T.shape) )
    
    print("\n\t Testing ")
    print("\t\t ---> Ptest: " + str( Ptest.shape ))
    print("\t\t ---> Ttest: " + str( Ttest.shape ))

    return  P, T, Ptest, Ttest
   
def loadDataset_Mnist():
    
    nb_classes = 10
    (P, T), (Ptest, Ttest) = mnist.load_data()
    
        
    P = P.reshape(60000, 784)
    Ptest = Ptest.reshape(10000, 784)
    P = P.astype('float32')
    Ptest = Ptest.astype('float32')
    
    P = preproc.MinMaxScaler().fit_transform(P)     
    Ptest = preproc.MinMaxScaler().fit_transform(Ptest)
    
    T = np_utils.to_categorical( T, nb_classes)
    Ttest = np_utils.to_categorical( Ttest, nb_classes)
    
    
    print("\n\t Dataset Mnist Loaded ")
    print("\n\t Training: ")
    print("\t\t ---> P: " + str ( P.shape) )  
    print("\t\t ---> T: " + str ( T.shape) )
    
    print("\n\t Testing ")
    print("\t\t ---> Ptest: " + str( Ptest.shape ))
    print("\t\t ---> Ttest: " + str( Ttest.shape ))

    return  P, T, Ptest, Ttest

def loadDataset_3D_Espiral( to_categorical = True ):
    
    abs_path = str(os.path.abspath(__file__) ) 
    abs_path =  abs_path[ 0 :    abs_path.rfind("MLNN") ]
    abs_path = abs_path + 'MLNN/Datasets/3D_Spiral/spiral_3D_class_2.mat'
    
    dict = loadmat( abs_path )
    
    
    t_p = dict['P']
    t_t = dict['T']
    
    P = np.zeros( [t_p.shape[1], t_p.shape[0]], np.float32 )
    T = np.zeros( [t_t.shape[1], t_t.shape[0]] , np.int )
    
    for idx in range( P.shape[0] ):
        P[idx ] = t_p[ :, idx ]
        T[idx] = t_t[:, idx ]

    P, Ptest, T, Ttest  =  train_test_split(P, T, test_size=0.2, random_state=4)

    if ( to_categorical ):
        T = T -1          # ESTO ES POR EL CATEGORICAL 
        Ttest = Ttest -1  # ESTO ES POR EL CATEGORICAL

        T = np_utils.to_categorical( T , 2 )
        Ttest = np_utils.to_categorical(Ttest, 2 )
    

    print("\t Dataset Loaded ")
    print("\n\t Training: ")
    print("\t\t ---> P: " + str ( P.shape) )  
    print("\t\t ---> T: " + str ( T.shape) )
    
    
    print("\n\t Testing ")
    print("\t\t ---> Ptest: " + str( Ptest.shape ))
    print("\t\t ---> Ttest: " + str( Ttest.shape ))
    
    return P, T, Ptest, Ttest
        
def loadDataset_Espiral_2Class_N_Loops (  to_categorical = True):

    abs_path = str(os.path.abspath(__file__) ) 
    abs_path =  abs_path[ 0 :    abs_path.rfind("MLNN") ]
    abs_path = abs_path + 'MLNN/Datasets/2_Class_5_Loops_Spiral/espiral_5.mat'
    
    dict = loadmat( abs_path )
    
    t_p = dict['P']
    t_t = dict['T']
    t_ptest = dict ['Ptest']
    t_ttest = dict ['Ttest']
    
    
    t_P = np.array( t_p, dtype = np.float32)
    P = np.array( t_p, dtype = np.float32)
    
    t_T = np.array( t_t, dtype = np.int )
    T = np.array( t_t, dtype = np.int )
    
    t_Ptest = np.array( t_ptest, dtype = np.float32 )
    Ptest = np.array( t_ptest, dtype = np.float32 )
    
    t_Ttest = np.array( t_ttest, dtype = np.int )
    Ttest = np.array( t_ttest, dtype = np.int )
    
    del t_p
    del t_t
    del t_ptest
    del t_ttest
    
    P = np.zeros( [t_P.shape[1], t_P.shape[0]], np.float32 )
    T = np.zeros( [t_T.shape[1], t_T.shape[0]] , np.int )
    
    Ptest = np.zeros( [t_Ptest.shape[1],t_Ptest.shape[0]], np.float32 )
    Ttest = np.zeros( [t_Ttest.shape[1],t_Ttest.shape[0]], np.int )
    
    for idx in range( P.shape[0] ):
        P[idx ] = t_P[ :, idx ] 
        T[idx] = t_T[:, idx ]

    for idx in range ( Ptest.shape[0] ):
        Ptest[ idx ] = t_Ptest[ :, idx ]
        Ttest[ idx ] = t_Ttest[ :, idx ]

    del t_P
    del t_T
    del t_Ptest
    del t_Ttest

    input_dim = P.shape[1]

    T = T -1          # ESTO ES POR EL CATEGORICAL 
    Ttest = Ttest -1  # ESTO ES POR EL CATEGORICAL

    if ( to_categorical ):
        T = np_utils.to_categorical( T , 2 )
        Ttest = np_utils.to_categorical(Ttest, 2 )
    
    #print("\n\n ")
    print("\t Dataset Loaded ")
    print("\n\t Training: ")
    print("\t\t ---> P: " + str ( P.shape) )  
    print("\t\t ---> T: " + str ( T.shape) )
    
    print("\n\t Testing ")
    print("\t\t ---> Ptest: " + str( Ptest.shape ))
    print("\t\t ---> Ttest: " + str( Ttest.shape ))
    
    return P, T, Ptest, Ttest



