'''
@author: robotica
'''

import numpy as np

from Datasource import Datasource as dts

from Training import BuildModel as bm

from Plot import Plot_Utils as plt_util


def classify_A():
    
    #Load Dataset
    P, T, Ptest, Ttest = dts.loadDataset_A( )

    input_shape = (P.shape[1],)     
    output_shape = T.shape[1]     # Number of Dense neurons at output layer    
    
    ### Build Model
    dendral_neurons = 9
    lr = 0.08629237289
    activation = 'tanh'
    
     
    model = bm.build_HybridModel_MLNN( dendral_neurons, activation, input_shape, output_shape  )
    
    [hist, train_time] = bm.train_HybridModel_MLNN( model, lr, P, T, Ptest, Ttest, batch_size = 512, nb_epoch = 100, v_verbose= False )
    
    print("\n\t Dataset A: ")
    
    print("\n\t Classificacion: " + str(hist.history['val_acc'][-1]) )
         
    plt_util.my_plot_train_loss(hist)

def classify_B():
    
    #Load Dataset
    P, T, Ptest, Ttest = dts.loadDataset_B( )

    input_shape = (P.shape[1],)     
    output_shape = T.shape[1]     # Number of Dense neurons at output layer    
    
    ### Build Model
    dendral_neurons = 90
    lr = 0.1829408228    
    activation = 'tanh'
    
     
    model = bm.build_HybridModel_MLNN( dendral_neurons, activation, input_shape, output_shape  )
    
    [hist, train_time] = bm.train_HybridModel_MLNN( model, lr, P, T, Ptest, Ttest, batch_size = 512, nb_epoch = 100, v_verbose= False )
    
    print("\n\t Dataset B: ")
    
    print("\n\t Classificacion: " + str(hist.history['val_acc'][-1]) )
         
    plt_util.my_plot_train_loss(hist)

def classify_XOR():
    #Load Dataset
    P, T, Ptest, Ttest = dts.loadDataset_XOR( )

    input_shape = (P.shape[1],)     
    output_shape = T.shape[1]     # Number of Dense neurons at output layer    
    
    ### Build Model
    dendral_neurons =  6
    lr = 0.08971484393708822      
    activation = 'tanh'
    

    model = bm.build_HybridModel_MLNN( dendral_neurons, activation, input_shape, output_shape  )
    
    [hist, train_time] = bm.train_HybridModel_MLNN( model, lr, P, T, Ptest, Ttest, batch_size = 512, nb_epoch = 100, v_verbose= False )
    
    print("\n\t Dataset XOR: ")
    
    print("\n\t Classificacion: " + str(hist.history['val_acc'][-1]) )
         
    plt_util.my_plot_train_loss(hist)
    
def classify_Iris():
     #Load Dataset
    P, T, Ptest, Ttest = dts.loadDataset_Iris()

    input_shape = (P.shape[1],)     
    output_shape = T.shape[1]     # Number of Dense neurons at output layer    
    
    ### Build Model
    dendral_neurons =  10
    lr = 0.05      
    activation = 'tanh'
    
    model = bm.build_HybridModel_MLNN( dendral_neurons, activation, input_shape, output_shape  )
    
    [hist, train_time] = bm.train_HybridModel_MLNN( model, lr, P, T, Ptest, Ttest, batch_size = 512, nb_epoch = 120, v_verbose= False )
    
    print("\n\t Dataset Iris: ")
    
    print("\n\t Classificacion: " + str(hist.history['val_acc'][-1]) )
         
    plt_util.my_plot_train_loss(hist)
    
def classify_2C_5L_Spiral():
    #Load Dataset
    P, T, Ptest, Ttest = dts.loadDataset_Espiral_2Class_N_Loops()

    input_shape = (P.shape[1],)     
    output_shape = T.shape[1]     # Number of Dense neurons at output layer    
    
    ### Build Model
    dendral_neurons =  250
    lr = 0.2
    activation = 'tanh'
    
    batch_size = 512
    
    model = bm.build_HybridModel_MLNN( dendral_neurons, activation, input_shape, output_shape  )
    
    [hist, train_time] = bm.train_HybridModel_MLNN( model, lr, P, T, Ptest, Ttest, batch_size = batch_size, nb_epoch = 1000, v_verbose= False )
    
    print("\n\t Dataset 2 class  5 Loops  spiral : ")
    
    print("\n\t Classificacion: " + str(hist.history['val_acc'][-1]) )
         
    plt_util.my_plot_train_loss(hist)
    
    
    plt_util.plot_decision_boundary_2_class(P, model, batch_size, h = 0.05, half_dataset = True, expand = 0.5, x_lim= 45, y_lim=45)

    
    
def classify_3D_2C_1L_Spiral():
    #Load Dataset
    P, T, Ptest, Ttest = dts.loadDataset_3D_Espiral( )

    input_shape = (P.shape[1],)     
    output_shape = T.shape[1]     # Number of Dense neurons at output layer    
    
    ### Build Model
    dendral_neurons =  100
    lr = 0.05
    activation = 'tanh'
    
    model = bm.build_HybridModel_MLNN( dendral_neurons, activation, input_shape, output_shape  )
    
    [hist, train_time] = bm.train_HybridModel_MLNN( model, lr, P, T, Ptest, Ttest, batch_size = 512, nb_epoch = 100, v_verbose= False )
    
    print("\n\t Dataset 3D spiral : ")
    
    print("\n\t Classificacion: " + str(hist.history['val_acc'][-1]) )
         
    plt_util.my_plot_train_loss(hist)
    
    
def classify_MNIST():
     #Load Dataset
    P, T, Ptest, Ttest = dts.loadDataset_Mnist()

    input_shape = (P.shape[1],)     
    output_shape = T.shape[1]     # Number of Dense neurons at output layer    
    
    ### Build Model
    dendral_neurons =  500
    lr = 0.01694132 
    activation = 'tanh'
    
    model = bm.build_HybridModel_MLNN( dendral_neurons, activation, input_shape, output_shape  )
    
    [hist, train_time] = bm.train_HybridModel_MLNN( model, lr, P, T, Ptest, Ttest, batch_size = 512, nb_epoch = 20, v_verbose= False )
    
    print("\n\t Dataset MNIST: ")
    
    print("\n\t Classificacion: " + str(hist.history['val_acc'][-1]) )
         
    plt_util.my_plot_train_loss(hist)
    
def main():

    np.random.seed(12345)
    
    #classify_A()
    
    #classify_B()
    
    #classify_XOR()
    
    #classify_Iris()
    
    classify_2C_5L_Spiral()
    
    #classify_3D_2C_1L_Spiral()
    
    #classify_MNIST()
    
    
    ###############################
    ####### PLOT RESULTS  AND  DESCISSION BOUNDARY
    
        
    #if (output_shape >= 3 and input_shape[0] == 2):
    #    plot_utils.plot_decision_boundary_3_class(P, model, batch_size, h = 0.05, half_dataset = True, path_save=path_save, expand = pr_expand, dashed = True )
    
    #if ( input_shape[0] == 3):
    #    #plot_utils.plot_decision_boundary_2_class_3D(P_ori, model, batch_size, h = 0.08, half_dataset = True, path_save=path_save, expand = pr_expand)
    #    #plot_utils.plot_decision_boundary_2_class_3D(P_ori, model, batch_size, h = 0.5, half_dataset = True, path_save=path_save, expand = pr_expand)
     
    
    print(" Done ... ")

if __name__ == "__main__":
    main()