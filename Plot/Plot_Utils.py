
import matplotlib.pyplot as plt
import numpy as np

def my_plot_train_loss( history):

    plt.figure(1)
    plt.subplot(211)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')

    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    

    plt.subplot(212)
    
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    
    plt.show()
    

def plot_decision_boundary_2_class(P, model, batch_size, h = 0.05, half_dataset = False, expand = 0.0, x_lim = 10, y_lim =10):
    xmin, xmax = P[:, 0].min(), P[:, 0].max()
    ymin, ymax = P[:, 1].min(), P[:, 1].max()
    
    xmin = xmin + xmin*expand
    xmax = xmax + xmax*expand
    
    ymin = ymin + ymin*expand
    ymax = ymax + ymax*expand
    
    #dx, dy = (xmax - xmin)*0.1, (ymax - ymin)*0.1
    
     #create mesh 
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h),
                         np.arange(ymin, ymax, h))
    
    z = model.predict(np.c_[xx.ravel(), yy.ravel()] , batch_size ) # default batch size 32

    z_class = []
    
    for idx in range(len(z) ):
        if ( z[idx, 0] >= z[idx, 1] ):
            z_class.append( 0 )
             
        if ( z[idx, 1] > z[idx, 0] ):
            z_class.append( 1 )
    
    z_t = np.array( z_class, dtype = np.float32 )
    z_t = z_t.reshape(xx.shape)
    
    plt.contour(xx, yy, z_t, colors='k') 
    
    if ( half_dataset ):
        half = int(P.shape[0]/2)
        plt.scatter(P[0: half, 0], P[0: half, 1]  , cmap=plt.cm.Spectral, s=1)
        plt.scatter(P[half:half*2, 0], P[half:half*2, 1]  , cmap=plt.cm.Spectral, s=1)
    else:
        plt.scatter(P[:, 0], P[:, 1], cmap=plt.cm.Spectral, s=1)
    
    plt.grid(axis= 'both')
    plt.xlim( -x_lim, x_lim)
    plt.ylim( -y_lim, y_lim)
    
    plt.show()
    
