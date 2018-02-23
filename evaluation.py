import numpy as np

def evaluation(orig, pred, aggre):
    
    # Relative error in total evergy
    E = np.sum(orig) # relative error in total energy
    Ep = np.sum(pred)
    RETE = abs(Ep-E)/max(Ep,E) 
    
    # Mean absolute error
    MAE = np.sum(np.abs(orig-pred))/(orig.shape[0]*6) # mean absolute error
    
    # Proportion of total energy correctly assigned
    PTECA = 1 - (np.sum(np.abs(orig-pred))/np.sum(aggre))/2
    
    return RETE, MAE, PTECA