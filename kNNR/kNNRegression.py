import numpy as np
import csv
import random
import time


# ##################
# Preprocessing
# ##################

def Extract_data(DB_SolarMining, IndVar, Target):
    N_features_input = len(IndVar)
    Train_data_in = np.zeros((len(DB_SolarMining[:,0]), N_features_input))
    Train_data_out = np.zeros((len(DB_SolarMining[:,0]), 1))
    for i in range(len(DB_SolarMining[:,0])):
        for ind0 in range(N_features_input):
            Train_data_in[i,ind0] = DB_SolarMining[i:i+1,IndVar[ind0]]
        Train_data_out[i,:] = DB_SolarMining[i,Target]
    return Train_data_in, Train_data_out


def Pred_k_closest(InData, Train_data_in, Train_data_out, kvalue, aVal, bVal):
    distArr = np.sum((Train_data_in - InData)**2, axis=1)
    pos = np.argpartition(distArr,kvalue)[:kvalue]
    Beta = (0.5*np.average(distArr[pos])+0.00000001)
    PhiElem = np.exp(-1*(distArr[pos])/Beta)
    LabEne = bVal*Train_data_out[pos]+aVal
    predEnergy = np.sum(np.multiply(LabEne[0],PhiElem))/np.sum(PhiElem)
    return predEnergy

# #####################
def kNN_regression(kvalue, IndVar, Target, Stats):
    
    # Parameters
    #N_features_input = len(IndVar)
    #N_categ = 1

    # Importing Data Base
    DB_Training = np.loadtxt("SAG_2_Train_Normalized.txt")
    DB_Testing = np.loadtxt("SAG_2_Validation_Normalized.txt")
    #DB_Training = DB_SolarMining
     
    # Preprocessing
    Train_data_in, Train_data_out = Extract_data(DB_Training, IndVar, Target)     
    Test_data_in, Test_data_out = Extract_data(DB_Testing, IndVar, Target) 

    aVal = 16710
    bVal = int(Stats)
    Num_Elements = int(Test_data_in.shape[0])
    Results = np.zeros((Num_Elements,4)) # day, real, estimated, diff
    for ind1 in range(Num_Elements):
        Sim_in_real = Test_data_in[ind1:(ind1+1),:][0]
        Sim_out_real = Test_data_out[ind1:(ind1+1),:]
        One_pred = Pred_k_closest(InData=Sim_in_real, Train_data_in=Train_data_in, Train_data_out=Train_data_out, kvalue=kvalue, aVal=aVal, bVal=bVal)
        Real = bVal*Sim_out_real[0]+aVal
        #print("Actual [%.1f] vs Predicted [%.1f]" % (Real, One_pred))
        Results[ind1,0], Results[ind1,1], Results[ind1,2], Results[ind1,3] = (ind1+1), int(Real), int(One_pred), (int(Real) - int(One_pred))
    np.savetxt('Models_1_3_4_7_8/Sim_Res_Validation_%s_%s.txt'%(kvalue,Target), Results, fmt='%3.4e', delimiter='\t', newline='\n')      
    return print("kNN regression done \n")


#0: Day	   1: FT [Tph]	   2: EC tn [kW]    3: BPr [Psi]	4: SSp [Rpm]	   5: Wtr [m3/h]	6: SPe [%]	
#7: DeltaFT [Tph]	8: DeltaSSp [Rpm]	9: EC tn+1 [kW]	      10: EC t_1hr	11: EC t_2hr	12: EC t_4hr	13: EC t_8hr
StatsDsv = [1504, 1410, 1321, 1242, 1174] # EC 0.5, EC 1, EC 2, EC 4, EC 8
#k_value = [1,2,3,4,5,6,7,8,9,10]
#k_value = [1,2,3,4,5]
k_value = np.linspace(1,100,100)
TargetVec = [9,10,11,12,13]

for ind_0 in range(len(TargetVec)):  
    for ind_1 in range(len(k_value)):
        #IndVar = [1, 2, 3, 4, 7, 8]
        #IndVar = [1, 2, 3, 4]   
        IndVar = [1, 3, 4, 7, 8]            
        Target = int(TargetVec[ind_0]) 
        ValueOfK = int(k_value[ind_1])
        print("Target: ", Target, "ValueOfK; ", ValueOfK)
        kNN_regression(kvalue=ValueOfK, IndVar=IndVar, Target=Target, Stats=StatsDsv[ind_0])
        