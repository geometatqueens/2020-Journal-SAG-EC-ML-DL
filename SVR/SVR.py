import numpy as np
import csv
import random
import time
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


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

# #####################
def Fit_SVR(c_val, IndVar, Target, Stats):
    #print("Polynomial degree:", degree)
    
    # Parameters
    #N_features_input = len(IndVar)
    #N_categ = 1

    # Importing Data Base
    DB_Training = np.loadtxt("SAG_2_Train_Normalized.txt")
    DB_Testing = np.loadtxt("SAG_2_Validation_Normalized.txt")
     
    # Preprocessing
    Train_data_in, Train_data_out = Extract_data(DB_Training, IndVar, Target)     
    Test_data_in, Test_data_out = Extract_data(DB_Testing, IndVar, Target) 
    
    clf = SVR(C=c_val, epsilon=0.1, gamma='scale')
   
    #clf.fit(Train_data_in, Train_data_out)
    clf.fit(Train_data_in, np.ravel(Train_data_out))

    # predicting on training data-set
    y_train_predicted = clf.predict(Train_data_in)

    # predicting on test data-set
    y_test_predict = clf.predict(Test_data_in)

    # evaluating the model on training dataset
    rmse_train = np.sqrt(mean_squared_error(Train_data_out, y_train_predicted))
    r2_train = r2_score(Train_data_out, y_train_predicted)
    coefcorr_train = np.corrcoef(Train_data_out[:,0], y_train_predicted[:])

    # evaluating the model on test dataset
    rmse_test = np.sqrt(mean_squared_error(Test_data_out, y_test_predict))
    r2_test = r2_score(Test_data_out, y_test_predict)
    coefcorr_test = np.corrcoef(Test_data_out[:,0], y_test_predict[:])
    

    a = 16710
    b = int(Stats)
    #print("T:",Target," a:",a," b:",b)
    # Preprocessing
    Num_Elements = int(Test_data_in.shape[0])
    Results = np.zeros((Num_Elements,4)) # day, real, estimated, diff
    for ind1 in range(Num_Elements):
        Sim_in_real = Test_data_in[ind1:(ind1+1),:]
        Sim_out_real = Test_data_out[ind1:(ind1+1),:]
        One_pred = clf.predict(Sim_in_real)
        Real = Sim_out_real[0]
        #print("Actual consumption [%.3f] vs Predicted consumption: [%.3f]" % (Real, One_pred))
        Results[ind1,0], Results[ind1,1], Results[ind1,2], Results[ind1,3] = (ind1+1), int(Real*b+a), int(One_pred*b+a), (int(Real*b+a) - int(One_pred*b+a))
    np.savetxt('Models_1_3_4_7_8/Sim_Res_Validation_%s_%s.txt'%(c_val,Target), Results, fmt='%3.4e', delimiter='\t', newline='\n')      
    
    return None #print("Fitting model done")



# "t" Parameters
#0: Days
#1: Feed Tonnage [Tph]	2: Energy Consumption [kW]	3: Support Pressure [Psi]	
#4: Spindle Speed [Rpm]	     5: Water [m3/h]      	6: Solid Percentage [%]
# Delta
#7: Feed Tonnage [Tph]	8: Spindle Speed [Rpm]	
# "t+1" parameters
#9: Energy n+1

#0: Day	   1: FT [Tph]	   2: EC tn [kW]    3: BPr [Psi]	4: SSp [Rpm]	   5: Wtr [m3/h]	6: SPe [%]	
#7: DeltaFT [Tph]	8: DeltaSSp [Rpm]	9: EC tn+1 [kW]	      10: EC t_1hr	11: EC t_2hr	12: EC t_4hr	13: EC t_8hr
StatsDsv = [1504, 1410, 1321, 1242, 1174] # EC 0.5, EC 1, EC 2, EC 4, EC 8
CconstVec = [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,5000,10000]
#CconstVec = [500,1000,5000,10000]
TargetVec = [9,10,11,12,13]

for ind_1 in range(len(CconstVec)):
    for ind_0 in range(len(TargetVec)):  
        IndVar = [1, 3, 4, 7, 8]
        #IndVar = [1, 2, 3, 4]   
        #IndVar = [1, 3, 4, 7, 8]            
        Target = int(TargetVec[ind_0]) 
        CValue = float(CconstVec[ind_1])
        print("Target: ", Target, "CValue; ", CValue, "\n")
        Fit_SVR(c_val=CValue, IndVar=IndVar, Target=Target, Stats=StatsDsv[ind_0])
        
        