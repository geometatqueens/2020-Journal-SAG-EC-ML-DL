import numpy as np

#0: Day	   1: FT [Tph]	   2: EC tn [kW]    3: BearingPressure [Psi]	4: SpindleSpeed [Rpm]	   5: Wtr [m3/h]	6: SPe [%]	
#7: DeltaFT [Tph]	8: DeltaSSp [Rpm]	9: EC tn+1 [kW]	      10: EC t_1hr	11: EC t_2hr	12: EC t_4hr	13: EC t_8hr
#14: FT05h	15: FT1h	16: FT2h	17: FT4h	18: FT8h


StatsDsv = [1500, 1390, 1292, 1209, 1137] # EC 0.5, EC 1, EC 2, EC 4, EC 8
#NumHiddenLayers = [4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120]
NumHiddenLayers = np.linspace(4,600,150)
rRMSE = np.zeros((len(StatsDsv),len(NumHiddenLayers)))

CorrCoefMatrix = np.zeros((len(NumHiddenLayers),len(StatsDsv)+1))
CorrCoefMatrix[:,0] = NumHiddenLayers

for ind_1 in range(len(NumHiddenLayers)):    
    for ind_0 in range(len(StatsDsv)):       
        n_hidden = int(NumHiddenLayers[ind_1])
        IndVar = [1, 3, 4, 7, 8]
        Target = int(9+ind_0)
        N_lag = 1
        training_iters = 120
        LocModel = 'Models_1_3_4_7_8/Lag%s_nh%s_targ%s_MLP_V1_3_4_7_8'%(N_lag, n_hidden, Target)     
        File = LocModel + '/Sim_Res_Testing.txt'
        ResFile = np.loadtxt(fname=File)
        temp_RMSE = 100*np.sqrt(np.sum((ResFile[:,1] - ResFile[:,2])**2)/len(ResFile[:,1]))/np.average(ResFile[:,1])
        print("EC: ", ind_0+1, "NumHid: ", n_hidden,  "RMSE: ", 100*np.sqrt(np.sum((ResFile[:,1] - ResFile[:,2])**2)/len(ResFile[:,1]))/np.average(ResFile[:,1]))
        rRMSE[ind_0,ind_1] = temp_RMSE
        temp_CorrCoef = np.corrcoef(ResFile[:,1], ResFile[:,2])
        CorrCoefMatrix[ind_1,ind_0+1] = temp_CorrCoef[1,0]
        print("EC: ", ind_0+1, "NumHid: ", n_hidden,  "CorrCoef: ", temp_CorrCoef[1,0])
  
np.savetxt(fname="CorrCoefMatrix_MLP.txt", X=CorrCoefMatrix, fmt='%.2f', delimiter='\t')        
np.savetxt(fname="rRMSE_MLP.txt", X=rRMSE, fmt='%.2f', delimiter='\t')
            
        