import numpy as np

#0: Day	   1: FT [Tph]	   2: EC tn [kW]    3: BearingPressure [Psi]	4: SpindleSpeed [Rpm]	   5: Wtr [m3/h]	6: SPe [%]	
#7: DeltaFT [Tph]	8: DeltaSSp [Rpm]	9: EC tn+1 [kW]	      10: EC t_1hr	11: EC t_2hr	12: EC t_4hr	13: EC t_8hr
#14: FT05h	15: FT1h	16: FT2h	17: FT4h	18: FT8h
      
k_value = np.linspace(1,100,100)
#k_value = [1,2,3,4,5]
TargetVec = [9,10,11,12,13]
rRMSE = np.zeros((len(TargetVec),len(k_value)))

CorrCoefMatrix = np.zeros((len(k_value),len(TargetVec)+1))
CorrCoefMatrix[:,0] = k_value

for ind_0 in range(len(TargetVec)):  
    for ind_1 in range(len(k_value)):         
        Target = int(TargetVec[ind_0]) 
        KVal = int(k_value[ind_1])
        LocModel = 'Models_1_3_4_7_8'
        File = LocModel + '/Sim_Res_Validation_%s_%s.txt'%(KVal,Target)
        ResFile = np.loadtxt(fname=File)
        temp_RMSE = 100*np.sqrt(np.sum((ResFile[:,1] - ResFile[:,2])**2)/len(ResFile[:,1]))/np.average(ResFile[:,1])
        print("KVal: ", KVal, "Target: ", Target,  "RMSE: ", temp_RMSE)
        rRMSE[ind_0,ind_1] = temp_RMSE
        temp_CorrCoef = np.corrcoef(ResFile[:,1], ResFile[:,2])
        CorrCoefMatrix[ind_1,ind_0+1] = temp_CorrCoef[1,0]
        print("EC: ", ind_0+1, "KVal: ", KVal,  "CorrCoef: ", temp_CorrCoef[1,0])        
np.savetxt(fname="rRMSE_kNNReg.txt", X=rRMSE, fmt='%.2f', delimiter='\t')
np.savetxt(fname="CorrCoefMatrix_kNNReg.txt", X=CorrCoefMatrix, fmt='%.2f', delimiter='\t')
