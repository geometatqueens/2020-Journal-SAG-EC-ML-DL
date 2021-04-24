import numpy as np

#0: Day	   1: FT [Tph]	   2: EC tn [kW]    3: BearingPressure [Psi]	4: SpindleSpeed [Rpm]	   5: Wtr [m3/h]	6: SPe [%]	
#7: DeltaFT [Tph]	8: DeltaSSp [Rpm]	9: EC tn+1 [kW]	      10: EC t_1hr	11: EC t_2hr	12: EC t_4hr	13: EC t_8hr
#14: FT05h	15: FT1h	16: FT2h	17: FT4h	18: FT8h

TargetVec = [9,10,11,12,13]
StatsDsv = [1500, 1390, 1292, 1209, 1137] # EC 0.5, EC 1, EC 2, EC 4, EC 8
CconstVec = [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,5000,10000]

CorrCoefMatrix = np.zeros((len(CconstVec),len(StatsDsv)+1))
CorrCoefMatrix[:,0] = CconstVec

rRMSE = np.zeros((len(StatsDsv),len(CconstVec)))
for ind_1 in range(len(CconstVec)):    
    for ind_0 in range(len(StatsDsv)):    
        Target = int(TargetVec[ind_0]) 
        CValue = float(CconstVec[ind_1])
        File = 'Models_1_3_4_7_8/Sim_Res_Validation_%s_%s.txt'%(CValue,Target) 
        ResFile = np.loadtxt(fname=File)
        rRMSE[ind_0,ind_1] = 100*np.sqrt(np.sum((ResFile[:,1] - ResFile[:,2])**2)/len(ResFile[:,1]))/np.average(ResFile[:,1])
        temp_CorrCoef = np.corrcoef(ResFile[:,1], ResFile[:,2])
        CorrCoefMatrix[ind_1,ind_0+1] = temp_CorrCoef[1,0]
  
np.savetxt(fname="CorrCoefMatrix_SVR.txt", X=CorrCoefMatrix, fmt='%.3f', delimiter='\t')
np.savetxt(fname="rRMSE_SVR.txt", X=rRMSE, fmt='%.3f', delimiter='\t')
            
        