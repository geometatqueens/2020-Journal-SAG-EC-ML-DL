# Do not display the AVX message about using GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import numpy as np
import csv
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time
import os

# Functions


# Forward pass for the recurrent neural network
def RNN(x, weights, biases, N_lag, N_units):

    x = tf.unstack(x, N_lag, 1)

    # 3 layeres LSTM definition
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(num_units=n) for n in N_units])    
    #rnn_cell = rnn.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper([rnn.BasicLSTMCell(num_units=n) for n in N_units], output_keep_prob=0.5)])    
    #rnn_cell = rnn.MultiRNNCell([rnn.GRUCell(num_units=n) for n in N_units])    
    #rnn_cell = rnn.MultiRNNCell([rnn.LayerNormBasicLSTMCell(num_units=n) for n in N_units])    
    
    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    pred_LSTM = tf.add(tf.matmul(outputs[-1], weights['out']), biases['out'], name="pred_LSTM")
    return pred_LSTM

# ##################
# Preprocessing
# ##################

def Extract_data(DB_SolarMining, N_lag, IndVar, Target):
    N_features_input = len(IndVar)
    Train_data_in = np.zeros((len(DB_SolarMining[:,0]) - N_lag + 1, N_lag, N_features_input))
    Train_data_out = np.zeros((len(DB_SolarMining[:,0]) - N_lag + 1, 1))
    for i in range(len(DB_SolarMining[:,0]) - N_lag+1):
        for ind0 in range(N_features_input):
            Train_data_in[i,:,ind0] = DB_SolarMining[i:i+N_lag,IndVar[ind0]]
        Train_data_out[i,:] = DB_SolarMining[i+N_lag-1,Target]
    return Train_data_in, Train_data_out

# #####################

def Simulation_SolarRecurrentModel(LocModel, DB_TotalDB, IndVar, Target, N_lag, Stats):
    tf.reset_default_graph() # A brand new graph each run
    with tf.device('/gpu:0'):
        a = 16710
        b = int(Stats)
        print("T:",Target," a:",a," b:",b)
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5 # Testing
        session = tf.InteractiveSession(config = config) 
        TempLocModel = LocModel+'/FeatMaps'
        saver = tf.train.import_meta_graph(TempLocModel+'.meta')
        LocModelOutput = LocModel + '/Sim_Res_Testing.txt'
        saver.restore(session, save_path=TempLocModel)
        graph = tf.get_default_graph()    
        # Preprocessing
        Simulation_data_in, Simulation_data_out = Extract_data(DB_TotalDB, N_lag, IndVar, Target) 
        X_Input = graph.get_tensor_by_name("X_Input:0")
        pred = graph.get_tensor_by_name("pred_LSTM:0")
        Num_Elements = int(Simulation_data_in.shape[0] - N_lag+1)
        Results = np.zeros((Num_Elements,4)) # day, real, estimated, diff
        for ind1 in range(Num_Elements):
            Sim_in_real = Simulation_data_in[ind1:(ind1+1),:,:]
            Sim_out_real = Simulation_data_out[ind1:(ind1+1),:]
            One_pred = session.run([pred], feed_dict={X_Input: Sim_in_real})[0]  
            Real = Sim_out_real[0]
            #print("Actual consumption [%.3f] vs Predicted consumption: [%.3f]" % (Real, One_pred))
            Results[ind1,0], Results[ind1,1], Results[ind1,2], Results[ind1,3] = (ind1+1), int(Real*b+a), int(One_pred*b+a), (int(Real*b+a) - int(One_pred*b+a))
        np.savetxt(LocModelOutput, Results, fmt='%3.4e', delimiter='\t', newline='\n')
        tf.get_default_graph().finalize()
        session.close()        
    return print("Simulation Completed!")


#0: Day	   1: FT [Tph]	   2: EC tn [kW]    3: BearingPressure [Psi]	4: SpindleSpeed [Rpm]	   5: Wtr [m3/h]	6: SPe [%]	
#7: DeltaFT [Tph]	8: DeltaSSp [Rpm]	9: EC tn+1 [kW]	      10: EC t_1hr	11: EC t_2hr	12: EC t_4hr	13: EC t_8hr
#14: FT05h	15: FT1h	16: FT2h	17: FT4h	18: FT8h

StatsDsv = [1504, 1410, 1321, 1242, 1174] # EC 0.5, EC 1, EC 2, EC 4, EC 8
NumHiddenLayers = [596, 376, 576, 260, 488]

#for ind_1 in range(len(NumHiddenLayers)):    
for ind_0 in range(5):    
    print("Goes in ", ind_0)
    n_hidden = int(NumHiddenLayers[ind_0])
    N_units = [n_hidden]
    NumRecurrent = len(N_units)        
    IndVar = [1, 3, 4, 7, 8]
    Target = int(9+ind_0)
    N_lag = 8
    training_iters = 51
    LocModel = 'Models_1_3_4_7_8/Lag%s_nh%sx%st_LSTM_V1_3_4_7_8_T%s_E%sv1'%(N_lag, n_hidden, NumRecurrent, Target, training_iters)     
    DB_TotalDB = np.loadtxt("SAG_2_Total_Normalized.txt")
    #DB_TotalDB = np.genfromtxt("SAG_2_Total_Normalized.txt")
    Simulation_SolarRecurrentModel(LocModel, DB_TotalDB, IndVar, Target, N_lag, Stats=StatsDsv[ind_0])
        
        
    