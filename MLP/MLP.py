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
def Train_MLP(IndVar, Target, N_lag, n_hidden, training_iters, LocModel):
    tf.reset_default_graph() # A brand new graph each run
    
    # Parameters
    learning_rate = 0.001
    N_features_input = len(IndVar)
    N_sampleByBatch = 128
    N_categ = 1

    # Output cross entropy
    CrossEntro = np.zeros((training_iters,2))
    # Importing Data Base
    DB_SolarMining = np.loadtxt("SAG_2_Train_Normalized.txt")
    DB_Testing = np.loadtxt("SAG_2_Validation_Normalized.txt")

    
    # Local Model to Save
    TempLocModel = LocModel+'/FeatMaps' #%(N_lag, n_hidden, NumRecurrent, Target, training_iters)
    LocModelOutput = LocModel+'/CostTotal.txt' #%(N_lag, n_hidden, NumRecurrent, Target, training_iters)
    
    # Preprocessing
    Train_data_in, Train_data_out = Extract_data(DB_SolarMining, N_lag, IndVar, Target) 

    Test_data_in, Test_data_out = Extract_data(DB_Testing, N_lag, IndVar, Target) 
    
    # Place holder for Mini-batch input output
    X_in = tf.placeholder("float", [None, N_lag, N_features_input], name="X_Input")
    X_in_reshaped = tf.reshape(X_in, [-1, N_lag*N_features_input], name="X_in_reshaped")
    
    y = tf.placeholder("float", [None, 1], name="Y_Known")
    
    # MLP structure
    W1 = tf.Variable(tf.truncated_normal(shape=[N_lag*N_features_input, n_hidden], stddev=0.1), dtype=tf.float32, name="W1")
    full_1 = tf.nn.leaky_relu(tf.matmul(X_in_reshaped, W1), name="full_1")  
    
    W2 = tf.Variable(tf.truncated_normal(shape=[n_hidden, n_hidden], stddev=0.1), dtype=tf.float32, name="W2")
    full_2 = tf.nn.leaky_relu(tf.matmul(full_1, W2), name="full_2")  
    
    W3 = tf.Variable(tf.truncated_normal(shape=[n_hidden, 1], stddev=0.1), dtype=tf.float32, name="W3")
    pred =  tf.matmul(full_2, W3, name="pred")      
    
    # Loss and optimizer
    cost = tf.losses.mean_squared_error(labels=y, predictions=pred)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initializing the variables
    init = tf.global_variables_initializer()
    
    # Launch the graph
    with tf.Session() as session:
        session.run(init)
        saver = tf.train.Saver()
        step_train = 0
        loss_total = 0
        N_batchs = int((Train_data_in.shape[0] - N_lag+1)/N_sampleByBatch)
        N_batchs_test = int((Test_data_in.shape[0] - N_lag+1)/N_sampleByBatch)
        N_step = int(Test_data_in.shape[0] - N_lag+1)
        Time_Start_Training = time.time()
        while step_train < training_iters:
            cost_total = 0
            cost_total_Test = 0
            for ind0 in range(N_batchs):
                Training_in = Train_data_in[ind0*N_sampleByBatch:(ind0+1)*N_sampleByBatch,:,:]
                Training_out = Train_data_out[ind0*N_sampleByBatch:(ind0+1)*N_sampleByBatch,:]
                _, TempCost = session.run([optimizer,cost], feed_dict={X_in: Training_in, y: Training_out})
                cost_total += TempCost
            # ###############################
            saver.save(session, TempLocModel)
            # ###############################
            for ind1 in range(N_batchs_test):
                Test_in_real = Test_data_in[ind1*N_sampleByBatch:(ind1+1)*N_sampleByBatch,:,:]
                Test_out_real = Test_data_out[ind1*N_sampleByBatch:(ind1+1)*N_sampleByBatch,:]
                TempoCost_Test = session.run(cost, feed_dict={X_in: Test_in_real, y: Test_out_real})
                cost_total_Test += TempoCost_Test          
            # ###############################
            print("Epoch: "+str(step_train+1)+", C.E. Train: " + "{:.2f}".format(np.around(cost_total, decimals=2))+", C.E. Test:" + "{:.2f}".format(np.around(cost_total_Test, decimals=2))+", Time (min): " + "{:.2f}".format((time.time() -  Time_Start_Training)/60))
            CrossEntro[step_train,0] = np.around(cost_total, decimals=3)
            CrossEntro[step_train,1] = np.around(cost_total_Test, decimals=3)
            # ###############################        
            step_train += 1
            np.savetxt(LocModelOutput, CrossEntro, fmt='%10.5f', delimiter='\t', newline='\n')
            # ###############################        
        tf.get_default_graph().finalize()
    return print("Training Completed!")


def Simulation_MLP(LocModel, DB_TotalDB, IndVar, Target, N_lag, Stats):
    tf.reset_default_graph() # A brand new graph each run
    with tf.device('/gpu:0'):
        a = 16710
        b = int(Stats)
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
        pred = graph.get_tensor_by_name("pred:0")
        Num_Elements = int(Simulation_data_in.shape[0] - N_lag+1)
        Results = np.zeros((Num_Elements,4)) # day, real, estimated, diff
        for ind1 in range(Num_Elements):
            Sim_in_real = Simulation_data_in[ind1:(ind1+1),:,:]
            Sim_out_real = Simulation_data_out[ind1:(ind1+1),:]
            One_pred = session.run([pred], feed_dict={X_Input: Sim_in_real})[0]  
            Real = Sim_out_real[0]
            V_r, V_p = int(np.clip(Real*b+a, a_min=0, a_max=None)), int(np.clip(One_pred*b+a, a_min=0, a_max=None))
            #print("Actual consumption [%.3f] vs Predicted consumption: [%.3f]" % (Real, One_pred))
            Results[ind1,0], Results[ind1,1], Results[ind1,2], Results[ind1,3] = (ind1+1), V_r, V_p, (V_r - V_p)
            
        np.savetxt(LocModelOutput, Results, fmt='%3.4e', delimiter='\t', newline='\n')
        tf.get_default_graph().finalize()
        session.close()        
    return print("Simulation Completed! \n")


#0: Day	   1: FT [Tph]	   2: EC tn [kW]    3: BearingPressure [Psi]	4: SpindleSpeed [Rpm]	   5: Wtr [m3/h]	6: SPe [%]	
#7: DeltaFT [Tph]	8: DeltaSSp [Rpm]	9: EC tn+1 [kW]	      10: EC t_1hr	11: EC t_2hr	12: EC t_4hr	13: EC t_8hr
#14: FT05h	15: FT1h	16: FT2h	17: FT4h	18: FT8h

StatsDsv = [1504, 1410, 1321, 1242, 1174] # EC 0.5, EC 1, EC 2, EC 4, EC 8
#NumHiddenLayers = [4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120]
NumHiddenLayers = np.linspace(4,600,150)

for ind_1 in range(len(NumHiddenLayers)):    
    for ind_0 in range(len(StatsDsv)):    
        n_hidden = int(NumHiddenLayers[ind_1])
        IndVar = [1, 3, 4, 7, 8]
        Target = int(9+ind_0)
        N_lag = 1
        training_iters = 120
        print("Target:",Target, "n_hidden", n_hidden)        
        LocModel = 'Models_1_3_4_7_8/Lag%s_nh%s_targ%s_MLP_V1_3_4_7_8'%(N_lag, n_hidden, Target)     
        Train_MLP(IndVar, Target, N_lag, n_hidden, training_iters, LocModel)
        DB_TotalDB = np.loadtxt("SAG_2_Validation_Normalized.txt")
        Simulation_MLP(LocModel, DB_TotalDB, IndVar, Target, N_lag, Stats=StatsDsv[ind_0])
