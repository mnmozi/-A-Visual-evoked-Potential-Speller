import random
import socket
import os
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import threading
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal
import scipy.fftpack as scipy
from sklearn.metrics import confusion_matrix
import os
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def save_csv(data,file_name):
    field_list = data.split(b',')
    if len(field_list) > 17:    
        with open (file_name,'a',newline='') as f:
            thewriter = csv.writer(f)
            thewriter.writerow(field_list)
        return True
    else:
        return False


def Generate_sequence(i):
    sequence = ""
    for i in range (1,i):
        rand_class = random.randint(1,5)
        sequence += str(rand_class)
    return (sequence)    






def NStart_unity(sequance,start,whatNow): #whatnow true = send to arrow of the sequance else = send to open all ............ start is the bool to start the while
    #sending data to the GUI
    host, port = "127.0.0.1", 25002
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host,port))

    while True:
        if (start[0]):
            if (whatNow[0]):
                if len(sequance) == 0:
                    break
                arrowPos = sequance[0]
                sendData= 'a'+arrowPos
                s.send(sendData.encode("utf-8"))
                start[0]=False
            else:
                sendData= 't'
                s.send(sendData.encode("utf-8")) 
                start[0]=False
                sequance= sequance[1:]

    sendData= 'y'
    s.send(sendData.encode("utf-8"))
    s.close()


def trials_in_Path(user_number):
    trials_count = 0 
    file_exists = True
    while file_exists:
        file_exists = os.path.exists(user_number+"/"+str(trials_count)+".csv")
        if file_exists:
            trials_count+=1
    return (trials_count)


def Rtrain_routine(user_number):
    #---------------------check the user folder if exists or not then check the trial number---------------------------------
    dir_exists = os.path.isdir(user_number)
    if not dir_exists:
        os.mkdir(user_number)

    trials_count=trials_in_Path(user_number)



    #-------------------------------------intialise the files------------------------------------------------
    file_name = user_number+"/"+str(trials_count)+".csv"
    event_file_name = user_number+"/event_"+str(trials_count)+".csv"


    seq_number = 25
    t_sequence = "1534212345312451425352413"    #Generate_sequence(seq_number)
    print (t_sequence)


    eeg_stream=[]
    event_data=[]
    buffer = b''
    Unwanted_finished = False
    unwanted_passed=0
    remove_newline = False
    sample_counter = 0 #The whole samples count
    Fseconds_count=0  #five seconds counter
    
    start=[True]  #Control the GUI
    whatNow =[True] #send to the GUI to appear a arrow or a box
    Unity_thread = threading.Thread(target=NStart_unity, args=(t_sequence,start,whatNow,))


    #------------------------------connect to the headset and save the data to the correct csv file-----------------------------------------------
    Hhost, Hport = "127.0.0.1", 54123
    BUFFER_SIZE =256
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((Hhost,Hport))
    s.send(b"\r\n")


    W_trail_time =(((seq_number*2)+1)*5)*128  
    print(W_trail_time)      
    while sample_counter < W_trail_time:
        # We read a chunk
        data = s.recv(BUFFER_SIZE)
        # If we have to remove \n at the begining
        if remove_newline:
            data = data[1:]
            remove_newline = False

        # Splitting the chunk into the end of the previous message and the begining of the next message
        msg_parts = data.split(b'\r')

        # If the second part ends with nothing when splitted we will have to remove \n next time
        if msg_parts[-1] == b'':
            remove_newline = True
            # Therefore the buffer for the next step is empty
            n_buffer = b''
        else:
            # otherwise we store the begining of the next message as the next buffer
            n_buffer = msg_parts[-1][1:]

        # We interprete a whole message (begining from the previous step + the end
        #fullBool = save_csv(buffer + msg_parts[0], file_name)
        full_data = (buffer + msg_parts[0])
        
        if ( Unwanted_finished and len(field_list) > 17  ):
            field_list = full_data.split(b',')
            eeg_stream.append(field_list)
            Fseconds_count+=1
            sample_counter+=1

   
        if ( not Unwanted_finished  ):
            field_list = full_data.split(b',')
            if (((str(field_list[0]).replace('b','')).replace('\'', '')) == "127"):
                if (unwanted_passed==2):
                    Unity_thread.start()
                    Unwanted_finished= True
                else:
                    unwanted_passed+=1


        if ( Fseconds_count == 640 ):
            whatNow[0]=not whatNow[0]
            start[0]=True
            Fseconds_count = 0
            event_data.append(sample_counter)
            print(sample_counter)
        # We setup the buffer for next step
        buffer = n_buffer


    header = ["COUNTER", "DATA-TYPE", "AF3", "F7", "F3", "FC5", "T7", "P7", "01", "02","P8" ,"T8" ,"FC6", "F4", "F8", "AF4", "DATALINE_1", "DATALINE_2"]
    pd.DataFrame(eeg_stream).to_csv(file_name, index=None , header=header) 
    pd.DataFrame(event_data).to_csv(event_file_name, header=["Events"], index=None)  
    print("closed")  
    s.close()


def csv_intEEG(eeg):
    fields = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', '01', '02','P8' ,'T8' ,'FC6', 'F4', 'F8', 'AF4']
    newEEG=[]
    for field in fields:
        NCol = [ float((o.replace('b','')).replace('\'', '')) for o in eeg[field]]
        newEEG.append(NCol)
    return (newEEG)    


def preprocessing(eeg):
    signals_num = len(eeg)
    num_of_samples_in_signals= len(eeg[0])

    print("Signal Length = ", num_of_samples_in_signals)
    #Common Average Reference (car)
    avg=[0] * num_of_samples_in_signals

    for i in range(signals_num):
        for j in range(num_of_samples_in_signals):
                avg[j] = avg[j]+eeg[i][j]


    avg[:] = [x / (len(eeg)) for x in avg]

    for i in range(signals_num):
        for j in range(num_of_samples_in_signals):
                eeg[i][j]=eeg[i][j]-avg[j]

    for i in range (6,8):
        sm=sum(eeg[i])//len(eeg[i])
        eeg[i] = [x - sm for x in eeg[i]]

    return(eeg)


def fft_filter(eeg,samples):
    
    #filter O1 and O2
    num_eff_samples=(len(samples))
    num_trials= num_eff_samples//2
    print("the user tried => ",num_trials)
    signals=[]  #The eeg in the trial periods
    signals_fft=[]
    freqences=[]
    psds=[]
    for i in range(6,8):
        print("Now in singla number -> ", i)
        for j in range(0,num_eff_samples,2):
            number_of_samples = samples[j+1] - samples[j]
            mid_point= number_of_samples//2

            print(samples[j] ,' to ', samples[j+1] , " = ", number_of_samples)

            signals.append( eeg[ i, samples[j]:(samples[j+1]) ])
            Ts=np.fft.fft ( eeg[ i, samples[j]:(samples[j+1]) ])
            freq= np.fft.fftfreq(number_of_samples,(1/128))

            signals_fft.append(Ts)
            freqences.append(freq)
            psd = []
            
            for x in range(1,mid_point):
                    #mprint("at freq " , freq[x] ," and " ,  freq[-x])
                    psd.append( (np.abs(Ts[x])**2) + (np.abs(Ts[-x]**2)) )
            psds.append(psd[40:240])

    plt.figure(3)
    for signal in signals:
            plt.plot(signal)

    plt.figure(4)
    for signal in signals_fft:
            plt.plot(signal)


    #print("............",len(freqences[0]))
    #print("............",len(psds[0]))

    ''' # in this part i get the frequencies and harmonics and the values around it without the rest if u used is remove the boundaries in line 248
    #this is signal O1 in index 6 almost all the values comes right
    
    o1_output=[]
    for i in range (0,num_trials): 
            outputs = fft_helper(psds[(0*num_trials)+i],freqences[0]) 
            o1_output.append(outputs)
    print(len(psds))
    print('o1 output is ',len(o1_output))
    

    
    #this is signal O2 in index 7 almost all the values comes right
    o2_output=[]
    for i in range (0,num_trials):
            outputs = fft_helper(psds[(1*num_trials)+i],freqences[0]) 
            o2_output.append(outputs)
    print('o2 output is ',len(o2_output))

    
    new_psds= []
    new_psds+=(o1_output)
    new_psds+=(o2_output)
    print('o1 and o2 ',len(new_psds))
    '''


    #print(conf_matrices)
    #print()

    return(psds)

def fft_helper(psd,freqs):
        default_freqs= [12.0 , 10.0 , 8.6, 7.4 , 6.6]
        output=[]
        for i in range(0,5):
                index =1
                for j in range(1,4):
                        while (True):
                                if( round(freqs[index],1) == round( ((default_freqs[i])*j),1)):
                                        #print(freqs[index] , " equal ", (default_freqs[i])*j)
                                        break
                                else:
                                        #print(freqs[index] , " not equal ", (default_freqs[i])*j)
                                        index +=1

                        output.append(psd[index-1])                
                        for i in range (1,5):
                                output.append(psd[index-i-1])
                                output.append(psd[index+i-1])
                        index=0
        return (output)


def Rtest_routine(user_number):
    dir_exists = os.path.isdir(user_number)
    if not dir_exists:
        return ("There is No User with this name")

    trials_count=trials_in_Path(user_number)

    O1_trials=[]
    O2_trials=[]


    for i in range (0,trials_count): #open trial trial 
        file_name = user_number+"/"+str(i)+".csv"
        event_file_name = user_number+"/event_"+str(i)+".csv"
        eeg = pd.read_csv(file_name)
        events = pd.read_csv(event_file_name)
        events = events['Events']
        eeg = csv_intEEG(eeg)
        eeg = np.array(eeg)
        events = np.array(events)
        events= events[:-1]

        plt.figure(40)
        for ee in eeg:
            plt.plot(ee)

        new_eeg = preprocessing(eeg)

        plt.figure(40)
        for ee in new_eeg:
            plt.plot(ee)
            
        trials= len(events)//2
        output = fft_filter(new_eeg,events)
        O1_trials+=output[0:trials]
        O2_trials+=output[trials:]


 
    X = O2_trials
    y =  [1,5,3,4,2,1,2,3,4,5,3,1,2,4,5,1,4,2,5,3,5,2,4,1,3]*(trials_count-0)


    #pca = PCA(n_components=30)
    #pricipalComponents = pca.fit_transform(np.concatenate((O1_trials,O2_trials),axis=1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =0)

    print (len(X_test))
    print(len(O1_trials))
    print(len(O2_trials))

    classifier = RandomForestClassifier(n_estimators=1000,n_jobs=2, random_state=0)

    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    print(y_test)
    print(y_predict)

    plt.figure(6)
    sns.heatmap(confusion_matrix(y_test, y_predict), annot= True, fmt='d')


    #print(classifier.predict_proba(X_test)[0:10])




    plt.show()


#get input to decide what to do Train or Test 

user_stage = input ("State Your stage train or test: ")
user_number = input ("State the user number: ")
if (user_stage == "train"):
    Rtrain_routine(user_number)
elif (user_stage == "test"):
    Rtest_routine(user_number)    


plt.show()
