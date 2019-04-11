import random
import socket
import os
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import threading



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
    sequence = "t"
    for i in range (1,i):
        rand_class = random.randint(1,5)
        sequence += str(rand_class)
    return (sequence)    




def Start_unity(sequance,time_passed):
    #sending data to the GUI
    host, port = "127.0.0.1", 25002
    BUFFER_SIZE =256
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host,port))
    s.send(sequance.encode("utf-8"))

    while True:
        data = s.recv(BUFFER_SIZE)
        time_passed[0]= True
        print(data)





def Rtrain_routine(user_number):
    #---------------------check the user folder if exists or not then check the trial number---------------------------------
    dir_exists = os.path.isdir(user_number)
    if not dir_exists:
        os.mkdir(user_number)
    trials_count = 0 
    file_exists = True
    while file_exists:
        file_exists = os.path.exists(user_number+"/"+str(trials_count)+".csv")
        if file_exists:
            trials_count+=1



    #-------------------------------------intialise the files------------------------------------------------
    file_name = user_number+"/"+str(trials_count)+".csv"
    event_file_name = user_number+"/event"+str(trials_count)+".csv"


    seq_number = 5
    t_sequence = Generate_sequence(seq_number)
    print (t_sequence)


    eeg_stream=[]
    event_data=[]
    buffer = b''
    remove_newline = False
    sample_counter = 0
    
    seconds_count=0
    Unwanted_finished = False
    time_passed = [False]
    t = threading.Thread(target=Start_unity, args=(t_sequence,time_passed,))




    #------------------------------connect to the headset and save the data to the correct csv file-----------------------------------------------
    Hhost, Hport = "127.0.0.1", 54123
    BUFFER_SIZE =256
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((Hhost,Hport))
    s.send(b"\r\n")
    start_trial= time.time()
    w_trail_time =((seq_number*2)+1)*5        
    while time.time() - start_trial < w_trail_time:
        # We read a chunk
        data = s.recv(BUFFER_SIZE, socket.MSG_WAITALL)
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
            seconds_count+=1
            sample_counter+=1

   
        if ( Unwanted_finished == False ):
            field_list = full_data.split(b',')
            if (((str(field_list[0]).replace('b','')).replace('\'', '')) == "127") :
                t.start()
                Unwanted_finished= True


        if ( time_passed[0] ):
            time_passed[0]= False
            seconds_count = 0
            event_data.append(sample_counter)
            print(sample_counter)
        # We setup the buffer for next step
        buffer = n_buffer


    header = ["COUNTER", "DATA-TYPE", "AF3", "F7", "F3", "FC5", "T7", "P7", "01", "02","P8" ,"T8" ,"FC6", "F4", "F8", "AF4", "DATALINE_1", "DATALINE_2"]
    pd.DataFrame(eeg_stream).to_csv(file_name, index=None , header=header) 
    pd.DataFrame(event_data).to_csv(event_file_name, header=None, index=None)    
    s.close()

    
def preprocessing(eeg):
    signals_num = len(eeg)
    print(signals_num)
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
    
    for i in range (2):
            sm=sum(eeg[i])//len(eeg[i])
            eeg[i] = [x - sm for x in eeg[i]]

    return(eeg)


def Rtest_routine():
    print ("hello")
    df = pd.read_csv('0/0.csv')
    O1 = df['01'] 
    O2 = df['02'] 
    NO1 = [ float((o.replace('b','')).replace('\'', '')) for o in O1]
    NO2 = [ float((o.replace('b','')).replace('\'', '')) for o in O2]
    

    eeg = []
    eeg.append(NO1[:])
    eeg.append(NO2[:])

    new_eeg = preprocessing(eeg)


    plt.figure(1)
    plt.plot(NO1)

    plt.figure(2)
    plt.plot(NO2)

    plt.figure(3)
    plt.plot(new_eeg[0])

    plt.figure(4)
    plt.plot(new_eeg[1])
    #print (new_eeg)


#get input to decide what to do Train or Test 

user_stage = input ("State Your stage train or test: ")
user_number = input ("State the user number: ")
if (user_stage == "train"):
    Rtrain_routine(user_number)
elif (user_stage == "test"):
    Rtest_routine()    


plt.show()
