import random
import socket
import os
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    #--------------------------------------------------send the sequance to the GUI----------------------------------------
    seq_number = 25
    t_sequence = Generate_sequence(seq_number)
    print (t_sequence)
    #sending data to the GUI
    host, port = "127.0.0.1", 25002
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host,port))
        sock.sendall(t_sequence.encode("utf-8"))
        sock.close() 
    finally: 
        sock.close()  

    #-------------------------------------intialise the files------------------------------------------------
    file_name = user_number+"/"+str(trials_count)+".csv"
    event_file_name = user_number+"/event"+str(trials_count)+".csv"
    with open (file_name,'w',newline='') as f:
        thewriter = csv.writer(f)
        thewriter.writerow(['COUNTER', 'DATA-TYPE', 'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', '01', '02','P8' ,'T8' ,'FC6', 'F4', 'F8', 'AF4', 'DATALINE_1', 'DATALINE_2'])

    with open (event_file_name,'w',newline='') as f:
        thewriter = csv.writer(f) 
        thewriter.writerow(['samples'])

    #------------------------------connect to the headset and save the data to the correct csv file-----------------------------------------------
    host, port = "127.0.0.1", 54123
    BUFFER_SIZE =256
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host,port))

    buffer = b''
    remove_newline = False
    sample_counter = 0
    start_trial= time.time()
    start_time = time.time()
    w_trail_time =((seq_number*2)+1)*5
    s.send(b"\r\n")
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
        fullBool = save_csv(buffer + msg_parts[0], file_name)
        if (fullBool):
            sample_counter+=1

        now_time = time.time()

        if (now_time - start_time >= 5 ):
            start_time = time.time()
            with open (event_file_name,'a',newline='') as f:
                thewriter = csv.writer(f)
                x = [sample_counter]
                print(x)
                thewriter.writerow(x)
        # We setup the buffer for next step
        buffer = n_buffer
        
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
