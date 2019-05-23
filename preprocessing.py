import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal
import scipy.fftpack as scipy
from sklearn.metrics import confusion_matrix
import os
from sklearn.metrics import classification_report
import seaborn as sns
#get the samples between each start and end
def event_codes(events):
        samples=[]
        for event in events:
                if(event[1] == 32779 or event[1] == 32780):
                        samples.append(event[2])
        return (samples)


def preprocessing(eeg):
        signals_num = len(eeg)-1
        num_of_samples_in_signals= len(eeg[0])

        print("Signal Length = ", num_of_samples_in_signals)
        #Common Average Reference (car)
        avg=[0] * num_of_samples_in_signals

        for i in range(signals_num):
                for j in range(num_of_samples_in_signals):
                        avg[j] = avg[j]+eeg[i][j]


        avg[:] = [x / (len(eeg)-1) for x in avg]

        for i in range(signals_num):
                for j in range(num_of_samples_in_signals):
                        eeg[i][j]=eeg[i][j]-avg[j]


        for i in range (6,8):
                sm=sum(eeg[i]) / len(eeg[i])
                eeg[i] = [x - sm for x in eeg[i]]
        
        return(eeg)


def fft_filter(eeg,samples):
        #filter O1 and O2
        num_eff_samples=(len(samples))
        num_trials= num_eff_samples//2
        print(num_trials)
        signals=[]
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
                        psds.append(psd)

        plt.figure(3)
        for signal in signals:
                plt.plot(signal)
        plt.title('Sample of The Signals')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')

        plt.figure(4)
        #for signal in signals_fft:
        plt.plot( freqences[0] ,signals_fft[0])
        plt.title('Frequency Domain Representation')
        plt.xlabel('FREQUENCY')
        plt.ylabel('INTENSITY')
        #print(freqences)
        plt.figure(9)
        plt.plot(psds[0])


        '''
        #this is the sum of all the psd of all the 14 signals 
        psds_sum = []
        for i in range(0,12):   #change the 13 to 12 based on the file
                period=[0] * 5
                for j in range(0,14):  
                        outputs = fft_helper(psds[(i+(j*12))],freqences[0])     #change the 13 to 12 based on the file
                        period = [x + y for x, y in zip(period, outputs)]
                psds_sum.append(period)
        print(psds_sum)
        for i in range (0,12):  #change the 13 to 12 based on the file
                plt.figure(8+i)
                plt.plot([1,2,3,4,5],psds_sum[i])
        '''


        #this is signal O1 in index 6 
        o1_output=[]
        for i in range (0,num_trials):  #replace the 13 to 12 based on the file
                outputs = fft_helper(psds[(0*num_trials)+i],freqences[0]) #replace the 12 to 13 or 13 to 12  based on the file
                o1_output.append(outputs)
        '''        
        for i in range (0,12):
                plt.figure(8+i)
                plt.plot([1,2,3,4,5],o2_output[i])
        '''
        
        
        #this is signal O2 in index 7 almost all the values comes right
        o2_output=[]
        for i in range (0,num_trials):  #replace the 13 to 12 based on the file
                outputs = fft_helper(psds[(1*num_trials)+i],freqences[0]) #replace the 12 to 13 or 13 to 12  based on the file
                o2_output.append(outputs)
        '''    
        for i in range (0,num_trials):
                plt.figure(8+i)
                plt.plot([1,2,3,4,5],o2_output[i])
        '''
        print("O1")
        conf_matrices=[]
        O1_labels = dominant_freq( o1_output,num_trials)
        O1_conf = conf_matrix(O1_labels,num_trials)
        conf_matrices.append(O1_conf)
        print("O2")
        O2_labels = dominant_freq( o2_output,num_trials)
        O2_conf = conf_matrix(O2_labels,num_trials)
        conf_matrices.append(O2_conf)
        

        #print(conf_matrices)
        #print()

        return(conf_matrices)


def fft_helper(psd,freqs):
        default_freqs= [12.0 , 10.0 , 8.6, 7.6 , 6.6]
        output=[]
        #print(freqs)
        for i in range(0,5):
                sum_psd=0
                index =0
                for j in range(1,4):
                        while (True):
                                if( round(freqs[index],1) == round( ((default_freqs[i])*j),1)):
                                        #print(freqs[index] , " equal ", (default_freqs[i])*j)
                                        break
                                else:
                                        #print(freqs[index] , " not equal ", (default_freqs[i])*j)
                                        index +=1
                                             
                        sum_psd = sum_psd + max (psd[index-3],psd[index-2],psd[index-1],psd[index],psd[index+1])
                        #print(psd[index-1], " of frequencey ", (default_freqs[i])*j)
                        index=0

                output.append(sum_psd)
                print(sum_psd)
        print(output)        
        return (output)


def dominant_freq(pow_freq,num_trials):
        trial_output=[]
        for i in range(num_trials):
                curr_trial = pow_freq[i]
                dominant_label = curr_trial.index(max(curr_trial)) + 1
                print(dominant_label , " => ", curr_trial)
                trial_output.append(dominant_label)
        return (trial_output)        


def conf_matrix(y_pred,num_trials):
        if num_trials == 12 :
                y_true=[4, 2, 3, 5, 1, 2, 5, 4, 2, 3, 1, 5]
                return (confusion_matrix(y_true, y_pred))
        else:
                y_true=[4, 3, 2, 4, 1, 2, 5, 3, 4, 1, 3, 1, 3]
                return (confusion_matrix(y_true, y_pred))






conf_matrices=[[[0,0,0,0,0]]*5,[[0,0,0,0,0]]*5]
print(conf_matrices)
i=0
for filename in os.listdir('EEG-SSVEP-Experiment3/3'):

    if filename.endswith(".mat") : 
        i+=1
        curr_file = os.path.join('EEG-SSVEP-Experiment3/3', filename)
        print(curr_file)
        data = sio.loadmat(curr_file)
        eeg = data['eeg']
        events = data['events']

        signals_num = len(eeg)-1
        if (i == 1 ):
                plt.figure(1)
                #for signal in signals_fft:
                

                plt.figure(1)
                for x in range(signals_num):
                        plt.plot(eeg[x])
                plt.title('Trial Before CAR filtering')
                plt.xlabel('Time')
                plt.ylabel('Amplitude')
        samples = event_codes(events)
        print("effective samples",len(samples))

        new_eeg = preprocessing(eeg)
        if (i == 1 ):
                plt.figure(1)
                #for signal in signals_fft:
                
                plt.figure(2)
                #for signal in signals_fft:
                plt.figure(2)
                for x in range(signals_num):
                        plt.plot( new_eeg[x])
                plt.title('Trial After CAR filtering')
                plt.xlabel('Time')
                plt.ylabel('Amplitude')

        output = fft_filter(new_eeg,samples)
        conf_matrices[0] = conf_matrices[0] + output[0]
        conf_matrices[1] = conf_matrices[1] + output[1]
        print(conf_matrices)

        continue
    else:
        continue

plt.figure(6)
sns.heatmap(conf_matrices[0], yticklabels=[1,2,3,4,5], xticklabels=[1,2,3,4,5], annot= True, fmt='d')
plt.title('Window Approach O1')
plt.xlabel('Actual Class')
plt.ylabel('Predicted Class')

plt.figure(5)
sns.heatmap(conf_matrices[1],yticklabels=[1,2,3,4,5],xticklabels=[1,2,3,4,5], annot= True, fmt='d')
#yticklabels=[1,2,3,4,5],xticklabels=[1,2,3,4,5]
print (i)


plt.show()