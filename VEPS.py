import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal
import scipy.fftpack as scipy

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


        return(eeg)


def fft_filter(eeg,samples):
        #filter O1 and O2
        num_eff_samples=(len(samples))
        signals=[]
        signals_fft=[]
        freqences=[]
        psds=[]
        for i in range(0,14):
                print("Now in singla number -> ", i)
                for j in range(0,num_eff_samples,2):
                        number_of_samples = samples[j+1] - samples[j]
                        mid_point= number_of_samples//2

                        print(samples[j] ,' to ', samples[j+1] ," = ",number_of_samples)

                        signals.append( eeg[ i, samples[j]:(samples[j+1]) ])
                        Ts=np.fft.fft ( eeg[ i, samples[j]:(samples[j+1]) ])
                        freq= np.fft.fftfreq(number_of_samples,(1/128))
                        '''
                        #freq in +ve
                        for x in range(1,mid_point):
                                print(freq[x])

                        #freq in -ve
                        for x in range(1,mid_point):
                                print(freq[-x])
                        '''
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

        plt.figure(4)
        for signal in signals_fft:
                plt.plot(signal)
        #print(freqences)






        print(len(psds))


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

        '''
        #this is signal O2 in index 7 almost all the values comes right
        o2_output=[]
        for i in range (0,12):  #replace the 13 to 12 based on the file
                outputs1 = fft_helper(psds[(7*12)+i],freqences[0]) #replace the 12 to 13 or 13 to 12  based on the file
                o2_output.append(outputs1)
        for i in range (0,12):
                plt.figure(8+i)
                plt.plot([1,2,3,4,5],o2_output[i])
        '''
        
        '''
        #this is signal O1 in index 6 almost all the values comes right
        o2_output=[]
        for i in range (0,12):  #replace the 13 to 12 based on the file
                outputs1 = fft_helper(psds[(6*12)+i],freqences[0]) #replace the 12 to 13 or 13 to 12  based on the file
                o2_output.append(outputs1)
        for i in range (0,12):
                plt.figure(8+i)
                plt.plot([1,2,3,4,5],o2_output[i])
        '''


        return(psds)


def fft_helper(psd,freqs):
        default_freqs= [12.0 , 10.0 , 8.6, 7.4 , 6.6]
        output=[]
        #print(freqs)
        for i in range(0,5):
                sum_psd=0
                index =1
                for j in range(1,4):
                        while (True):
                                if(index < 640):
                                        if( round(freqs[index],1) == round( ((default_freqs[i])*j),1)):
                                                #print(freqs[index] , " equal ", (default_freqs[i])*j)
                                                break
                                        else:
                                                #print(freqs[index] , " not equal ", (default_freqs[i])*j)
                                                index +=1
                                else:
                                        break
                        sum_psd = sum_psd + psd[index-1]
                        #print(psd[index], " of frequencey ", (default_freqs[i])*j)
                        index=0
                output.append(sum_psd)
        #print("mm",output)
        return (output)









data = sio.loadmat('EEG-SSVEP-Experiment3/U001bi.mat')
eeg = data['eeg']
events = data['events']

signals_num = len(eeg)-1


samples = event_codes(events)
print("effective samples",len(samples))


#plotting the signals before preprocessing
plt.figure(1)
for x in range(signals_num+1):
    plt.plot(eeg[x])

new_eeg = preprocessing(eeg)

#after preprocessing
plt.figure(2)
for x in range(signals_num):
        # print (new_eeg[x])
        plt.plot(new_eeg[x])

#power spectral density
psds = fft_filter(new_eeg,samples)
#print(psds[0])
#plt.figure(6)
#for psd in psds:
#plt.plot(psds[3])





#plt.axis([0, 18000,0,30*10**4])
plt.show()