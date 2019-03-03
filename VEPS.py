import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio
import scipy.signal
import scipy.fftpack

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
        for i in range(7,8):
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
                                psd.append( 2*(abs(Ts[x])**2) )
                                      

                        psds.append(psd)
        plt.figure(3)
        for signal in signals:
                plt.plot(signal)

        plt.figure(4)
        for signal in signals_fft:
                plt.plot(signal)
        #print(freqences)


        
        '''
        outputs = fft_helper(psds[0],freqences[0])
        plt.figure(5)
        plt.plot([1,2,3,4,5],outputs)
        '''

       
        return(psds)


def fft_helper(psd,freqs):
        default_freqs= [12.0 , 10.0 , 8.6, 7.4 , 6.6]
        output=[]
        #print(freqs)
        for i in range(0,5):
                sum_psd=0
                index =0 
                for j in range(1,4):
                        while (True):
                                if(index < 640):
                                        if( round(freqs[index],1) == round((default_freqs[i])*j,1)):
                                                
                                                break
                                        else:
                                                #print(freqs[index] , " not equal ", (default_freqs[i])*j)
                                                index +=1 
                                else:
                                        break                     
                        sum_psd = sum_psd + psd[index]
                        index=0
                output.append(sum_psd)
        print(output)        
        return (output)









data = sio.loadmat('EEG-SSVEP-Experiment3/U001ai.mat')
eeg = data['eeg']
events = data['events']

signals_num = len(eeg)-1


samples = event_codes(events)
print("effective samples",len(samples))


#plotting the signals before preprocessing
plt.figure(1)
for x in range(signals_num):
    plt.plot(eeg[x])

eeg = preprocessing(eeg)

#after preprocessing
plt.figure(2)
for x in range(signals_num):
        # print (eeg[x])
        plt.plot(eeg[x])

#power spectral density
psds = fft_filter(eeg,samples)
#print(psds[0])
plt.figure(6)
#for psd in psds:
plt.plot(psds[11])





#plt.axis([0, 18000,0,30*10**4])
plt.show()