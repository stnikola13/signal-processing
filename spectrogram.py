import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sounddevice as sd
from scipy.io import wavfile
from scipy import signal
from scipy.fftpack import fft
from scipy.fftpack import fftshift, fftfreq

samplerate_sequence, sequence = wavfile.read('20200015.wav')
dt = 1 / samplerate_sequence
t = np.arange(0, dt * len(sequence), dt)

plt.figure(figsize = (8,6))
plt.plot(t, sequence)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Audio signal - vremenski domen')

#plt.savefig('5-1.png')

plt.show()

#scaled1 = np.int16(sequence / np.max (np.abs (sequence)) * 32767)
#sd.play(scaled, samplerate_sequence)
#wavfile.write('5-1.wav', samplerate_sequence, scaled1)

sd.wait()


########################################


'''data = sequence
fs = samplerate_sequence
freq_axis = fftshift(fftfreq(len(data), 1/fs))
freq_axis_positive = np.arange(0, fs/2, fs/len(data))

x = data
X_fft = fft(x)
Xa = np.abs(X_fft)
#Xa=Xa[0:int(np.ceil(len(Xa)/2))]

plt.figure(figsize = (8,6))
plt.plot(freq_axis, Xa)
plt.xlim([-200,max(freq_axis_positive)+100])
plt.ylim([0,6.2e8])
plt.show()'''

data = sequence
fs = samplerate_sequence
X_fft = fft(x)
Xa = np.abs(X_fft)
freq_axis = np.arange(-fs/2, fs/2, fs/len(data))

plt.figure(figsize = (8,6))
plt.xlabel('$w$')
plt.ylabel('$|X(jw)|$')
plt.title('Audio signal - frekvencijski domen')

plt.plot(freq_axis, Xa)

#plt.savefig('5-2.png')

#plt.xlim([-200,max(freq_axis_positive)+100])
#plt.ylim([0,6.2e8])
plt.show()


############################################


f, t, Sxx = signal.spectrogram(sequence, samplerate_sequence)

plt.figure(figsize = (8,6))
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('$Frequency [Hz]$')
plt.xlabel('$Time [s]$')
plt.title('Spectrogram')
plt.ylim([400,1700])

#plt.savefig('5-3.png')

plt.show()


plt.figure(figsize = (8,6))
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('$Frequency [Hz]$')
plt.xlabel('$Time [s]$')
plt.title('Spectrogram')
plt.ylim([400,1700])

plt.axhline(1336, xmin=0.05, xmax=0.71, color='red', linestyle='--', label='')
plt.axhline(1336, xmin=0.86, xmax=0.95, color='red', linestyle='--', label='')
plt.axhline(1209, xmin=0.75, xmax=0.83, color='orange', linestyle='--', label='')
plt.axhline(770, xmin=0.87, xmax=0.95, color='lime', linestyle='--', label='')
plt.axhline(941, xmin=0.17, xmax=0.25, color='yellow', linestyle='--', label='')
plt.axhline(941, xmin=0.40, xmax=0.72, color='yellow', linestyle='--', label='')
plt.axhline(697, xmin=0.05, xmax=0.13, color='cyan', linestyle='--', label='')
plt.axhline(697, xmin=0.28, xmax=0.36, color='cyan', linestyle='--', label='')
plt.axhline(697, xmin=0.75, xmax=0.83, color='cyan', linestyle='--', label='')

#plt.savefig('5-4.png')

plt.show()


######################################


samplerate_sequence2, sequence2 = wavfile.read('klavir16k.wav')
#print(samplerate_sequence)

t2 = np.arange(0, 5, 1/16000)

plt.figure(figsize = (8,6))

#scaled2 = np.int16(sequence2 / np.max (np.abs (sequence2)) * 32767)
#sd.play(scaled, samplerate_sequence)
#wavfile.write('5-2.wav', samplerate_sequence2, scaled2)

plt.plot(t2, sequence2)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Audio signal - vremenski domen')
#plt.savefig('5-5.png')

f2, t2, Sxx2 = signal.spectrogram(sequence2, 26000)

plt.figure(figsize = (8,6))
plt.pcolormesh(t2, f2, Sxx2, shading='gouraud')


plt.ylabel('$Frequency [Hz]$')
plt.xlabel('$Time [s]$')
plt.title('Spectrogram')
plt.ylim([0,1000])

#plt.savefig('5-6.png')
plt.show()



###########################################


samplerate_sequence3, sequence3 = wavfile.read('klavir44k.wav')
#print(samplerate_sequence)

t3 = np.arange(0, 5, 1/44100)

plt.figure(figsize = (8,6))


plt.plot(t3, sequence3)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Audio signal - vremenski domen')

#plt.savefig('5-7.png')

f3, t3, Sxx3 = signal.spectrogram(sequence3, 16000)

plt.figure(figsize = (8,6))
plt.pcolormesh(t3, f3, Sxx3, shading='gouraud')
plt.ylabel('$Frequency [Hz]$')
plt.xlabel('$Time [s]$')
plt.title('Spectrogram')
plt.ylim([0,1000])
#plt.savefig('5-8.png')
plt.show()
