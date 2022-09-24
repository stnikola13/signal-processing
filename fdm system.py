import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sounddevice as sd
from scipy.io import wavfile
from scipy import signal
from scipy.fftpack import fft
from scipy.fftpack import fftshift, fftfreq
import scipy.signal as sci

samplerate1, y1 = wavfile.read('sekvencane.wav')
samplerate2, y2 = wavfile.read('sekvencane2.wav')

t = np.arange(0, 3, 1/44100)

# Vremenski oblik inicijalnih signala
plt.figure(figsize=(16,6))

plt.subplot(1,2,1)
plt.plot(t, y1)
plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.title('Audio signal (govor) - vremenski domen')

plt.subplot(1,2,2)
plt.plot(t, y2)
plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.title('Audio signal (klavir) - vremenski domen')

#plt.savefig('6-1.png')
#scaled1 = np.int16(y1 / np.max (np.abs (y1)) * 32767)
#wavfile.write('6-1.wav', samplerate1, scaled1)
#scaled2 = np.int16(y2 / np.max (np.abs (y2)) * 32767)
#wavfile.write('6-2.wav', samplerate2, scaled2)

plt.show()

# Amplitudski spektar inicijalnih signala
plt.figure(figsize=(16,6))

f1 = fftshift(fftfreq(len(y1), 1 / samplerate1))
plt.subplot(1,2,1)
plt.xlim(-15000, 15000)
plt.xlabel('$w$')
plt.ylabel('$|X(jw)|$')
plt.title('Audio signal (govor) - frekvencijski domen')
plt.plot(f1, fftshift(np.abs(fft(y1))))

f2 = fftshift(fftfreq(len(y2), 1 / samplerate2))
plt.subplot(1,2,2)
plt.xlim(-10000, 10000)
plt.xlabel('$w$')
plt.ylabel('$|X(jw|)$')
plt.title('Audio signal (klavir) - frekvencijski domen')
plt.plot(f2, fftshift(np.abs(fft(y2))))

#plt.savefig('6-12.png')

plt.show()


#####################################


n = 6
freqnf1 = 7000
freqnf2 = 3000

b1, a1 = sci.butter(n, freqnf1 / (samplerate1 / 2), btype = 'lowpass')
y1f = sci.filtfilt(b1, a1, y1)


b2, a2 = sci.butter(n, freqnf2 / (samplerate2 / 2), btype = 'lowpass')
y2f = sci.filtfilt(b2, a2, y2)


# Vremenski oblik posle NF
plt.figure(figsize=(16,6))

plt.subplot(1,2,1)
plt.plot(t, y1f)
plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.title('Signal $y_1^f$ - vremenski domen')

plt.subplot(1,2,2)
plt.plot(t, y2f)
plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.title('Signal $y_2^f$ - vremenski domen')

#plt.savefig('6-2.png')
#scaled3 = np.int16(y1f / np.max (np.abs (y1f)) * 32767)
#wavfile.write('6-3.wav', samplerate1, scaled3)
#scaled4 = np.int16(y2f / np.max (np.abs (y2f)) * 32767)
#wavfile.write('6-4.wav', samplerate2, scaled4)

plt.show()

# Amplitudski spektar posle NF
plt.figure(figsize=(16,6))

f1f = fftshift(fftfreq(len(y1f), 1 / samplerate1))
plt.subplot(1,2,1)
plt.xlim(-15000, 15000)
plt.xlabel('$w$')
plt.ylabel('$|X(jw)|$')
plt.title('Signal $y_1^f$ - frekvencijski domen')
plt.plot(f1f, fftshift(np.abs(fft(y1f))))

f2f = fftshift(fftfreq(len(y2f), 1 / samplerate2))
plt.subplot(1,2,2)
plt.xlim(-10000, 10000)
plt.xlabel('$w$')
plt.ylabel('$|X(jw|)$')
plt.title('Signal $y_2^f$ - frekvencijski domen')
plt.plot(f2f, fftshift(np.abs(fft(y2f))))

#plt.savefig('6-22.png')

plt.show()


######################################


fc = 11000
#t2m = np.linspace(0, len(y2f) / samplerate2, num=len(y2f) )
t2m = t
y2m = y2f * np.cos(2 * np.pi * fc * t)

plt.figure(figsize=(16,6))

# Vremenski oblik posle modulacije
plt.subplot(1,2,1)
plt.plot(t2m, y2m)
plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.title('Signal $y_2^m$ - vremenski domen')


#scaled5 = np.int16(y2m / np.max (np.abs (y2m)) * 32767)
#wavfile.write('6-5.wav', samplerate2, scaled5)

# Amplitudski spektar posle modulacije
plt.subplot(1,2,2)
f2m = fftshift(fftfreq(len(y2m), 1 / samplerate2))
plt.xlim(-15000, 15000)
plt.xlabel('$w$')
plt.ylabel('$|X(jw)|$')
plt.title('Signal $y_2^m$ - frekvencijski domen')
plt.plot(f2m, fftshift(np.abs(fft(y2m))))

#plt.savefig('6-3.png')

plt.show()



#######################################


yt = y1f + y2m
tt = t

plt.figure(figsize = (16,6))

# Vremenski oblik posle spajanja
plt.subplot(1,2,1)
plt.plot(tt, yt)
plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.title('Signal $y_T$ - vremenski domen')

#plt.savefig('6-4.png')
#scaled6 = np.int16(yt / np.max (np.abs (yt)) * 32767)
#wavfile.write('6-6.wav', samplerate1, scaled6)

# Amplitudski spektar posle spajanja
plt.subplot(1,2,2)
ft = fftshift(fftfreq(len(yt), 1 / samplerate1))
plt.xlim(-15000, 15000)
plt.xlabel('$w$')
plt.ylabel('$|X(jw)|$')
plt.title('Signal $y_T$ - frekvencijski domen')
plt.plot(ft, fftshift(np.abs(fft(yt))))

#plt.savefig('6-4.png')

plt.show()


#####################################


n = 6
freqkv = 14000

b3, a3 = sci.butter(n, freqkv / (samplerate1 / 2), btype = 'lowpass')
yr = sci.filtfilt(b3, a3, yt)

plt.figure(figsize = (16,6))

# Vremenski oblik posle KV
plt.subplot(1,2,1)
plt.plot(t, yr)
plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.title('Signal $y_R$ - vremenski domen')

#plt.savefig('6-5.png')
#scaled7 = np.int16(yr / np.max (np.abs (yr)) * 32767)
#wavfile.write('6-7.wav', samplerate2, scaled7)

# Amplitudski spektar posle KV
plt.subplot(1,2,2)
fr = fftshift(fftfreq(len(yr), 1 / samplerate2))
plt.xlim(-15000, 15000)
plt.xlabel('$w$')
plt.ylabel('$|X(jw)|$')
plt.title('Signal $y_R$ - frekvencijski domen')
plt.plot(fr, fftshift(np.abs(fft(yr))))

#plt.savefig('6-5.png')

plt.show()



##################################


y2b = bandpass(10, sampleRate1, freqBP_l,freqBP_h, yr)

n = 6
freqpo1 = fc - freqnf2
freqpo2 = fc + freqnf2

b4, a4 = sci.butter(n, np.array((freqpo1 / (samplerate1 / 2) ,freqpo2 / (samplerate1 / 2))), btype = 'bandpass')
y2b = sci.filtfilt(b4, a4, yr)

plt.figure(figsize = (16,6))

# Vremenski oblik posle PO
plt.subplot(1,2,1)
plt.plot(t, y2b)
plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.title('Signal $y_2^b$ - vremenski domen')

#plt.savefig('6-6.png')
#scaled8 = np.int16(y2b / np.max (np.abs (y2b)) * 32767)
#wavfile.write('6-8.wav', samplerate2, scaled8)

# Amplitudski spektar posle PO
plt.subplot(1,2,2)
f2b = fftshift(fftfreq(len(y2b), 1 / samplerate2))
plt.xlim(-15000, 15000)
plt.xlabel('$w$')
plt.ylabel('$|X(jw)|$')
plt.title('Signal $y_2^b$ - frekvencijski domen')
plt.plot(f2b, fftshift(np.abs(fft(y2b))))

#plt.savefig('6-6.png')

plt.show()


#####################################


fc = 11000
#t2m = np.linspace(0, len(y2f) / samplerate2, num=len(y2f) )
t2d = t
y2d = y2b * np.cos(2 * np.pi * fc * t)

plt.figure(figsize=(16,6))

# Vremenski oblik posle modulacije
plt.subplot(1,2,1)
plt.plot(t2d, y2d)
plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.title('Signal $y_2^d$ - vremenski domen')

#plt.savefig('6-7.png')
#scaled9 = np.int16(y2d / np.max (np.abs (y2d)) * 32767)
#wavfile.write('6-9.wav', samplerate2, scaled9)

# Amplitudski spektar posle modulacije
plt.subplot(1,2,2)
f2d = fftshift(fftfreq(len(y2d), 1 / samplerate2))
plt.xlim(-15000, 15000)
plt.xlabel('$w$')
plt.ylabel('$|X(jw)|$')
plt.title('Signal $y_2^d$ - frekvencijski domen')
plt.plot(f2d, fftshift(np.abs(fft(y2d))))

#plt.savefig('6-7.png')

plt.show()


###############################


n = 6
freqnf1 = 7000
freqnf2 = 3000

b5, a5 = sci.butter(n, freqnf1 / (samplerate1 / 2), btype = 'lowpass')
y1r = sci.filtfilt(b5, a5, yr)


b6, a6 = sci.butter(n, freqnf2 / (samplerate2 / 2), btype = 'lowpass')
y2r = sci.filtfilt(b6, a6, y2d)


# Vremenski oblik posle NF
plt.figure(figsize=(16,6))

plt.subplot(1,2,1)
plt.plot(t, y1r)
plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.title('Signal $y_1^r$ - vremenski domen')

plt.subplot(1,2,2)
plt.plot(t, y2r)
plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.title('Signal $y_2^r$ - vremenski domen')

#plt.savefig('6-8.png')

plt.show()

#plt.savefig('6-8.png')
#scaledA = np.int16(y1r / np.max (np.abs (y1r)) * 32767)
#wavfile.write('6-A.wav', samplerate1, scaledA)
#scaledB = np.int16(y2r / np.max (np.abs (y2r)) * 32767)
#wavfile.write('6-B.wav', samplerate2, scaledB)

# Amplitudski spektar posle NF
plt.figure(figsize=(16,6))

f1r = fftshift(fftfreq(len(y1r), 1 / samplerate1))
plt.subplot(1,2,1)
plt.xlim(-15000, 15000)
plt.xlabel('$w$')
plt.ylabel('$|X(jw)|$')
plt.title('Signal $y_1^r$ - frekvencijski domen')
plt.plot(f1r, fftshift(np.abs(fft(y1r))))

f2r = fftshift(fftfreq(len(y2r), 1 / samplerate2))
plt.subplot(1,2,2)
plt.xlim(-10000, 10000)
plt.xlabel('$w$')
plt.ylabel('$|X(jw|)$')
plt.title('Signal $y_2^r$ - frekvencijski domen')
plt.plot(f2r, fftshift(np.abs(fft(y2r))))

#plt.savefig('6-9.png')

plt.show()


#################################


#Poredjenje signala
plt.figure(figsize = (16,12))

plt.subplot(2,2,1)
plt.plot(t, y1)
plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.title('Audio signal na ulazu - vremenski domen')

plt.subplot(2,2,2)
plt.plot(t, y2)
plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.title('Audio signal na ulazu - vremenski domen')

plt.subplot(2,2,3)
plt.plot(t, y1r)
plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.title('Audio signal na izlazu - vremenski domen')

plt.subplot(2,2,4)
plt.plot(t, y2r)
plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.title('Audio signal na izlazu - vremenski domen')

#plt.savefig('6-A.png')

plt.show()