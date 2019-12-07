import matplotlib.pyplot as plt 
import numpy as np 
import wave 

from scipy.io.wavfile import write 

spf = wave.open("helloworld.wav", "r")
signal = spf.readframes(-1)
signal = np.frombuffer(signal, "int16")
print("Numpy Signal Shape : ", signal.shape)

plt.plot(signal)
plt.title("Sample Music Without of the echo effect.")
plt.show()

delta = np.array([1., 0., 0.])
no_echo = np.convolve(signal, delta)
print("No Echo Signal Shape: ", no_echo.shape)

no_echo = no_echo.astype(np.int16)
write("helloWorldnoecho.wav", 16000, no_echo) 

filt = np.zeros(16000)
filt[0] = 1
filt[4000] = .6
filt[8000] = .3
filt[12000] = .2
filt[15999] = .1
out = np.convolve(signal,filt)

out = out.astype(np.int16)
write("helloWorld_echo.wav", 16000, out) 
