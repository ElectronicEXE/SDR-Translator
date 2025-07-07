# SDR-Translator

A simple python program that "live" translate audio.

There are 2 versiions: There is one that uses google translate, and the other runs fully offfline after downloading required models to work.The google translate one performs better, but it needs internet.

To use it with an SDR, like the RTL-SDR, configure your SDR program of choice to output audio in to a Virtual Audio Cable.

In the softwer, select the VB Cable Output and adjust settings if you want. 

# !!!!! IMPORTANT !!!!!

The first time you run the code it needs to download the Whisper translator model, and it will take some time.


# Dependencies:
'''
 sounddevice numpy webrtcvad whisper deep_translator queue threading time ttkbootstrap tkinter 
'''
 ![image](https://github.com/user-attachments/assets/e2b9089e-e275-4aa8-8109-228c023a8d25)
