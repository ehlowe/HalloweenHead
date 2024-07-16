import openai 
from pvrecorder import PvRecorder
import time
import numpy as np
import wave
import struct
import queue
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import cv2
from scipy.fftpack import fft

import os
with open("working_files/SECRET.txt",'r') as f:
    OPENAI_SECRET=f.read().split("\n")[0]
os.environ['OPENAI_API_KEY'] = OPENAI_SECRET
openai.api_key=OPENAI_SECRET

times = []
loudnesses = []
digitalized_loudnesses=[]
loud_counts=[]
digialized_threshold=210
record_list=[]
dominant_frequency_list=[]
freq_magnitude=[]
freq_magnitudes=[]
xf=[]

data_queue = queue.Queue()

#recorder
def main_record_audio():
    recorder = PvRecorder(device_index=0, frame_length=512)
    audio = []
    path = 'audio_recording.wav'
    avg_loudness=0
    t_start=time.time()
    tlt_100=time.time()

    time_before_saving=1.5
    time_before_exiting=35

    path_i=0
    hearing_something=False
    prev_hearing_something=False
    t_hearing_start=time.time()

    actively_loud=False
    prev_actively_loud=False
    t_actively_loud=time.time()

    actively_loud_count=0

    record_val=600
    global times, loudnesses, digitalized_loudnesses, digialized_threshold,loud_counts, record_list, dominant_frequency_list, freq_magnitude

    global freq_magnitudes, xf
    t_print=time.time()

    
    loops_count=0
    loops_print_count=0
    try:
        # Start recording
        recorder.start()
        print("Starting Recording")
        while True:
            # Get audio frame
            frame = recorder.read()
            audio.extend(frame)
            #if audio is longer than 10 seconds then cutoff old audio frames
            if len(audio)>16000*10:
                audio=audio[len(audio)-16000*10:]

            N = len(frame)
            yf = fft(frame)
            xf = np.linspace(0.0, 16000/2.0, N//2)  # Assuming a sampling rate of 16000 Hz
            freq_magnitude = 2.0/N * np.abs(yf[0:N//2])
            data_queue.put(freq_magnitude)
            
            # Get the dominant frequency
            #dominant_frequency = xf[np.argmax(freq_magnitude)]
            #dominant_frequency_list.append(dominant_frequency)

            # Calculate average loudness
            avg_loudness+=np.mean(np.abs(frame))*0.3
            avg_loudness*=0.7
            #print("Average Loudness: ", avg_loudness,"     ", end='\r')

            t_current = time.time() - t_start
            #times.append(t_current)
            #loudnesses.append(avg_loudness)

            #digitalize loudness and save audio if speech segment
            if avg_loudness>digialized_threshold:
                #digitalized_loudnesses.append(650)
                actively_loud=True
                actively_loud_count+=1
                if actively_loud_count>14:
                    actively_loud_count=14
            else:
                #digitalized_loudnesses.append(100)
                actively_loud=False
                actively_loud_count-=1
                if actively_loud_count<0:
                    actively_loud_count=0
            #loud_counts.append(actively_loud_count*250)

            #If not hearing something set the start time as current but at threshold of loudness start hearing
            if not hearing_something:
                if actively_loud_count>=4:
                    hearing_something=True
                t_hearing_start=time.time()-0.5
            #If hearing something but not actively loud for set period then stop hearing
            else:
                record_val=800
                if actively_loud:
                    t_actively_loud=time.time()
                if ((time.time()-t_actively_loud)>1):
                    hearing_something=False

            if (time.time()-t_print)>1:
                print("ACTIVELY LOUD COUNT: ",actively_loud_count," Loops count in 1 second: ",loops_count-loops_print_count,"\n")
                loops_print_count=loops_count
                print("FM: ",freq_magnitude)
                t_print=time.time()

            
            if not hearing_something and prev_hearing_something:
                record_val=600
            #record_list.append(record_val)
            

            prev_hearing_something=hearing_something
            loops_count+=1


    except KeyboardInterrupt:
        # Stop recording and save to file
        recorder.stop()
        with wave.open(path, 'w') as f:
            f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
            f.writeframes(struct.pack("h" * len(audio), *audio))


MAX_MAGNITUDE=500
spec_data=[]
def plot_spectrogram():
    fig, ax = plt.subplots()

    def update():
        if not data_queue.empty():  # Check if there's any data in the list

            # If the data becomes too large, you might want to limit its size
            # For example, keep only the latest 100 frames:
            while not data_queue.empty():
                current_magnitude = data_queue.get()  # Get the earliest added magnitude
                spec_data.append(current_magnitude)
            
            while len(spec_data) > 100:
                del spec_data[0]

            ax.clear()
            # Display the spectrogram
            ax.imshow(np.transpose(spec_data), origin="lower", aspect="auto",
                      cmap="inferno", extent=[0, len(spec_data), 0, 16000/2.0],vmin=0, vmax=MAX_MAGNITUDE)
            
            ax.set_ylabel("Frequency (Hz)")
            ax.set_xlabel("Time (frames)")

            #get the ax image
            im = ax.get_images()[0]
            #get shape of im
            shape = im.get_array().shape
            print(shape)

            #convert to a 3d array
            im_3d = np.dstack((im.get_array(),)*3)
            print(im_3d.shape)

            # plot it
            plt.show(block=False)
            plt.pause(0.001)


    while True:
        update()

    

# Start audio recording in a separate thread
threading.Thread(target=main_record_audio, args=()).start()


plot_spectrogram()