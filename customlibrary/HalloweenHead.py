#recorder
def recorder(shared_dict):
    from pvrecorder import PvRecorder
    import wave
    import struct
    import numpy as np
    import openai
    import time
    recorder = PvRecorder(device_index=0, frame_length=512)
    audio = []
    shared_dict["audio"]=audio
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

    #initialize variables
    times=[]
    loudnesses=[]
    digitalized_loudnesses=[]
    digialized_threshold=200
    loud_counts_before_hearing=3

    t_print=time.time()
    try:
        # Start recording
        recorder.start()
        print("Starting Recording")
        while True:
            # Get audio frame
            frame = recorder.read()
            audio.extend(frame)

            # Calculate average loudness
            avg_loudness+=np.mean(np.abs(frame))*0.3
            avg_loudness*=0.7
            #print("Average Loudness: ", avg_loudness,"     ", end='\r')

            t_current = time.time() - t_start
            times.append(t_current)
            loudnesses.append(avg_loudness)

            #digitalize loudness and save audio if speech segment
            if avg_loudness>digialized_threshold:
                digitalized_loudnesses.append(650)
                actively_loud=True
                actively_loud_count+=1
                if actively_loud_count>14:
                    actively_loud_count=14
            else:
                digitalized_loudnesses.append(100)
                actively_loud=False
                actively_loud_count-=1
                if actively_loud_count<0:
                    actively_loud_count=0

            #Turn off recording when speaking
            if shared_dict["playingsound"]:
                actively_loud=False
                t_hearing_start=time.time()-0.25
                actively_loud_count=0

            #If not hearing something set the start time as current but at threshold of loudness start hearing
            if not hearing_something:
                if actively_loud_count>=loud_counts_before_hearing:
                    hearing_something=True
                t_hearing_start=time.time()-0.5
            #If hearing something but not actively loud for set period then stop hearing
            else:
                if actively_loud:
                    t_actively_loud=time.time()
                if ((time.time()-t_actively_loud)>1):
                    hearing_something=False

            if (time.time()-t_print)>1:
                #print("ACTIVELY LOUD COUNT: ",actively_loud_count,"\n")
                t_print=time.time()

            if not shared_dict["playingsound"] and not hearing_something and prev_hearing_something:
                print("\nSAVEBOOL TRUE\n\n")
                shared_dict["audio"]=audio.copy()
                path_i+=1
                shared_dict["savebool"]=True
                shared_dict["saveduration"]=time.time()-t_hearing_start
                audio=[]
                #save_audio(shared_dict["audio"],6,"audiofolder/"+str(path_i)+".wav")

            prev_hearing_something=hearing_something


            # If the average loudness is below 120 for 0.35 seconds after 1 second, stop recording
            if (time.time()-t_start)>1.0:
                if avg_loudness<120:
                    if (time.time()-tlt_100)>time_before_exiting:
                        pass
                        #raise KeyboardInterrupt
                else:
                    tlt_100=time.time()
            else:
                tlt_100=time.time()
    except KeyboardInterrupt:
        # Stop recording and save to file
        recorder.stop()
        with wave.open(path, 'w') as f:
            f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
            f.writeframes(struct.pack("h" * len(audio), *audio))
        with wave.open("lasthhalf.wav", 'w') as f:
            f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
            halved_audio=audio[len(audio)//2:]
            f.writeframes(struct.pack("h" * len(halved_audio), *halved_audio))
        recorder.delete()
        print("Recording Finished")
    finally:
        # Transcribe audio
        t_start=time.time()
        audio_file= open(path, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        t_end=time.time()
        print("Time to transcribe: ",t_end-t_start, " Text:",transcript["text"])

        return(transcript["text"])
    


#save audio
def save_audio(raw_audio, seconds_from_live,path):
    import wave
    import struct    
    
    with wave.open(path, 'w') as f:
        f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
        if len(raw_audio)>seconds_from_live*16000:
            halved_audio=raw_audio[int(len(raw_audio)-(seconds_from_live*16000)):-16000]
            f.writeframes(struct.pack("h" * len(halved_audio), *halved_audio))
        else:
            halved_audio=raw_audio[:-16000]
            f.writeframes(struct.pack("h" * len(halved_audio), *halved_audio))
                    


# Speech file maker
def tts_generator(shared_dict):
    from gtts import gTTS
    import time

    so_far=0
    fi=0

    time_diff=0.5

    t_start=time.time()
    first_time=True


    ending_marks=[" ",".","!","?"," ","\n"]

    ct_prev=""

    pause_index=0
    while True:
        ct=shared_dict["text"].split("You: ")[-1].split("</s>")[0]
        if ct!=ct_prev:
            if len(ct)>0:
                print("CT -1:",ct[-1],"ct:",ct)
        ct_prev=ct
        llm_finished=shared_dict["llm_finished"]
        if not shared_dict["tts_finished"]:
            shared_dict["tts_finished"]=False
            if llm_finished or (((len(ct)-so_far)>20) and (time.time()-t_start)>time_diff):
                pause_index=len(ct)-1
                while (pause_index>so_far) and (ct[pause_index] not in ending_marks):
                    pause_index-=1

                if len(ct)>0:
                    print("GTTS PRINT: ",so_far,fi,": ",ct[so_far:pause_index], " Pause index: ",pause_index," Len ct: ",len(ct))
                    if shared_dict["llm_finished"]:
                        tts = gTTS(ct[so_far:])
                        so_far=len(ct)
                    else:
                        tts = gTTS(ct[so_far:pause_index])
                        so_far=pause_index
                    
                    #Save tts
                    fi+=1
                    tts.save('working_files/tts_files/tts'+str(fi)+'.mp3')
                    shared_dict["file queue"]+=['working_files/tts_files/tts'+str(fi)+'.mp3']

                    t_start=time.time()

                    if llm_finished:
                        print("Full: ",ct)
                        shared_dict["tts_finished"]=True
                        first_time=True
                        so_far=0
                        pause_index=0
                        fi=0

        time.sleep(0.01)




#sound player
def tts_player(shared_dict):
    import pygame
    import time
    pygame.init()

    mic_active=True

    t_mouth=time.time()

    while True:
        if len(shared_dict["file queue"])>0:
            shared_dict["playingsound"]=True
            if mic_active:
                print("TIME TO START SPEAKING: ",time.time()-shared_dict["time_started"])
                shared_dict["eye"]=1
            mic_active=False
            file=shared_dict["file queue"][0]
            print("PLAYING: ",file)
            shared_dict["file queue"]=shared_dict["file queue"][1:]
            sound = pygame.mixer.Sound(file)
            sound.play()
            #print("duration of audio file: ",sound.get_length())
            while pygame.mixer.get_busy():
                time.sleep(0.05)
                if time.time()-t_mouth>0.2:
                    shared_dict["mouth"]=1-shared_dict["mouth"]
                    t_mouth=time.time()
            shared_dict["mouth"]=0
                
        else:
            time.sleep(0.01)

        if shared_dict["tts_finished"]:
            if len(shared_dict["file queue"])==0:
                if not mic_active:
                    shared_dict["text"]=""
                    shared_dict["eye"]=0
                    time.sleep(0.5)
                    shared_dict["eye"]=1
                    shared_dict["playingsound"]=False
                    print("\n\nMIC ACTIVE\n\n")
                    mic_active=True
        time.sleep(0.01)




