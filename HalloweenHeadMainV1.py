#motion
import torch
import serial
import time
import cv2
import ultralytics
import multiprocessing

#conversation
import threading
import os
import openai
import time
from pvrecorder import PvRecorder
import wave
import struct
import numpy as np


#initialize variables
times=[]
loudnesses=[]
digitalized_loudnesses=[]
digialized_threshold=200

#speech file maker
def make_speech_file(shared_dict):
    from gtts import gTTS
    import time

    so_far=0
    fi=0

    time_diff=0.5

    t_start=time.time()
    first_time=True


    while True:
        ct=shared_dict["text"].split("You: ")[-1].split("</s>")[0]
        if not shared_dict["tts_finished"] or not shared_dict["finished_llm"]:
            shared_dict["tts_finished"]=False
            if shared_dict["finished_llm"] or (((len(ct)-so_far)>20) and (time.time()-t_start)>time_diff):
                if (shared_dict["last_token_size"]>3) or shared_dict["finished_llm"]:
                    first_time=False
                    tts = gTTS(ct[so_far:])
                    fi+=1
                    tts.save('tts'+str(fi)+'.mp3')
                    shared_dict["file queue"]+=['tts'+str(fi)+'.mp3']

                    print(so_far,fi,": ",ct[so_far:])

                    so_far=len(ct)
                    t_start=time.time()

                if shared_dict["finished_llm"]:
                    print("Full: ",ct)
                    shared_dict["tts_finished"]=True
                    first_time=True
                    so_far=0
                    fi=0

        time.sleep(0.01)

#sound player
def play_from_queue(shared_dict):
    import pygame
    import time
    pygame.init()

    #create pygame window
    #screen = pygame.display.set_mode((640,480))

    #set window caption
    #pygame.display.set_caption("Text to Speech")

    mic_active=True

    t_mouth=time.time()

    while True:
        if len(shared_dict["file queue"])>0:
            shared_dict["playingsound"]=True
            if mic_active:
                print("TIME TO START SPEAKING: ",time.time()-shared_dict["time_started"])
                shared_dict["eyeV"]=1
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
                    shared_dict["mouthV"]=1-shared_dict["mouthV"]
                    t_mouth=time.time()
            shared_dict["mouthV"]=0
                
        else:
            time.sleep(0.01)

        if shared_dict["tts_finished"]:
            if len(shared_dict["file queue"])==0:
                time.sleep(0.5)
                shared_dict["playingsound"]=False
                if not mic_active:
                    print("\n\nMIC ACTIVE\n\n")
                    mic_active=True
        time.sleep(0.01)


#recorder
def record_audio(shared_dict):
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

    global times, loudnesses, digitalized_loudnesses, digialized_threshold

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
                if actively_loud_count>=10:
                    hearing_something=True
                t_hearing_start=time.time()-0.5
            #If hearing something but not actively loud for set period then stop hearing
            else:
                if actively_loud:
                    t_actively_loud=time.time()
                if ((time.time()-t_actively_loud)>1):
                    hearing_something=False

            if (time.time()-t_print)>1:
                print("ACTIVELY LOUD COUNT: ",actively_loud_count,"\n")
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
    with wave.open(path, 'w') as f:
        f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
        if len(raw_audio)>seconds_from_live*16000:
            halved_audio=raw_audio[int(len(raw_audio)-(seconds_from_live*16000)):-16000]
            f.writeframes(struct.pack("h" * len(halved_audio), *halved_audio))
        else:
            halved_audio=raw_audio[:-16000]
            f.writeframes(struct.pack("h" * len(halved_audio), *halved_audio))













#arduino comm thread
def main_arduino(shared_motion,shared_dict):
    ser = serial.Serial('COM6', 115200)

    #set ser timeout
    ser.timeout=5
    time.sleep(3)
    t_var=time.time()
    line=""
    while True:
        mouthV=shared_dict["mouthV"]
        eyeV=shared_dict["eyeV"]
        x_pos_write=shared_motion["x_pos_write"]
        y_pos_write=shared_motion["y_pos_write"]
        command = f"{str(int(x_pos_write))},{str(int(y_pos_write))},{str(mouthV)},{str(eyeV)}\n"
        ser.write(command.encode())


        time.sleep(0.01)


        if (time.time()-t_var)>1:
            print(line,command)
            # if mouthV==1:
            #     mouthV=0
            # else:
            #     mouthV=1
            
            # if eyeV>=2:
            #     shared_dict["eyeV"]=0
            # else:
            #     shared_dict["eyeV"]+=1

            t_var=time.time()

        while ser.in_waiting>0:
            try:
                line=ser.readline().decode('utf-8').rstrip()
                #print("arduino: ",line)
            except:
                print("error")
                pass















#motion functions
def main_motion(shared_motion):

    #setup processess
    servo_process = multiprocessing.Process(target=servo_handling, args=(shared_motion,))
    servo_process.start()

    model = torch.hub.load("ultralytics/yolov5", 'custom', path="crowdhuman_yolov5m.pt")
    #model=yolov5.load("crowdhuman_yolov5m.pt")
    cam=cv2.VideoCapture(1)
    
    x_pos=0
    y_pos=0
    failed=False
    found_times=0

    acc_threshold=0.6
    t_print=time.time()
    t_start=time.time()
    while True:
        try:
            acc_threshold=shared_motion["acc_threshold"]
            ret,frame=cam.read()
            #show frame
            cv2.imshow("frame",frame)
            cv2.imwrite("shi.jpg",frame)
            
            cv2.waitKey(1)
            #hide output of model inference
            with torch.no_grad():
                results = model(frame)
                try:
                    #get results and put on cpu
                    result_vals=results.pred[0][0][:].cpu().numpy()
                    if time.time()-t_print>1:
                        print("ACC: ",result_vals[4])
                        t_print=time.time()
                    #if confidence is too low, don't move
                    if result_vals[4]<acc_threshold:
                        #throw exception to go to except block
                        raise Exception("Confidence too low")


                    x_pos=int((result_vals[0]+result_vals[2])/2)-(320+shared_motion["x_cam_offset"])
                    y_pos=int((result_vals[1]+result_vals[3])/2)-(240+shared_motion["y_cam_offset"])
                    #print(x_pos,y_pos, x_pos_write, y_pos_write ," hs pred: ",result_vals,end="\r")
                    fps=1/(time.time()-t_start)
                    #format to 2 decimal places
                    fps=round(fps,2)

                    #print(fps,x_pos,y_pos,result_vals[4],end="\r")
                    found_times+=1
                    if found_times>=3:
                        failed=False
                except:
                    failed=True
                    found_times=0
            t_start=time.time()
            shared_motion["x_pos"]=x_pos
            shared_motion["y_pos"]=y_pos
            shared_motion["failed"]=failed
        except ValueError:
            print("Please enter a valid number")
        
        except KeyboardInterrupt:
            print("Exiting program")
            #ser.close()  # close the serial connection
            break
        



def servo_handling(shared_motion):
    
    #ser = serial.Serial('COM6', 2400)  # replace 'COM3' with the port where your Arduino is connected
    time.sleep(0.1)  # wait for the serial connection to initialize

    while shared_motion["x_pos"] == None:
        time.sleep(0.1)
    print("Servo process started")


    x_pos=shared_motion["x_pos"]
    y_pos=shared_motion["y_pos"]
    failed=shared_motion["failed"]

    x_pos_write=1500
    y_pos_write=1500
    x_old_write_pos=x_pos_write
    y_old_write_pos=y_pos_write


    MAX_ACCEL=6000
    MAX_VELOCITY=1000

    ACC_DEC=0.1

    LOOP_T=0.05
    MAX_ACCEL*=LOOP_T**2
    MAX_VELOCITY*=LOOP_T
    stop_multiplier=2

    exponent_var=2

    x_velocity=0
    y_velocity=0
    x_accel=0
    y_accel=0
    deadzone=30


    x_dir_val=0
    y_dir_val=0


    person_x_vel=0
    person_y_vel=0

    failed_times=0
    t_settings=time.time()

    panning_dir=1

    failed_3=True
    failed_3_counter=0

    #servo loop
    while True:
        if (time.time()-t_settings)>5:
            with open("settings.txt","r") as f:
                settings=f.read().split("\n")[1:]
                MAX_ACCEL=int(settings[0])
                MAX_VELOCITY=int(settings[1])
                MAX_ACCEL*=LOOP_T**2
                MAX_VELOCITY*=LOOP_T
                deadzone=int(settings[2])
                stop_multiplier=float(settings[3])
                exponent_var=float(settings[4])
                shared_motion["acc_threshold"]=float(settings[5])
                shared_motion["y_cam_offset"]=int(settings[6])
                shared_motion["x_cam_offset"]=int(settings[7])

            t_settings=time.time()
            print("Settings updated",MAX_ACCEL,MAX_VELOCITY,deadzone)

        loop_t=time.time()

        x_pos_prev=x_pos
        y_pos_prev=y_pos

        #get data from main loop
        failed=shared_motion["failed"]
        x_pos=shared_motion["x_pos"]
        y_pos=shared_motion["y_pos"]


        if x_pos!=x_pos_prev:
            person_x_vel=int(x_pos-x_pos_prev)*17
        if y_pos!=y_pos_prev:
            person_y_vel=int(y_pos-y_pos_prev)*17




        # get old_write positions
        if not failed:
            x_old_write_pos=x_pos_write
            y_old_write_pos=y_pos_write
            failed_3_counter+=1
            if failed_3_counter>2:
                failed_3=False
        else:
            failed_3_counter=0
            failed_3=True

        

        #pid to get velocity and change the x_pos_write and y_pos_write
        if not failed_3:
            #x axis
            prev_x_velocity=x_velocity

            #get the sign of the x_pos
            if x_pos>0:
                x_dir_val=1
            elif x_pos<0:
                x_dir_val=-1

            x_velocity=((abs(x_pos)/320)**exponent_var)*MAX_VELOCITY*x_dir_val
            if abs(x_pos)<deadzone:
                x_velocity=0
            
            x_accel=x_velocity-prev_x_velocity
            if abs(x_velocity)>abs(prev_x_velocity):
                if x_accel>MAX_ACCEL:
                    x_accel=MAX_ACCEL
                    x_velocity=x_accel+prev_x_velocity
                elif x_accel<-MAX_ACCEL:
                    x_accel=-MAX_ACCEL
                    x_velocity=x_accel+prev_x_velocity
            else:
                if x_accel>MAX_ACCEL*stop_multiplier:
                    x_accel=MAX_ACCEL*stop_multiplier
                    x_velocity=x_accel+prev_x_velocity
                elif x_accel<-MAX_ACCEL*stop_multiplier:
                    x_accel=-MAX_ACCEL*stop_multiplier
                    x_velocity=x_accel+prev_x_velocity

            if x_velocity>MAX_VELOCITY:
                x_velocity=MAX_VELOCITY
            elif x_velocity<-MAX_VELOCITY:
                x_velocity=-MAX_VELOCITY
            x_pos_write-=x_velocity



            #y axis
            prev_y_velocity=y_velocity

            #get the sign of the y_pos
            if y_pos>0:
                y_dir_val=-1
            elif y_pos<0:
                y_dir_val=1
            
            y_velocity=((abs(y_pos)/240)**exponent_var)*MAX_VELOCITY*y_dir_val
            if abs(y_pos)<deadzone:
                y_velocity=0
            
            y_accel=y_velocity-prev_y_velocity

            if abs(y_velocity)>abs(prev_y_velocity):
                if y_accel>MAX_ACCEL:
                    y_accel=MAX_ACCEL
                    y_velocity=y_accel+prev_y_velocity
                elif y_accel<-MAX_ACCEL:
                    y_accel=-MAX_ACCEL
                    y_velocity=y_accel+prev_y_velocity
            else:
                if y_accel>MAX_ACCEL*2:
                    y_accel=MAX_ACCEL*2
                    y_velocity=y_accel+prev_y_velocity
                elif y_accel<-MAX_ACCEL*2:
                    y_accel=-MAX_ACCEL*2
                    y_velocity=y_accel+prev_y_velocity
            
            if y_velocity>MAX_VELOCITY:
                y_velocity=MAX_VELOCITY
            elif y_velocity<-MAX_VELOCITY:
                y_velocity=-MAX_VELOCITY
            y_pos_write+=y_velocity

            failed_times=0
        else:
            failed_times+=1
            if failed_times>5:
                x_add=person_x_vel*2
                if x_add>200:
                    x_add=200
                elif x_add<-200:
                    x_add=-200
                
                y_add=person_y_vel*2
                if y_add>200:
                    y_add=200
                elif y_add<-200:
                    y_add=-200


                if failed_times<50:
                    x_pos_write-=x_add/50
                    y_pos_write+=y_add/50

                if (failed_times>70)and(failed_times<120):
                    x_pos_write+=x_add/50
                    y_pos_write-=y_add/50
                
                if failed_times==120:
                    x_pos_write=1500
                    y_pos_write=1500
                if failed_times>120 and failed_times<300:
                    if x_pos_write>=2200:
                        panning_dir=-1
                    elif x_pos_write<=1300:
                        panning_dir=1
                    x_pos_write+=panning_dir*MAX_VELOCITY/7
                    if panning_dir==1:
                        y_pos_write=2200-(x_pos_write/2)#1100
                    else:
                        y_pos_write=2300-(x_pos_write/3)
                elif failed_times>=300:
                    y_pos_write=1100
                    x_pos_write=1900

        if not failed:
            if x_pos_write>1500:
                shared_motion["approaching"]=True
                shared_motion["at_door"]=False
            else:
                shared_motion["approaching"]=False
                shared_motion["at_door"]=True
        elif (failed_times)>10:
            shared_motion["approaching"]=False
            shared_motion["at_door"]=False


        # Set bounds
        if x_pos_write<700:
            x_pos_write=700
        elif x_pos_write>2300:
            x_pos_write=2300

        if y_pos_write<700:
            y_pos_write=700
        elif y_pos_write>2400:
            y_pos_write=2400



        # Send the position to the Arduino
        # if not failed:
        #     command = f"{str(int(x_pos_write))},{str(int(y_pos_write))}\n"
        # else:
        #     command = f"{str(int(x_old_write_pos))},{str(int(y_old_write_pos))}\n"
        command = f"{str(int(x_pos_write))},{str(int(y_pos_write))}\n"
        #ser.write(command.encode('utf-8'))
        shared_motion["x_pos_write"]=x_pos_write
        shared_motion["y_pos_write"]=y_pos_write

        #sleep to keep loop time constant
        loop_d=(time.time()-loop_t)
        if loop_d<LOOP_T:
            time.sleep(LOOP_T-loop_d)        
        fps=int(1/(time.time()-loop_t))
        #print("  ",command.split("\n")[0],"  ",fps,"  ",x_velocity,x_accel,failed,"  ",person_y_vel,"  ",person_x_vel,"   ",failed_times," ",failed_3)#,end="\r")


# Vision Description
def description_task(shared_dict):
    vision_output = replicate.run(
        "yorickvp/llava-13b:2facb4a474a0462c15041b78b1ad70952ea46b5ec6ad29583c0b29dbd4249591",
        input={"image": open("shi.jpg", "rb"),"prompt":"In this image is: "}
    )
    description_string=""
    for item in vision_output:
        description_string+=item
        shared_dict["visual_description"]=description_string

    


if __name__ == "__main__":
    #create thread for motion
    import threading
    import keyboard

    # Text to speech
    from gtts import gTTS

    # Read all keys
    with open("SECRET.txt",'r') as f:
        list_of_keys=f.read().split("\n")
        OPENAI_SECRET=list_of_keys[0]
        TOGETHER_SECRET=list_of_keys[1]
        REPLICATE_SECRET=list_of_keys[2]
    
    # Setup OpenAI for transcription
    os.environ['OPENAI_API_KEY'] = OPENAI_SECRET
    openai.api_key=OPENAI_SECRET

    # Setup together LLM
    model_name="togethercomputer/llama-2-13b-chat"
    model_name="togethercomputer/llama-2-70b-chat"
    import together
    together.api_key = TOGETHER_SECRET

    # Setup replicate Vision LLM
    import replicate
    os.environ["REPLICATE_API_TOKEN"]=REPLICATE_SECRET

    # Setup shared data 
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()
    shared_dict["audio"]=[]
    shared_dict["savebool"]=True
    shared_dict["saveduration"]=0
    shared_dict["playingsound"]=True
    shared_dict["text"]=""
    shared_dict["file queue"]=[]
    shared_dict["last_token_size"]=0
    shared_dict["ready"]=True
    shared_dict["finished_llm"]=False
    shared_dict["tts_finished"]=False
    shared_dict["mouthV"]=0
    shared_dict["eyeV"]=1
    shared_dict["visual_description"]=""


    #motion thread
    shared_motion=manager.dict()
    shared_motion["x_pos"]=None
    shared_motion["y_pos"]=None
    shared_motion["x_pos_write"]=1500
    shared_motion["y_pos_write"]=2000
    shared_motion["acc_threshold"]=0.6
    shared_motion["x_cam_offset"]=0
    shared_motion["y_cam_offset"]=0
    shared_motion["failed"]=True
    shared_motion["approaching"]=False
    shared_motion["at_door"]=False
    activate_motion=True
    if activate_motion:
        # Motion thread
        motion_process = threading.Thread(target=main_motion, args=(shared_motion,))
        motion_process.start()

        # Arduino thread
        arduino_process = threading.Thread(target=main_arduino, args=(shared_motion,shared_dict))
        arduino_process.start()
        time.sleep(3)
        print("arduino on")


    being_smart=True

    #start threads
    recording_thread = threading.Thread(target=record_audio, args=(shared_dict,))
    recording_thread.start()

    #sound thread
    sound_thread=threading.Thread(target=play_from_queue, args=(shared_dict,))
    sound_thread.start()

    #gTTS thread
    gtts_thread=threading.Thread(target=make_speech_file,args=(shared_dict,))
    gtts_thread.start()



    chat_log=""

    path_i=0
    #while recording thread is running, do this
    approaching_to_door=False
    prev_approaching_to_door=False
    approach_ready=True

    t_print=time.time()
    off_bool=True

    t_failed_start=time.time()

    while recording_thread.is_alive():
        time.sleep(0.05)

        if shared_motion["approaching"]:
            approaching_to_door=True

        if approaching_to_door and not prev_approaching_to_door:
            description_thread=threading.Thread(target=description_task, args=(shared_dict,))
            description_thread.start()
                
            
        
        prev_approaching_to_door=approaching_to_door

        if approaching_to_door:
            if shared_motion["at_door"]:
                approach_ready=True
                approaching_to_door=False

        if shared_motion["failed"]:
            if (time.time()-t_failed_start)>2:
                approach_ready=False
                shared_dict["savebool"]=True
                chat_log=""
                t_failed_start=time.time()
        else:
            t_failed_start=time.time()

        if keyboard.is_pressed('r'):
            approach_ready=True
            print("\nAPPROACH READY ON\n")
        if keyboard.is_pressed('c'):
            off_bool=True
        if keyboard.is_pressed('v'):
            off_bool=False

        if time.time()-t_print>1:
            print("Failed: ", shared_motion['failed'], "Approaching: ", shared_motion["approaching"], "At Door: ", shared_motion["at_door"], "Approach Ready: ", approach_ready," Approaching to door: ", approaching_to_door)
            t_print=time.time()

        if off_bool:
            shared_dict["text"]=""
            shared_dict["savebool"]=False

        #if recording thread has saved audio, do this
        if shared_dict["savebool"] and approach_ready and not off_bool:
            path_i+=1
            shared_dict["savebool"]=False
            save_audio(shared_dict["audio"],shared_dict["saveduration"],"audiofiles/"+str(path_i)+".wav")
            print(shared_dict["saveduration"]," Saving audio ", path_i)

            #stop listening
            t_thinking_start=time.time()
            shared_dict["playingsound"]=True


            if being_smart:
                #transcribe
                t_start=time.time()
                if chat_log!="":
                    shared_dict["eyeV"]=2
                    audio_file= open("audiofiles/"+str(path_i)+".wav", "rb")
                    try:
                        transcript = openai.Audio.transcribe("whisper-1", audio_file)
                    except:
                        transcript="You couldn't hear me"
                        break
                else:
                    shared_dict["eyeV"]=2
                    transcript={"text":""}
                t_end=time.time()
                print("Time to transcribe: ",t_end-t_start, " Text:",transcript["text"])

                ts=time.time()
                # generate response
                if chat_log=="":
                    chat_log="""You are in a spooky skeleton head on halloween night being witty and humorous at the front door of a house while keeping what you say short. Make sure to make the convesation go back and forth.
Rules:
    Your response will be 1 message only which is 'You: <message>'.
    Your response should be short and concise unless the context calls for a longer response.
    You do not use emojis in your response.

Description of what you see right now: \""""+shared_dict["visual_description"]+".\""
                    the_prompt=chat_log+"\n\nYou: <message>\n\nWhat do you want to say for the <message> to the user to which will start the conversation? This can be something funny, a comment on the appearance of the people at the door or a introductory statement, ex: \"Trick or treat, smell my feet. I hope you didn't forget any candy because I'm dying for a sugar rush!\". Provide a witty, humorous greeting or comment to the children. Your answer will be used as <message>."
                    the_prompt="Imagine you are a witty and humorous character inside a spooky skeleton head, greeting trick-or-treaters on Halloween night at the front door of a house. This is what you see in front of you: "+shared_dict["visual_description"]+". Respond with a single, concise message starting with 'You:'. No emojis. Provide a witty, humorous introduction to the user, be sure to make it very short and to the point with very little fluff ie: not starting with well, well, well. Your answer will be used as <message>."
                else:
                    chat_log=chat_log+"\nUser: "+str(transcript["text"])
                    the_prompt=chat_log+"\nYou: <message>\n\nWhat should you say for the <message> to the user?\nAnswer: "
                print("the prompt test: ",the_prompt)
                #the_prompt="I've seen more candy in a single bag! But, hey"
                ts=time.time()
                # generate response
                result_l=[]
                shared_dict["file queue"]=[]
                shared_dict["time_started"]=time.time()
                result_s=""
                shared_dict["finished_llm"]=False
                shared_dict["text"]=""
                for token in together.Complete.create_streaming(prompt=the_prompt,model=model_name,max_tokens=100,temperature=0.6):
                    print(token, end="", flush=True)
                    shared_dict["last_token_size"]=len(token)
                    result_l.append(token)
                    result_s="".join(result_l).strip()
                    shared_dict["text"]=result_s
                shared_dict["finished_llm"]=True
                chat_log=chat_log+"\nYou: "+result_s.split("You: ")[-1].split("</s>")[0]
                print("\n")
                print("TIME: ",time.time()-ts)


                print("LLM Time: ",time.time()-ts)


    recording_thread.join()
    sound_thread.join()
    gtts_thread.join()
    print("Done")