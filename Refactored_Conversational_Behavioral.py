# Motion
import serial
import time
import multiprocessing
import socket

# Conversation
import threading
import os
import openai
from openai import OpenAI
import time
from pvrecorder import PvRecorder
import wave
import struct
import numpy as np


# Import the functions for conversation
import CustomImports as hh


# Socket connection
def socket_connection(shared_dict):
    import socket
    import select

    print(socket.gethostname())

    hostname = socket.gethostname()
    port=1234
    s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    #set timeout
    s.settimeout(2)

    s.bind((hostname, port))
    s.listen(1)
    try:
        conn_obj, addr = s.accept()
    except:
        pass

    fail_check_time=time.time()
    failed_check_times=0


    while True:
        try:
            data=conn_obj.recv(1024)
            data_string=data.decode()
            if data_string!="":
                numbers_l=data_string.split(",")
                shared_dict["x_pos"]=int(numbers_l[0])
                shared_dict["y_pos"]=int(numbers_l[1])
                if int(numbers_l[2])==1:
                    shared_dict["failed"]=True
                else:
                    shared_dict["failed"]=False

            mouth=shared_dict["mouth"]
            eye=shared_dict["eye"]
            send_message=f"{str(mouth)},{str(eye)}\n"
            conn_obj.sendall(send_message.encode())

            # Update the approaching and at_door variables
            if (time.time()-fail_check_time)>0.2:
                if not shared_dict["failed"]:
                    failed_check_times=0
                    if shared_dict["x_pos"]>1500:
                        shared_dict["approaching"]=True
                        shared_dict["at_door"]=False
                    else:
                        shared_dict["approaching"]=False
                        shared_dict["at_door"]=True
                else:
                    failed_check_times+=1

                if (failed_check_times)>10:
                    shared_dict["approaching"]=False
                    shared_dict["at_door"]=False
                
                fail_check_time=time.time()


            
        except Exception as e:
            print("read or send error: ", e, " line: ", e.__traceback__.tb_lineno)
            try:
                conn_obj, addr = s.accept()
            except:
                print("Tried to reconnect, but failed")



# def servo_handling(shared_motion):
    
#     #ser = serial.Serial('COM6', 2400)  # replace 'COM3' with the port where your Arduino is connected
#     time.sleep(0.1)  # wait for the serial connection to initialize

#     while shared_motion["x_pos"] == None:
#         time.sleep(0.1)
#     print("Servo process started")


#     x_pos=shared_motion["x_pos"]
#     y_pos=shared_motion["y_pos"]
#     failed=shared_motion["failed"]

#     x_pos_write=1500
#     x_pos_write_prev=1500
#     x_test_vel=0
#     x_old_write_pos=x_pos_write

#     y_pos_write=1500
#     y_old_write_pos=y_pos_write


#     MAX_ACCEL=6000
#     MAX_VELOCITY=1000

#     ACC_DEC=0.1

#     LOOP_T=0.03
#     MAX_ACCEL*=LOOP_T**2
#     MAX_VELOCITY*=LOOP_T
#     stop_multiplier=2

#     exponent_var=2

#     x_velocity=0
#     y_velocity=0
#     x_accel=0
#     y_accel=0
#     deadzone=30


#     x_dir_val=0
#     y_dir_val=0


#     person_x_vel=0
#     person_y_vel=0

#     failed_times=0
#     t_settings=time.time()

#     panning_dir=1

#     failed_3=True
#     failed_3_counter=0

#     t_print=time.time()

#     #servo loop
#     while True:
#         if (time.time()-t_print)>0.5:
#             print("CONTROLS: ",((person_x_vel-x_test_vel)*LOOP_T),person_x_vel,x_test_vel,x_pos_write,x_pos)
#             t_print=time.time()
#         if (time.time()-t_settings)>5:
#             with open("working_files/settings.txt","r") as f:
#                 settings=f.read().split("\n")[1:]
#                 MAX_ACCEL=int(settings[0])
#                 MAX_VELOCITY=int(settings[1])
#                 MAX_ACCEL*=LOOP_T**2
#                 MAX_VELOCITY*=LOOP_T
#                 deadzone=int(settings[2])
#                 stop_multiplier=float(settings[3])
#                 exponent_var=float(settings[4])
#                 shared_motion["acc_threshold"]=float(settings[5])
#                 shared_motion["y_cam_offset"]=int(settings[6])
#                 shared_motion["x_cam_offset"]=int(settings[7])

#             t_settings=time.time()
#             print("Settings updated",MAX_ACCEL,MAX_VELOCITY,deadzone)

#         loop_t=time.time()

#         x_pos_prev=x_pos
#         y_pos_prev=y_pos

#         #get data from main loop
#         failed=shared_motion["failed"]
#         x_pos=shared_motion["x_pos"]
#         y_pos=shared_motion["y_pos"]


#         if x_pos!=x_pos_prev:
#             person_x_vel=((x_pos-x_pos_prev)/(LOOP_T*1.76))*0.5+0.5*person_x_vel
#         if y_pos!=y_pos_prev:
#             person_y_vel=int(y_pos-y_pos_prev)*17




#         # get old_write positions
#         if not failed:
#             x_old_write_pos=x_pos_write
#             y_old_write_pos=y_pos_write
#             failed_3_counter+=1
#             if failed_3_counter>2:
#                 failed_3=False
#         else:
#             failed_3_counter=0
#             failed_3=True

        
#         #pid to get velocity and change the x_pos_write and y_pos_write
#         if not failed_3:
#             #x axis
#             prev_x_velocity=x_velocity

#             #get the sign of the x_pos
#             if x_pos>0:
#                 x_dir_val=1
#             elif x_pos<0:
#                 x_dir_val=-1

#             x_velocity=((abs(x_pos)/320)**exponent_var)*MAX_VELOCITY*x_dir_val
#             if abs(x_pos)<deadzone:
#                 x_velocity=0
            
#             x_accel=x_velocity-prev_x_velocity
#             if abs(x_velocity)>abs(prev_x_velocity):
#                 if x_accel>MAX_ACCEL:
#                     x_accel=MAX_ACCEL
#                     x_velocity=x_accel+prev_x_velocity
#                 elif x_accel<-MAX_ACCEL:
#                     x_accel=-MAX_ACCEL
#                     x_velocity=x_accel+prev_x_velocity
#             else:
#                 if x_accel>MAX_ACCEL*stop_multiplier:
#                     x_accel=MAX_ACCEL*stop_multiplier
#                     x_velocity=x_accel+prev_x_velocity
#                 elif x_accel<-MAX_ACCEL*stop_multiplier:
#                     x_accel=-MAX_ACCEL*stop_multiplier
#                     x_velocity=x_accel+prev_x_velocity

#             if x_velocity>MAX_VELOCITY:
#                 x_velocity=MAX_VELOCITY
#             elif x_velocity<-MAX_VELOCITY:
#                 x_velocity=-MAX_VELOCITY


#             #add player speed:


#             # Old pixel diff method
#             #x_pos_write-=x_velocity

#             x_pos_write=(x_pos_write-((abs(x_pos)**exponent_var)*x_dir_val*1.76*LOOP_T*1.05))#-((person_x_vel-x_test_vel)*LOOP_T*0.15)
#             x_test_vel=(((x_pos_write-x_pos_write_prev)/LOOP_T)*0.5)+0.5*x_test_vel
#             x_pos_write_prev=x_pos_write
#             #person_x_vel



#             #y axis
#             prev_y_velocity=y_velocity

#             #get the sign of the y_pos
#             if y_pos>0:
#                 y_dir_val=-1
#             elif y_pos<0:
#                 y_dir_val=1
            
#             y_velocity=((abs(y_pos)/240)**exponent_var)*MAX_VELOCITY*y_dir_val
#             if abs(y_pos)<deadzone:
#                 y_velocity=0
            
#             y_accel=y_velocity-prev_y_velocity

#             if abs(y_velocity)>abs(prev_y_velocity):
#                 if y_accel>MAX_ACCEL:
#                     y_accel=MAX_ACCEL
#                     y_velocity=y_accel+prev_y_velocity
#                 elif y_accel<-MAX_ACCEL:
#                     y_accel=-MAX_ACCEL
#                     y_velocity=y_accel+prev_y_velocity
#             else:
#                 if y_accel>MAX_ACCEL*2:
#                     y_accel=MAX_ACCEL*2
#                     y_velocity=y_accel+prev_y_velocity
#                 elif y_accel<-MAX_ACCEL*2:
#                     y_accel=-MAX_ACCEL*2
#                     y_velocity=y_accel+prev_y_velocity
            
#             if y_velocity>MAX_VELOCITY:
#                 y_velocity=MAX_VELOCITY
#             elif y_velocity<-MAX_VELOCITY:
#                 y_velocity=-MAX_VELOCITY
#             y_pos_write+=y_velocity

#             failed_times=0
#         else:
#             failed_times+=1
#             if failed_times>5:
#                 x_add=person_x_vel*2
#                 if x_add>200:
#                     x_add=200
#                 elif x_add<-200:
#                     x_add=-200
                
#                 y_add=person_y_vel*2
#                 if y_add>200:
#                     y_add=200
#                 elif y_add<-200:
#                     y_add=-200


#                 if failed_times<50:
#                     x_pos_write-=x_add/50
#                     y_pos_write+=y_add/50

#                 if (failed_times>70)and(failed_times<120):
#                     x_pos_write+=x_add/50
#                     y_pos_write-=y_add/50
                
#                 if failed_times==120:
#                     x_pos_write=1500
#                     y_pos_write=1500
#                 if failed_times>120 and failed_times<300:
#                     if x_pos_write>=2200:
#                         panning_dir=-1
#                     elif x_pos_write<=1300:
#                         panning_dir=1
#                     x_pos_write+=panning_dir*MAX_VELOCITY/7
#                     if panning_dir==1:
#                         y_pos_write=2200-(x_pos_write/2)#1100
#                     else:
#                         y_pos_write=2300-(x_pos_write/3)
#                 elif failed_times>=300:
#                     y_pos_write=1600
#                     x_pos_write=1900

#         if not failed:
#             if x_pos_write>1500:
#                 shared_motion["approaching"]=True
#                 shared_motion["at_door"]=False
#             else:
#                 shared_motion["approaching"]=False
#                 shared_motion["at_door"]=True
#         elif (failed_times)>10:
#             shared_motion["approaching"]=False
#             shared_motion["at_door"]=False


#         # Set bounds
#         if x_pos_write<700:
#             x_pos_write=700
#         elif x_pos_write>2300:
#             x_pos_write=2300

#         if y_pos_write<700:
#             y_pos_write=700
#         elif y_pos_write>2400:
#             y_pos_write=2400



#         # Send the position to the Arduino
#         # if not failed:
#         #     command = f"{str(int(x_pos_write))},{str(int(y_pos_write))}\n"
#         # else:
#         #     command = f"{str(int(x_old_write_pos))},{str(int(y_old_write_pos))}\n"
#         command = f"{str(int(x_pos_write))},{str(int(y_pos_write))}\n"
#         #ser.write(command.encode('utf-8'))
#         shared_motion["x_pos_write"]=x_pos_write
#         shared_motion["y_pos_write"]=y_pos_write

#         #sleep to keep loop time constant
#         loop_d=(time.time()-loop_t)
#         if loop_d<LOOP_T:
#             time.sleep(LOOP_T-loop_d)        
#         fps=int(1/(time.time()-loop_t))
#         #print("  ",command.split("\n")[0],"  ",fps,"  ",x_velocity,x_accel,failed,"  ",person_y_vel,"  ",person_x_vel,"   ",failed_times," ",failed_3)#,end="\r")





# Vision Description
def description_task(shared_dict):
    import cv2
    while True:
        im=cv2.imread("working_files/shi.jpg")
        initial_size = os.path.getsize("working_files/shi.jpg")
        time.sleep(0.01)
        if (os.path.getsize("working_files/shi.jpg") == initial_size) and (initial_size>300):
            cv2.imwrite("working_files/vision_description_image.jpg",im)
            break

    vision_output = replicate.run(
        "yorickvp/llava-13b:2facb4a474a0462c15041b78b1ad70952ea46b5ec6ad29583c0b29dbd4249591",
        input={"image": open("working_files/vision_description_image.jpg", "rb"),"prompt":"In this image is: "}
    )
    description_string=""
    for item in vision_output:
        description_string+=item
        shared_dict["visual_description"]=description_string


    


if __name__ == "__main__":
    if not os.path.isdir("working_files/audio_files"):
        os.mkdir("working_files/audio_files")
    if not os.path.isdir("working_files/tts_files"):
        os.mkdir("working_files/tts_files")

    #create thread for motion
    import threading
    import keyboard
    import cv2

    # Read all keys
    with open("working_files/SECRET.txt",'r') as f:
        list_of_keys=f.read().split("\n")
        OPENAI_SECRET=list_of_keys[0]
        TOGETHER_SECRET=list_of_keys[1]
        REPLICATE_SECRET=list_of_keys[2]
    
    # Setup OpenAI for transcription
    os.environ['OPENAI_API_KEY'] = OPENAI_SECRET
    openai.api_key=OPENAI_SECRET
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    # Setup together LLM
    model_name="togethercomputer/llama-2-13b-chat"
    together_model_name="togethercomputer/llama-2-70b-chat"
    import together
    together.api_key = TOGETHER_SECRET

    # Setup replicate Vision LLM
    import replicate
    os.environ["REPLICATE_API_TOKEN"]=REPLICATE_SECRET

    # Setup shared data 
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()
    #Audio
    shared_dict["audio"]=[]
    shared_dict["savebool"]=True
    shared_dict["saveduration"]=0
    shared_dict["playingsound"]=True

    #TTs
    shared_dict["text"]=""
    shared_dict["file queue"]=[]
    shared_dict["last_token_size"]=0
    shared_dict["llm_finished"]=False
    shared_dict["tts_finished"]=False

    # Animation
    shared_dict["mouth"]=0
    shared_dict["eye"]=1

    # Vision
    shared_dict["visual_description"]=""

    # Motion
    shared_dict["x_pos"]=None
    shared_dict["y_pos"]=None
    shared_dict["failed"]=True
    shared_dict["approaching"]=False
    shared_dict["at_door"]=False



    socket_thread = threading.Thread(target=socket_connection, args=(shared_dict,))
    socket_thread.start()

    being_smart=True

    #start threads
    recording_thread = threading.Thread(target=hh.recorder, args=(shared_dict,))
    recording_thread.start()

    #sound thread
    sound_thread=threading.Thread(target=hh.tts_player, args=(shared_dict,))
    sound_thread.start()

    #gTTS thread
    tts_thread=threading.Thread(target=hh.tts_generator,args=(shared_dict,))
    tts_thread.start()



    chat_log=""

    path_i=0
    #while recording thread is running, do this
    approaching_to_door=False
    prev_approaching_to_door=False
    approach_ready=False

    t_print=time.time()
    off_bool=False

    t_failed_start=time.time()

    ran_vision_description=False

    openai_bool=False
    together_bool=True
    if openai_bool:
        together_bool=False
    if together_bool:
        openai_bool=False

    import base64
    import os



    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_openai_response(the_prompt):
        client = OpenAI()
        image_path="working_files/vision_description_image.jpg"
        while True:
            im=cv2.imread("working_files/shi.jpg")
            initial_size = os.path.getsize("working_files/shi.jpg")
            time.sleep(0.01)
            if (os.path.getsize("working_files/shi.jpg") == initial_size) and (initial_size>300):
                cv2.imwrite(image_path,im)
                break

        base64_image = encode_image(image_path)
        image_url=f"data:image/jpeg;base64,{base64_image}"

        try:
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {'role': 'user', 'content': [{"type": "text", "text":the_prompt},{"type": "image", "image": image_url},],}
                ],
                temperature=0.4,
                max_tokens=500,
                stream=True
            )
            return response
        except Exception as e:
            print(f"An exception occurred: {e}")
            return None
    

    while recording_thread.is_alive():
        time.sleep(0.05)

        if shared_dict["approaching"]:
            approaching_to_door=True

        if not ran_vision_description:
            if (time.time()-t_failed_start)>0.4:
                description_thread=threading.Thread(target=description_task, args=(shared_dict,))
                #description_thread.start()
                ran_vision_description=True
        
        prev_approaching_to_door=approaching_to_door

        if approaching_to_door:
            if shared_dict["at_door"]:
                approach_ready=True
                approaching_to_door=False

        if shared_dict["failed"]:
            if (time.time()-t_failed_start)>2:
                approach_ready=False
                shared_dict["savebool"]=True
                chat_log=""
                t_failed_start=time.time()
                ran_vision_description=False
        else:
            t_failed_start=time.time()

        if keyboard.is_pressed('r'):
            approach_ready=True
            print("\nAPPROACH READY ON\n")
        # if keyboard.is_pressed('c'):
        #     off_bool=True
        # if keyboard.is_pressed('v'):
        #     off_bool=False

        if time.time()-t_print>1:
            print("Failed: ", shared_dict['failed'], "Approaching: ", shared_dict["approaching"], "At Door: ", shared_dict["at_door"], "Approach Ready: ", approach_ready," Approaching to door: ", approaching_to_door)
            t_print=time.time()

        # if off_bool:
        #     shared_dict["text"]=""
        #     shared_dict["savebool"]=False

        #if recording thread has saved audio, do this
        if shared_dict["savebool"] and approach_ready and not off_bool:
            if not ran_vision_description:
                description_thread=threading.Thread(target=description_task, args=(shared_dict,))
                #description_thread.start()
                ran_vision_description=True
            shared_dict["playingsound"]=True
            path_i+=1
            shared_dict["savebool"]=False
            hh.save_audio(shared_dict["audio"],shared_dict["saveduration"],"working_files/audio_files/"+str(path_i)+".wav")
            print(shared_dict["saveduration"]," Saving audio ", path_i)

            #stop listening
            t_thinking_start=time.time()


            if being_smart:
                #transcribe
                t_start=time.time()
                if chat_log!="":
                    shared_dict["eye"]=2
                    audio_file= open("working_files/audio_files/"+str(path_i)+".wav", "rb")
                    try:
                        transcript = openai.Audio.transcribe("whisper-1", audio_file)
                    except:
                        transcript="You couldn't hear me"
                        break
                else:
                    shared_dict["eye"]=2
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
                    chat_log="Imagine you are a witty and humorous character inside a spooky skeleton head, greeting trick-or-treaters on Halloween night at the front door of a house. This is what you see in front of you: "+shared_dict["visual_description"]+". Respond with a single, concise message starting with 'You:'. No emojis. Provide a witty, humorous introduction to the user, be sure to make it very short and to the point with very little fluff ie: not starting with well, well, well. Your answer will be used as <message>."
                    #the_prompt="Imagine you are a witty and humorous character inside a spooky skeleton head, greeting trick-or-treaters on Halloween night at the front door of a house. This is what you see in front of you: "+shared_dict["visual_description"]+". Respond with a single, concise message starting with 'You:'. Provide a witty, humorous introduction to the user, be sure to make it short and to the point with very little fluff ie: not starting with well, well, well. Do not be inappropriate. Also be sure to keep the conversation up and make it easy for the user to respond. Your answer will be used as <message>."
                    #the_prompt="I want you to role play as a talking skeleton head who is whitty and funny. You are greeting trick-or-treaters on Halloween night at the front door of a house. I want this conversation to be engauging for the trick or treaters so make sure to respond in a way that keeps a path for conversation open. This is what you see in front of you: "+shared_dict["visual_description"]
                    the_prompt="I want you to role play as a talking skeleton head who is whitty and funny. You are greeting trick-or-treaters on Halloween night at the front door of a house. I want this conversation to be engauging for the trick or treaters and for it to be funny. Try to be a little crazy and try not to repeat yourself or say things that are repetitive and don't ramble on for a long time because this is a back and forth conversation. This is what you see in front of you: "+shared_dict["visual_description"]
                    chat_log=the_prompt
                    the_prompt+="\nYou: <message>\n\nWhat should you say for the <message> to the user?\nAnswer: "
                else:
                    chat_log=chat_log+"\nUser: "+str(transcript["text"])
                    the_prompt=chat_log+"\nYou: <message>\n\nWhat should you say for the <message> to the user?\nAnswer: "
                
                #open a yaml file to write the prompt to the end
                with open("working_files/prompt.yaml","a") as f:
                    #get rid of emojis from prompt
                    write_prompt=the_prompt.encode('ascii', 'ignore').decode('ascii')
                    f.write(write_prompt+"\n\n\n\n\n")

                print("the prompt test: ",the_prompt)
                #the_prompt="I've seen more candy in a single bag! But, hey"
                ts=time.time()
                # generate response
                result_l=[]
                shared_dict["file queue"]=[]
                shared_dict["time_started"]=time.time()
                result_s=""
                shared_dict["llm_finished"]=False
                shared_dict["tts_finished"]=False
                shared_dict["text"]=""

                timeout_seconds = 2
                response=None
                if openai_bool:
                    while response==None:
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(get_openai_response, the_prompt)
                            try:
                                response = future.result(timeout=timeout_seconds)
                            except TimeoutError:
                                print(f"Timeout after {timeout_seconds} seconds")
                                response = None

                    print("MADE RESPONSE")
                    collected_messages=[]
                    for chunk in response:
                        try:
                            shared_dict["last_token_size"]=len(chunk['choices'][0]['delta']["content"])
                            print(chunk['choices'][0]['delta']["content"])
                            collected_messages.append(chunk['choices'][0]['delta']["content"])
                            shared_dict["text"]="".join(collected_messages)
                        except Exception as e:
                            print(e)
                    result_s="".join(collected_messages)


                if together_bool:
                    # Together llamma response:
                    for token in together.Complete.create_streaming(prompt=the_prompt,model=together_model_name,max_tokens=500,temperature=0.9):
                        print(token, end="", flush=True)
                        shared_dict["last_token_size"]=len(token)
                        result_l.append(token)
                        result_s="".join(result_l).strip()
                        shared_dict["text"]=result_s

                print("LLM RESULT: ",result_s)

                shared_dict["llm_finished"]=True
                chat_log=chat_log+"\nYou: "+result_s.split("You: ")[-1].split("</s>")[0]
                with open("working_files/prompt.yaml","a") as f:
                    #get rid of emojis from prompt
                    write_prompt=chat_log.encode('ascii', 'ignore').decode('ascii')
                    f.write(write_prompt+"\n\n\n\n\n")
                print("\n")
                print("TIME: ",time.time()-ts)


                print("LLM Time: ",time.time()-ts)

    print("Done")