#camera thread
def camera(shared_dict):
    # Open camera
    cam=cv2.VideoCapture(1, cv2.CAP_MSMF) 

    t_start=time.time()
    t_print=time.time()

    while True:
        try:
            # Set start time for loop speed
            t_start=time.time()
            
            # Get frame
            ret,frame=cam.read()

            # Write frame to file and notify that frame is ready to read
            if ret:
                cv2.imwrite("working_files/shi.jpg",frame)
                shared_dict["im_done"]=True
                time.sleep(0.020)
                shared_dict["im_done"]=False
                time.sleep(0.008)
                
                # Print fps every second if ret
                if time.time()-t_print>1:
                    print("CAM fps: ",1/(time.time()-t_start))
                    t_print=time.time()

        # Print any errors
        except Exception as e:
            print("CAM error: ",e)
            time.sleep(0.028)


def arduino_controls(shared_dict):
    # Setup serial connection to arduino
    import serial
    ser = serial.Serial('COM3', 115200)
    ser.timeout=5
    time.sleep(1.5)

    # Arduino's response
    recv_string=""

    # 0 is closed, 1 is power open
    mouth=0

    # 0 is off, 1 is red, 2 is blue
    eye=2

    failed_prev=False

    t_loop=time.time()
    t_print=time.time()

    # Initialize position and velocity values
    x_micros=shared_dict["x_micros"]
    prev_x_micros=x_micros
    x_micro_vel=0
    prev_x_micro_vel=0

    y_micros=shared_dict["y_micros"]
    prev_y_micros=y_micros
    y_micro_vel=0
    prev_y_micro_vel=0

    while True:
        # Get the values to send from the shared
        mouth=shared_dict["mouth"]
        eye=shared_dict["eye"]
        prev_x_micros=x_micros
        prev_y_micros=y_micros

        # calculate how to move head based on the pixel values of the prediction
        x_micros, y_micros=calculate_writes(x_micros,y_micros,prev_x_micro_vel,prev_y_micro_vel,x_micro_vel,y_micro_vel,shared_dict["x_pos"],shared_dict["y_pos"],shared_dict["failed"],failed_prev,(time.time()-t_loop),shared_dict["exp"],shared_dict["em1"],shared_dict["em2"])

        # Calculate the velocity of the head
        prev_x_micro_vel=x_micro_vel
        prev_y_micro_vel=y_micro_vel
        y_micro_vel=(y_micros-prev_y_micros)/(time.time()-t_loop)
        x_micro_vel=(x_micros-prev_x_micros)/(time.time()-t_loop)

        # update the write values
        shared_dict["x_micros"]=int(x_micros)
        shared_dict["y_micros"]=int(y_micros)
        t_loop=time.time()
        failed_prev=shared_dict["failed"]

        # Send the values to the arduino
        command = f"{str(int(x_micros))},{str(int(y_micros))},{str(mouth)},{str(eye)}\n"
        ser.write(command.encode())
        time.sleep(0.01)

        # Read the serial
        while ser.in_waiting>0:
            try:
                recv_string=ser.readline().decode('utf-8').rstrip()
            except:
                print("Error reading serial")

        # Print what is sent and recieved every second
        if (time.time()-t_print)>1:
            print("Recieving: ",recv_string," and Sending: ",command)
            t_print=time.time()


def calculate_writes(x_write_value,y_write_value,prev_x_micro_vel,prev_y_micro_vel,x_micro_vel,y_micro_vel,x_pos,y_pos,failed,failed_prev,loop_t,exp,em1,em2):
    if (x_pos!=None) and (y_pos!=None):
        x_dir_val=0
        y_dir_val=0

        # Get x dir
        if x_pos>0:
            x_dir_val=1
        elif x_pos<0:
            x_dir_val=-1

        # Get y dir
        if y_pos>0:
            y_dir_val=1
        elif y_pos<0:
            y_dir_val=-1



        if not failed_prev or not failed:
            x_delta=((abs(x_pos/320)**exp)*x_dir_val*loop_t*em1)
            x_write_value=(x_write_value-x_delta)

            y_delta=((abs(y_pos/240)**exp)*y_dir_val*loop_t*em2)
            y_write_value=(y_write_value-y_delta)
            
            # Set x bounds
            if x_write_value<710:
                x_write_value=710
            elif x_write_value>2290:
                x_write_value=2290

            # Set y bounds
            if y_write_value<710:
                y_write_value=710
            elif y_write_value>2290:
                y_write_value=2290

            # Return the values to write
            return(int(x_write_value),int(y_write_value))
    return(int(x_write_value),int(y_write_value))

#motion functions
def object_detection(shared_dict):
    # Import libraries
    import torch

    # Load model
    model = torch.hub.load("ultralytics/yolov5", 'custom', path="working_files/crowdhuman_yolov5m.pt")
    
    x_pos=0
    y_pos=0

    failed=False
    found_times=0

    acc_threshold=0.6
    t_print=time.time()
    t_error=time.time()
    t_start=time.time()


    
    while True:
        t_start=time.time()
        
        # Get frame when it is ready
        while not shared_dict["im_done"]:
            time.sleep(0.003)
        frame=cv2.imread("working_files/shi.jpg")

        # Check size
        frame_file_size=os.path.getsize("working_files/shi.jpg")
        if (frame_file_size>20000):
            try:
                with torch.no_grad():
                    # Get results
                    results = model(frame)

                    try:
                        # Put results on cpu
                        result_vals=results.pred[0][0][:].cpu().numpy()

                        # If confidence is too low, don't move
                        if result_vals[4]<acc_threshold:
                            raise Exception("Confidence too low")

                        # Calculate x and y position difference from center of frame
                        x_pos=int((result_vals[0]+result_vals[2])/2)-(320+shared_dict["x_cam_offset"])
                        y_pos=int((result_vals[1]+result_vals[3])/2)-(240+shared_dict["y_cam_offset"])
                        shared_dict["x_pos"]=x_pos
                        shared_dict["y_pos"]=y_pos
                        
                        # Calculate fps
                        fps=1/(time.time()-t_start)
                        fps=round(fps,2)
                        
                        # Debounce failed=False state
                        found_times+=1
                        if found_times>=2:
                            failed=False

                        # Print detection information every second
                        if time.time()-t_print>1:
                            print("ACC: ",result_vals[4], x_pos, y_pos, "FPS: ",fps, found_times)
                            t_print=time.time()
                    
                    # Exit when keyboard interrupt
                    except KeyboardInterrupt:
                        print("Exiting program")
                        exit()
                    # Print the every second that it has failed to detect someone
                    except Exception as e:
                        if (time.time()-t_error)>1:
                            print(e," Line: ",e.__traceback__.tb_lineno)
                            t_error=time.time()
                        failed=True
                        found_times=0
                        
                # Update detection status   
                shared_dict["failed"]=failed

                #show frame with a red or green circle in the top left indicating detection status
                if failed:
                    cv2.rectangle(frame,(0,0),(10,10),(0,0,255),-1)
                else:
                    cv2.rectangle(frame,(0,0),(10,10),(0,255,0),-1)
                cv2.imshow("frame",frame)
                cv2.waitKey(1)

            # Exit when keyboard interrupt
            except KeyboardInterrupt:
                print("Exiting program")
                exit()
            
            # If something goes wrong when displaying frame, print error
            except Exception as e:
                print(e)



def websocket_connection(shared_dict):
    import socket

    #create socket
    hostname=socket.gethostname()
    port=1234
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)

    t_print=time.time()
    sending_data="0,0,1"
    data=""

    while True:
        try:
            # Setup data to send
            if shared_dict["x_pos"]!=None:
                if shared_dict["failed"]:
                    sending_data=str(shared_dict["x_micros"])+","+str(shared_dict["y_micros"])+",1"
                else:
                    sending_data=str(shared_dict["x_micros"])+","+str(shared_dict["y_micros"])+",0"
            else:
                sending_data="0,0,1"
            
            # Send data
            s.sendall(sending_data.encode())

            # Recieve data
            recieved_data=s.recv(1024)
            data=recieved_data.decode("utf-8")
            if data!="":
                shared_dict["eye"]=data.split(",")[1]
                shared_dict["mouth"]=data.split(",")[0]

            # Print data sent and recieved every second
            if (time.time()-t_print)>1:
                print(data)
                print(recieved_data)
                t_print=time.time()

        # Exit if keyboard interrupt and print error
        except Exception as e:
            if e==KeyboardInterrupt:
                print("Exiting program")
                exit()
            print(e)

            # Continuously try to reconnect
            while True:
                try:
                    s.close()
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(2)
                    time.sleep(1)
                    s.connect((hostname, port))
                    break
                except Exception as e:
                    #if keyboard interrupt, exit otherwise print error
                    if e==KeyboardInterrupt:
                        print("Exiting program")
                        exit()
                    else:
                        print(e)
                    print("Failed to reconnect")
        time.sleep(0.02)





        
        

if __name__=="__main__":
    import time
    import threading
    import multiprocessing
    import time
    import os
    import cv2
    manger=multiprocessing.Manager()
    shared_dict=manger.dict()
    shared_dict["x_pos"]=None
    shared_dict["y_pos"]=None
    shared_dict["failed"]=False


    # Initialize head eye color and mouth position
    shared_dict["eye"]=1
    shared_dict["mouth"]=0

    # Initialize the head position write value
    shared_dict["x_micros"]=1500
    shared_dict["y_micros"]=1900

    # Set the camera offset values
    shared_dict["x_cam_offset"]=0
    shared_dict["y_cam_offset"]=0
    shared_dict["im_done"]=False

    # Set movement characteristics
    shared_dict["acc_threshold"]=0.7
    shared_dict["exp"]=1.1
    shared_dict["em1"]=0.5
    shared_dict["em2"]=0.5

    # Start Thread to connect to the websocket
    socket_thread=threading.Thread(target=websocket_connection,args=(shared_dict,))
    socket_thread.start()

    detection_thread=threading.Thread(target=object_detection,args=(shared_dict,))
    detection_thread.start()

    camera_thread=threading.Thread(target=camera,args=(shared_dict,))
    camera_thread.start()

    arduino_thread=threading.Thread(target=arduino_controls,args=(shared_dict,))
    arduino_thread.start()

    while True:
        try:
            print("Detection status: ",detection_thread.is_alive(), ", Socket status: ",socket_thread.is_alive(), ", Camera status: ",camera_thread.is_alive(), ", Arduino status: ",arduino_thread.is_alive())
            time.sleep(3)
            try:
                with open("working_files/settings.txt","r") as f:
                    data=f.read()
                    data=data.split("\n")
                    shared_dict["acc_threshold"]=float(data[6])
                    shared_dict["x_cam_offset"]=int(data[7])
                    shared_dict["y_cam_offset"]=int(data[8])
                    shared_dict["exp"]=float(data[5])
                    shared_dict["em1"]=float(data[9])
                    shared_dict["em2"]=float(data[10])
            except Exception as e:
                print(e)
                print("Failed to read settings file")
        except KeyboardInterrupt:
            print("Exiting program")
            os._exit(1)
