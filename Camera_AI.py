#motion functions
def main_motion_processing(shared_motion):
    import torch
    import cv2
    model = torch.hub.load("ultralytics/yolov5", 'custom', path="working_files/crowdhuman_yolov5m.pt")
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
            acc_threshold=0.6#shared_motion["acc_threshold"]
            ret,frame=cam.read()
            #show frame
            cv2.imshow("frame",frame)
            cv2.imwrite("working_files/shi.jpg",frame)
            cv2.waitKey(1)

            #hide output of model inference
            with torch.no_grad():
                results = model(frame)
                try:
                    #get results and put on cpu
                    result_vals=results.pred[0][0][:].cpu().numpy()
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

                    if time.time()-t_print>1:
                        print("ACC: ",result_vals[4], x_pos, y_pos, "FPS: ",fps, found_times)
                        t_print=time.time()
                #get line number of error
                except KeyboardInterrupt:
                    print("Exiting program")
                    exit()
                except Exception as e:
                    #if keyboard interrupt, exit
                    print(e," Line: ",e.__traceback__.tb_lineno)
                    failed=True
                    found_times=0
                    
                    
            t_start=time.time()
            shared_motion["x_pos"]=x_pos
            shared_motion["y_pos"]=y_pos
            shared_motion["failed"]=failed
        except KeyboardInterrupt:
            print("Exiting program")
            exit()
        except Exception as e:
            print(e)

def websocket_connection(shared_motion):
    import socket
    import time

    hostname=socket.gethostname()
    port=1234

    #create socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)

    data_vals=[0,0]

    t_print=time.time()

    while True:
        data_vals[0]+=1
        data_vals[1]+=1
        try:
            if shared_motion["x_pos"]!=None:
                if shared_motion["failed"]:
                    data=str(shared_motion["x_pos"])+","+str(shared_motion["y_pos"])+",1"
                else:
                    data=str(shared_motion["x_pos"])+","+str(shared_motion["y_pos"])+",0"
            else:
                data="0,0,1"
            s.sendall(data.encode())

            recieved_data=s.recv(1024)
            if (time.time()-t_print)>1:
                print(data)
                print(recieved_data)
                t_print=time.time()
        except Exception as e:
            if e==KeyboardInterrupt:
                print("Exiting program")
                exit()
            print("Timeout")
            while True:
                try:
                    s.close()
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(2)
                    time.sleep(1)
                    s.connect((hostname, port))
                    break
                except Exception as e:
                    #if keyboard interrupt, exit
                    if e==KeyboardInterrupt:
                        print("Exiting program")
                        exit()
                    else:
                        print(e)
                    print("Failed to reconnect")
        time.sleep(0.02)
        

if __name__=="__main__":
    import threading
    import multiprocessing
    import time
    import os
    manger=multiprocessing.Manager()
    shared_motion=manger.dict()
    shared_motion["x_pos"]=None
    shared_motion["y_pos"]=None
    shared_motion["failed"]=False
    shared_motion["acc_threshold"]=0.7
    shared_motion["x_cam_offset"]=0
    shared_motion["y_cam_offset"]=0

    socket_thread=threading.Thread(target=websocket_connection,args=(shared_motion,))
    socket_thread.start()

    motion_thread=threading.Thread(target=main_motion_processing,args=(shared_motion,))
    motion_thread.start()

    while True:
        try:
            print("Motion status: ",motion_thread.is_alive(), ", Socket status: ",socket_thread.is_alive())
            time.sleep(3)
            try:
                with open("working_files/settings.txt","r") as f:
                    data=f.read()
                    data=data.split("\n")
                    shared_motion["acc_threshold"]=float(data[6])
                    shared_motion["x_cam_offset"]=int(data[7])
                    shared_motion["y_cam_offset"]=int(data[8])
            except Exception as e:
                print(e)
                print("Failed to read settings file")
        except KeyboardInterrupt:
            print("Exiting program")
            os._exit(1)


