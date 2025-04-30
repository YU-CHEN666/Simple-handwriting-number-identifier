import cv2
import numpy
from operator import itemgetter
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from keras.models import load_model
from keras import Layer
from keras.ops import argmax,convert_to_numpy

img_write = cv2.imread("write_area.bmp")
img_option = cv2.imread("use_information.bmp") 
img_waring = cv2.imread("waring.bmp")
result_img = cv2.imread("result.bmp")
position = []
def mouse_draw(event,x,y,flags,param):
    global position
    if event == cv2.EVENT_LBUTTONUP:
        position=[]
    if flags == cv2.EVENT_FLAG_LBUTTON:
        position.append((x,y))
        position_array = numpy.array(position,dtype="int32")
        cv2.polylines(img_write,[position_array],False,(0,0,0),30,cv2.LINE_AA)
        cv2.imshow("Main",img_write)
        

def get_ROI_area(contours,picture):
    number_list = []
    check_intreval = []
    for i in range(len(contours)):
        x,y,w,h=cv2.boundingRect(contours[i])
        check_intreval.append((x,y,w,h))
        check_intreval = sorted(check_intreval,key=itemgetter(0))
    if check_intreval[1][0] < check_intreval[0][0]+check_intreval[0][2]:
        cv2.namedWindow("Waring")
        cv2.imshow("Waring",img_waring)
        Main_position = cv2.getWindowImageRect("Main")
        cv2.moveWindow("Waring",Main_position[0]-77,Main_position[1]+18)
        cv2.waitKey(0)
        state = cv2.getWindowProperty("Waring",cv2.WND_PROP_VISIBLE)
        if state == 0.0:
            pass
        else:
            cv2.destroyWindow("Waring")
    else:
        for x,y,w,h in check_intreval:
            number = picture[y:y+h,x:x+w]
            background = numpy.zeros((420,420),"uint8")
            left_x = 210-int(w/2)
            left_y = 210-int(h/2)
            background[left_y:left_y+h,left_x:left_x+w] = number
            number_list.append(background)
        return number_list


class dim_splitter(Layer):  
    def __init__(self,**kawgs):
        super().__init__(**kawgs)
    def call(self,inputs):
        inputs_split_1 = inputs[:,:,:,0:100:2]
        inputs_split_2 = inputs[:,:,:,1:100:2]
        return inputs_split_1,inputs_split_2

        
def preprocessing(picture):
    picture_list = []
    for img in picture:
        img = img.astype("float32")
        img_resize = cv2.resize(img,(28,28),0,0,cv2.INTER_AREA)
        _,img_resize = cv2.threshold(img_resize,30,255,cv2.THRESH_BINARY)
        img_resize = img_resize/255
        picture_list.append(numpy.expand_dims(img_resize, axis=2))
    return numpy.array(picture_list)




def detection(picture):
    model = load_model("mymodel.keras",custom_objects={"dim_splitter":dim_splitter})
    result = argmax(model(picture,training=False),axis=1)
    return result
    


def display_result(result):
    text = ""
    result_img_display = result_img.copy()
    for i in convert_to_numpy(result):
        text+=str(i)
    cv2.putText(result_img_display,text,(350,109),cv2.FONT_HERSHEY_TRIPLEX,3,(0,0,0),2,cv2.LINE_AA)
    cv2.namedWindow("Result")
    cv2.imshow("Result",result_img_display) 
    Main_position = cv2.getWindowImageRect("Main")
    cv2.moveWindow("Result",Main_position[0]-48,Main_position[1]+18)
    cv2.waitKey(0)
    state = cv2.getWindowProperty("Result",cv2.WND_PROP_VISIBLE)
    if state == 0.0:
        pass
    else:
        cv2.destroyWindow("Result") 

first_start = True
cv2.namedWindow("Information")
cv2.namedWindow("Main")
cv2.setMouseCallback('Main',mouse_draw)
while(True):   
    cv2.imshow("Main",img_write)
    cv2.imshow("Information",img_option)
    if first_start:
        Main_position = cv2.getWindowImageRect("Main")
        cv2.moveWindow("Information",Main_position[0]+480,Main_position[1]-5)
    k = cv2.waitKey(0)
    if k==67 or k==99:
        img_write[:]=255
    elif k==-1 or k==76 or k==108:
        break
    elif k==78 or k==110:
        writearea = cv2.cvtColor(img_write,cv2.COLOR_BGR2GRAY)
        _,reverse = cv2.threshold(writearea,100,255,cv2.THRESH_BINARY_INV)
        contours,_ = cv2.findContours(reverse,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 1:
            model_input = preprocessing([reverse])
            result = detection(model_input)    
            display_result(result)
        elif len(contours) == 2:
            model_inputs = get_ROI_area(contours,reverse)
            if type(model_inputs) == type(None):
                pass
            else:
                model_inputs = preprocessing(model_inputs)
                result = detection(model_inputs)
                display_result(result)
    first_start = False  
cv2.destroyAllWindows()
