import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import urllib.request


def display_lines(image,lines):
    lst = []
    
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            lst.append([x1,y1,x2,y2])
        #for i in range(x1,x2,1):  
            cv2.line(image,(x1,y1),(x2,y2),(0,255,0),10)
        
        lst = np.array(lst)
        
        try:
            x1_1,y1_1 , x2_1,y2_1 , x1_2,y1_2 , x2_2,y2_2 = np.array(lst).flatten()
            x_axis = int((x2_1+x2_2)/2)
            y_axis = int(y1_1/2)
            cv2.circle(image , (x_axis  ,y_axis ) , 10 ,(0,255,0), 1 )
        except:
            pass
    
    return image


def make_coordinates(img ,line_parameters):
    slope,intercept = line_parameters #we are getting slope and intercept from  each left and right line
    y1 = img.shape[0]
    y2 = int(y1/(1.7))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])




def average_slope_intercept(img , lines):
    left_fit = []
    right_fit = []
    for line in lines:
        
        x1,y1,x2,y2 = line.reshape(4,)
        parameters = np.polyfit((x1,x2),(y1,y2),1) #slope,intercept
        #print(parameters)
        slope = parameters[0]
        intercept = parameters[1]
        if slope<0:
            #print('go left')
            left_fit.append([slope,intercept])
        elif slope>0:
            #print('go right')
            right_fit.append([slope,intercept])

    left_fit_average = np.average(left_fit,axis = 0)
    right_fit_average = np.average(right_fit,axis = 0)
    left_line = make_coordinates(img,left_fit_average)
    right_line = make_coordinates(img,right_fit_average)


    return np.array([left_line,right_line])

def canny_(img):
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5),0)
    canny = cv2.Canny(blur,50,150)
    return canny


def region_of_interest(img):
    height = img.shape[0]
    #triangle = np.array([[(200,height),(1100,height),(550,250)]])
    triangle = np.array([[(0,height),(1085,height),(789,402),(430,402)]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,triangle,255)
    cropped_image = cv2.bitwise_and(img, mask)
    return cropped_image
def main():
    cap = cv2.VideoCapture('test4.mp4')
    while True:

       # image = cv2.imread('test.jpg')
       try:
            ret,image = cap.read()
            #cv2.imwrite('1.jpg',image)
            lane_image = np.copy(image)
            canny = canny_(lane_image)
            cropped_image = region_of_interest(canny)
            #cv2.imshow('-',cropped_image)
            lines = cv2.HoughLinesP(cropped_image,5, np.pi/180,50,np.array([]), minLineLength = 50 , maxLineGap=1) #5 pixel rho 1 degree and 50 is number of bin votes
            averaged_lines = average_slope_intercept(lane_image,lines)
            #print(averaged_lines)
            line_image = display_lines(lane_image,averaged_lines)

            cv2.imshow('-',line_image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
                cap.release()
       except:
            pass
    cv2.destroyAllWindows()
main()
