import cv2
import numpy as np
train_data = list(np.load('Car-Dataset/Final.npy' , allow_pickle = True))
print(len(train_data))
for image,out in train_data:
    image = cv2.resize(image , (150,150))
    cv2.imshow('img',image)
    print(out)
    #print(choice)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()