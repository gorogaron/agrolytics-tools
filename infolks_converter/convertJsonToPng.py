import json
import numpy as np
import cv2

imgPath = './images/'
jsonPath = './json/'
contourPath = './contour/'

from os import listdir
from os.path import isfile, join
imgFilenames = [f for f in listdir(imgPath) if isfile(join(imgPath, f))]

for filename in imgFilenames:
    print('Processing image:' + filename)
    image = cv2.imread(imgPath+filename)
    height, width = image.shape[:2]


    with open(jsonPath + filename[0:-4] + '.json') as json_file:
        data = json.load(json_file)
        
        for objectElement in np.asarray(data['output']['objects']):
            contours = np.round(np.asarray(objectElement['points']['exterior'], dtype=np.int32))
            cv2.drawContours(image, [contours], 0, (0,0,255), 3)

            #x,y,w,h = cv2.boundingRect(contours)
            #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imwrite('./contours/' + filename ,image)