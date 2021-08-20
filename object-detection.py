import numpy as np
import cv2
windowName = "Output"
cv2.namedWindow(windowName)
thres = 0.5 # Threshold to detect object
nms_threshold = 0.2 #(0.1 to 1) 1 means no suppress , 0.1 means high suppress
cap = cv2.VideoCapture('highway-traffic.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH,800) #width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,500) #height
cap.set(cv2.CAP_PROP_BRIGHTNESS,150) #brightness

classNames = []
with open('coco.names','r') as f:
    classNames = f.read().splitlines()

font = cv2.FONT_HERSHEY_PLAIN
Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

weightsPath = "frozen_inference_graph.pb"
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    #print(type(confs[0]))
    #print(confs)

    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)


    if len(classIds) != 0:
        for i in indices:
            i = i[0]
            box = bbox[i]

            confidence = str(round(confs[i], 2))
            color = Colors[classIds[i][0]-1]

            x, y, w, h = box[0], box[1], box[2], box[3]
            """
            #Piksellerden elde edilen RGB değerlerinin ortalamasının alınması.
            r = 0
            g = 0
            b = 0
            r_l = []
            g_l = []
            b_l = []

            for k in range(w+1):

                r_l.append(img[x+k, y][0])
                g_l.append(img[x+k, y][1])
                b_l.append(img[x+k, y][2])

                for j in range(h+1):

                    r_l.append(img[x, y+j][0])
                    g_l.append(img[x, y+j][1])
                    b_l.append(img[x, y+j][2])
                    j += 1
                k += 1

            r_sum = sum(r_l)
            g_sum = sum(g_l)
            b_sum = sum(b_l)
            r_ort = int(r_sum / len(r_l))
            g_ort = int(g_sum / len(g_l))
            b_ort = int(b_sum / len(b_l))
            print(r_ort, g_ort, b_ort)
            """
            cv2.rectangle(img, (x,y), (x+w,y+h), color, thickness=2)
            cv2.putText(img, "color",(x-10,y-10),font,1,color,3)

    cv2.imshow(windowName,img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



