import cv2
import matplotlib.pyplot as plt

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'  # importing

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = [0]

file_name = 'coco.names'

with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
    classLabels.append(fpt.read())

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

img = cv2.imread('golf.jpg')
plt.imshow(img)
cv2.imshow("Image", img)

classIndex, confidece, bbox = model.detect(img, confThreshold=0.5)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for classInd, conf, boxes in zip(classIndex.flatten(), confidece.flatten(), bbox):
    cv2.rectangle(img, boxes, (255, 0, 0), 2)
    cv2.putText(img, classLabels[classInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale,
                color=(0, 255, 0), thickness=3)


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()