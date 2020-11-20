import airsim
import os
import cv2
import numpy as np
import pickle

def yolo(img, net, confidence_threshold, threshold):
    print("In YOLO method")
    image = cv2.imread(img)
    print("shape of image is:",image.shape)
    #since image is 4 channel, we consider height and width
    (H, W) = image.shape[:2]
    layerNames = net.getLayerNames()
    layerNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    ##construct a blob from the input image
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0,(416,416),swapRB=True,crop=False)
    net.setInput(blob)
    netOutputs = net.forward(layerNames)
    # Constructing bounding box
    boxes = []
    confidences = []
    classIDs = []

    for output in netOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if(confidence>confidence_threshold):
                box = detection[0:4] * np.array([W, H, W, H])
                (Xcenter, Ycenter, width, height) = box.astype("int")
                #obtain coordinates for top left corner
                tl_x = int(Xcenter - (width/2))
                tl_y = int(Ycenter - (height/2))
                boxes.append([tl_x,tl_y,int(width),int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    #applying non maxima suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold,
                            threshold)
    close = dict()
    if(len(idxs)>0):
        for i in idxs.flatten():
            (x,y) = (boxes[i][0], boxes[i][1])
            (w,h) = (boxes[i][2], boxes[i][3])
            cv2.rectangle(image, (x,y), (x+w,y+h), (100,220,210), 2)
            text = LABELS[classIDs[i]]
            cv2.putText(image, text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.25, (180,100,70), 1)
            cl = get_closeness(image,x,y,w,h)
            close.update({(LABELS[classIDs[i]],confidences[i]):cl})
    #df = pd.DataFrame(columns=['Object','Confidence','Closeness'])
    close_metric = sorted(close.items(), key=lambda x: x[1], reverse=True)
    return image, close_metric
    
    
client = airsim.CarClient()
client.confirmConnection()

#Loading YOLO
print("loading YOLO")
LABELS = open("yolo/coco.names").read().strip().split("\n")
path_weights = "yolo/yolov3.weights"
path_config = "yolo/yolov3.cfg"
np.random.seed(42)
#load the trained YOLO net using dnn library in cv2
net = cv2.dnn.readNetFromDarknet(path_config, path_weights)

print("applying YOLO")
filepath = "yolo/image_outputs/"
responses2 = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
cam_image = responses2[0]
# get numpy array
img1d = np.fromstring(cam_image.image_data_uint8, dtype=np.uint8) 

# reshape array to 4 channel image array H X W X 4
img_rgb = img1d.reshape(cam_image.height, cam_image.width, 3)

# original image is fliped vertically
#img_rgb = np.flipud(img_rgb)
filename = str(img_idx)+'.png'
airsim.write_png(os.path.normpath(filepath + filename), img_rgb) 
#cam_image = client.simGetImage("0", airsim.ImageType.Scene)

image_name = os.path.normpath(filepath + filename)
#yolores = yolo(cam_image.image_data_uint8,net,0.5,0.5)
yolores, closeness = yolo(image_name,net,0.5,0.5)
print(closeness)
#yolores.save(filepath+str(img_idx)+"yolo.png")
cv2.imwrite(filepath+str(img_idx)+"yolo.png",yolores)
print("saved image")
img_idx+=1


