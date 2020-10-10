import setup_path
import airsim
import cv2
import numpy as np
import os
import time
import tempfile
from PIL import Image


def yolo(img, net, confidence_threshold, threshold):
    print("In YOLO method")
    image = cv2.imread(img)
    #since image is 4 channel, we consider height and width
    (H, W) = image.shape[:2]
    layerNames = net.getLayerNames()
    layerNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    ##construct a blob from the input image
    blob = cv2.dnn.blobFromImage(image,1/255.0,(416,46),swapRB=True,crop=False)
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
                tl_x = int(center - (width/2))
                tl_y = int(center - (height/2))
                boxes.append([tl_x,tl_y,int(width),int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    #applying non maxima suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold,
                            threshold)
    if(len(idxs)>0):
        for i in idxs.flatten():
            (x,y) = (boxes[i][0], boxes[i][1])
            (w,h) = (boxes[i][2], boxes[i][3])
            cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
            text = LABELS[classIDs[i]]
            cv2.putText(image, text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, 5, 2)
    image.save("yolo/image-outputs/image"+str(i)+".png")


# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
print("API Control enabled: %s" % client.isApiControlEnabled())
car_controls = airsim.CarControls()

#Loading YOLO

LABELS = open("yolo/coco.names").read().strip().split("\n")
path_weights = "yolo/yolov3.weights"
path_config = "yolo/yolov3.cfg"
#load the trained YOLO net using dnn library in cv2
net = cv2.dnn.readNetFromDarknet(path_config, path_weights)

tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_car")
print ("Saving images to %s" % tmp_dir)
try:
    os.makedirs(tmp_dir)
except OSError:
    if not os.path.isdir(tmp_dir):
        raise

for idx in range(3):
    # get state of the car
    car_state = client.getCarState()
    print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))

    # go forward
    car_controls.throttle = 0.5
    car_controls.steering = 0
    client.setCarControls(car_controls)
    print("Go Forward")
    time.sleep(3)   # let car drive a bit

    # Go forward + steer right
    car_controls.throttle = 0.5
    car_controls.steering = 1
    client.setCarControls(car_controls)
    print("Go Forward, steer right")
    time.sleep(3)   # let car drive a bit

    # go reverse
    car_controls.throttle = -0.5
    car_controls.is_manual_gear = True
    car_controls.manual_gear = -1
    car_controls.steering = 0
    client.setCarControls(car_controls)
    print("Go reverse, steer right")
    time.sleep(3)   # let car drive a bit
    car_controls.is_manual_gear = False # change back gear to auto
    car_controls.manual_gear = 0

    # apply brakes
    car_controls.brake = 1
    client.setCarControls(car_controls)
    print("Apply brakes")
    time.sleep(3)   # let car drive a bit
    car_controls.brake = 0 #remove brake

    # get camera images from the car
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthVis),  #depth visualization image
        airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True), #depth in perspective projection
        airsim.ImageRequest("1", airsim.ImageType.Scene), #scene vision image in png format
        airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])  #scene vision image in uncompressed RGB array
    print('Retrieved images: %d' % len(responses))

    for response_idx, response in enumerate(responses):
        filename = os.path.join(tmp_dir, f"{idx}_{response.image_type}_{response_idx}")

        if response.pixels_as_float:
            print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
            airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
        elif response.compress: #png format
            print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
            #applying YOLO to the image
            yolo(response.image_data_uint8,net,0.5,0.5)
            airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
        else: #uncompressed array
            print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
            img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 3 channel image array H X W X 3
            cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png

#restore to original state
client.reset()

client.enableApiControl(False)
