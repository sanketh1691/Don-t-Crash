
from keras.models import load_model
import numpy as np
import airsim


## SET PATH to the model
MODEL_PATH = 'models/City.h5'
# confirms that model is imported
print('Using model {0} for testing.'.format(MODEL_PATH))

model = load_model(MODEL_PATH)

# connect to AirSim server
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()
print('Connection established!')

car_controls.steering = 0
car_controls.throttle = 0
car_controls.brake = 0

image_buf = np.zeros((1, 59, 255, 3))
state_buf = np.zeros((1,4))

# read the image from AirSim and process it
def get_image():
   image_response = client.simGetImages([airsim.ImageRequest("0",airsim.ImageType.Scene,False,False)])[0]
   #image_response = client.simGetImages([client.simGetImage("0",airsim.ImageType.Scene,False)])[0]
   #image_response = client.simGetImages([client.simGetImage(0, 0, False, False)])[0]
   image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
   image_rgba = image1d.reshape(image_response.height, image_response.width, 3)

   return image_rgba[76:135, 0:255, 0:3].astype(float)

# Control block to run the car
try:
    while (True):
       car_state = client.getCarState()

       car_controls.throttle = max(min(7,(car_state.speed-7)/-3),0)

       image_buf[0] = get_image()
       state_buf[0] = np.array([car_controls.steering, car_controls.throttle, car_controls.brake, car_state.speed])
       model_output = model.predict([image_buf, state_buf])
       car_controls.steering = round(0.5 * float(model_output[0][0]) * 0.1, 2)

       print('Sending steering = {0}, throttle = {1}'.format(car_controls.steering, car_controls.throttle))

       client.setCarControls(car_controls)

except:
    print('client broken')
                
#restore to original state
#client.reset()

client.enableApiControl(False)
