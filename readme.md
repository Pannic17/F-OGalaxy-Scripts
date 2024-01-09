# O Galaxy
Jiangyun Pan, 22044483
### Python Script for the Final Project
## Environment
- Python 3.10
- Cuda 11.8
- Pytorch 2.0.1
- diffuser
- numpy
- opencv
- matplotlib
## Overview
The project is structured into two main components: the Real-World Scene and the Virtual World Game. The Real-World Scene features a wooden frame that forms a stand, creating a playground-like environment. Within this space, participants can interact with lenses that are placed on an acrylic board. This hands-on engagement is further enhanced by the option to place a white acrylic board on top. By pressing a key on a controller or computer, users can initiate complex calculations that occur behind the scenes, seamlessly blending physical interaction with digital technology. This integration of physical interaction and digital computation creates a unique and immersive experience.  

The server is now running the pipeline to process and analyze the image taken from infrared camera underneath. Meanwhile, the generated image will project onto the White acrylic board. Approximately 30 seconds later, the game scene will change accordingly.  
On the Django server, once it received the request and celery sent the task to Redis waiting for start. The scripts then save the raw image from camera. Using computer vision algorithms, the area of target board can be located and transformed quadrilateral into square. Then using k-Means algorithms, the center of lenses can be clustered.   


