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

### Real-World Scene

The project is structured into two distinct sections: the Real-World Scene and the Virtual World Game. In the Real-World Scene, a wooden frame is constructed to create a stand that forms an interactive playground. Within this space, participants can interact with lenses that are placed on an acrylic board. This hands-on engagement is further enhanced by the option to place a white acrylic board on top. By pressing a key on a controller or computer, users can initiate complex calculations that occur behind the scenes, seamlessly blending physical interaction with digital technology.   

Simultaneously, a server operates a pipeline that processes and analyzes images captured by an infrared camera positioned underneath. The processed image is then projected onto the white acrylic board. Approximately 30 seconds later, the visual display on the board adjusts, reflecting changes in the game scene, thereby creating a dynamic and immersive gaming experience.

### Python Django Server

Upon receiving a request, the Django server delegates the task to Celery, which then queues it in Redis for execution. The initial step involves saving the raw image captured by the camera. Utilizing computer vision algorithms, the system identifies and locates the target board's area, transforming the identified quadrilateral shape into a square. Following this, the k-Means algorithm clusters the centers of the lenses.

Meanwhile, the stable diffusion neural network is tasked with creating the skybox. This involves a trained LoRA stable diffusion model working in conjunction with the Canny ControlNet to generate a starry skybox, incorporating elements from the captured photo. The resulting image is then processed by the inpaint stable diffusion technique to produce an enlarged image (1024x768 pixels) featuring skybox elements. To enhance the image to a qualified HDR for universe skybox, the ESRGAN model is employed to upscale the image by four times and a mask is added onto the image to ensure seamless integration.

Finally, the k-Means algorithm is once again used, this time to cluster the dominant colors of the generated image. These color data are then transmitted to the Unreal Engine, which performs further calculations to adjust the shader's appearance, thus completing the process.

### Unreal Engine

During the operation, Unreal Engine waits for 30 seconds before polling every second, as the computational process averages 32-40 seconds. 
Once the calculations and image generations are complete, procedural generation commences within Unreal Engine. This phase intricately combines the use of both Blueprint and C++ coding. 
The process involves the creation of game objects, assignment of animations, and setting of shader parameters to achieve specific visual effects. Key elements include a blackhole, which is composed of a center, a ring, and a shader that produces a lens effect. Additionally, the star in the game features a dynamic surface shader and a Niagara particle system, which together create a captivating plasma effect. The planets are also intricately designed, boasting up to 13 surface textures inspired by the solar system, and each planet is accentuated with a unique shader ring, further enhancing the visual diversity and depth of the game environment.