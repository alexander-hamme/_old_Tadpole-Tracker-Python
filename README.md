# computer-vision-xenopus-tadpole-tracker
Undergraduate Thesis project that tracks and records movement data of many *Xenopus laevis* tadpoles in real time. 

Project is in collaboration with the neuroscience department at Bard College.

Program will be implemented in both Java and Python, to increase portability and allow wider access for biology researchers.

-----

There are two major components of this tracker program: **Detection** and **Tracking**.
  * detection is the process of finding regions of interest (ROI) in each frame (image) from the video input stream
  * tracking is the process of connecting where each animal was in previous frames to its new position in sequential frames, 
    i.e. connecting ROIs to the corresponding tadpoles. This becomes complicated when tracking multiple animals, because of the potential for collisions and collusions. Therefore, trajectory prediction algorithms need to be implemented.

Approaches:

  * Detection: Convolutional neural networks will be the building block for the tadpole detection system. Currently, I am embedding an implementation of a [YOLO](https://pjreddie.com/darknet/yolo/) model trained on my dataset.

  * Tracking (specifically, trajectory prediction): I will train a Long Short-Term Memory (LSTM) recurrent neural network on recorded tadpole movement data. 

-----

Current Progress:

My most recent results were achieving 7fps with accurate detection on tadpoles from a video stream, running on a GTX 1070 GPU. The next step is to try to optimize this process to hit a higher fps rate, and then hook up the tracking system to the tadpole detection bounding boxes.


###### Proof of Concept gif:

![Uh oh, it appears the gif didn't load. Please find the gif in the images folder of this repositiory.](/images/proof_of_concept.gif?raw=true "Proof of Concept")

<br>

###### Detection results of custom trained Yolo

![Uh oh, it appears the gif didn't load. Please find it as "yolo_detections.jpg" in the images folder of this repositiory.](/images/yolo_detections.jpg?raw=true "Detection Results")

###### More files will be added soon.
