# computer-vision-xenopus-tadpole-tracker
Research project with OpenCV (Python / C++) and Tensorflow that tracks and records movement data of many *Xenopus laevis* tadpoles in real time. 

Project is in collaboration with the neuroscience department at Bard College.

-----

There are two major components of this tracker program: **Detection** and **Tracking**.
  * detection is the process of finding regions of interest (ROI) in each frame (image) from the video input stream
  * tracking is the process of connecting where each animal was in previous frames to its new position in sequential frames, 
    i.e. connecting ROIs to the corresponding tadpoles. This becomes complicated when tracking multiple animals, because of the potential for collisions and collusions. Therefore, trajectory prediction algorithms need to be implemented.

Approaches:

  * Detection: I will use a convolutional neural network. This program will implement the Faster R-CNN algorithm, retrained for detection of Xenopus tadpoles. 

  * Tracking (specifically, trajectory prediction): I will train an LSTM (Long Short-Term Memory) recurrent neural network on recorded tadpole movement data. 


###### Proof of Concept gif:

![Uh oh, it appears the gif didn't load. Please find the gif in the images folder of this repositiory.](/images/proof_of_concept.gif?raw=true "Proof of Concept")




###### More files will be added soon.
