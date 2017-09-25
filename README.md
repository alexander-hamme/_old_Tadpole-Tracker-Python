# computer-vision-xenopus-tadpole-tracker
Program with OpenCV (Python / C++) and Tensorflow that tracks and records data of multiple *Xenopus laevis* tadpoles at once. 

There are two major components of this tracker program: **Detection** and **Tracking**.
  * detection is the process of finding regions of interest (ROI) in each frame (image) from the video input stream
  * tracking is the process of connecting where each animal was in previous frames to each sequential frame; 
    i.e. connecting the ROIs to individual tadpoles. This becomes complicated when tracking multiple animals, because collisions and collusions become possible. Therefore, trajectory prediction and other algorithms need to be implemented.


Detection: The best approach is to use a convolutional neural network. This program will implement the Faster R-CNN algorithm,  retrained for detection of Xenopus tadpoles. 

Tracking (trajectory prediction specifically): The best approach to this challenge is to train a sequential neural network on recorded tadpole movement data. 

Project is in collaboration with neuroscience researchers at Bard College.

###### Proof of Concept gif:

![Uh oh, it appears the gif didn't load. Please find the gif in the images folder of this repositiory.](/images/proof_of_concept.gif?raw=true "Proof of Concept")




###### ![Sample output file](https://github.com/alexander-hamme/Computer_Vision_Xenopus_Tadpole_Tracker/blob/master/data.csv) (Side note, acceleration calculations are not yet incorporated into final data file)


###### More files will be added soon.
