# TrafficLightAI

This project I made a AI that finds the traffic light if it is green, yellow, or red by using Jetson Resnet-18.

For this project to be working on your nano, you will need Jetson-Inference. The brief instruction is first we will use detectnet and find traffic light. After that we will find the coordinates and crop the picture. With the imagenet, we will find the trafficlight to be either green, yellow, or red. 

# Instruction

## Using Detectnet to crop the Image

We will first locate ourselves to Jetson-Inference using cd
```
nvidia@ubuntu:~$ cd jetson-inference/
```
After that, we will use detectnet.
```
nvidia@ubuntu:~/jetson-inference$ detectnet python/training/classification/data/trafficlight/trafficlightcrop/ python/training/classification/data/trafficlight/trafficlightcrop/detect/example_detected1.jpg
```
After you run the command, scroll up and you will find the coordinates of the traffic light like one below:
```
detected obj 3  class #10 (traffic light)  confidence=0.677246
bounding box 3  (362.304688, 49.493408)  (419.140625, 125.024414)  w=56.835938  h=75.531006
```
You will need to copy the two coordiantes. These represents the edges of the trafficlight in the picture. Paste it somewhere so it will not be lost.

## Cropping the image
First few commands to do before cropping:
```
nvidia@ubuntu:~/jetson-inference$ docker/run.sh
root@ubuntu:/jetson-inference# cd python/training/classification/data/trafficlight/trafficlightcrop/
```
After you get yourself into the trafficlightcrop folder do:
```
python3 crop.py (your original picture) crop/example_cropped1.jpg 362.304688 49.493408 419.140625 125.024414
```
## Imagenet the cropped image
You will need to get out of some of the folders so do:
```
cd ../../../
```
Double check that you are in the classification folder and now you can do this command to find if its green, yellow, or red:
```
imagenet --model=models/trafficlight/resnet18.onnx --labels=data/trafficlight/labels.txt --input_blob=input_0 --output_blob=output_0 data/trafficlight/trafficlightcrop/crop/example_cropped1.jpg data/trafficlight/trafficlightcrop/output
```
Because the pictures are small, it is likely that it wouldn't show the color AI thinks it is so scroll up and you can find it out.

