AITrain: detecting obstacles and railway infrastructure components
=================================

Competition of algorithms for detecting obstacles and railway infrastructure components.  

Creating smart systems that warn train drivers about possible collisions with potentially hazardous objects is a way to enhance rail traffic safety. Competition participants should create an algorithm to detect potentially hazardous objects and railway infrastructure components that are crucial for rail traffic – rails, switches, and traffic signals.

The input data are RGB images taken by various cameras installed on the train, with image descriptions.


## Task setting

Based on the photos, you should create the detection algorithm for the following objects:
- "Car" (car);
- "Human" (human);
- "Wagon" (wagon)*;
- "FacingSwitchL" (turnout switch in the course of movement, to the left);
- "FacingSwitchR" (turnout switch in the course of movement, to the right);
- "FacingSwitchNV" (turnout switch in the course of movement, out of sight);
- "TrailingSwitchL" (turnout switch against the course of movement, to the left);
- "TrailingSwitchR" (turnout switch against the course of movement, to the right);
- "TrailingSwitchNV" (turnout switch against the course of movement, out of sight);
- "SignalE" (traffic light permitting);
- "SignalF" (no traffic light).

Besides, you need to put segmentation masks for the following elements:
 - 6 - "MainRailPolygon";
 - 7 - "AlternativeRailPolygon";
 - 10 - "Train"*.

Note: * - the first and last railcars are to be detected, and other railcars need segmentation masks.


## Solution format

Participants should send the algorithm code in ZIP format to the testing system. Solutions shall be run by Docker in the offline mode. The testing time and resources are limited. Participants do not need to study the Docker technology.

### Container content

The archive root must contain the metadata.json file containing the following:
```json
{
    "image": "cr.msk.sbercloud.ru/aicloud-base-images-test/custom/aij2021/aitrain:f66e1b5f-1269",
    "entrypoint": "python3 /home/jovyan/solution.py"
}
```

Where `image` is a field with the docker image name, in which the solution will be run, entrypoint is a command that runs the solution. For solution, the archive root will be the current directory. 

To run solutions, existing environments can be used:

- `cr.msk.sbercloud.ru/aicloud-base-images-test/custom/aij2021/aitrain:f66e1b5f-1269` — [Dockerfile](https://github.com/sberbank-ai/railway_infrastructure_detection_aij2021/blob/main/Dockerfile) with the description of the image and [requirements](https://github.com/sberbank-ai/railway_infrastructure_detection_aij2021/blob/main/requirements.txt) with libraries

Any other image which is available in `sbercloud` will be suitable. If necessary, you can prepare your own image, add necessary software and libraries to it (see [the manual on creating Docker images for `sbercloud`](https://github.com/sberbank-ai/railway_infrastructure_detection_aij2021/blob/main/sbercloud_instruction.md)); to use it, you will need to publish it on `sbercloud`.

### Limitations

The solution container will be run under the following conditions:

- 16 GB RAM;
- 4 vCPU;
- 1 GPU Tesla V100;
- Time for performance: 30m;
- Offline solution;
- Maximal size of your solution archive compressed and decompressed: 10 GB;
- Maximal size of the Docker image used: 15 GB.

## Quality check

Panoptic quality (PQ) is a quality metric of the task AITrain:

![Panoptic quality](https://github.com/sberbank-ai/railway_infrastructure_detection_aij2021/blob/main/images/pq_1.png)  

Which is equivalent to:  

![Panoptic quality](https://github.com/sberbank-ai/railway_infrastructure_detection_aij2021/blob/main/images/pq_2.png)  

PQ is the metric used for segmentation model performance assessment (Panoptic Segmentation). The numerator of the fraction is the sum of  Intersection over Union (IoU) ratios for all True Positive model solutions. The denominator is the sum of absolute values of all True Positive model results, half of all False Positive and False Negative results.

Explanations: Intersection over Union (IoU) is the ratio showing how accurately the model determined object location in a particular image (accepts values from 0 to 1).  
The solution is considered True Positive, if IoU>0.5.  
The solution is considered False Positive, if 0<IoU<0.5.  
The solution is considered False Negative, if 0=IoU.  

Your solution will be assessed against a lazy fetch of RGB images in the training data format. Based on panoptic quality values, The Leaderboard is formed.  

If several Participants have the same panoptic quality metric values, their solutions are assessed based on processing time metric (the time spent on processing problems).  

You may choose three solutions to submit for the final evaluation. By default, these are solutions with the best Leaderboard metric.
