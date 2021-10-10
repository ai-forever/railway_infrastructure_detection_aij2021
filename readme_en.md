AITrain: detecting obstacles and railway infrastructure components
=================================
[link to github](https://github.com/sberbank-ai/railway_infrastructure_detection_aij2021/blob/main/readme_en.md)

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
    "image": "cr.msk.sbercloud.ru/aijcontest2021/aitrain-base:v0.0.1",
    "entrypoint": "python3 /home/jovyan/solution.py"
}
```

Where `image` is a field with the docker image name, in which the solution will be run, `entrypoint` is a command that runs the solution. For solution, the archive root will be the current directory. 

To run solutions, existing environments can be used:

- [Basic images](https://docs.sbercloud.ru/aicloud/mlspace/concepts/environments__basic-images-for-training.html).

Any other image which is available in `sbercloud` will be suitable. If necessary, you can prepare your own image, add necessary software and libraries to it (see [the manual on creating Docker images for `sbercloud`](https://github.com/sberbank-ai/no_fire_with_ai_aij2021/blob/main/sbercloud_instruction.md)); to use it, you will need to publish it on `sbercloud`.

### Limitations

During one day, a Participant or a Team of Participants can upload no more than three solutions for evaluation. Only valid attempts that have received a numerical estimate are taken into account.  

The solution container will be run under the following conditions:

- 94 GB RAM;
- 3 vCPU;
- 1 GPU Tesla V100 32 Gb.
- Time for performance: 30m;
- Offline solution;
- Maximal size of your solution archive compressed and decompressed: 10 GB;
- Maximal size of the Docker image used: 15 GB.

## Quality check

Weighted average of `mAP@.5` and `meanIoU` is a quality metric of the task AITrain:

[mAP description](https://cocodataset.org/#detection-eval)  

[meanIoU description](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanIoU)

```
competition_metric = 0.7 * mAP@.5 + 0.3 * meanIoU
```

You may choose three solutions to submit for the final evaluation. By default, these are solutions with the best Leaderboard metric.


##
[Terms of use](https://api.dsworks.ru/dsworks-transfer/api/v1/public/file/terms_of_use_en.pdf/download)  
[Rules](https://api.dsworks.ru/dsworks-transfer/api/v1/public/file/rules_en.pdf/download)
