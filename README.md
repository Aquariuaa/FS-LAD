# FSLog: Adversarial Margin for Cross-System Few-shot Log Anomaly Detection

## Pre-requisites:

The experiment is based on a Pytorch implementation running in the following environment

```
torch==2.2.0.dev20231016+cu121
torchvision==0.17.0.dev20231016+cu121
python-dateutil==2.9.0.post0
pytorch-lightning==2.5.1.post0
numpy==1.24.1
pandas==2.2.1
```
## Dataset
Our approach follows the work in LogRobust and LogADEmpirical. Therefore, all experiments were performed on three public log datasets, HDFS, BGL and TBird. 
The open source datasets and their structured versions are available in [LogADEmpirical](https://github.com/LogIntelligence/LogADEmpirical/tree/icse2022). 

## Log Parsing
The log parser used by LogOnline is [Spell](https://github.com/pfeak/spell)

## LAD Methods of comparison

DeepLog | LogAnomaly:  [Code](https://github.com/xUhEngwAng/LogOnline)

LogRobust | PLELog|CNN: [Code](https://github.com/LogIntelligence/LogADEmpirical/tree/icse2022)

LogADEmpirical: [Code](https://github.com/LogIntelligence/LogADEmpirical/tree/icse2022)

## Few-shot Methods of comparison
LibFewShot: [Code](https://github.com/rl-vig/libfewshot)

## Acknowledgements
We acknowledge the work done by the LogOnline approach, and our code is implementation based on the [LogOnline](https://github.com/xUhEngwAng/LogOnline) .

