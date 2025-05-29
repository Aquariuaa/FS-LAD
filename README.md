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


## Pre-training Model
The normality detection model is derived from LogOnline, and we re-trained the normality detection model based on the Spell parsed dataset according to its source code. 
The code to implement the normality detection model is placed in src/aemodeltrain.py and src/aefeature.py.
Also, wiki-news-300d-1M.vec is available for download at [wiki-news-300d-1M-subword.vec](https://fasttext.cc/docs/en/english-vectors.html)

## Running of OMLog
You can run the code by clicking on OMLog.py directly in the root directory after installing the environment. 
All the parameters can be adjusted in OMLog.py.

```
python OMLog.py
```

## Methods of comparison

DeepLog | LogAnomaly:  [Code](https://github.com/xUhEngwAng/LogOnline)

LogRobust | PLELog|CNN: [Code](https://github.com/LogIntelligence/LogADEmpirical/tree/icse2022)

ROEAD: [Code](https://github.com/JasonHans/ROEAD-core-code)


## Acknowledgements
We acknowledge the work done by the LogOnline approach, and our code is implementation based on the [LogOnline](https://github.com/xUhEngwAng/LogOnline) .

