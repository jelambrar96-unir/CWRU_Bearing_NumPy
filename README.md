# CWRU Bearing Dataset in .npz format

The present repository presents the CWRU Bearing dataset in .npz format, corrected and reduced as far as some metadata are concerned. All credits for the dataset belong to CWRU.

## Dataset Overview

The CWRU Bearing dataset is an open-source dataset provided [here](https://engineering.case.edu/bearingdatacenter) by the Case School of Engineering of the Case Western Reserve University. The data correspond to time-series measured at locations near to and remote from motor bearings of a 2 HP Reliance Electric motor. As far as the experimental procedure is concerned, the CWRU webpage states:

> Motor bearings were seeded with faults using electro-discharge machining (EDM). Faults ranging from 0.007 inches in diameter to 0.040 inches in diameter were introduced separately at the inner raceway, rolling element (i.e. ball) and outer raceway. Faulted bearings were reinstalled into the test motor and vibration data was recorded for motor loads of 0 to 3 horsepower (motor speeds of 1797 to 1720 RPM).

A more detailed presentation of the methodology can be found [here](https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures).

## Data Files

The original data files are in .mat format. 

## Motivation

The present repository was created for two main reasons:

1. The dataset is given in .mat format, as MATLAB was (and still is, to some extent) the mainstream tool for data analysis in Engineering problems. Nonetheless, with the advances that Deep Learning has seen in the past decade, this dataset is being widely used to train, evaluate and deploy DL models. The mainstream frameworks for such tasks are written in Python, so converting the files directly to .npz format allows DL researchers and enthusiasts to easily and quickly load and subsequently convert them to tensor objects and feed them to NNs or other models.

2. The original files contain some inconsistencies and (perhaps) redundancies when it comes to their metadata (see the next section for more information). The version presented here contains only the time-series data necessary for analysis and DL.

## Changes & Corrections

TODO

## Attribution

All credits for the dataset belong to CWRU. This corrected and reduced version of the dataset was developed for the purposes of the following publication:

```
TODO
```

Nonetheless, citing this paper is not required for researchers who used this version of the dataset.
