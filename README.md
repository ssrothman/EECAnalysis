# EECAnalysis

Environment setup:

```
conda create --file envlist --name EEC
conda activate EEC
pip install fastjet==3.3.4.0rc9 energyflow 
git submodule init
git submodule update
cd EECs
pip install -e .
cd ../
```
