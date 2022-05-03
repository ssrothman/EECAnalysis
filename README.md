# EECAnalysis

Environment setup:

```
conda create --file envlist --name EEC
conda activate EEC
pip install fastjet==3.3.4.0rc9 energyflow 
git clone https://github.com/pkomiske/EnergyEnergyCorrelators.git
cd EnergyEnergyCorrelators
git submodule init
git submodule update
pip install -e .
```
