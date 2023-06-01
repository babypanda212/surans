# Stochastic Unsteady Reynolds Average Navier-Stokes Model
In this code a stochastic unsteady Reynolds average Navier-Stokes is defined. The corresponding publications is 
"Simulating the unsteady stable boundary layer with a stochastic stability equation" by Vyacheslav Boyko and Nikki Vercauteren.

## How to setup and run the model
1. Clone repository
```bash
git clone git@github.com:BoundaryLayerVercauteren/surans.git
cd surans
```

2. Create and activate environment
```bash
conda create --name surans --file requirements.txt
conda activate surans
```
or use docker
```bash
docker build . -t surans # build docker image
docker run -ti -v "$PWD:/home/fenics/shared" surans # run image
cd shared
```
Installation instructions for docker: https://docs.docker.com/engine/install/

3. Run the model
```bash
python main.py # conda
python3 main.py # docker
```

## Model run types
Option 1: Run an initialization run
Set in the file 'lib/params_class_base.py' the value of 'save_ini_cond' to True. The values of the last time step 
will be stored be stored in the directory 'init_condition'. The file name can be specified with 'solStringName' in 'lib/params_class_base.py'.

Option 2: Run stochastic model with initial conditions given in a file
Set in the file 'lib/params_class_base.py' the value of 'load_ini_cond' and 'stochastic_phi' to True and specify the file path for the initial conditions with 'initCondStr'.

## Note:
* The model output is stored in the directory 'solution'. 
* The directory 'supplementary_code' includes scripts which exemplary show how to visualize the model output.
* Please be aware that these scripts were only tested on a WSL (Windows subsystem for Linux) system. Fenics can not be installed with conda on a
Windows machine. A workaround is using WSL.