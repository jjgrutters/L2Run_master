=======================================
Setting up python opensim on your computer
=======================================

a. Download and install anaconda3
b. Create a virtual environment (step 7.)
c. install opensim packages (step 9.)
d. git clone our repository (step 10.) 

=======================================
Using the CTIT cluser for
OpenSim and stable-baselines
=======================================

1. Ask access to the server:
>> email: 
2. download and install Putty
Putty will be used to command the server, 
for installing and running only

3. download and install WinSCP
WinSCP gives you access to the server, 
this GUI will be used to drag and drop files in your personal folder

4. Connect to edurome, or connect to vpn
5. Open Putty and connect to server:
- server name: korenvliet.ewi.utwente.nl
- port: 22

6. setting up anaconda 
>> module load anaconda3
7. make virtual env: 
>> conda create -n <env_name> python=3.6
8. activate venv: 
>> source activate <env_name>
9. Install opensim-rl
>> conda install -c kidzik opensim 
>> conda install -c conda-forge lapack git
>> pip install git+https://github.com/stanfordnmbl/osim-rl.git

10. Gitclone our repository
link: https://github.com/jjgrutters/L2Run_master

### Important packages for DDPG algrithm ###
>> pip install stable-baselines==2.10.1 (or 2.8.0 if older)
>> pip install tensorflow==1.15.3
>> pip install mpi4py==3.0.3

### For mpi4py you should setup MPI ###
### If MPI is NOT setup correct do step 11.###
11. Setup MPI:
>> pip uninstall mpi4py
>> rm -r .cache/pip
>> module load openmpi/1.10.2 
>> export MPICC=$../../usr/bin/mpicc
>> env MPICC=../../usr/bin/mpicc pip install mpi4py

12. submitting jobs
Use the included .sbatch template to submit your jobs
Use WINSCP to setup our experiment:
- server name: korenvliet.ewi.utwente.nl
- port: 22
- File protocol: SCP
 
To submit a job to the server on Putty, use: >> sbatch, so:
>> sbatch job.sbatch

Visit http://korenvliet.ewi.utwente.nl/slurm/ for job status
Visit http://korenvliet.ewi.utwente.nl/wiki/doku.php for more information

================================
Using Tensorboard 
================================
open CMD on your computer
tensorboard file should be stored localy
>> activate anaconda3
>> conda activate <env_name>
>> tensorboard --logdir=C:/name/of/log/folder
chrome: localhost:6006



