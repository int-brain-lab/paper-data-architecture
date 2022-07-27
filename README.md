# Learning Variability
## Install Instructions

Create new environment and activate
```
conda create -n learning_variability python=3.8
conda activate learning_variability
```

Install requirements and code
```
pip install -r requirements.txt
pip install -e .
```

## Set up access to Datajoint tables
See [these](https://github.com/int-brain-lab/IBL-pipeline-light#install-package-and-set-up-the-configuration) instructions for how to connect to the datajoin database. Below are the credentials required
```
import datajoint as dj
dj.config['database.host'] = "datajoint-public.internationalbrainlab.org" 
dj.config['database.user'] = "ibl-public"
dj.config['database.password'] = "ibl-public"  
dj.config.save_global()
```
