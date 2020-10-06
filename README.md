# captain-caption

-- (photo illustration goes here) --
---

## Initial steps (NECESSARY STEPS)

- install miniconda (or anaconda)
- then git clone this repo
- now you might want to run this code sniplet `conda env create -f environment.yml` (if your cudatoolkit version doesn't match your Nvidia driver version, uninstall and reinstall it)
- just get into the env created `conda activate captain-caption`


## Understanding the config.py file

It's just a file to tune hyperparams and some other things--
- checkout the file first
- `cpt` is to activate (or deactivate) storing of checkpoints for the model
- `wb` is for turning on wandb logging and tracking

*for training only*
- `npy` is activate for first time to store all feature vectors (it'll take 41G for the whole dataset, so have enough disk space available before proceeding to train)
One can change the sample_size in `config.py` to around 30000 for test training, *you-know-what-I-mean!*


## If you are interested in training from ground up, follow these steps:
- here you need to download the train data (COCO dataset 2014), use [this link](http://images.cocodataset.org/zips/train2014.zip)
- ITS ~13G file
- extract it into the repo's root dir
- run `python train.py` with `npy = True` for generation of feature vectors
- after that set `npy = False`
- finally you need to run `python train.py` and see the ***magic happen*** . . .

### In case you need visuals for your model, signup at wandb.com
Create an experiment there with name `captioning` or your own custom name, in that case you need to change `line no. 61` of `train.py` file
then you need to run this command
`wandb login <YOUR API KEY from the website>`
-- OR -- simply `wandb login`
that's it!



## In the case you are looking to just check out the performace of the model -- not ready yet
- put images to caption in `test_images` folder
- run `python inference.py` with `npy = False` in `config.py`

`exit()`
