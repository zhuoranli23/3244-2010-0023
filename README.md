## Usage

* Download datasets: use [this script](https://github.com/arnab39/cycleGAN-PyTorch/blob/master/download_dataset.sh) via `sh ./download_dataset.sh horse2zebra`
* Train the model: `python main.py`
* Test the model
    * rename the latest checkpoint in `checkpoints` folder into `latest.ckpt`
    * run `test.ipynb`

**Note**: training will pick up the latest checkpoint by the name `latest.ckpt`

## To Start Tensorboard/ Visualise it 
* Type in your terminal: `tensorboard  --logdir=runs/` (i.e. the directory where your log files are stored)
