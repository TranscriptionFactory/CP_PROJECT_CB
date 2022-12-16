
***************************\
dehaze.yml - conda environment specs\
***************************\
-src: all scripts \
	- my_models.py has all models\
	- various helper functions imported by my_models\
***************************\
BATCH SUBMIT: slurm scripts to train all models at once. \
- run_more.slurm submits run_model1-7.slurm \
- run_model.py loads each model\
\
***************************\
-DEHAZEFORMER: Github repo from DehazeFormer paper. /src/ imports DehazeFormer.utils.models\
\
***************************\
-LIGHT-DEHAZENET: Light-Dehazenet repo from paper\
\
***************************\
WEIGHT FOLDERS: All are organized by model name, with weights from each epoch and .csv files with training/validation loss\
\
weights_batch8: weights from using batch_size = 8
weights_batch4: weights from using batch_size = 4
