{\rtf1\ansi\ansicpg1252\cocoartf2706
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 ***************************\
dehaze.yml - conda environment specs\
***************************\
-src: all scripts \
	- my_models.py has all models\
	- various helper functions imported by my_models\
***************************\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0
\cf0 BATCH SUBMIT: slurm scripts to train all models at once. \
- run_more.slurm submits run_model1-7.slurm \
- run_model.py loads each model\
\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0
\cf0 ***************************\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0
\cf0 -DEHAZEFORMER: Github repo from DehazeFormer paper. /src/ imports DehazeFormer.utils.models\
\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0
\cf0 ***************************\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0
\cf0 -LIGHT-DEHAZENET: Light-Dehazenet repo from paper\
\
***************************\
WEIGHT FOLDERS: All are organized by model name, with weights from each epoch and .csv files with training/validation loss\
\
weights_batch8: weights from using batch_size = 8\
weights_batch4: weights from using batch_size = 4}