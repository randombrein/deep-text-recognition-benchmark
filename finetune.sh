# https://github.com/clovaai/deep-text-recognition-benchmark/issues/98


#1 - Prepare your dataset as described, create lmdb. Easiest way is just preserver the same structure like MJ dataset prepared by authors
#2 - Download pretrained model, You'd like to finetune. For example TPS-ResNet-BiLSTM-Attn.pth; rename it to TPS-ResNet-BiLSTM-Attn_<iteration to start over>.pth (for example TPS-ResNet-BiLSTM-Attn_15000.pth to resume training from 15000 iteration).
#3 - Just train with --FT flag.
#python3 train.py --train_data train/MJ --valid_data validation --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --FT --saved_model pretrained/TPS-ResNet-BiLSTM-Attn_15000.pth

python3 train.py \
  --train_data /home/vi/Datasets/UFPR-ALPR-lmdb/training \
  --valid_data /home/vi/Datasets/UFPR-ALPR-lmdb/validation \
  --select_data / \
  --batch_ratio 1 \
  --Transformation TPS \
  --FeatureExtraction ResNet \
  --SequenceModeling BiLSTM \
  --Prediction Attn \
  --FT \
  --valInterval 1000 \
  --data_filtering_off \
  --saved_model pretrained_model/TPS-ResNet-BiLSTM-Attn_15000.pth

# finetune img size issue
# https://github.com/clovaai/deep-text-recognition-benchmark/issues/96

# python3 create_lmdb_dataset.py \
#   --inputPath=/home/vi/Datasets/UFPR-ALPR/training \
#   --outputPath=/home/vi/Datasets/UFPR-ALPR-lmdb/training

# python3 create_lmdb_dataset.py \
#   --inputPath=/home/vi/Datasets/UFPR-ALPR/validation \
#   --outputPath=/home/vi/Datasets/UFPR-ALPR-lmdb/validation