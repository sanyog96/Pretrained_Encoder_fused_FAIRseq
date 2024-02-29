1)  Please clone the commit id (**100cd91db19bb27277a06a25eb4154c805b10189**) from fairseq
2)  Please clone the above repository in the same way and add the path of custom directory to PYTHONPATH
```
PYTHONPATH="./custom:$PYTHONPATH"
``` 
3)  Please keep track of all the variables to help simplify the process of execution

```    
source_lang=eng
target_lang=hin
date_format=2024_02_14_16_38
model_desc_folder_name=freezed_bert_base_t2500
epoch_name=pat_500_enc12_e768_f768_h12_dec6_e512_f1024_h4
model_desc=$model_desc_folder_name/$epoch_name
model_folder_name=eng_hin
```

4)  Run bert_tokenize.py or roberta_tokenize.py on source files to add `([CLS], [SEP])` and `(<s>, </s>)` tokens to source sentences.
    You are supposed to pass the path to the pre-trained model directory to pick the necessary files
    
    `--model` should be used to pass this path
5)  Run preprocess.py to create binary and indexing files for source and target files in the data-bin directory
    Since we are using a pre-trained model we should have an existing dictionary for the source language from that model. We can use that dictionary by keeping it in the data-bin folder where we want all the remaining binary and indexing files to generate.
```    
python3 custom/preprocess.py --source-lang $source_lang --target-lang $target_lang \
--srcdict data-bin/$model_folder_name/$model_desc/dict.eng.txt \
--trainpref raw_corpus/$model_folder_name/train \
--validpref raw_corpus/$model_folder_name/valid \
--testpref raw_corpus/$model_folder_name/test \
--destdir data-bin/$model_folder_name/$model_desc \
--log-file logs/$model_folder_name/$model_desc/preprocess_$date_format.log
``` 
6)  After this you can run the below command to train the model by replacing the parameters as required.
    For finetuning you need to add, the `--finetune` parameter to the below command
```
CUDA_VISIBLE_DEVICES=0 \
fairseq-train \
data-bin/$model_folder_name/$model_desc \
--max-source-positions=210 --max-target-positions=210 --max-update 200000 \
-s $source_lang \
-t $target_lang \
--user-dir custom \
--task translation_with_pretrained_encoder_model \
--model [model_path] \
--model-name [BERT/RoBERTa] \
--arch [model_architecture] \
--log-format simple \
--share-decoder-input-output-embed \
--optimizer adam --adam-betas '(0.9, 0.99)' \
--warmup-init-lr 1e-07 --lr 5e-4 \
--lr-scheduler inverse_sqrt --warmup-updates 4000 \
--weight-decay 0.001 --criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 --ddp-backend legacy_ddp --clip-norm 1.0 \
--patience 500 --batch-size 4096 --keep-last-epochs 1 --skip-invalid-size-inputs-valid-test \
--wandb-project eng_hin_freezed \
--fp16 --save-dir $MODEL_STAGE1/$model_folder_name/$model_desc/$date_format \
--log-file log_custom_fldr/$model_folder_name/$model_desc/train_$date_format.log
```
