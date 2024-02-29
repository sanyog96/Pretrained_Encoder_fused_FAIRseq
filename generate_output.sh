#!/bin/bash
echo $#
if [ $# -ne 20 ]; then
    echo "Mismatch in # of arguments provided"
    echo "Please provide -n model_unique_identifier_name ; -d model_unique_description; -s source_language; -t target_lang -a cuda_device"
    exit 1
fi

while getopts n:d:s:t:a:f:u:k:m:c: flag
do
    case "${flag}" in
        n) model_folder_name=${OPTARG};;
        m) bert_model=${OPTARG};;
        u) user_dir=${OPTARG};;
        k) task=${OPTARG};;
        d) model_desc=${OPTARG};;
    	s) source_lang=${OPTARG};;
	    t) target_lang=${OPTARG};;
        a) cuda_device=${OPTARG};;
        f) date_format=${OPTARG};;
        c) checkpoint=${OPTARG};;
    esac
done

src=$source_lang
tgt=$target_lang
lang=$source_lang-$target_lang

if [ ! -d "output_custom_fldr/$model_folder_name/$model_desc/$date_format/$checkpoint" ]; then
    mkdir output_custom_fldr/$model_folder_name/$model_desc/$date_format/$checkpoint
fi

echo "This code generates outputs for Training, Validation and Testing datasets for $model_folder_name, $model_desc..."

echo "Generating output for testing data in ./output_custom_fldr/$model_folder_name/$model_desc/$date_format"
#Generate output for test
CUDA_VISIBLE_DEVICES=$cuda_device fairseq-generate data/$model_folder_name/$model_desc --bert-model $bert_model -s $source_lang -t $target_lang --user-dir $user_dir --task $task --log-file log_custom_fldr/$model_folder_name/$model_desc/generate_test_$date_format.log --sacrebleu --path model.stage2/$model_folder_name/$model_desc/$date_format/checkpoint_$checkpoint.pt --batch-size 32 --beam 5 --gen-subset test --remove-bpe --skip-invalid-size-inputs-valid-test > output_custom_fldr/$model_folder_name/$model_desc/$date_format/$checkpoint/output_test_$lang_$date_format.txt

#Generate output for valid
CUDA_VISIBLE_DEVICES=$cuda_device fairseq-generate data/$model_folder_name/$model_desc --bert-model $bert_model -s $source_lang -t $target_lang --user-dir $user_dir --task $task --log-file log_custom_fldr/$model_folder_name/$model_desc/generate_valid_$date_format.log --sacrebleu --path model.stage2/$model_folder_name/$model_desc/$date_format/checkpoint_$checkpoint.pt --batch-size 32 --beam 5 --gen-subset valid --remove-bpe --skip-invalid-size-inputs-valid-test > output_custom_fldr/$model_folder_name/$model_desc/$date_format/$checkpoint/output_valid_$lang_$date_format.txt

#Generate output for train
# CUDA_VISIBLE_DEVICES=$cuda_device fairseq-generate data/$model_folder_name/$model_desc --bert-model $bert_model -s $source_lang -t $target_lang --user-dir $user_dir --task $task --log-file log_custom_fldr/$model_folder_name/$model_desc/generate_train_$date_format.log --sacrebleu --path model.stage2/$model_folder_name/$model_desc/$date_format/checkpoint_$checkpoint.pt --batch-size 32 --beam 5 --gen-subset train --remove-bpe --skip-invalid-size-inputs-valid-test > output_custom_fldr/$model_folder_name/$model_desc/$date_format/$checkpoint/output_train_$lang_$date_format.txt
