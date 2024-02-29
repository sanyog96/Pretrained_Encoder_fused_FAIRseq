#!/bin/bash

if [ $# -ne 10 ]; then
    echo "Mismatch in # of arguments provided. Please provide -n model_folder_name -d model_desc_folder_name -e epoch_folder_name -f date_format"
    exit 1
fi

while getopts n:d:e:f:s: flag
do
    case "${flag}" in
        n) model_folder_name=${OPTARG};;
        d) model_desc_folder_name=${OPTARG};;
        e) epoch_folder_name=${OPTARG};;
        f) date_format=${OPTARG};;
        s) stage=${OPTARG};;
    esac
done


cd data
if [ ! -d "$model_folder_name" ]; then
    mkdir $model_folder_name
fi
if [ ! -d "$model_folder_name/$model_desc_folder_name" ]; then
    mkdir $model_folder_name/$model_desc_folder_name
fi
if [ ! -d "$model_folder_name/$model_desc_folder_name/$epoch_folder_name" ]; then
    mkdir $model_folder_name/$model_desc_folder_name/$epoch_folder_name
fi

if [ "$stage" == "stage1" ]; then
    cd ../model.stage1
    if [ ! -d "$model_folder_name" ]; then
        mkdir $model_folder_name
    fi
    if [ ! -d "$model_folder_name/$model_desc_folder_name" ]; then
        mkdir $model_folder_name/$model_desc_folder_name
    fi
    if [ ! -d "$model_folder_name/$model_desc_folder_name/$epoch_folder_name" ]; then
        mkdir $model_folder_name/$model_desc_folder_name/$epoch_folder_name
    fi
    if [ ! -d "$model_folder_name/$model_desc_folder_name/$epoch_folder_name/$date_format" ]; then
        mkdir $model_folder_name/$model_desc_folder_name/$epoch_folder_name/$date_format
    fi
fi

if [ "$stage" == "stage2" ]; then
    cd ../model.stage2
    if [ ! -d "$model_folder_name" ]; then
        mkdir $model_folder_name
    fi
    if [ ! -d "$model_folder_name/$model_desc_folder_name" ]; then
        mkdir $model_folder_name/$model_desc_folder_name
    fi
    if [ ! -d "$model_folder_name/$model_desc_folder_name/$epoch_folder_name" ]; then
        mkdir $model_folder_name/$model_desc_folder_name/$epoch_folder_name
    fi
    if [ ! -d "$model_folder_name/$model_desc_folder_name/$epoch_folder_name/$date_format" ]; then
        mkdir $model_folder_name/$model_desc_folder_name/$epoch_folder_name/$date_format
    fi
fi

cd ../log_custom_fldr
if [ ! -d "$model_folder_name" ]; then
    mkdir $model_folder_name
fi
if [ ! -d "$model_folder_name/$model_desc_folder_name" ]; then
    mkdir $model_folder_name/$model_desc_folder_name
fi
if [ ! -d "$model_folder_name/$model_desc_folder_name/$epoch_folder_name" ]; then
    mkdir $model_folder_name/$model_desc_folder_name/$epoch_folder_name
fi

cd ../output_custom_fldr
if [ ! -d "$model_folder_name" ]; then
    mkdir $model_folder_name
fi
if [ ! -d "$model_folder_name/$model_desc_folder_name" ]; then
    mkdir $model_folder_name/$model_desc_folder_name
fi
if [ ! -d "$model_folder_name/$model_desc_folder_name/$epoch_folder_name" ]; then
    mkdir $model_folder_name/$model_desc_folder_name/$epoch_folder_name
fi
if [ ! -d "$model_folder_name/$model_desc_folder_name/$epoch_folder_name/$date_format" ]; then
    mkdir $model_folder_name/$model_desc_folder_name/$epoch_folder_name/$date_format
fi
