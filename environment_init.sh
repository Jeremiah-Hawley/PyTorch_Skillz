#!/bin/env bash
if [[ $# -ne 0 ]]; then
  echo " === creating conda environment === ";
  conda create --name $1 python=3.11 numpy pytorch torchvision numpy matplotlib scikit-learn &&
  conda activate $1 &&
  mkdir data;
else
  echo "please pass an argument containing a name for the environment"
fi

