#!/bin/env bash
apt update &&
apt upgrade &&
apt install vim &&
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh &&
bash Miniconda3-latest-Linux-*.sh &&
source ~/.bashrc;
