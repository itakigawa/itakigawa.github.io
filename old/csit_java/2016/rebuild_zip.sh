#!/bin/bash

rm -rf data_src
mkdir data_src
cp *.java data_src/
cp file.txt *.c table.dat table.csv data_src/
zip -r data_src.zip data_src


