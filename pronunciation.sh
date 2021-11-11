#!/bin/bash

Pronunciation -ifmt UNICODE8 -ofmt UNICODE8 -log ./pronunciation/log.txt -logfmt UNICODE8 $1 ./pronunciation/converted.txt

#Pronunciation -ifmt UNICODE8 -ofmt ROMANIZEDHANGUL -log ./pronunciation/log.txt -logfmt UNICODE8 -use_syllable T $1 ./pronunciation/converted.txt