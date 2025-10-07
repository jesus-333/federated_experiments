#!/bin/sh
#
# Author : Alberto  Zancanaro (Jesus)
# Organization: Luxembourg Centre for Systems Biomedicine (LCSB)
# Contact : alberto.zancanaro@uni.lu
 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

root_certificate_path="certificates/ca.crt"
superlink_ip="10.240.18.119:9092"

# POSITIONAL_ARGS=()
#
# while [[ $# -gt 0 ]]; do
# 	case $1 in
# 		-c|--root_certificate_path)
# 		EXTENSION="$2"
#
# 		root_certificate_path="certificates/ca.crt"
# 		shift # past argument
# 		shift # past value
# 		;;
# 	-s|--searchpath)
# 		SEARCHPATH="$2"
# 		shift # past argument
# 		shift # past value
# 		;;
# 		--default)
# 	DEFAULT=YES
# 		shift # past argument
# 		;;
# 		-*|--*)
# 		echo "Unknown option $1"
# 		exit 1
# 		;;
# 		*)
# 	POSITIONAL_ARGS+=("$1") # save positional arg
# 		shift # past argument
# 		;;
# 	esac
# done
#
# set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

flower-supernode \
	--root-certificates ${root_certificate_path} \
	--superlink ${superlink_ip} \
	--clientappio-api-address 0.0.0.0:9094 \
	--node-config="partition-id=0 num-partitions=2"

