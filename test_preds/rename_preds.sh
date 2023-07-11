#!/bin/bash

root=$(readlink -f $1)

for lang in arp git lez nyb ddo usp ntu; do
	mv "$root/${lang}_output_preds" "$root/${lang}-test-track2-covered.txt"
done