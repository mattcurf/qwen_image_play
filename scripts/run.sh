#!/bin/bash

python scripts/download.py

rm output_folder/*

counter=0
PROMPTS_FILE=/project/scripts/prompts.txt
while IFS= read -r prompt || [[ -n "$prompt" ]]; do
  if [[ -z "$prompt" ]]; then
    continue
  fi

  python scripts/qwen_image.py --prompt "$prompt" --output output_folder/"$counter.png"
  counter=$((counter + 1))

done < "$PROMPTS_FILE"
