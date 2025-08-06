#!/bin/bash

#huggingface-cli login --token $HF_TOKEN
python scripts/download.py

PROMPTS_FILE=/project/scripts/prompts.txt

counter=0
while IFS= read -r prompt || [[ -n "$prompt" ]]; do
  if [[ -z "$prompt" ]]; then
    continue
  fi

  python scripts/qwen_image.py --prompt "$prompt" --output "$counter.png"
  counter=$((counter + 1))

done < "$PROMPTS_FILE"
