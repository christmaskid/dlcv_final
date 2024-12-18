source ./configs/custombranch/watercolor_ori.sh

python sample.py \
  --ref_prompt "$REF_PROMPT" \
  --base_prompt "$BASE_PROMPT" \
  --custom_prompts "$CUSTOM_PROMPTS_1" "$CUSTOM_PROMPTS_2" \
  --ref_image_path "$REF_IMAGE_PATH" \
  --ref_mask_paths "$REF_MASK_PATHS_1" "$REF_MASK_PATHS_2" \
  --edlora_paths "$EDLORA_PATHS_1" "$EDLORA_PATHS_2" \
  --start_seed "$START_SEED" \
  --batch_size "$BATCH_SIZE" \
  --n_batches "$N_BATCHES"