REF_PROMPT="A cat wearing weable_glasses in a watercolor style."
BASE_PROMPT="A cat wearing weable_glasses in a watercolor style."
CUSTOM_PROMPTS_1="A <cat2> wearing <cat2> in a <cat2> style."
CUSTOM_PROMPTS_2="A <wearable_glasses> wearing <wearable_glasses> in a <wearable_glasses> style."
CUSTOM_PROMPTS_3="A <watercolor> wearing <watercolor> in a <watercolor> style."
REF_IMAGE_PATH="prompt3/A cat wearing wearable glasses in a watercolor style_14.jpg"
REF_MASK_PATHS_1="prompt3/A cat wearing wearable glasses in a watercolor style_14_mask1.png"
REF_MASK_PATHS_2="prompt3/A cat wearing wearable glasses in a watercolor style_14_mask1.png"
REF_MASK_PATHS_3="examples/cat_glasses_color.png"
EDLORA_PATHS_1="experiments/cat2/models/edlora_model-latest.pth"
EDLORA_PATHS_2="experiments/wearable_glasses/models/edlora_model-latest.pth"
EDLORA_PATHS_3="experiments/watercolor/models/edlora_model-latest.pth"
START_SEED=0
BATCH_SIZE=4
N_BATCHES=1
# A <cat2> wearing <wearable_glasses> in a <watercolor> style.