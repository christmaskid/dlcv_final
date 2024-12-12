combined_model_root="experiments/composed_edlora/stable-diffusion-v1-4/"
expdir="cat2+dog6"

context_prompt="A <cat2> on the right and a <dog6> on the left."
python inference/mix_of_show_sample.py \
  --pretrained_model="experiments/pretrained_models/stable-diffusion-v1-4" \
  --combined_model="${combined_model_root}/${expdir}/combined_model_.pth" \
  --save_dir="results/multi-concept/${expdir}" \
  --pipeline_type="sd_pplus" \
  --prompt="${context_prompt}" \
  --suffix="_241207" \
  --n_samples=20
