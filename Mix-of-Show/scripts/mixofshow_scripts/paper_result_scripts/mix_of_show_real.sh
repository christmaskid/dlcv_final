combined_model_root="experiments/composed_edlora/chilloutmix/"
expdir="potter+hermione+thanos+hinton+lecun+bengio+catA+dogA+chair+table+dogB+vase+pyramid+rock_chilloutmix"

#---------------------------------------------samoke potter_rock---------------------------------------------
potter_rock=1

if [ ${potter_rock} -eq 1 ]
then
  context_prompt='<potter1> <potter2> and a woman, a cat and a dog, at <rock1> <rock2>, 4K, high quality, high resolution, best quality'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  # region1_prompt='[<potter1> <potter2>, in hogwarts school uniform, holding hands, at <rock1> <rock2>, 4K, high quality, high resolution, best quality]'
  # region1_neg_prompt="[${context_neg_prompt}]"
  # region1='[0, 315, 512, 530]'

  # region2_prompt='[<hermione1> <hermione2>, in hogwarts school uniform, holding hands, at <rock1> <rock2>, 4K, high quality, high resolution, best quality]'
  # region2_neg_prompt="[${context_neg_prompt}]"
  # region2='[0, 502, 512, 747]'

  # region3_prompt='[<dogA1> <dogA2>, 4K, high quality, high resolution, best quality]'
  # region3_neg_prompt="[${context_neg_prompt}]"
  # region3='[221, 43, 512, 258]'

  # region4_prompt='[<catA1> <catA2>, 4K, high quality, high resolution, best quality]'
  # region4_neg_prompt="[${context_neg_prompt}]"
  # region4='[228, 752, 512, 1016]'

  # prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}|${region4_prompt}-*-${region4_neg_prompt}-*-${region4}"

  keypose_condition='datasets/validation_spatial_condition/characters-objects/harry_heminone_scene_pose.png'
  keypose_adaptor_weight=1.0
  region_keypose_adaptor_weight=""

  # sketch_condition='datasets/validation_spatial_condition/characters-objects/harry_heminone_scene_sketch.png'
  # sketch_adaptor_weight=0.5
  # region_sketch_adaptor_weight="${region3}-0.8|${region4}-0.8"

  python inference/mix_of_show_sample.py \
    --pretrained_model="experiments/pretrained_models/chilloutmix" \
    --combined_model="${combined_model_root}/${expdir}/combined_model.pth" \
    --save_dir="results/multi-concept/${expdir}" \
    --pipeline_type="sd_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --suffix="baseline" \
    --seed=641
fi
