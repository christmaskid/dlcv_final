import os
import json
import random
import argparse

prompts = [
	"a photo of <TOK>"
]

semantics = {
	"cat2": "cat",
	"pet_cat1": "cat",
	"dog": "dog",
	"dog6": "dog",
	"flower_1": "flower",
	"wearable_glasses": "glasses",
	"watercolor": "watercolor",
	"vase": "vase"
}

all_region_prompts = {
	"0": [
		("A <cat2> on the right.", '[0,0,512,256]'), 
		("A <dog6> on the left.", '[0,256,512,512]')
		],
	"1": [
		("A <flower_1>.", "[30,200,320,256]"), 
		("A <vase>", "[168, 144, 512, 384]")
		],
	"2": [
		("A <dog> near a forest.", "[119,30,385,151]"), 
		("A <pet_cat1> near a forest.", "[111,197,384,314]"),
		("A <dog6> near a forest.","[276,139,384,480]")
		],
	"3": [
		("A <cat2> in a <watercolor> style.","[60,90,512,452]"), 
		("A <wearable_glasses> in a <watercolor> style.","[111,142,215,386]")
		]
}

new_concepts_tokens = {}

def check_path(_path):
	if not os.path.exists(_path):
		os.mkdir(_path)

def convert(args):
	print(args)
	check_path(args.out_captions_path)
	check_path(args.out_jsons_path)

	f = open(args.prompts_path, "w")
	for prompt in prompts:
		f.write(prompt+"\n")
	f.close()

	for token_name in os.listdir(args.input_images_path):
		print(token_name)
		caption_dir = os.path.join(args.out_captions_path, token_name)
		check_path(caption_dir)

		image_dir = os.path.join(args.input_images_path, token_name)
		for img_fn in os.listdir(image_dir):
			txt_file = open(os.path.join(caption_dir, img_fn.split(".")[0]+".txt"), "w")
			txt_file.write(random.choice(prompts))
			txt_file.close()

		output_json_file = [{
			"instance_prompt": "<TOK>",
			"instance_data_dir": image_dir,
			"caption_dir": caption_dir
		}]
		out_json_path = os.path.join(args.out_jsons_path, token_name+".json")
		json.dump(output_json_file, open(out_json_path, "w"))

		template_yaml_file = open(args.template_yaml, "r")
		token_yaml_file = open(token_name+".yml", "w")

		yaml_content = ''.join(template_yaml_file.readlines())

		new_concepts = ["<{}_1>".format(token_name), "<{}_2>".format(token_name)]
		new_concepts_tokens[token_name] = new_concepts

		yaml_content = yaml_content.replace("<name>", token_name)
		yaml_content = yaml_content.replace("<replace_mapping>", " ".join(new_concepts))
		yaml_content = yaml_content.replace("<new_concept_token>", "+".join(new_concepts))
		yaml_content = yaml_content.replace("<concept_list>", out_json_path)
		yaml_content = yaml_content.replace("<prompts_path>", args.prompts_path)

		yaml_content = yaml_content.replace("<semantic>", semantics[token_name])

		yaml_content = yaml_content.replace("<embedding_enable_tuning>", args.embedding_enable_tuning)
		yaml_content = yaml_content.replace("<text_encoder_enable_tuning>", args.text_encoder_enable_tuning)
		yaml_content = yaml_content.replace("<unet_enable_tuning>", args.unet_enable_tuning)
		yaml_content = yaml_content.replace("<embedding_lr>", str(args.embedding_lr))
		yaml_content = yaml_content.replace("<text_encoder_lr>", str(args.text_encoder_lr))
		yaml_content = yaml_content.replace("<unet_lr>", str(args.unet_lr))
		yaml_content = yaml_content.replace("<n_iterations>", str(args.n_iterations))
		yaml_content = yaml_content.replace("<lora_rank>", str(args.lora_rank))
		yaml_content = yaml_content.replace("<lora_alpha>", str(args.lora_alpha))
		yaml_content = yaml_content.replace("<latent_size>", str(args.latent_size))

		yaml_content = yaml_content.replace("<num_inference_steps>", str(args.num_inference_steps))
		yaml_content = yaml_content.replace("<guidance_scale>", str(args.guidance_scale))

		yaml_content = yaml_content.replace("<save_checkpoint_freq>", str(args.save_checkpoint_freq))
		
		token_yaml_file.write(yaml_content)
		token_yaml_file.close()
		template_yaml_file.close()

	if args.input_json_path is not None:
		val_json = json.load(open(args.input_json_path, "r"))
		for k, v in val_json.items():
			token_names = v["token_name"]
			merge_json_file = []
			fn = "-".join(token[1:-1] for token in token_names)

			for token_name in token_names:
				merge_json_file.append({
						"lora_path": "experiments/"+token_name[1:-1]+"/models/net_g_1000.pth",
						"unet_alpha": 1.0, 
						"text_encoder_alpha": 1.0,
						"concept_name": " " .join(new_concepts_tokens[token_name[1:-1]])
					})

			json.dump(merge_json_file, 
				open(args.merge_json_path_prefix+"_"+fn+".json", "w"), indent=4)

			inf_bash_file = open("mix_of_show_"+fn+".sh", "w")
			prompt = v["prompt"]
			for token in new_concepts_tokens:
				prompt = prompt.replace("<"+token+">", " ".join(new_concepts_tokens[token]))

			region_prompts = ""
			prompt_rewrite = ""
			for j, (region_prompt,bbox) in enumerate(all_region_prompts[k]):
				if j>0:
					prompt_rewrite += "|"
				region_prompts += "region{}_prompt='[{}]'\n".format(j+1, region_prompt)
				region_prompts += "region{}_neg_prompt=\"[${{context_neg_prompt}}]\"\n".format(j+1)
				region_prompts += "region{}='{}'\n\n".format(j+1, bbox)
				prompt_rewrite += "${{region{}_prompt-*-${{region{}_neg_prompt}}-*-${{region{}}}".format(j+1,j+1,j+1)

			s = """
combined_model_root="experiments/composed_edlora/stable-diffusion-v1-4"
expdir="{0}"

context_prompt="{1}"
context_neg_prompt="{2}"

{3}

prompt_rewrite = "{4}"

python inference/mix_of_show_sample.py \\
  --pretrained_model="experiments/pretrained_models/stable-diffusion-v1-4" \\
  --combined_model="${{combined_model_root}}/${{expdir}}/combined_model_.pth" \\
  --save_dir="results/multi-concept/${{expdir}}" \\
  --pipeline_type="sd_pplus" \\
  --prompt="${{context_prompt}}" \\
  --negative_prompt="${{context_neg_prompt}}" \\
  --prompt_rewrite="${{prompt_rewrite}}" \\
  --suffix="" \\
  --n_samples=20""".format(
  				"+".join(token[1:-1] for token in token_names), 
  				prompt,
  				args.neg_prompt,
  				region_prompts,
  				prompt_rewrite
  			)
			inf_bash_file.write(s)




if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_images_path", required=True)
	parser.add_argument("--input_json_path", required=True)
	parser.add_argument("--out_captions_path", required=True)
	parser.add_argument("--out_jsons_path", required=True)
	parser.add_argument("--template_yaml", required=True)
	parser.add_argument("--prompts_path", required=True)
	parser.add_argument("--merge_json_path_prefix", required=True)

	parser.add_argument("--neg_prompt", default="")

	parser.add_argument("--embedding_enable_tuning", default="true")
	parser.add_argument("--text_encoder_enable_tuning", default="true")
	parser.add_argument("--unet_enable_tuning", default="true")
	parser.add_argument("--embedding_lr", default=1e-3)
	parser.add_argument("--text_encoder_lr", default=1e-5)
	parser.add_argument("--unet_lr", default=1e-4)
	parser.add_argument("--n_iterations", default=1000)
	parser.add_argument("--lora_rank", default=4)
	parser.add_argument("--lora_alpha", default=1)
	parser.add_argument("--latent_size", default=[4,64,64])

	parser.add_argument("--num_inference_steps", default=50)
	parser.add_argument("--guidance_scale", default=7.5)

	parser.add_argument("--save_checkpoint_freq", default=200)

	args = parser.parse_args()

	convert(args)

