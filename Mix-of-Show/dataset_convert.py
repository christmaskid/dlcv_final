import os
import json
import random
import argparse

prompts = [
	"a photo of <TOK>"
]

def check_path(_path):
	if not os.path.exists(_path):
		os.mkdir(_path)

def convert(args):
	print(args)
	check_path(args.out_captions_path)
	check_path(args.out_jsons_path)

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
		yaml_content = yaml_content.replace("<name>", token_name)
		yaml_content = yaml_content.replace("<new_concept_token>", "<"+token_name+">")
		yaml_content = yaml_content.replace("<concept_list>", out_json_path)
		yaml_content = yaml_content.replace("<prompts>", args.prompts_path)
		
		token_yaml_file.write(yaml_content)
		token_yaml_file.close()
		template_yaml_file.close()

	if args.input_json_path is not None:
		val_json = json.load(open(args.input_json_path, "r"))
		for k, v in val_json.items():
			token_names = v["token_name"]
			merge_json_file = []

			for token_name in token_names:
				merge_json_file.append({
						"lora_path": "experiments/"+token_name[1:-1]+"/models/net_g_1000.pth",
						"unet_alpha": 1.0, 
						"text_encoder_alpha": 1.0,
						"concept_name": token_name
					})

			json.dump(open(merge_json_file, "r"), 
				args.merge_json_path_prefix+"_"+"-".join(token_names), indent=4)


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_images_path")
	parser.add_argument("--input_json_path")
	parser.add_argument("--out_captions_path")
	parser.add_argument("--out_jsons_path")
	parser.add_argument("--template_yaml")
	parser.add_argument("--prompts_path")
	parser.add_argument("--merge_json_path_prefix")
	args = parser.parse_args()

	convert(args)

