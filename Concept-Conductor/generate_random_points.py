import sys
import random
import subprocess
from PIL import Image, ImageDraw

task_name = "dog-cat-dog_{}_".format(sys.argv[1])
img_width = 512
img_height = 512
n_concept = 3

bboxes = []

def between(val, m, M):
	return (val >= m) and (val <= M)

def collide(a, b):
	# x: vertical, y: horizontal
	ax1, ay1, ax2, ay2 = a
	bx1, by1, bx2, by2 = b
	return (between(ax1, bx1, bx2) or between(bx1, ax1, ax2)) and (between(ay1, by1, by2) or between(by1, ay1, ay2))

def create_and_save_mask(bboxes, save_fn):
	mask = Image.new("L", (img_height, img_width), 0)
	draw = ImageDraw.Draw(mask)
	for bbox in bboxes:
		draw.rectangle(bbox, fill=255)
	mask.save(save_fn)
	print("Save to {}".format(save_fn))


while len(bboxes) < n_concept:
	center = [random.randint(img_height//6, img_height*5//6-1), 
			  random.randint(img_width//6, img_width*5//6-1)]
	h = random.randint(img_height//5, img_height//3-1)
	w = random.randint(img_width//5, img_width//3-1)
	bbox = [center[0]-h//2, center[1]-w//2, center[0]+h//2, center[1]+w//2]

	flag = False
	for other in bboxes:
		flag = collide(bbox, other)
		if flag: break
	if not flag: bboxes.append(bbox)

for i, bbox in enumerate(bboxes):
	mask = create_and_save_mask([bbox], "examples/{}_{}.png".format(task_name, i+1))
create_and_save_mask(bboxes, "examples/{}_mask.png".format(task_name))


cmd = """
python sample.py \
--sd_ckpt "/content/dlcv_final/Concept-Conductor/experiments/pretrained_models/stable-diffusion-v1-5" \
--outroot "outputs/dog_pet_cat1_dog6_mask_foreground1" \
--ref_prompt 'A dog, a pet cat and a dog near a forest.' \
--base_prompt 'A dog, a pet cat and a dog near a forest.' \
--custom_prompts 'A <dog_1> <dog_2>, a <dog_1> <dog_2> and a <dog_1> <dog_2> near a forest.' \
  'A <pet_cat1_1> <pet_cat1_2>, a <pet_cat1_1> <pet_cat1_2> and a <pet_cat1_1> <pet_cat1_2> near a `forest.' \
  'A <dog6_1> `<dog6_2>, a <dog6_1> <dog6_2> and a <dog6_1> <dog6_2> near a forest.' \
--ref_image_path "examples/{}_mask.png" \
--ref_mask_paths "examples/{}_{}.png" "examples/{}_{}.png" "examples/{}_{}.png" \
--edlora_paths "experiments/dog/models/edlora_model-latest.pth" "experiments/pet_cat1/models/edlora_model-latest.pth" "experiments/dog6/models/edlora_model-latest.pth" \
--start_seed 0 \
--batch_size 4 \
--n_batches 1 \
--use_loss_mask \
--visualization \
--mask_update_interval 10
""".format(task_name, task_name, 1, task_name, 2, task_name, 3)
print(cmd)
# subprocess.run(cmd.replace("\n", "").split(" "))
