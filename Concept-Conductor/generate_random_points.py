import sys
import random
import subprocess
from PIL import Image, ImageDraw

task_name = "dog-cat-dog_{}".format(sys.argv[1])
img_width = 768 #512
img_height = 768 #512
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


if len(sys.argv) > 2 and isinstance(eval(sys.argv[2]), list):
	bboxes = eval(sys.argv[2])

margin_h = img_height//3
margin_w = img_width//(n_concept+1)
while len(bboxes) < n_concept:
	center = [
			  # random.randint(margin_w, img_height-1-margin_w), 
			  # random.randint(
			  # 	max(margin_w, img_width//n_concept*len(bboxes)), 
			  # 	min(img_width-margin_w, img_width//n_concept*(len(bboxes)+1))
			  # ),
			  int(img_width//n_concept*(len(bboxes)+0.5)),
			  random.randint(img_height//2, img_height-1-margin_h),
			  ]
	h = random.randint(margin_h, img_height//2-1)
	w = img_width//(n_concept+1) #random.randint(img_width//(n_concept+1), img_width//n_concept)
	bbox = [
			max(0, center[0]-w//2), max(0, center[1]-h//2), 
			min(img_width-1, center[0]+w//2), min(img_height-1, center[1]+h//2), 
		]
	print(
			  	max(margin_w, img_width//n_concept*len(bboxes)), 
			  	min(img_width-margin_w, img_width//n_concept*(len(bboxes)+1)))
	print(center, w, h,  bbox, bboxes)

	flag = False
	for other in bboxes:
		flag = collide(bbox, other)
		if flag: break
	if not flag: bboxes.append(bbox)

random.shuffle(bboxes)
for i, bbox in enumerate(bboxes):
	mask = create_and_save_mask([bbox], "examples/{}_{}.png".format(task_name, i+1))
create_and_save_mask(bboxes, "examples/{}_mask.png".format(task_name))


cmd = """
python sample.py \
--config_file dog_cat_dog_config.yaml \
--ref_image_path examples/{}_mask.png \
--ref_mask_paths examples/{}_{}.png examples/{}_{}.png examples/{}_{}.png \
--height {} \
--width {} \
--outroot outputs/{}
""".format(task_name, task_name, 1, task_name, 2, task_name, 3, img_height, img_width, task_name)
print(cmd)
subprocess.run(cmd.replace("\n", "").split(" "))
