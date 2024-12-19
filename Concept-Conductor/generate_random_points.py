import sys
import random
import subprocess
from PIL import Image, ImageDraw

task_name = "dog-cat-dog_{}".format(sys.argv[1])
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
	return (between(ax1, bx1, bx2) or between(bx1, ax1, ax2)) #and (between(ay1, by1, by2) or between(by1, ay1, ay2))

def create_and_save_mask(bboxes, save_fn):
	mask = Image.new("L", (img_height, img_width), 0)
	draw = ImageDraw.Draw(mask)
	for bbox in bboxes:
		draw.rectangle(bbox, fill=255)
	mask.save(save_fn)
	print("Save to {}".format(save_fn))


if sys.argc >= 2 and isinstance(eval(sys.argv[2]), list):
	bboxes = eval(sys.argv[2])

else:
	while len(bboxes) < n_concept:
		center = [random.randint(img_height//6, img_height*5//6-1), 
				  random.randint(img_width//6, img_width*5//6-1)]
		h = random.randint(img_height//5, img_height//2-1)
		w = random.randint(img_width//5, img_width//3-1)
		bbox = [center[0]-h//2, center[1]-w//2, center[0]+h//2, center[1]+w//2]
		print(bbox)

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
--config_file dog_cat_dog_config.yaml \
--ref_image_path examples/{}_mask.png \
--ref_mask_paths examples/{}_{}.png examples/{}_{}.png examples/{}_{}.png \
--outroot outputs/{}
""".format(task_name, task_name, 1, task_name, 2, task_name, 3, task_name)
print(cmd)
subprocess.run(cmd.replace("\n", "").split(" "))
