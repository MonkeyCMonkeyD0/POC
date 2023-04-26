import os
from PIL import Image
from pynput import keyboard
import time

import matplotlib.pyplot as plt


def main():

	dataset_dir = input("Enter dataset directory:")
	files = os.listdir(dataset_dir)

	if not os.path.exists(dataset_dir + 'removed_files/'):
		os.makedirs(dataset_dir + 'removed_files/')

	for file in files:
		if not os.path.isfile(dataset_dir + "/" + file):
			continue
		img = Image.open(dataset_dir + "/" + file)
		plt.imshow(img)
		plt.ion()
		plt.show()
		plt.pause(0.001)

		time.sleep(0.1)
		with keyboard.Events() as events:
			for event in events:
				if event.key == keyboard.Key.right:
					print("Keeping {}".format(file))
					break
				elif event.key == keyboard.Key.left:
					print("Removing {}".format(file))
					os.rename(dataset_dir + file, dataset_dir + "removed_files/" + file)
					break
				elif event.key == keyboard.Key.esc:
					print('Qutting the program')
					exit()
		
		plt.close()

if __name__ == '__main__':
	main()