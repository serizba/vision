import matplotlib.pyplot as plt

def grid(imgs, titles, rows, cols):
	f, axes = plt.subplots(rows, cols, figsize=(15, 15))
	axes = axes.flatten()
	for i in range(rows * cols):
	    axes[i].title.set_text(titles[i])
	    if imgs[i].ndim == 2:
	    	axes[i].imshow(imgs[i], cmap='gray')
	    else:
	    	axes[i].imshow(imgs[i])
	    axes[i].axis('off')
	plt.show()
