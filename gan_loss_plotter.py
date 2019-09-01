import matplotlib.pyplot as plt
import pickle

file = open('gan_plot_data.txt','rb')
plot_data = pickle.load(file)
file.close()

def make_plot(plot_data):
	x = [x[0] for x in plot_data]
	y1 = [x[1] for x in plot_data]
	y2 = [x[2] for x in plot_data]
	fig, ax = plt.subplots()
	ax.grid(True, which='both')
	ax.plot(x,y1,label='Generator')
	ax.plot(x,y2,label='Discriminator')
	ax.spines['bottom'].set_position('zero')
	plt.xlabel('Iterations')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig('gan_training_plot.png')
	plt.show()
	plt.close()

make_plot(plot_data)