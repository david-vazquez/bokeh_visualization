import time
import numpy as np
from skimage import io
from skimage.transform import rotate
from bokeh.plotting import figure, output_server, cursession, show, hplot, gridplot, vplot
from bokeh.palettes import RdBu6 as palette


class visualize_data:
    def __init__(self, title, legends, n_images_train=0, n_images_test=0, n_cols_images=3, axisx="Epochs",
                 axisy="Values", line_width=4, alpha=0.8, line_dash=[4, 4], plot_width=800, plot_height=400):

        # Save some parameters
        self.n_lines_plot = len(legends)
        self.n_images_train = n_images_train
        self.n_images_test = n_images_test

        # prepare output to server
        output_server(title)
        cursession().publish()

        # Create the figure plot
        if self.n_lines_plot > 0:
            p_plot, self.data_plot = self.create_figure_plot(title, legends, axisx=axisx, axisy=axisy,
                                                             line_width=line_width, alpha=alpha, line_dash=line_dash,
                                                             plot_width=plot_width, plot_height=plot_height)

        if n_images_train > 0:
            # Create black images as initialization
            img = np.zeros((64, 256, 3))
            img_batch = []
            for i in xrange(n_images_train):
                img_batch.append(img)

            # Create the training image grid
            p_images_train, self.data_img_train = self.create_grid_images(img_batch, name='img_train', n_cols=n_cols_images)

        if n_images_test > 0:
            # Create black images as initialization
            img = np.zeros((64, 256, 3))
            img_batch = []
            for i in xrange(n_images_test):
                img_batch.append(img)

            # Create the testing image grid
            p_images_test, self.data_img_test = self.create_grid_images(img_batch, name='img_test', n_cols=n_cols_images)

        # Create a vertical grid with the plot and the train and test images
        if self.n_lines_plot > 0 and n_images_train > 0 and n_images_test > 0:
            p = vplot(p_plot, p_images_train, p_images_test)
        elif self.n_lines_plot > 0 and n_images_train > 0 and n_images_test <= 0:
            p = vplot(p_plot, p_images_train)
        else:
            print 'ERROR: Not implemented combination. Please, do it!'

        # Show the plot
        show(p)


    # Convert the image to Bokeh format
    def convert_image(self, img):
        # Get image shape
        im_size = img.shape

        # Rotate the image 180 degrees
        img = rotate(img, 180)

        # Adjust the image to 0-255 range
        if np.max(img) <= 1:
            img *= 255.

        # Create an array of RGBA data in the format required by Bokeh
        image = np.empty((im_size[0], im_size[1]), dtype=np.uint32)
        view = image.view(dtype=np.uint8).reshape((im_size[0], im_size[1], 4))

        # Copy the data
        view[:, :, 0:3] = img[:, :, 0:3]
        view[:, :, 3] = 255

        # return the bokeh formated image
        return image


    # Create the figure to show an image
    def create_figure_image(self, img, name, title=None):
        # Get image size
        im_size = img.shape

        # Convert the image to Bokeh format
        image = self.convert_image(img)

        # Create the figure to show the image there
        p = figure(plot_width=im_size[1], plot_height=im_size[0], x_range=(0, im_size[1]),
                   y_range=(0, im_size[0]), title=title)

        # Show image
        r = p.image_rgba(image=[image], x=[0], y=[0], dw=[im_size[1]], dh=[im_size[0]], name=name)

        # Remove axis, grid and border lines
        p.xaxis.visible = None
        p.yaxis.visible = None
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.min_border_left = 10
        p.min_border_right = 10
        p.min_border_top = 10
        p.min_border_bottom = 10

        # return image plot and the data sources
        return p, r.data_source


    # Create a grid with the figure to the image plots
    def create_grid_images(self, img_list, name, n_cols=3):
        # Create a list with the image figures and image data sources
        p_images = []
        data_sources = []
        for i in xrange(len(img_list)):
            p_img, data_img = self.create_figure_image(img_list[i], name=name + str(i))
            p_images.append(p_img)
            data_sources.append(data_img)

        # Add the needed Nones so we can transform in a square matrix
        for i in xrange(n_cols - len(img_list) % n_cols):
            p_images.append(None)

        # Reshape as a matrix (List format)
        p_images = np.asarray(p_images).reshape(-1, 3).tolist()

        # Plot the images
        p = gridplot(p_images)

        # return the grid
        return p, data_sources


    # Create the figure to show the error plot
    def create_figure_plot(self, title, legends, axisx="Epochs", axisy="Values", line_width=4, alpha=0.8,
                           line_dash=[4, 4], plot_width=800, plot_height=400):
        # Create the figure
        p = figure(plot_width=plot_width, plot_height=plot_height, title=title)

        # Set axis labels
        p.xaxis.axis_label = axisx
        p.yaxis.axis_label = axisy

        # Create the lines to plot
        nan = float('nan')
        for i in xrange(len(legends)):
            p.line([nan], [0], name='line' + str(i), legend=legends[i], color=palette[i], alpha=alpha,
                   line_width=line_width, line_dash=line_dash)

        # Get data
        data_sources = []
        for i in xrange(len(legends)):
            data_sources.append(p.select(dict(name='line' + str(i)))[0].data_source)
        return p, data_sources


    # Appends data to the plot
    def append_data_plot(self, values):
        if self.n_lines_plot != len(values):
            print 'ERROR: Number of values incorrect. It should be: ' + str(self.n_lines_plot) + ' and got: ' \
                  + str(len(values))
            exit()
        else:
            for i in xrange(len(self.data_plot)):
                self.data_plot[i].data["x"].append(len(self.data_plot[i].data["x"]))
                self.data_plot[i].data["y"].append(values[i])
                cursession().store_objects(self.data_plot[i])

    # Change images of the plots
    def set_data_images(self, data_img, img_list):
        if len(data_img) != len(img_list):
            print 'ERROR: Number of images incorrect. It should be: ' + str(len(data_img)) + ' and got: ' \
                  + str(len(img_list))
            exit()
        else:
            for i in xrange(len(data_img)):
                data_img[i].data['image'] = [self.convert_image(img_list[i])]
                cursession().store_objects(data_img[i])

    def set_images_train(self, img_list):
        self.set_data_images(self.data_img_train, img_list)

    def set_images_test(self, img_list):
        self.set_data_images(self.data_img_test, img_list)


if __name__ == "__main__":

    # Define the lines to plot
    legends = ["Training loss", "Training accuracy", "Training jaccard", "Validation loss", "Validation accuracy",
               "Validation jaccard"]

    n_images_train = 7
    n_images_test = 7

    # Create a bokeh plot
    vis_data = visualize_data("_Experiment_2", legends, n_images_train=n_images_train, n_images_test=n_images_test)


    # Update the plot
    while True:
        # Compute the values to plot
        values = np.random.rand(len(legends)) + xrange(len(legends))

        # Add data to the plot
        vis_data.append_data_plot(values)

        # Read the images train
        indx = np.random.random_integers(0, 15, n_images_train)
        img_batch_train = []
        for i in xrange(n_images_train):
            print 'out' + str(indx[i]) + '.png'
            img = io.imread('out'+ str(indx[i]) + '.png')
            img = io.imread('out0.png')
            img_batch_train.append(img)

        #time.sleep(0.5)
        # Read the images test
        img = io.imread('out1.png')
        img_batch_test = []
        for i in xrange(n_images_test):
            img_batch_test.append(img)

        # Change training images of the plot
        vis_data.set_images_train(img_batch_train)

        # Change testing images of the plot
        vis_data.set_images_test(img_batch_test)