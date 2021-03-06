import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
#import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
from skimage.transform import warp, AffineTransform

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))

def create_image_labels(image, num_classes):
    background_color = np.array([255, 0, 0])
    main_road_color = np.array([255, 0, 255])
    background_color2 = np.array([255, 255, 255])
    gt_bg = np.logical_or(np.all(image == background_color, axis=2), np.all(image == background_color2, axis=2))
    #gt_bg = np.all(image == background_color2, axis=2)
    gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
    if num_classes == 3:
        gt_main_road = np.all(image == main_road_color, axis=2)
        gt_main_road = gt_main_road.reshape(*gt_main_road.shape, 1)
        #gt_main_road = np.logical_and(gt_main_road, np.invert(gt_bg)) # make sure one hot
        gt_side_road = np.logical_or(gt_bg, gt_main_road)
        gt_side_road = np.invert(gt_side_road)
        gt_bg = gt_bg.astype(int) * 255
        gt_main_road = gt_main_road.astype(int) * 255
        gt_side_road = gt_side_road.astype(int) * 255
        imageLabels = np.concatenate((gt_bg, gt_main_road, gt_side_road), axis=2)
    else:
        imageLabels = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
    return imageLabels

save_augumented = True
def gen_batch_function(data_folder, image_shape, num_classes=2, augmentation_args=None):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :param augmentation_args: the image augmentation arguments, None for no augmentation
    :return: a generator that returns a sequence of (image, label) 
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        global save_augumented
        save_augumented = augmentation_args is not None
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        random.shuffle(image_paths)
        if save_augumented:
            augmented_folder = os.path.join(data_folder, 'augmented', time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
            os.makedirs(augmented_folder)
            os.makedirs(augmented_folder + "-labels")
        if augmentation_args is not None:
            random.seed(augmentation_args['seed'] if 'seed' in augmentation_args else 0)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]
                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
                if augmentation_args is not None:
                    scalex = 1.0 + random.randint(-100, 100) / 100.0 * (augmentation_args['scale'] if 'scale' in augmentation_args else 0)
                    scaley = 1.0 + random.randint(-100, 100) / 100.0 * (augmentation_args['scale'] if 'scale' in augmentation_args else 0)
                    rotation = random.randint(-100, 100) / 100.0 * (augmentation_args['rotate'] if 'rotate' in augmentation_args else 0)
                    shear = random.randint(-100, 100) / 100.0 * (augmentation_args['shear'] if 'shear' in augmentation_args else 0)
                    translationx = int(random.randint(-100, 100) / 100.0 * (augmentation_args['translate'] if 'translate' in augmentation_args else 0))
                    translationy = int(random.randint(-100, 100) / 100.0 * (augmentation_args['translate'] if 'translate' in augmentation_args else 0))
                    transformation = AffineTransform(scale=(scalex, scaley), rotation=rotation, shear=shear, translation=(translationx, translationy)).inverse
                    image = warp(image, transformation, output_shape=image.shape, order=1, mode='constant', cval=255, preserve_range=True).astype(int)
                    gt_image = warp(gt_image, transformation, output_shape=image.shape, order=0, mode='constant', cval=255, preserve_range=True).astype(int)
                    if save_augumented:
                        scipy.misc.imsave(os.path.join(augmented_folder, os.path.basename(image_file)), image)
                        scipy.misc.imsave(os.path.join(augmented_folder, os.path.basename(gt_image_file)), gt_image)
                gt_image = create_image_labels(gt_image, num_classes)
                if save_augumented and num_classes == 3:
                    scipy.misc.imsave(os.path.join(augmented_folder + "-labels", os.path.basename(gt_image_file)), gt_image)
                gt_images.append(gt_image)
                images.append(image)
            yield np.array(images), np.array(gt_images)
        save_augumented = False
    return get_batches_fn

def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape, num_classes=2):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        main_mask = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        main_mask = (main_mask > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(main_mask, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        if num_classes == 3:
            side_mask = im_softmax[0][:, 2].reshape(image_shape[0], image_shape[1])
            side_mask = (side_mask > 0.5).reshape(image_shape[0], image_shape[1], 1)
            mask = np.dot(side_mask, np.array([[0, 0, 255, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")
            street_im.paste(mask, box=None, mask=mask)
        yield os.path.basename(image_file), np.array(street_im)

def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, num_classes=2):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape, num_classes)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

def save_model(sess, output_dir, name, step) :
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(output_dir, name), global_step=step)

def load_model(sess, model):
    saver = tf.train.import_meta_graph(model + '.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    return graph.get_tensor_by_name('segmentation_logits:0'), graph.get_tensor_by_name('image_input:0'), graph.get_tensor_by_name('keep_prob:0')
