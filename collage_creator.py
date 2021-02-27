from anytree import NodeMixin, RenderTree
import os
import glob
import random
import cv2 as cv
from PIL import Image
import numpy as np
from bisect import bisect_left

# configuration
INPUT_DIR = r'/Users/arturschmidt/Documents/Bilder/final_pictures_summary'
OUTPUT_FILE_NAME = 'collage'
OUTPUT_FILE_EXTENSION = '.png'
TARGET_ASPECT_RATIO = float(16)/9
ASPECT_RATIO_TOLERANCE = 0.05
CANVAS_TARGET_HEIGHT = 5500
PNG_COMPRESSION = 9
COLLAGE_COUNT = 3

# constants
ALIGNMENT = ['left', 'right']
SUPPORTED_IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png']
SPLIT_TYPES = ['H', 'V']


class ImageClass:
    def __init__(self, image_path):
        self.path = image_path
        self.width, self.height = Image.open(image_path).size
        self.aspect_ratio = float(self.width) / self.height

    def to_cv_image(self):
        # load image ignoring exif orientation and transparency
        image_cv = cv.imread(self.path, flags=cv.IMREAD_COLOR+cv.IMREAD_IGNORE_ORIENTATION)
        return image_cv

    def to_resized_cv_image(self, target_height, target_width):
        image_cv = self.to_cv_image()
        height_scale_factor = target_height / self.height
        width_scale_factor = target_width / self.width
        image_cv = cv.resize(image_cv, None, fx=width_scale_factor, fy=height_scale_factor, interpolation=cv.INTER_AREA)
        return image_cv


class TreeNode(NodeMixin):
    def __init__(self, parent=None, children=None, alignment=''):
        super(TreeNode, self).__init__()
        self.parent = parent
        if children:
            self.children = children
        else:
            self.children = list()
        self.alignment = alignment
        self.split_type = ''
        self.aspect_ratio = 0
        self.target_aspect_ratio = 0
        self.origin_x = 0
        self.origin_y = 0
        self.target_height = 0
        self.target_width = 0
        self.image_obj = None

    def assign_image(self, image_obj):
        self.image_obj = image_obj
        self.aspect_ratio = image_obj.aspect_ratio


class Tree:
    def __init__(self, canvas_target_aspect_ratio, image_object_list):
        self.converged = False
        self.adjustment_count = 0
        while not self.converged:
            self.root_node = TreeNode()
            self.root_node.target_aspect_ratio = canvas_target_aspect_ratio
            self.image_object_list = image_object_list
            self.n = len(image_object_list)
            self.tree_nodes = [self.root_node]
            self.recursively_assign_images(self.image_object_list, self.root_node.target_aspect_ratio, self.n,
                                           self.root_node)

            # adjust the tree a maximum of 100 times or until the canvas aspect ratio (root.target.aspect_ratio)
            # is acceptable as defined by ASPECT_RATIO_TOLERANCE
            self.adjustment_count = 0
            for i in range(100):
                self.recursively_calc_aspect_ratio(self.root_node)
                if abs(self.root_node.target_aspect_ratio - self.root_node.aspect_ratio) <= \
                        ASPECT_RATIO_TOLERANCE*self.root_node.target_aspect_ratio:
                    self.converged = True
                    break
                self.adjust_tree(self.root_node)
                self.adjustment_count += 1
        print('{} tree adjustment(s) made'.format(self.adjustment_count))
        self.set_canvas_dimensions()
        self.calc_image_positions_top_down()

    def recursively_assign_images(self, image_object_list, target_aspect_ratio, n, tree_node):
        if n == 1:
            index = take_closest([image_obj.aspect_ratio for image_obj in image_object_list],
                                 target_aspect_ratio)
            tree_node.assign_image(image_object_list[index])
            del image_object_list[index]
        elif n == 2:
            tree_node.split_type = random.choice(SPLIT_TYPES)
            i, j = find_image_pair(image_object_list, target_aspect_ratio, tree_node.split_type)
            # print index_pair
            # print len(image_object_list)
            left_node = TreeNode(parent=tree_node, children=None, alignment='left')
            left_node.assign_image(image_obj_list[i])
            right_node = TreeNode(parent=tree_node, children=None, alignment='right')
            right_node.assign_image(image_obj_list[j])
            self.tree_nodes.append(left_node)
            self.tree_nodes.append(right_node)
            del image_object_list[j]
            del image_object_list[i]
        else:
            tree_node.split_type = random.choice(SPLIT_TYPES)
            new_ns = [n/2, n - n/2]
            n_left = random.choice(new_ns)
            n_right = new_ns[not new_ns.index(n_left)]
            left_node = TreeNode(parent=tree_node, children=None, alignment='left')
            right_node = TreeNode(parent=tree_node, children=None, alignment='right')
            if tree_node.split_type == 'V':
                left_node.target_aspect_ratio = target_aspect_ratio / 2
                right_node.target_aspect_ratio = target_aspect_ratio / 2
            else:
                left_node.target_aspect_ratio = target_aspect_ratio * 2
                right_node.target_aspect_ratio = target_aspect_ratio * 2
            self.tree_nodes.append(left_node)
            self.tree_nodes.append(right_node)
            self.recursively_assign_images(image_object_list, left_node.target_aspect_ratio, n_left, left_node)
            self.recursively_assign_images(image_object_list, right_node.target_aspect_ratio, n_right, right_node)

    def recursively_calc_aspect_ratio(self, tree_node):
        if not tree_node.is_leaf:
            children_aspect_ratios = [self.recursively_calc_aspect_ratio(child) for child in tree_node.children]
            if tree_node.split_type == 'V':
                tree_node.aspect_ratio = sum(children_aspect_ratios)
            else:
                tree_node.aspect_ratio = 1 / sum([1 / aspect_ratio for aspect_ratio in children_aspect_ratios])
        return tree_node.aspect_ratio

    def adjust_tree(self, tree_node):
        threshold = 1+ASPECT_RATIO_TOLERANCE
        if not tree_node.is_leaf:
            if tree_node.aspect_ratio > tree_node.target_aspect_ratio * threshold:
                tree_node.split_type = 'H'
            elif tree_node.aspect_ratio < tree_node.target_aspect_ratio / threshold:
                tree_node.split_type = 'V'
        if tree_node.split_type == 'V':
            for child in tree_node.children:
                child.target_aspect_ratio = tree_node.target_aspect_ratio / 2
        else:
            for child in tree_node.children:
                child.target_aspect_ratio = tree_node.target_aspect_ratio * 2
        for child in tree_node.children:
            self.adjust_tree(child)

    def set_canvas_dimensions(self, height=CANVAS_TARGET_HEIGHT, width=0):
        if self.root_node.aspect_ratio == 0:
            print('Error: Root aspect ratio not set')
        else:
            if height != 0:
                self.root_node.target_height = int(round(height))
                self.root_node.target_width = int(round(height*self.root_node.aspect_ratio))
            elif width != 0:
                self.root_node.target_width = int(round(width))
                self.root_node.target_height = int(round(width/self.root_node.aspect_ratio))
            else:
                print('Error: Height and width are both 0')

    def calc_image_positions_top_down(self):
        for tree_node in self.tree_nodes[1:]:
            if tree_node.parent.split_type == 'V':
                tree_node.target_height = tree_node.parent.target_height
                tree_node.target_width = tree_node.target_height * tree_node.aspect_ratio
            else:
                tree_node.target_width = tree_node.parent.target_width
                tree_node.target_height = tree_node.target_width / tree_node.aspect_ratio
            if tree_node.alignment == 'left':
                tree_node.origin_x = tree_node.parent.origin_x
                tree_node.origin_y = tree_node.parent.origin_y

            else:
                if tree_node.parent.split_type == 'H':
                    tree_node.origin_y = \
                        tree_node.parent.origin_y + tree_node.parent.target_height - tree_node.target_height
                    tree_node.origin_x = tree_node.parent.origin_x
                else:
                    tree_node.origin_x = \
                        tree_node.parent.origin_x + tree_node.parent.target_width - tree_node.target_width
                    tree_node.origin_y = tree_node.parent.origin_y

            tree_node.origin_x = int(round(tree_node.origin_x))
            tree_node.origin_y = int(round(tree_node.origin_y))
            assert tree_node.target_width != 0
            assert tree_node.target_height != 0


def find_image_pair(image_object_list, target_aspect_ratio, split_type):
    left_pointer = 0
    right_pointer = len(image_object_list) - 1
    i = left_pointer
    j = right_pointer
    left_aspect_ratio = image_object_list[left_pointer].aspect_ratio
    right_aspect_ratio = image_object_list[right_pointer].aspect_ratio
    if split_type == 'H':
        left_aspect_ratio = 1 / left_aspect_ratio
        right_aspect_ratio = 1 / right_aspect_ratio
        target_aspect_ratio = 1 / target_aspect_ratio
    minimal_error = abs(left_aspect_ratio + right_aspect_ratio - target_aspect_ratio)
    while left_pointer < right_pointer:
        current_error = abs(left_aspect_ratio + right_aspect_ratio - target_aspect_ratio)
        if left_aspect_ratio + right_aspect_ratio > target_aspect_ratio:
            if current_error < minimal_error:
                minimal_error = current_error
                i = left_pointer
                j = right_pointer
            right_pointer -= 1
        elif left_aspect_ratio + right_aspect_ratio <= target_aspect_ratio:
            if current_error < minimal_error:
                minimal_error = current_error
                i = left_pointer
                j = right_pointer
            left_pointer += 1
        else:
            break
    return i, j


def take_closest(input_list, input_number):
    """
    Assumes myList is sorted. Returns index of closest value to myNumber.
    If two numbers are equally close, return the index of the smallest number.
    Source: https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
    """
    pos = bisect_left(input_list, input_number)
    if pos == 0:
        return pos
    elif pos == len(input_list):
        return pos - 1
    else:
        if input_list[pos] - input_number < input_number - input_list[pos - 1]:
            return pos
        else:
            return pos - 1


if __name__ == '__main__':
    # set image directory
    image_dir_path = INPUT_DIR
    # construct filters from supported image extensions
    filters = [os.path.join(image_dir_path, '*.'+extension) for extension in SUPPORTED_IMAGE_EXTENSIONS]
    # list for storing found images
    image_file_paths = list()
    # populate image_file_paths with image paths
    for image_filter in filters:
        for image in glob.glob(image_filter):
            image_file_paths.append(image)
    image_count = len(image_file_paths)

    # main loop for each collage
    for collage_index in range(COLLAGE_COUNT):
        # get image resolution and compute aspect ratio
        image_obj_list = [ImageClass(path) for path in image_file_paths]
        # set sorting function
        try:
            import operator
        except ImportError:
            key_function = lambda key: key.aspect_ratio
        else:
            key_function = operator.attrgetter('aspect_ratio')
        # pre-sort by ascending aspect_ratio
        image_obj_list.sort(key=key_function)

        # create tree and assign images
        my_tree = Tree(TARGET_ASPECT_RATIO, image_obj_list)

        # render tree and count leaves
        leaf_counter = 0
        for pre, _, node in RenderTree(my_tree.root_node):
            tree_str = '{} {} is: {:.2f} target: {:.2f}'.format(pre, node.split_type, node.aspect_ratio,
                                                                node.target_aspect_ratio)
            if node.is_leaf:
                leaf_counter += 1
                try:
                    tree_str += ' {}'.format(node.image_obj.path)
                except UnicodeDecodeError:
                    # print node.image_obj.path
                    tree_str += 'path could not be decoded'
            # print tree_str.ljust(100), node.is_leaf, node.depth

        # make sure the tree has as many leaves as the image paths provided
        assert leaf_counter == len(image_file_paths)

        # print canvas dimensions
        print('canvas dimensions %d %d' % (my_tree.root_node.target_height, my_tree.root_node.target_width))

        # create empty canvas
        canvas_numpy = np.zeros((my_tree.root_node.target_height, my_tree.root_node.target_width, 3), dtype=np.float32)

        # fill canvas
        print('Filling canvas')

        for node in my_tree.tree_nodes:
            if node.is_leaf:
                image_np = node.image_obj.to_resized_cv_image(node.target_height, node.target_width)
                try:
                    target_height, target_width, _ = canvas_numpy[node.origin_y:node.origin_y+image_np.shape[0],
                                                                  node.origin_x:node.origin_x+image_np.shape[1],
                                                                  :].shape
                    source_height, source_width, _ = image_np.shape
                    height_diff = target_height - source_height
                    width_diff = target_width - source_width
                    node.origin_y += height_diff
                    node.origin_x += width_diff
                    canvas_numpy[node.origin_y:node.origin_y+image_np.shape[0],
                                 node.origin_x:node.origin_x+image_np.shape[1], :] = image_np
                except Exception as e:
                    print()
                    print(e)
                    print('Image insertion failed for %s', node.image_obj.path)
                    print('Target Shape:', canvas_numpy[node.origin_y:node.origin_y + image_np.shape[0],
                                           node.origin_x:node.origin_x + image_np.shape[1], :].shape)
                    print('Source Shape:', image_np.shape)
        # write collage to disk
        cv.imwrite(OUTPUT_FILE_NAME+'_'+str(collage_index)+OUTPUT_FILE_EXTENSION, canvas_numpy,
                   [cv.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])

        print('Collage %d done' % collage_index)

    print('All done')
