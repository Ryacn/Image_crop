# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 21:42:09 2019

@author: ys
"""

import crop_morphology
import cv2
import numpy as np
from scipy import stats

from enum import Enum
class PageSetting(Enum): #PageSetting refers to location of page number
    TOP = 0
    BOTTOM = 1
    NONE = 2
    BOTH = 3
    
def seperate_main_Content(img, page_setting):
    #trims out pages (ex. trims title and page numbers)
    pixel_padding = 5
    padding = 5
    pixel_thresh = 100
    image_thresh = 100
    
    img = np.pad(img, ((padding,padding), (padding,padding), (0,0)), 'constant', constant_values=255)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    hist = cv2.reduce(threshed,1, cv2.REDUCE_AVG).reshape(-1)
    if sum(1 for value in hist if value > pixel_thresh) > image_thresh:
        return None
    
    th = 1
    H,W = img.shape[:2]
    uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
    lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]
    
    if len(uppers) > 3 and len(lowers) > 3:
        if page_setting == PageSetting.BOTTOM:
            img = (img[uppers[0] - pixel_padding:lowers[-2] + pixel_padding,::])
            gray = (gray[uppers[0] - pixel_padding:lowers[-2] + pixel_padding,::])
        elif page_setting is PageSetting.TOP:
            if uppers[2] - uppers[1] > 5:
                img = (img[uppers[2] - pixel_padding:lowers[-1] + pixel_padding,::])
                gray = (gray[uppers[1] - pixel_padding:lowers[-1] + pixel_padding,::])
            else:
                img = (img[uppers[1] - pixel_padding:lowers[-1] + pixel_padding,::])
                gray = (gray[uppers[1] - pixel_padding:lowers[-1] + pixel_padding,::])
        elif page_setting is PageSetting.NONE:
            img = (img[uppers[0] - pixel_padding:lowers[-1] + pixel_padding,::])
            gray = (gray[uppers[0] - pixel_padding:lowers[-1] + pixel_padding,::])
        elif page_setting is PageSetting.BOTH:
            img = (img[uppers[1] - pixel_padding:lowers[-2] + pixel_padding,::])
            gray = (gray[uppers[1] - pixel_padding:lowers[-2] + pixel_padding,::])
    else:
        return None
    
    gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    hist = cv2.reduce(threshed,1, cv2.REDUCE_AVG).reshape(-1)
    
    th = 1
    H,W = gray.shape[:2]
    uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
    lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]
    
    img = (img[::, uppers[0] - pixel_padding:lowers[-1] + pixel_padding])
    return img

from enum import Enum
class PageDirection(Enum):
    VERTICAL = 0
    HORIZONTAL = 1
def seperate_lines(img, read_direction):
    #coverts block of text images to lines of text images
    padding = 5
    buffer = 2
    
    if read_direction is PageDirection.VERTICAL:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = np.pad(img, ((padding,padding), (padding,padding), (0,0)), 'constant', constant_values=255)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    hist = cv2.reduce(threshed,1, cv2.REDUCE_AVG).reshape(-1)
    
    th = 1
    H,W = img.shape[:2]
    uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
    lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]

    line_elements = []
    my_iter = iter(range(len(uppers)))
    for i in my_iter:
        if read_direction is PageDirection.VERTICAL:
            if i < len(uppers) - 1:
                if lowers[i] - uppers[i] + buffer * 2 > (lowers[i+1] - uppers[i+1] + buffer *2) * 1.5:
                    line_elements.append(cv2.rotate(img[uppers[i]-buffer:lowers[i+1]+buffer,::],cv2.ROTATE_90_COUNTERCLOCKWISE))
                    next(my_iter,None)
                else:
                    line_elements.append(cv2.rotate(img[uppers[i]-buffer:lowers[i]+buffer,::],cv2.ROTATE_90_COUNTERCLOCKWISE))
            else:    
                line_elements.append(cv2.rotate(img[uppers[i]-buffer:lowers[i]+buffer,::],cv2.ROTATE_90_COUNTERCLOCKWISE))
        else:
            if i < len(uppers) - 1:
                if lowers[i] - uppers[i] + buffer * 2 > (lowers[i+1] - uppers[i+1] + buffer *2) * 1.5:
                    line_elements.append(img[uppers[i]-buffer:lowers[i+1]+buffer,::])
                    next(my_iter,None)
                else:
                    line_elements.append(img[uppers[i]-buffer:lowers[i]+buffer,::])
            else:    
                line_elements.append(img[uppers[i]-buffer:lowers[i]+buffer,::])
    if read_direction is PageDirection.VERTICAL:
        line_elements = line_elements[::-1]
    return line_elements

def seperate_words(img, read_direction):
    #converts line of text to words of text
    padding = 3
    buffer = 2
    
    if read_direction is PageDirection.HORIZONTAL:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = np.pad(img, ((padding,padding), (padding,padding), (0,0)), 'constant', constant_values=255)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    hist = cv2.reduce(threshed,1, cv2.REDUCE_AVG).reshape(-1)
    
    th = 1
    H,W = img.shape[:2]
    uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
    lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]

    word_elements = []
    my_iter = iter(range(len(uppers)))
    for i in my_iter:
        if read_direction is PageDirection.HORIZONTAL:
            if i < len(uppers) - 1:
                if lowers[i] - uppers[i] + buffer * 2 > (lowers[i+1] - uppers[i+1] + buffer *2) * 1.5:
                    word_elements.append(cv2.rotate(img[uppers[i]-buffer:lowers[i+1]+buffer,::],cv2.ROTATE_90_COUNTERCLOCKWISE))
                    next(my_iter,None)
                else:
                    word_elements.append(cv2.rotate(img[uppers[i]-buffer:lowers[i]+buffer,::],cv2.ROTATE_90_COUNTERCLOCKWISE))
            else:    
                word_elements.append(cv2.rotate(img[uppers[i]-buffer:lowers[i]+buffer,::],cv2.ROTATE_90_COUNTERCLOCKWISE))
        else:
            if i < len(uppers) - 1:
                if lowers[i] - uppers[i] + buffer * 2 > (lowers[i+1] - uppers[i+1] + buffer *2) * 1.5:
                    word_elements.append(img[uppers[i]-buffer:lowers[i+1]+buffer,::])
                    next(my_iter,None)
                else:
                    word_elements.append(img[uppers[i]-buffer:lowers[i]+buffer,::])
            else:    
                word_elements.append(img[uppers[i]-buffer:lowers[i]+buffer,::])
    return word_elements

#experimentation after this point

import scipy.misc
import shutil
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

def process_images(img_dirs, save_dir, new_page_dimension, page_setting, read_direction):#links of imgs to lines of image
    height, width = new_page_dimension
    pixel_height = 0
    pixel_width = 0
    page_number = 1
    for img_dir in img_dirs:
        new_image_pages = []
        img = imread(img_dir)
        img = seperate_main_Content(img, page_setting)
        if img is None:
            shutil.copyfile(img_dir, save_dir + '/' + str(page_number) + '.jpg')
            page_number += 1
            continue
        lines = seperate_lines(img, read_direction)
        words = []
        if read_direction is PageDirection.VERTICAL:
            for line in lines:
                words += seperate_words(line, read_direction)
            new_lines = []
            cur_word_index = 0
            if pixel_height == 0 or pixel_width == 0:
                pixel_height = max(words[:int(len(words)*.3)] ,key=(lambda x: np.size(x,0))).shape[0] * height
                pixel_width = max(words[:int(len(words)*.3)] ,key=(lambda x: np.size(x,1))).shape[1] * width
            while cur_word_index < len(words):
#                words_in_interest = words[cur_word_index:min(len(words),cur_word_index+height)]
                line_size_pixel = 0
                line_size_index = 0
                while line_size_pixel < pixel_height and cur_word_index + line_size_index < len(words):
                    line_size_pixel += np.size(words[cur_word_index + line_size_index],0)
                    line_size_index += 1
                words_in_interest = words[cur_word_index:min(len(words),cur_word_index+line_size_index)]
                resize_value = max(words_in_interest ,key=(lambda x: np.size(x,1))).shape[1]
                for word_index in range(len(words_in_interest)):
                    padding_value = resize_value - np.size(words_in_interest[word_index],1)
                    if padding_value > 0:
                        words_in_interest[word_index]=np.pad(words_in_interest[word_index], ((0,0),(0,padding_value),(0,0)), 'constant', constant_values=255)
                new_lines.append(np.vstack(tuple(words_in_interest)))
#                cur_word_index += height
                cur_word_index += line_size_index
            new_lines = new_lines[::-1]
            cur_line_index = 0
            while cur_line_index < len(new_lines):
                lines_in_interest = new_lines[cur_line_index:min(len(new_lines), cur_line_index+width)]
                resize_value = max(lines_in_interest ,key=(lambda x: np.size(x,0))).shape[0]
                for line_index in range(len(lines_in_interest)):
                    padding_value = resize_value - np.size(lines_in_interest[line_index],0)
                    if padding_value > 0:
                        lines_in_interest[line_index]=np.pad(lines_in_interest[line_index], ((0,padding_value),(0,0),(0,0)), 'constant', constant_values=255)
                new_image_pages.append(np.hstack(tuple(lines_in_interest)))
                cur_line_index += width
            new_image_pages = new_image_pages[::-1]
            for page in new_image_pages:
                Image.fromarray(page).save(save_dir + '/' + str(page_number) + '.jpg')
                page_number += 1
        elif read_direction is PageDirection.HORIZONTAL:
            for line in lines:
                words += seperate_words(line, read_direction)
            new_lines = []
            cur_word_index = 0
            while cur_word_index < len(words):
                words_in_interest = words[cur_word_index:min(len(words),cur_word_index+width)]
                resize_value = max(words_in_interest ,key=(lambda x: np.size(x,0))).shape[0]
                for word_index in range(len(words_in_interest)):
                    padding_value = resize_value - np.size(words_in_interest[word_index],0)
                    if padding_value > 0:
                        words_in_interest[word_index]=np.pad(words_in_interest[word_index], ((0,padding_value),(0,0),(0,0)), 'constant', constant_values=255)
                new_lines.append(np.hstack(tuple(words_in_interest)))
                cur_word_index += width
            cur_line_index = 0
            while cur_line_index < len(new_lines):
                lines_in_interest = new_lines[cur_line_index:min(len(new_lines), cur_line_index+height)]
                resize_value = max(lines_in_interest ,key=(lambda x: np.size(x,0))).shape[0]
                for line_index in range(len(lines_in_interest)):
                    padding_value = resize_value - np.size(lines_in_interest[line_index],0)
                    if padding_value > 0:
                        lines_in_interest[line_index]=np.pad(lines_in_interest[line_index], ((0,0),(0,padding_value),(0,0)), 'constant', constant_values=255)
                new_image_pages.append(np.vstack(tuple(lines_in_interest)))
                cur_line_index += height
            for page in new_image_pages:
                Image.fromarray(page).save(save_dir + '/' + str(page_number) + '.jpg')
                page_number += 1
    return new_image_pages

def process_images2(img_dirs, save_dir, new_page_dimension, page_setting, read_direction):
#    only center cut editing
    height, width = new_page_dimension
    pixel_height = 0
    pixel_width = 0
    page_number = 1
    for img_dir in img_dirs:
        new_image_pages = []
        img = imread(img_dir)
        img = seperate_main_Content(img, page_setting)
        if img is None:
            shutil.copyfile(img_dir, save_dir + '/' + str(page_number) + '.jpg')
            page_number += 1
            continue
        Image.fromarray(img).save(save_dir + '/' + str(page_number) + '.jpg')

def process_images3(img_dirs, save_dir, new_page_dimension, page_setting, read_direction):
#    reconstruction using only lines
    height, width = new_page_dimension
    pixel_height = 0
    pixel_width = 0
    page_number = 1
    for img_dir in img_dirs:
        new_image_pages = []
        img = imread(img_dir)
        img = seperate_main_Content(img, page_setting)
        if img is None:
            shutil.copyfile(img_dir, save_dir + '/' + str(page_number) + '.jpg')
            page_number += 1
            continue
        lines = seperate_lines(img, read_direction)
        lines = lines[::-1]
        resize_value = max(lines ,key=(lambda x: np.size(x,0))).shape[0]
        for line_index in range(len(lines)):
            padding_value = resize_value - np.size(lines[line_index],0)
            if padding_value > 0:
                lines[line_index]=np.pad(lines[line_index], ((0,0),(0,padding_value),(0,0)), 'constant', constant_values=255)
        Image.fromarray(np.hstack(tuple(lines))).save(save_dir + '/' + str(page_number) + '.jpg')
        page_number += 1
        
import math
def process_images4(img_dirs, save_dir, new_page_dimension, page_setting, read_direction):
#    reconstruction from modifying lines
    pixel_height, pixel_width = new_page_dimension
    padding = 5
    page_number = 1
    for img_dir in img_dirs:
        img = imread(img_dir)
        img = seperate_main_Content(img, page_setting)
        if img is None:
            shutil.copyfile(img_dir, save_dir + '/' + str(page_number) + '.jpg')
            page_number += 1
            continue
        lines = seperate_lines(img, read_direction)
        lines = lines[::-1]
        new_lines = []
        asdf = 0
        for line in lines:
            asdf=asdf+1
            new_line = np.pad(line, ((padding,padding), (padding,padding), (0,0)), 'constant', constant_values=255)
            gray = cv2.cvtColor(new_line, cv2.COLOR_BGR2GRAY)
            
            th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            
            hist = cv2.reduce(threshed,1, cv2.REDUCE_AVG).reshape(-1)
            
            th = 1
            H,W = new_line.shape[:2]
            uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
            lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]
            
            cut_index = 0
            temp_new_line = []
            
            for new_line_count in range(math.ceil(np.size(line,0)/pixel_height)):
                cut_index_to = 0
                while cut_index + cut_index_to < len(uppers) and cut_index + cut_index_to < len(lowers) and lowers[cut_index + cut_index_to] - uppers[cut_index] < pixel_height:
                    cut_index_to += 1
                cut_index_to -= 1
                temp_new_line.append(line[uppers[cut_index] - padding:lowers[cut_index+cut_index_to] + padding])
                cut_index += cut_index_to
                if cut_index > len(lowers):
                    break;
            for lines in temp_new_line[::-1]:
                new_lines.append(lines)
                
        new_pages = []
        
        new_image_pages_component = []
        new_width_pixel = 0
        for line in new_lines:
            if new_width_pixel < pixel_width:
                new_width_pixel += np.size(line,1)
                new_image_pages_component.append(line)
            else:
                new_width_pixel = 0
                resize_value = max(new_image_pages_component ,key=(lambda x: np.size(x,0))).shape[0]
                for line_index in range(len(new_image_pages_component)):
                    padding_value = resize_value - np.size(new_image_pages_component[line_index],0)
                    if padding_value > 0:
                        new_image_pages_component[line_index]=np.pad(new_image_pages_component[line_index], ((0,padding_value),(0,0),(0,0)), 'constant', constant_values=255)
                new_pages.append(np.hstack(tuple(new_image_pages_component)))
                new_image_pages_component = []
        if new_width_pixel is not 0:
            new_width_pixel = 0
            resize_value = max(new_image_pages_component ,key=(lambda x: np.size(x,0))).shape[0]
            for line_index in range(len(new_image_pages_component)):
                padding_value = resize_value - np.size(new_image_pages_component[line_index],0)
                if padding_value > 0:
                    new_image_pages_component[line_index]=np.pad(new_image_pages_component[line_index], ((0,padding_value),(0,0),(0,0)), 'constant', constant_values=255)
            new_pages.append(np.hstack(tuple(new_image_pages_component)))
            new_image_pages_component = []
        new_pages = new_pages[::-1]
        for page in new_pages:
            Image.fromarray(page).save(save_dir + '/' + str(page_number) + '.jpg')
            page_number += 1

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None
    
from matplotlib import pyplot as plt
import cv2
import glob
if __name__ == '__main__':
    process_images3(glob.glob('testfile1/*'),'testfile',(20,7), PageSetting.NONE, PageDirection.VERTICAL)
#    process_images4(['IMG_0011.jpg'],'testfile',(600,300), PageSetting.NONE, PageDirection.VERTICAL)

#print(sys.argv)
#if __name__ == '__main__':
#    if len(sys.argv) == 2 and '*' in sys.argv[1]:
#        files = glob.glob(sys.argv[1])
#        random.shuffle(files)
#    else:
#        files = sys.argv[1:]
#
#    for path in files:
#        out_path = path.replace('.jpg', '.crop.png')
#        #out_path = path.replace('.png', '.crop.png')  # .png as input
#        if os.path.exists(out_path): continue
#        try:
#            process_image(path, out_path)
#        except Exception as e:
#            print('%s %s' % (path, e))