import cv2
import os
import numpy as np

class OpencvText():
    def __init__(self, folder='./', input_path='input/', output_path='output/'):
        self.folder = folder
        self.input_path = input_path
        self.output_path = output_path
    

    def write_text(self, img, text): # returns (extension of image, modified image)
        
        thickness = 2
        ratio_sin = 1

        hei, wid, _ = img.shape
        if hei <= 128 or wid <= 128: 
            thickness = 1
            ratio_sin = 0.1
        elif hei < 512 or wid < 512:
            thickness = 1
            ratio_sin = 0.3
        elif hei < 1024 or wid < 1024:
            thickness = 1
            ratio_sin = 0.5

        font_scale = 0.8 * ratio_sin
        rectangle_bgr = (255, 255, 255)
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, fontScale=font_scale, thickness=thickness)[0]
        text_offset_x = int(30 * ratio_sin)
        text_offset_y = img.shape[0] - int(25 * ratio_sin)
        # make the coords of the box with a small padding of five pixels
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + int(15 * ratio_sin), text_offset_y - text_height - int(15 * ratio_sin)))
        cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
        cv2.putText(img, text, (text_offset_x, text_offset_y-int(10 * ratio_sin)), cv2.FONT_HERSHEY_COMPLEX, fontScale=font_scale, color=(51, 153, 255), thickness=thickness)
        return img
        # cv2.imwrite(os.path.join(self.folder, self.output_path, text)[:-1]+os.path.splitext(img_name)[1], img)

    def read_img(self, image_path):
        img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        if img is None: raise 'Not an image or not in correct format!'
        return img

    def image_filenames(self, folder): # list all valid images
        img_files = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                img_files.append(os.path.join(folder,filename))
        return img_files

    def run_folder(self, text_array=None):
        i = 0
        for im in self.image_filenames(os.path.join(self.folder, self.input_path)):
            if text_array != None:
                for txt in text_array:
                    img = self.read_img(im)
                    img = self.write_text(img, txt)
                    cv2.imwrite(os.path.join(self.folder, self.output_path, text)[:-1]+os.path.splitext(im)[1], img)
            else:
                i += 1
                text = f'{i}hello this is just a test for checking out opencv write on image!'
                img = self.read_img(im)
                img = self.write_text(img, text)
                cv2.imwrite(os.path.join(self.folder, self.output_path, text)[:-1]+os.path.splitext(im)[1], img)

    def run_image(self, image_name, text):
        if image_name != None and text != None:
            img = self.read_img(image_name)
            img = self.write_text(img, text)
            print(os.path.join(os.path.dirname(image_name), text)+os.path.splitext(image_name)[1])
            cv2.imwrite(os.path.join(os.path.dirname(image_name), text)+os.path.splitext(image_name)[1], img)
        else:
            raise 'Input image path and text to write'

    def run_numpy_image(self, image, text, output_dir='./', ext='jpg'):
        img = self.write_text(image, text)
        cv2.imwrite(os.path.join(output_dir, text)+f'.{ext}', img)


# op = OpencvText()
# op.run_folder()
# op.run_image('input/6t8Zh249QiFmVnkQdCCtHK.jpg', 'test image')
# op.run_numpy_image(op.read_img('input/6t8Zh249QiFmVnkQdCCtHK.jpg'), 'numpy test on sample image', output_dir='input', ext='png')
