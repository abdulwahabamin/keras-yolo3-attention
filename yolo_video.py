import sys
import argparse
import os
from yolo import YOLO, detect_video
from PIL import Image
from tqdm import tqdm

def detect_img(yolo,base_path,folders,split,savedir):
    for folder in folders:
        txt_file = os.path.join(base_path,folder,'ImageSets/Main',split + '.txt')
        f = open(txt_file,'r')
        lines = f.readlines()
        for line in tqdm(lines):
            img_name = line.strip()
            img = os.path.join(base_path,folder,'JPEGImages',img_name + '.jpg')
            # print('testing image ' + img + '\n')
            try:
                image = Image.open(img)
            except:
                print(img + ' does not exist')
                continue
            else:
                r_image, annot = yolo.detect_image(image)
                
                if not os.path.exists(os.path.join(savedir,folder)):
                    os.mkdir(os.path.join(savedir,folder))
                f = open(os.path.join(savedir,folder,img_name +'.txt'),'w+')
                for annotation in annot:
                    f.write(annotation + '\n')
                f.close()
                r_image.save(os.path.join(savedir,folder,img_name + '.jpg'))
    yolo.close_session()

# def detect_img(yolo,base_path,folders,split,savedir):
#     for folder in folders:
#         lines = os.listdir(os.path.join(base_path,folder))
     
#         for line in lines:
#             img_name = line
#             img = os.path.join(base_path,folder,img_name)     
#             print('testing image ' + img + '\n')
#             try:
#                 image = Image.open(img)
#             except:
#                 print(img + ' does not exist')
#                 continue
#             else:
#                 r_image, annot = yolo.detect_image(image)
                
#                 if not os.path.exists(os.path.join(savedir,folder)):
#                     os.mkdir(os.path.join(savedir,folder))
#                 r_image.save(os.path.join(savedir,img_name))
#     yolo.close_session()

# def detect_img(yolo,base_path,folders,split,savedir):
#     for folder in folders:
#         # txt_file = os.path.join(base_path,folder,'ImageSets/Main',split + '.txt')
#         lines = os.listdir(os.path.join(base_path, folder))
#         lines = [k for k in lines if '.jpg' or '.png' in k]
#         for line in tqdm(lines):
#             img = os.path.join(base_path, folder, line)
#             print('testing image ' + img + '\n')
#             try:
#                 image = Image.open(img)
#             except:
#                 print(img + ' does not exist')
#                 continue
#             else:
#                 r_image, annot = yolo.detect_image(image)
#                 im = line.split('.')[0]
#                 print(im)
#                 if not os.path.exists(savedir):
#                     os.mkdir(savedir)
#                 f = open(os.path.join(savedir,im +'.txt'),'w+')
#                 for annotation in annot:
#                     f.write(annotation + '\n')
#                 f.close()
#                 if not os.path.exists(os.path.join(savedir,folder)):
#                     os.mkdir(os.path.join(savedir,folder))
#                 r_image.save(os.path.join(savedir,folder,line))
#     yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()
    # base_path = '../../datasets/temp'
    # folders = ['images']
    base_path= '/Ted/LUMS/Thesis/Datasets/VOC_Test_'
    # folders = ['VOC_COMSATS_1','VOC_LUMS_1','VOC_LUMS_2','VOC_SKP_1']
    folders = ['VOC_Test_Easy', 'VOC_Test_Hard']
    split = 'test' #can be train, train_val or test
    savedir = '/Ted/LUMS/Thesis/results/yolo-attention'
    # savedir = '/home/cvlab/Desktop/Ted/results/supplementary-1/yolo'
    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)),base_path,folders,split,savedir)
 #   elif "input" in FLAGS:
 #       detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
