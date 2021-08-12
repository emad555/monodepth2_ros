#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function
import rospy
import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from sensor_msgs.msg import Image
import torch
from torchvision import transforms, datasets
from std_msgs.msg import String
import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
import cv2
from cv_bridge import CvBridge,CvBridgeError

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', default="assets/test_image.jpg")
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"],default="mono+stereo_640x192")
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only '
                             'makes sense for stereo-trained KITTI models).',
                        action='store_true')

    return parser.parse_args()

def pred_image(np_img, header_stamp):
    global encoder,depth_decoder,output_directory,args
    #print("image_path  is",image_path)
    with torch.no_grad():
        # Load image and preprocess
        #input_image = pil.open(image_path).convert('RGB')
        input_image= pil.fromarray(np_img)
        original_width, original_height = input_image.size
        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)
         # PREDICTION
        input_image = input_image.to(device)
        features = encoder(input_image)
        outputs = depth_decoder(features)
        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)
        # Saving numpy file
        #output_name = os.path.splitext(os.path.basename(image_path))[0]
        #print("output name is",output_name)
        scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
        #if args.pred_metric_depth:
        #    #name_dest_npy = os.path.join(output_directory, "{}_depth.npy".format(output_name))
        #    metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()
        #    np.save(name_dest_npy, metric_depth)
        #else:
        #    name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
        #    np.save(name_dest_npy, scaled_disp.cpu().numpy())
        # Saving colormapped depth image
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)
        #name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
        #print("name_dest_im name is",name_dest_im)
        #im.save(name_dest_im)
        im_np = np.asarray(im)
        im_np = cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR)
        im_np_gray = cv2.cvtColor(im_np, cv2.COLOR_BGR2GRAY)## convert to gray
        try:
            image_message=CvBridge().cv2_to_imgmsg(im_np_gray)
            #timestampe
            image_message.header.stamp=header_stamp
            # print("header_stamp",header_stamp)
            pub_prediction.publish(image_message)
        except CvBridgeError as e:
            print(e)
        
        #image_message = CvBridge.cv2_to_imgmsg(im_np_gray, encoding='bgr8')
        #image_message = CvBridge.cv2_to_imgmsg(im_np, encoding="bgr8")
        
        #cv2.imshow('img', im_np)
        #cv2.waitKey(1)

    #print('-> Done!')

def img_callback(msg):
    header_stamp = msg.header.stamp
    ##frame = CvBridge.imgmsg_to_cv2(msg.data, "bgr8")
    #try:
    np_img = np.fromstring(msg.data, dtype=np.uint8).reshape((480, 640 , 3))#921600   307200 480, 640  
            
    #except Exception as E:
    #    print(" -- exception! terminating...")
    #    print(E,"\n"*5)
            
    #np_img = np_img[:, :, :3]
    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    #print('call pred')
    pred_image(np_img, header_stamp)

    #cv2.imshow('img', np_img)
    #cv2.waitKey(2)











#def talker():
#    pub = rospy.Publisher('chatter', String, queue_size=10)
#    
#    rate = rospy.Rate(10) # 10hz
#    while not rospy.is_shutdown():
#        hello_str = "hello world %s" % rospy.get_time()
#        rospy.loginfo(hello_str)
#        pub.publish(hello_str)
#        rate.sleep()
##################################################################
##### beginning of the code
#########################################################
args = parse_args()#### get arguments
if torch.cuda.is_available():#### check cuda
    device = torch.device("cuda")
    print('yes cuda')
else:
    device = torch.device("cpu")
    print('no cuda')
############ download dataset
download_model_if_doesnt_exist(args.model_name)

######## get model
model_path = os.path.join("models", args.model_name)
print("-> Loading model from ", model_path)
encoder_path = os.path.join(model_path, "encoder.pth")
depth_decoder_path = os.path.join(model_path, "depth.pth")

# LOADING PRETRAINED MODEL
print("   Loading pretrained encoder")
encoder = networks.ResnetEncoder(18, False)
loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
feed_height = loaded_dict_enc['height']
feed_width = loaded_dict_enc['width']
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)
encoder.to(device)
encoder.eval()

print("   Loading pretrained decoder")
depth_decoder = networks.DepthDecoder(
    num_ch_enc=encoder.num_ch_enc, scales=range(4))
loaded_dict = torch.load(depth_decoder_path, map_location=device)
depth_decoder.load_state_dict(loaded_dict)
depth_decoder.to(device)
depth_decoder.eval()
#print("args.image_path",os.path.join("/home/pc-18/catkin_ws/src/monodepth2/scripts",args.image_path))


#paths = [args.image_path]
#output_directory = "/home/pc-18/catkin_ws/src/monodepth2/scripts/assets"

######## call pred function
#pred_image(os.path.join("/home/pc-18/catkin_ws/src/monodepth2/scripts",args.image_path))
rospy.init_node('monodepth2', anonymous=True)
img_sub = rospy.Subscriber('/camera/image_raw', Image, callback=img_callback, queue_size=1)
pub_prediction=rospy.Publisher('monodepth2/output_image', Image, queue_size=1)

#rospy.Subscriber('/orb_slam2_mono/debug_image', Image, img_callback)
rate = rospy.Rate(10)
if __name__ == "__main__":
    rospy.logwarn("Monodepth2 prediction started")
    while not rospy.is_shutdown():
        #img_sub = rospy.Subscriber('/camera/image_raw', Image, callback=img_callback, queue_size=1)
        rate.sleep()
