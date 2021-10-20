import argparse
from torchvision import transforms
import cv2
import torch
import numpy as np
from PIL import Image
from matplotlib import cm
import torch.nn as nn
from mmdet.apis import inference_detector, init_detector


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('--config',default = '/home/maxim/Age_detection_pipline/insightface/detection/scrfd/configs/scrfd/scrfd_500m.py', help='test config file path')
    parser.add_argument('--checkpoint',default = '/home/maxim/Age_detection_pipline/insightface/detection/scrfd/model.pth', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cpu', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.7, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    val_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    trans_tensor = transforms.ToTensor()

    mask_detector = torch.hub.load('mit-han-lab/ProxylessNAS', "proxyless_cpu" , pretrained=True)
    mask_detector.classifier = nn.Sequential(nn.Linear(in_features=1432,out_features=2, bias =True))
    mask_detector.load_state_dict(torch.load('/home/maxim/Age_detection_pipline/FaceMask_proxyless.pth',map_location=device))
    mask_detector.to(device)
    mask_detector.eval()
    class_names = ['with_mask', 'wo_mask']
    detector = init_detector(args.config, args.checkpoint, device=device)

    camera = cv2.VideoCapture(args.camera_id)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (255, 0, 0)
    thickness = 1
    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        ret_val, img = camera.read()
        result = inference_detector(detector, img)
        
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break
        for bbox in result[0]:
            if bbox[4] >= args.score_thr:
                frame = img[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_rgb = Image.fromarray(np.uint8(img_rgb))
                inputs = val_transforms(img_rgb)
 
                img_rgb = trans_tensor(img_rgb).unsqueeze(0)
                inputs = inputs.to(device)

                out = mask_detector(img_rgb)
                _,pred = torch.max(out,1)
   
        
            
                img = cv2.putText(img, str(class_names[int(pred.item())]),(int(bbox[0]), int(bbox[1])+25), font, fontScale, color, thickness, cv2.LINE_AA)#
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        cv2.imshow('Video', img)
	

if __name__ == '__main__':
    main()
