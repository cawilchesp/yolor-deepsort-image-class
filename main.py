from yolor_video_class import YOLOR_DEEPSORT

def main():
    yolor_options = {
        'cfg': 'yolor_p6.cfg',
        'weights': 'yolor_p6.pt',
        'names_file': 'coco.names',
        'inference_size': 1280,
        'use_gpu': True,
    }

    video_options = {
        'source': "D:\Data\Videos_Bogotá\CHICO1_1.avi",
        # 'source': 0,
        'output': "D:\Data\Videos_Bogotá\ouput",
        'view_image': True,
        'save_text': True,
        'frame_save': 300,
        'trail': 64,
        'class_filter': [0,1,2,3,5,7], # Based on coco.names
        'show_boxes': True,
        'show_trajectories': True,
        'save_video': True
    }

    yolor = YOLOR_DEEPSORT(yolor_options, video_options)
    yolor.detect()

if __name__ == '__main__':
    main()