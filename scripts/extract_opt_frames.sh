python extract_frames.py \
    --model='models/raft-things.pth' \
    --path='D:/data/pose_detection_sample/kinect' \
    --flow_output_path='D:/data/pose_detection_sample/kinectflows' \
    --rgb_output_path='D:/data/pose_detection_sample/kinectrgbs' \
    --batch_size=1 \
    --workers=4 \
    --center_crop=224 \
    --start=1100 \
    --end=1159