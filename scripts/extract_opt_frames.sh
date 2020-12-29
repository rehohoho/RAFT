python extract_frames.py \
    --model='models/raft-things.pth' \
    --path='D:/data/pose_detection_sample/kinect' \
    --output_path='D:/data/pose_detection_sample/kinectflows' \
    --batch_size=1 \
    --workers=4 \
    --resize=2 \
    --orig_size="1080,1920"