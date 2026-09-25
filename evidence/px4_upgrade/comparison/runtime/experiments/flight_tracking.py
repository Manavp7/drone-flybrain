"""RGB-D camera-motion compensation before ordinary short-term association.

This transforms old observations using measured camera pose and their registered
depth. It never refreshes observation time, creates detections, or changes a
selected identity. Actual new YOLO boxes and clothing checks still decide matches.
"""
import numpy as np
from perception.pipeline import ShortTermTracker


class EgoMotionTracker(ShortTermTracker):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.anchors={}
        self.prepared_frame=None
        self.last_reprojections=[]

    def reset(self):
        super().reset()
        self.anchors.clear()
        self.last_reprojections=[]
        # prepare() supplies the current frame before PerceptionPipeline's first
        # stream reset. Keep only that pending frame, never any old track anchor.

    def prepare(self,frame):
        frame.validate()
        self.prepared_frame=frame

    def update(self,detections,timestamp,image_rgb=None):
        frame=self.prepared_frame
        if frame is None or abs(frame.capture_time_s-timestamp)>1e-8:
            raise ValueError('Camera pose/depth must belong to the exact tracking capture')
        h,w=frame.rgb.shape[:2]
        fx,fy,cx,cy=frame.intrinsics
        self.last_reprojections=[]
        # Reproject from the last actually observed world anchor every time,
        # never compound a previously predicted pixel box or renew its age.
        for tid,track in self.tracks.items():
            anchor=self.anchors.get(tid)
            if anchor is None or timestamp-track['time']>self.max_age_s:
                continue
            optical=(anchor['corners']-frame.position_world_camera)@frame.rotation_world_camera
            record=dict(track_id=tid,anchor_capture_time_s=anchor['capture_time_s'],
                        previous_observation_time_s=track['time'],valid=False)
            if np.isfinite(optical).all() and np.min(optical[:,2])>.1:
                uv=optical[:,:2]/optical[:,2:]*[fx,fy]+[cx,cy]
                box=np.r_[uv.min(0),uv.max(0)]
                box=np.clip(box,[0,0,0,0],[w,h,w,h])
                if box[2]-box[0]>=1 and box[3]-box[1]>=1:
                    track['bbox']=box.tolist()
                    record.update(valid=True,predicted_bbox_xyxy=box.tolist())
            self.last_reprojections.append(record)
        result=super().update(detections,timestamp,image_rgb)
        self.anchors={k:v for k,v in self.anchors.items() if k in self.tracks}
        for detection in result:
            tid=detection['track_id'];box=np.asarray(detection['bbox_xyxy'])
            # A matched observation supersedes the old anchor even when its new
            # depth is unusable. Do not attach old depth to newly observed pixels.
            self.anchors.pop(tid,None)
            if not frame.registration_verified:continue
            bw,bh=box[2:]-box[:2]
            left,right=max(0,int(box[0]+.3*bw)),min(w,int(np.ceil(box[2]-.3*bw)))
            top,bottom=max(0,int(box[1]+.3*bh)),min(h,int(np.ceil(box[3]-.3*bh)))
            region=frame.depth_m[top:bottom,left:right]
            valid=np.isfinite(region)&(region>=.3)&(region<=20.)
            if region.size<4 or np.count_nonzero(valid)<4 or np.mean(valid)<.8:continue
            values=region[valid]
            if np.quantile(values,.9)-np.quantile(values,.1)>.35:continue
            z=float(np.median(values))
            pixels=np.array([[u,v] for u in (box[0],box[2]) for v in (box[1],box[3])])
            optical=np.column_stack([(pixels[:,0]-cx)*z/fx,(pixels[:,1]-cy)*z/fy,np.full(4,z)])
            world=optical@frame.rotation_world_camera.T+frame.position_world_camera
            self.anchors[tid]=dict(corners=world,capture_time_s=float(timestamp))
        return result
