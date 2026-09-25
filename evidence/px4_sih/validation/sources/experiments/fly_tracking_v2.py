"""Train motion readouts, select on validation, then run fresh Flyvis tracking tests."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from experiments.runtime import VideoFlyvisAdapter
from experiments.fly_tracking_core import FlowBoxTracker, evaluate_tracking, visible_response_indices
from experiments.fly_motion_readout import RidgeReadout, LearnedFlowTracker, flow_features
from experiments.fly_neural_template import NeuralTemplateTracker
from experiments.fly_tracking_v2_data import CELL_TYPES, DT, generate, specs, real_clip, REAL_FIRST
from flybrain_sim.research_model import validate_clip, validate_manifest

ROOT=Path(__file__).resolve().parents[1]
MODES=("median","statistics")
RIDGES=(.0001,.01,.1)
BANKS={"early3":[0,1,2],"all16":list(range(16))}
TEMPLATE_CONFIGS=[{"bank":bank,"template_update":update,"minimum_score":.35,
    "grid_step":4,"search_radius_px":24} for bank in BANKS for update in (0.,.02)]


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def announce(value):print(json.dumps(value),flush=True)


def neural_binding(adapter):
    nodes=adapter.network.connectome.nodes
    types=nodes.type[:].astype(str);u=np.asarray(nodes.u[:]);v=np.asarray(nodes.v[:])
    lattice=np.array([(a,b) for a in range(-15,16) for b in range(max(-15,-15-a),min(15,15-a)+1)])
    indices=[]
    for name in CELL_TYPES:
        cells=np.flatnonzero(types==name)
        by_coord={(int(u[i]),int(v[i])):int(i) for i in cells}
        if len(by_coord)!=721:raise ValueError(f"Unsupported cell lattice: {name}")
        indices.append([by_coord[tuple(x)] for x in lattice])
    centers=adapter.eye.receptor_centers.cpu().numpy()+[195,195]
    expected=np.array([[int(13*(a+b/2))+195,13*b+195] for a,b in lattice])
    np.testing.assert_array_equal(centers,expected)
    return np.asarray(indices),centers,lattice


def infer(adapter,clip,indices,output):
    folder=output/"data"/clip["name"];folder.mkdir(parents=True,exist_ok=False)
    sampling=validate_clip(clip["frames"],clip["times"],DT)
    receipt,activity=adapter.infer_clip(clip["frames"],clip["times"],dt=DT)
    flow=np.concatenate([adapter.decode_flow(activity[j:j+32]) for j in range(0,len(activity),32)])
    features=activity[:,indices]
    with adapter.torch.inference_mode():
        native=adapter.torch.as_tensor(clip["frames"][None],device=adapter.device)
        retina=adapter.eye(native).cpu().numpy()[0,:,0,:][sampling.indices,None,:]
    np.save(folder/"activity.npy",activity)
    np.savez_compressed(folder/"features.npz",flow=flow,neural=features,retina=retina)
    np.savez_compressed(folder/"input.npz",gray_u8=np.round(clip["frames"]*255).astype(np.uint8),timestamps=clip["times"])
    meta={"name":clip["name"],"kind":clip["kind"],"initial_box":initial_box(clip).tolist(),
          "spec":clip.get("spec"),"stimulus_times":sampling.stimulus_timestamps.tolist(),
          "response_times":sampling.response_timestamps.tolist(),"input_indices":sampling.indices.tolist()}
    save(folder/"meta.json",meta);save(folder/"neural_receipt.json",receipt)
    if clip["kind"]=="synthetic":
        save(folder/"truth.json",{"boxes_xyxy":clip["truth_boxes"].tolist(),
             "velocities_xy":clip["velocities"].tolist(),"timestamps":clip["times"].tolist()})
    else:save(folder/"source.json",clip["provenance"])
    announce({"inference":clip["name"],"steps":len(activity),"neural_wall_s":receipt["elapsed_wall_s"]})
    return {"meta":meta,"flow":flow,"neural":features,"retina":retina,"clip":clip}


def initial_box(clip):
    return clip["initial_box"] if clip["kind"]=="real_video" else clip["truth_boxes"][0]


def load_record(output,spec):
    folder=output/"data"/spec["name"]
    with np.load(folder/"features.npz") as f:arrays={k:f[k] for k in f.files}
    return {**arrays,"meta":json.loads((folder/"meta.json").read_text()),"clip":generate(spec)}


def align_trace(records,meta,clip):
    indices=visible_response_indices(meta["response_times"],clip["times"])
    initial=initial_box(clip)
    boxes=np.array([initial if i<0 else records[i]["box_xyxy"] for i in indices])
    status=np.array(["initialized" if i<0 else records[i]["status"] for i in indices])
    for i,row in enumerate(records):
        row.update(stimulus_time_s=meta["stimulus_times"][i],response_time_s=meta["response_times"][i],
                   source_input_index=meta["input_indices"][i])
    return {"boxes":boxes,"statuses":status,"trace":records}


def motion_track(record,centers,model,mode):
    tracker=LearnedFlowTracker(centers,initial_box(record["clip"]),model,mode=mode)
    rows=[tracker.step(f,DT) for f in record["flow"]]
    return align_trace(rows,record["meta"],record["clip"])


def old_motion(record,centers):
    mapping=np.array(json.loads((ROOT/"results/fly_tracking_run01/calibration.json").read_text())["mapping"])
    tracker=FlowBoxTracker(centers,initial_box(record["clip"]),mapping)
    return align_trace([tracker.step(f,DT) for f in record["flow"]],record["meta"],record["clip"])


def template_track(record,centers,config,scale,arm="neural"):
    config=dict(config);bank=config.pop("bank");channels=BANKS[bank]
    data=record["retina"] if arm=="raw" else record["neural"][:,channels,:]
    selected_scale=np.ones(1) if arm=="raw" else scale[channels]
    if arm=="zero":data=np.zeros_like(data)
    elif arm=="shuffle":data=data[:,:,np.random.default_rng(9401).permutation(data.shape[-1])]
    tracker=NeuralTemplateTracker(centers,initial_box(record["clip"]),data[0],
                                  channel_scale=selected_scale,**config)
    rows=[{"box_xyxy":initial_box(record["clip"]).tolist(),"status":"initialized","score":None}]
    rows.extend(tracker.step(f,DT) for f in data[1:])
    return align_trace(rows,record["meta"],record["clip"])


def score(trajectory,clip):
    boxes,status=trajectory["boxes"],trajectory["statuses"]
    if clip["kind"]=="synthetic":
        times=clip["times"];truth=clip["truth_boxes"]
        # Initialization ends at the first neural response; skip that supplied state.
        mask=times>DT+1e-9
        is_stationary=clip["spec"]["kind"]=="stationary"
        if not is_stationary:mask &= times>=.6
        result=evaluate_tracking(boxes[mask],truth[mask],status[mask]=="tracking")
        prefix=(times>DT+1e-9)&(times<.6)
        centers=(boxes[:,:2]+boxes[:,2:])/2;target_centers=(truth[:,:2]+truth[:,2:])/2
        result["stationary_prefix_max_center_error"]=float(np.linalg.norm(centers[prefix]-target_centers[prefix],axis=1).max())
        result["max_center_error"]=float(np.linalg.norm(centers[mask]-target_centers[mask],axis=1).max())
        static=np.broadcast_to(initial_box(clip),boxes[mask].shape)
        static_error=evaluate_tracking(static,truth[mask],np.ones(mask.sum(),bool))["mean_center_error"]
        result["static_mean_center_error"]=static_error
        result["improvement_vs_static"]=None if static_error<1e-9 else 1-result["mean_center_error"]/static_error
        result["acceptance_pass"]=bool(result["fraction_active_iou_ge_0_5"]>=.75 and (
            result["max_center_error"]<=5 if is_stationary else
            result["stationary_prefix_max_center_error"]<=5 and result["improvement_vs_static"]>=.2))
    else:
        labels=json.loads((ROOT/"results/fly_tracking_v2_annotations01/annotations.json").read_text())
        selected=[r for r in labels["frames"] if r["scorable"] and r["sequence"]>REAL_FIRST]
        ix=np.array([r["sequence"]-REAL_FIRST for r in selected])
        truth=(np.array([r["bbox_xyxy"] for r in selected])-[420,0,420,0])*391/1080
        result=evaluate_tracking(boxes[ix],truth,status[ix]=="tracking")
        result["acceptance_pass"]=None
        result["annotation_limit"]="Nine approximate agent checks; new target in same previously seen scene"
    return result


def ranking(results):
    return (sum(not r["acceptance_pass"] for r in results),
            -float(np.mean([r["fraction_active_iou_ge_0_5"] for r in results])),
            float(np.mean([r["mean_center_error"] for r in results])))


def train_readouts(output,training,centers):
    features={mode:[] for mode in MODES};targets=[];weights=[]
    channel_sum=np.zeros(16);channel_squares=np.zeros(16);channel_count=0
    for spec in training:
        record=load_record(output,spec);clip=record["clip"]
        idx=np.array(record["meta"]["input_indices"])
        valid=np.array(record["meta"]["stimulus_times"])>=.2
        # Flow describes preceding image motion. At timestamp j, image position
        # reflects velocity from interval j-1, rather than future interval j.
        y=clip["velocities"][np.maximum(0,idx-1)][valid]
        w=np.zeros(len(y));stationary=np.linalg.norm(y,axis=1)<1e-9
        groups=[g for g in (stationary,~stationary) if g.any()]
        for group in groups:w[group]=1/(len(groups)*group.sum()*len(training))
        targets.extend(y);weights.extend(w)
        for mode in MODES:
            features[mode].extend(flow_features(f,centers,clip["truth_boxes"][i],mode=mode)
                                  for f,i,keep in zip(record["flow"],idx,valid) if keep)
        # Equal-length training sequences only; no validation/test values here.
        x=record["neural"].astype(np.float64)
        channel_sum+=x.sum((0,2));channel_squares+=(x*x).sum((0,2));channel_count+=x.shape[0]*x.shape[2]
    scale=np.sqrt(np.maximum(channel_squares/channel_count-(channel_sum/channel_count)**2,1e-8))
    np.save(output/"training_channel_scale.npy",scale)
    candidates=[]
    for mode in MODES:
        np.savez_compressed(output/f"training_{mode}.npz",X=features[mode],y=targets,weights=weights)
        for ridge in RIDGES:
            model=RidgeReadout.fit(np.array(features[mode]),np.array(targets),np.array(weights),ridge=ridge)
            name=f"{mode}_{ridge}"
            save(output/f"readout_{name}.json",model.to_dict());candidates.append((name,mode,model))
    return candidates,scale


def render_case(writer,clip,trajectories,metrics,selected,stills):
    times=clip["times"];frames=np.searchsorted(times,np.arange(times[0],times[-1],DT),side="right")-1
    snapshots=set(np.linspace(0,len(frames)-1,3).round().astype(int))
    for ordinal,i in enumerate(frames):
        image=clip["color"][i].copy() if "color" in clip else cv2.cvtColor(np.round(clip["frames"][i]*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
        view=cv2.resize(image,(520,520));factor=520/391
        for arm,color in [("old_motion",(110,110,240)),("raw_template",(130,230,130)),("selected",(255,224,70))]:
            b=np.rint(trajectories[arm]["boxes"][i]*factor).astype(int)
            cv2.rectangle(view,tuple(b[:2]),tuple(b[2:]),color,2)
        if "truth_boxes" in clip:
            b=np.rint(clip["truth_boxes"][i]*factor).astype(int);cv2.rectangle(view,tuple(b[:2]),tuple(b[2:]),(255,255,255),1)
        status=trajectories["selected"]["statuses"][i]
        if status not in ("tracking","initialized"):
            cv2.putText(view,"UNCERTAIN - last position",(8,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,224,70),2,cv2.LINE_AA)
        canvas=np.full((640,1024,3),(24,20,18),np.uint8);canvas[55:575,20:540]=view
        main=metrics["selected"]
        lines=[clip["name"].replace('_',' ').upper(),"Flyvis target tracking v2",f"State: {status}",
               "CYAN: selected neural tracker","GREEN: same matcher on pixels","RED: previous neural motion", "Manually initialized; fixed box",
               f"Mean center error: {main['mean_center_error']:.1f}px",f"Active IoU >= .5: {100*main['fraction_active_iou_ge_0_5']:.0f}%",
               "PASS" if main["acceptance_pass"] else ("FAIL" if main["acceptance_pass"] is False else "9 approximate real-video checks")]
        for n,line in enumerate(lines):
            cv2.putText(canvas,line,(560,78+n*43),cv2.FONT_HERSHEY_SIMPLEX,.57,(225,225,225),1,cv2.LINE_AA)
        cv2.putText(canvas,"Saved causal trajectories | offline experiment; playback is not inference speed",(20,29),cv2.FONT_HERSHEY_SIMPLEX,.55,(220,220,220),1,cv2.LINE_AA)
        credit="Synthetic ground truth: white | unseen texture and trajectory" if clip["kind"]=="synthetic" else "Video: Vicente Quintero / QuinteroP | CC BY 3.0 | cropped, muted, annotated"
        cv2.putText(canvas,credit,(20,610),cv2.FONT_HERSHEY_SIMPLEX,.52,(180,180,180),1,cv2.LINE_AA)
        writer.write(canvas)
        if ordinal in snapshots:stills.append(canvas)


def run(output):
    output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(2);started=time.perf_counter()
    all_specs=specs();training=[s for s in all_specs if s["split"]=="train"]
    validation=[s for s in all_specs if s["split"]=="validation"];testing=[s for s in all_specs if s["split"]=="test"]
    sources=[Path(__file__),ROOT/"experiments/fly_tracking_v2_data.py",ROOT/"experiments/fly_motion_readout.py",ROOT/"experiments/fly_neural_template.py"]
    save(output/"definition.json",{"created_utc":datetime.now(timezone.utc).isoformat(),"specs":all_specs,
        "source_hashes":{str(p.relative_to(ROOT)):sha(p) for p in sources},"cell_types":CELL_TYPES,
        "motion_modes":MODES,"ridge_candidates":RIDGES,"template_candidates":TEMPLATE_CONFIGS,
        "selection":"validation failed-case count, then active overlap, then center error",
        "template_branch_only_if_motion_validation_fails":True,
        "acceptance":{"active_iou_fraction":.75,"moving_improvement_vs_static":.2,"stationary_max_drift_px":5},
        "real_annotations_sha256":sha(ROOT/"results/fly_tracking_v2_annotations01/annotations.json"),
        "real_source_frames":[315,360],"test_inference_after_selection_freeze":True})
    snapshot=output/"sources";snapshot.mkdir()
    for p in sources:(snapshot/p.name).write_bytes(p.read_bytes())
    adapter=VideoFlyvisAdapter(ROOT/"models/flyvis_0000_000.manifest.json")
    indices,centers,lattice=neural_binding(adapter)
    np.savez(output/"neural_binding.npz",indices=indices,centers_rc=centers,lattice_uv=lattice,cell_types=np.array(CELL_TYPES))
    for spec in training+validation:
        record=infer(adapter,generate(spec),indices,output);del record
    candidates,scale=train_readouts(output,training,centers)
    validation_results={}
    for name,mode,model in candidates:
        results=[]
        for spec in validation:
            record=load_record(output,spec);trajectory=motion_track(record,centers,model,mode)
            results.append(score(trajectory,record["clip"]))
        validation_results[name]=results
        announce({"motion_candidate":name,"validation_ranking":ranking(results)})
    best_name,best_mode,best_model=min(candidates,key=lambda c:ranking(validation_results[c[0]]))
    save(output/"motion_validation.json",validation_results)
    selected={"method":"motion","name":best_name,"mode":best_mode}
    template_results={};template_config=TEMPLATE_CONFIGS[0]
    if ranking(validation_results[best_name])[0]:
        for number,config in enumerate(TEMPLATE_CONFIGS):
            results=[]
            for spec in validation:
                record=load_record(output,spec)
                trajectory=template_track(record,centers,config,scale)
                results.append(score(trajectory,record["clip"]))
            template_results[str(number)]={"config":config,"results":results}
            announce({"template_candidate":number,"validation_ranking":ranking(results)})
        winner=min(template_results,key=lambda k:ranking(template_results[k]["results"]))
        template_config=template_results[winner]["config"]
        if ranking(template_results[winner]["results"])<ranking(validation_results[best_name]):
            selected={"method":"template","name":winner,"config":template_config}
    save(output/"template_validation.json",template_results)
    save(output/"selection_frozen.json",{"created_utc":datetime.now(timezone.utc).isoformat(),
        "selected":selected,"best_motion":{"name":best_name,"mode":best_mode,"model":best_model.to_dict()},
        "template_comparison_config":template_config,"training_scale_sha256":sha(output/"training_channel_scale.npy"),
        "final_test_inference_started":False})
    announce({"selection_frozen":selected})
    results=[];stills=[]
    intermediate=output/"preview.avi"
    writer=cv2.VideoWriter(str(intermediate),cv2.VideoWriter_fourcc(*"MJPG"),50.,(1024,640))
    if not writer.isOpened():raise ValueError("Preview writer failed")
    try:
        # The held-out branch starts only after the immutable selection file exists.
        for item in testing+[None]:
            clip=real_clip(ROOT) if item is None else generate(item)
            record=infer(adapter,clip,indices,output)
            learned=motion_track(record,centers,best_model,best_mode)
            neural=template_track(record,centers,template_config,scale)
            trajectories={"selected":neural if selected["method"]=="template" else learned,
                "learned_motion":learned,"neural_template":neural,"old_motion":old_motion(record,centers),
                "raw_template":template_track(record,centers,template_config,scale,arm="raw"),
                "zero_neural":template_track(record,centers,template_config,scale,arm="zero"),
                "shuffled_neural":template_track(record,centers,template_config,scale,arm="shuffle")}
            if selected["method"]=="motion":
                trajectories["zero_neural"]=motion_track({**record,"flow":np.zeros_like(record["flow"])},centers,best_model,best_mode)
                permutation=np.random.default_rng(9401).permutation(len(centers))
                trajectories["shuffled_neural"]=motion_track({**record,"flow":record["flow"][:,:,permutation]},centers,best_model,best_mode)
            metrics={name:score(trajectory,clip) for name,trajectory in trajectories.items()}
            folder=output/"evaluation"/clip["name"];folder.mkdir(parents=True)
            np.savez_compressed(folder/"display_boxes.npz",timestamps=clip["times"],**{k:v["boxes"] for k,v in trajectories.items()})
            save(folder/"traces.json",{k:v["trace"] for k,v in trajectories.items()})
            save(folder/"display_statuses.json",{k:v["statuses"].tolist() for k,v in trajectories.items()})
            save(folder/"metrics.json",metrics)
            results.append({"case":clip["name"],"kind":clip["kind"],"metrics":metrics})
            announce({"test":clip["name"],"selected":metrics["selected"],"raw_template":metrics["raw_template"]})
            render_case(writer,clip,trajectories,metrics,selected,stills)
    finally:writer.release()
    subprocess.run(["ffmpeg","-v","error","-i",str(intermediate),"-an","-c:v","libx264","-crf","20",
                    "-pix_fmt","yuv420p","-movflags","+faststart",str(output/"tracking_preview.mp4")],check=True)
    intermediate.unlink()
    for i,frame in enumerate(stills):cv2.imwrite(str(output/f"preview_{i:02d}.jpg"),frame)
    validate_manifest(ROOT/"models/flyvis_0000_000.manifest.json")
    summary={"status":"experiment_completed","created_utc":datetime.now(timezone.utc).isoformat(),
        "selected":selected,"cases":results,"elapsed_wall_s":time.perf_counter()-started,
        "all_synthetic_tests_pass":all(r["metrics"]["selected"]["acceptance_pass"] for r in results if r["kind"]=="synthetic"),
        "actual_flyvis_inference":True,"neural_weights_changed":False,"control_authority":False,
        "automatic_object_recognition":False,"real_scene_independent":False}
    save(output/"summary.json",summary)
    save(output/"artifact_hashes.json",{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    announce({"completed":str(output),"selected":selected,"all_tests_pass":summary["all_synthetic_tests_pass"],"elapsed_s":summary["elapsed_wall_s"]})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():parser.error("Output must be a new directory")
    try:run(args.output)
    except Exception as exc:
        if args.output.is_dir():save(args.output/"failure.json",{"error_type":type(exc).__name__,"error":str(exc)})
        raise
