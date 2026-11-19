#!/usr/bin/env python3
"""Real fixed-asset gsplat → native CUDA guides → NR → CPU output."""
from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[2]
STORAGE = Path(os.environ["PROJECT_STORAGE_ROOT"])
PUBLIC = Path(os.environ["PROJECT_DATASETS_ROOT"])
sys.path.insert(0, str(SOURCE/"python"))
sys.path.insert(0, str(SOURCE/"tools/graph"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


renderer = load_module("rw_gsplat_renderer", SOURCE/"adapters/gsplat/render_mip360.py")
motion_cpu = load_module("rw_gsplat_cpu_motion", SOURCE/"adapters/mip360/motion.py")
quality_module = load_module("rw_gsplat_quality", SOURCE/"adapters/gsplat/quality.py")
io = renderer.io


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def registered_inputs(scene, views, input_manifest=None, asset_manifest=None):
    path = input_manifest or STORAGE/"runs/RW-20261101-012-mip-inputs-r02/mip360_inputs.json"
    inputs = json.loads(path.read_text())
    base = next(x for x in inputs["frames"] if x["scene_id"] == scene)
    directory = io.public_input(Path(base["input_path"]).parent)
    images = sorted(p for p in directory.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    # Spaced capture views; no capture chronology or source PTS is invented.
    selected = images[::8][:views]
    scene_root = directory.parent
    cameras_path, poses_path = scene_root/"sparse/0/cameras.bin",scene_root/"sparse/0/images.bin"
    cameras = io.read_cameras(cameras_path)
    poses = io.read_selected_images(poses_path, {p.name for p in selected})
    frames = []
    for image in selected:
        width,height,_ = io.image_dimensions(image)
        pose = poses[image.name];camera = cameras[pose["camera_id"]]
        frames.append({"input_path":str(image),"sha256":io.sha256(image),"width":width,"height":height,
            "camera":{"intrinsics":camera,"camera_matrix_for_input":io.pinhole_camera(camera,width,height),
                      "world_to_camera":io.world_to_camera(pose)}})
    if asset_manifest is None and scene == "room":
        asset_manifest = STORAGE/"runs/RW-20261101-019-room-mirror-r02/gaussian_assets.json"
    elif asset_manifest is None:
        candidates = sorted((STORAGE/"runs").glob("*/gaussian_assets.json"))
        asset_manifest = next(p for p in candidates if any(a.get("scene_id")==scene for a in json.loads(p.read_text()).get("assets",[])))
    assets = json.loads(asset_manifest.read_text())
    asset = next(a for a in assets["assets"] if a["scene_id"]==scene)
    ply = io.public_input(Path(asset["path"]))
    if asset["coordinate_space"] != "mip360_colmap_world" or io.sha256(ply) != asset["sha256"]:
        raise ValueError("registered Gaussian identity or coordinate frame differs")
    return frames,asset,{"input_manifest":str(path),"input_manifest_sha256":io.sha256(path),
        "asset_manifest":str(asset_manifest),"asset_manifest_sha256":io.sha256(asset_manifest),
        "cameras_sha256":io.sha256(cameras_path),"poses_sha256":io.sha256(poses_path),
        "scene":scene,"split":inputs["split"],"split_id":inputs["split_id"],
        "quality_reference":"independent captured photo at the exact registered camera; Gaussian training-view membership unverified"}


def exact_plan(width, height, work, context):
    import nr_semantic_plan as planner
    args = Namespace(source_root=STORAGE/"runtime/reference-src/3beef7935a8ef976b47ab78b906e75600c52d2de",
        network="ht",width=width,height=height,
        weights=PUBLIC/"weights/dlss5/310.9.16.0/ht/weights.safetensors",slots=1,sessions=1,
        workspace_bytes=1<<30,reserve_bytes=2<<30,graph_instance_bytes=0,
        device_budget_bytes=96<<30,effective_flops_per_second=None,effective_transfer_bytes_per_second=None)
    report = planner.make_report(args,{**context,"shared_assets_root":str(PUBLIC)})
    path = work/f"plan-{width}x{height}.json"
    if not path.exists():save(path,report)
    return path


def make_runtime(library, plan):
    from renderweave.native_nr import NativeRuntime
    weights = PUBLIC/"weights/dlss5/310.9.16.0/ht/weights.safetensors"
    parameters = STORAGE/"runs/RW-20261101-011-parameters-r02/native_nr_parameters_ht.json"
    return NativeRuntime(weights=str(weights),weights_sha256="4f236d8a58a48653cde064d4a526d194cd1d9f97c65b26bba3e2e49e598ce353",
        parameter_manifest=str(parameters),parameter_manifest_sha256=io.sha256(parameters),
        plan=str(plan),plan_sha256=io.sha256(plan),library=str(library),experimental=True,
        device_ordinal=0,device_budget_bytes=96<<30,host_budget_bytes=8<<30,
        max_sessions=1,max_inflight=1,trace_capacity=128)


def render(arrays, resource, intrinsic, view, width, height):
    import torch
    from gsplat import rasterization
    k = torch.tensor([intrinsic],device="cuda:0",dtype=torch.float32)
    v = torch.tensor([view],device="cuda:0",dtype=torch.float32)
    image,alpha,_ = rasterization(*arrays,v,k,width,height,sh_degree=resource["sh_degree"],
        render_mode="RGB+ED",packed=False,rasterize_mode="classic",
        backgrounds=torch.zeros((1,3),device="cuda:0"))
    return image[0],alpha[0]


def collect_output(job):
    import numpy as np
    result = job.wait(-1)
    info = job.info()
    if info["status"] or not info["physical_completion_confirmed"]:
        raise RuntimeError("native frame did not physically complete")
    view = result.tensor_view()
    data = np.frombuffer(view,dtype=np.float32).reshape(result.metadata()["height"],result.metadata()["width"],3).copy()
    view.release();result.close();job.close()
    return data,info


def public_reference(library,work,context,batched_ab,*,sdr_temporal_head=False):
    """Same-input CUDA comparison with RW651's unchanged CPU reference."""
    import numpy as np
    import torch
    from PIL import Image
    fixture=PUBLIC/"references/renderweave/nr/ht-310.8.0-1080p"
    reference_dir=STORAGE/"runs/RW-20261118-651-public-reference-cpu-r2-reference"
    cpu_path=reference_dir/"reference-quarter.f32.npy"
    image=np.asarray(Image.open(fixture/"input.png").convert("RGB"),dtype=np.float32)/255
    vendor=np.asarray(Image.open(fixture/"vendor.png").convert("RGB"),dtype=np.float32)/255
    cpu=np.load(cpu_path,allow_pickle=False)
    height,width=image.shape[:2]
    plan=exact_plan(width,height,work,context)
    metrics=quality_module.QualityMetrics(PUBLIC/"weights")
    report={"input_sha256":io.sha256(fixture/"input.png"),"vendor_png_sha256":io.sha256(fixture/"vendor.png"),
        "cpu_output_sha256":io.sha256(cpu_path),"cpu_report_sha256":io.sha256(reference_dir/"report.json"),
        "dimensions":[width,height],"reference_role":"author_reported_vendor_capture",
        "profiles":{},"scope":"same-input single-frame/reset; original HT controls and quarter residual gain"}
    previous=os.environ.get("RW_NR_HOPPER_BATCHED_MMA");control=None
    profiles=[("default",False)]+([("batched_mma",True)] if batched_ab else [])
    if sdr_temporal_head:
        profiles=[("sdr_temporal_reset",previous=="1")]
    try:
        for name,enabled in profiles:
            os.environ["RW_NR_HOPPER_BATCHED_MMA"]="1" if enabled else "0"
            runtime=make_runtime(library,plan);session=None
            try:
                session=runtime.session(temporal=sdr_temporal_head)
                if sdr_temporal_head:
                    color=torch.from_numpy(image).to("cuda:0")
                    ready=torch.cuda.Event();ready.record()
                    output,info=collect_output(session.submit_cuda(color,frame_id=0,ready_event=ready))
                else:
                    output,info=collect_output(session.submit(image,frame_id=0))
                identity=runtime.query()
            finally:
                if session:session.close()
                runtime.close()
            output_path=work/("public-reference-"+name+".f32.npy")
            np.save(output_path,output,allow_pickle=False)
            quantized=np.rint(output*255).astype(np.uint8)
            Image.fromarray(quantized).save(work/("public-reference-"+name+".png"))
            score=metrics.pair(quantized.astype(np.float32)/255,vendor)
            report["profiles"][name]={"vs_vendor_png":score,"vs_original_cpu_float":metrics.pair(output,cpu),
                "output_path":str(output_path),"output_file_sha256":io.sha256(output_path),
                "job_info":info,"runtime_identity":identity,
                "single_fixture_45db_0995_met":score["psnr_db"]>=45 and score["ssim"]>=.995}
            if control is None:control=output
            else:
                exact=np.array_equal(control,output);report["batched_mma_output_exact"]=exact
                if not exact:raise AssertionError("batched WGMMA changes public reference output")
    finally:
        if previous is None:os.environ.pop("RW_NR_HOPPER_BATCHED_MMA",None)
        else:os.environ["RW_NR_HOPPER_BATCHED_MMA"]=previous
    save(work/("public-reference-temporal-reset.json" if sdr_temporal_head else "public-reference.json"),report)
    return report


def photo_quality(arrays, resource, frames, adapter, library, work, context):
    import numpy as np
    import torch
    from PIL import Image
    from renderweave.gsplat_cuda import PreparedFrame
    records = [];current_shape=None;runtime=None;session=None
    try:
        for index,frame in enumerate(frames):
            width,height = frame["width"],frame["height"]
            if current_shape != (height,width):
                if session:session.close();runtime.close()
                runtime = make_runtime(library,exact_plan(width,height,work,context))
                session = runtime.session(temporal=True)
                current_shape = (height,width);epoch = 1
            else:
                epoch+=1;session.reset(epoch)
            intrinsic,view = frame["camera"]["camera_matrix_for_input"],frame["camera"]["world_to_camera"]
            output = PreparedFrame.allocate(height,width,intrinsic,view,device="cuda:0")
            rendered,alpha = render(arrays,resource,intrinsic,view,width,height)
            adapter.prepare(rendered,alpha,output,stream=torch.cuda.current_stream())
            ready = torch.cuda.Event();ready.record()
            enhanced,info = collect_output(session.submit_cuda(output.color,frame_id=index,ready_event=ready))
            color = output.color.cpu().numpy()
            epoch+=1;session.reset(epoch)
            cpu,control = collect_output(session.submit(color,frame_id=index,native_history=True))
            exact = np.array_equal(cpu,enhanced)
            if not exact:raise AssertionError("real gsplat CPU/CUDA input transport differs")
            photo = np.asarray(Image.open(frame["input_path"]).convert("RGB"),dtype=np.float32)/255.0
            np.save(work/f"quality-{index}.render.f32.npy",color)
            np.save(work/f"quality-{index}.nr.f32.npy",enhanced)
            Image.fromarray(np.rint(enhanced*255).astype(np.uint8)).save(work/f"quality-{index}.nr.png")
            save(work/f"quality-{index}.json",{"frame":frame,"cpu_cuda_exact":exact,
                "job_info":info,"cpu_control_info":control,
                "render_sha256":io.sha256(work/f"quality-{index}.render.f32.npy"),
                "nr_sha256":io.sha256(work/f"quality-{index}.nr.f32.npy")})
            records.append({"frame":frame,"render":color,"enhanced":enhanced,"reference":photo,
                            "cpu_cuda_exact":exact,"job_info":info,"cpu_control_info":control})
    finally:
        if session:session.close()
        if runtime:runtime.close()
    return records


def motion_control(adapter, device):
    """Independent NumPy reference, including translation/depth/alpha masks."""
    import numpy as np
    import torch
    from renderweave.gsplat_cuda import PreparedFrame
    h,w = 31,47
    k = [[50.,0.,w/2],[0.,51.,h/2],[0.,0.,1.]]
    view = np.eye(4);old_view = np.eye(4);old_view[0,3] = .03
    rendered = np.zeros((h,w,4),dtype=np.float32);rendered[:,:,:3] = [.2,.5,.8];rendered[:,:,3] = 2
    alpha = np.ones((h,w,1),dtype=np.float32);alpha[3:5,4:8] = .2
    old = PreparedFrame.allocate(h,w,k,old_view,device=device)
    old.depth.fill_(2);old.alpha.fill_(1);old.depth[7:9,8:13] = 9
    output = PreparedFrame.allocate(h,w,k,view,device=device)
    adapter.prepare(torch.from_numpy(rendered).to(device),torch.from_numpy(alpha).to(device),output,
                    previous=old,stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    motion,valid = motion_cpu.derive_camera_motion(rendered[:,:,3],alpha[:,:,0],k,view,
        old.depth.cpu().numpy()[:,:,0],old.alpha.cpu().numpy()[:,:,0],k,old_view)
    motion*=np.array([w,h],dtype=np.float32)
    actual = output.motion.cpu().numpy();mask = output.validity.cpu().numpy()[:,:,0]>0.5
    maximum = float(np.max(np.abs(actual-motion)))
    if not np.array_equal(mask,valid) or maximum>1e-5:
        raise AssertionError(f"CUDA guide differs from CPU: max={maximum}, mask={np.count_nonzero(mask!=valid)}")
    return {"shape":[h,w],"mask_exact":True,"motion_max_abs_pixels":maximum,
            "valid_pixels":int(valid.sum()),"nonzero_motion":bool(np.any(actual))}


def continuous(arrays, resource, anchor, adapter, library, work, context, count, *, pipelined=False):
    import hashlib
    import numpy as np
    import torch
    from renderweave.gsplat_cuda import PreparedFrame
    width,height = 1920,1080
    intrinsic = io.pinhole_camera(anchor["camera"]["intrinsics"],width,height)
    base_view = np.asarray(anchor["camera"]["world_to_camera"],dtype=np.float64)
    buffers = [PreparedFrame.allocate(height,width,intrinsic,base_view.copy(),device="cuda:0") for _ in range(2)]
    producer_stream = torch.cuda.Stream()
    events = [[torch.cuda.Event(enable_timing=True) for _ in range(3)] for _ in range(2)]
    runtime = make_runtime(library,exact_plan(width,height,work,context));session = runtime.session(temporal=True)
    frames=[];warmup=16;started=None;previous_delivery=None
    def produce(index, previous):
        begin=time.monotonic_ns()
        output=buffers[index%2];output.view=base_view.copy();output.view[0,3]+=index*.0005
        start,stop,ready=events[index%2]
        with torch.cuda.stream(producer_stream):
            start.record()
            rendered,alpha=render(arrays,resource,intrinsic,output.view,width,height)
            stop.record()
            adapter.prepare(rendered,alpha,output,previous=previous,stream=producer_stream)
            ready.record()
        # Hold renderer inputs through the producer event/NR completion.
        return {"output":output,"begin":begin,"events":(start,stop,ready),"inputs":(rendered,alpha)}
    try:
        current=produce(0,None)
        for index in range(warmup+count):
            if index==warmup:started=time.monotonic_ns()
            output=current["output"];start,stop,ready=current["events"]
            job=session.submit_cuda(output.color,frame_id=index,
                motion_hwc2=output.motion if index else None,
                motion_validity=output.validity if index else None,ready_event=ready)
            # Camera/depth guides do not depend on the NR output. Prepare the
            # next producer frame while this one runs, on separate buffers.
            upcoming=produce(index+1,output) if pipelined and index+1<warmup+count else None
            if previous_delivery is not None:
                old_view,old_record=previous_delivery
                if old_record is not None:old_record["output_sha256"]=hashlib.sha256(old_view.cast("B")).hexdigest()
                old_view.release();previous_delivery=None
            result=job.wait(-1);info=job.info();view=result.tensor_view()
            if info["status"] or not info["physical_completion_confirmed"]:raise RuntimeError("real frame failed")
            completed=time.monotonic_ns();record=None
            if index>=warmup:
                record={"frame_id":index,"pts":index,"time_base":[1,30],"reset":False,
                    "expected_predecessor":index-1,"render_gpu_ms":start.elapsed_time(stop),
                    "guide_gpu_ms":stop.elapsed_time(ready),"nr_gpu_ms":info["device_stream_span_ms"],
                    "producer_to_cpu_ms":(completed-current["begin"])/1e6,
                    "host_output_ready_ms":(completed-started)/1e6,"job_info":info}
                frames.append(record)
            result.close();job.close();previous_delivery=(view,record)
            if index+1<warmup+count:
                current=upcoming if pipelined else produce(index+1,output)
        elapsed=(time.monotonic_ns()-started)/1e9
        if previous_delivery is not None:
            old_view,old_record=previous_delivery
            if old_record is not None:old_record["output_sha256"]=hashlib.sha256(old_view.cast("B")).hexdigest()
            old_view.release();previous_delivery=None
        # Guides are observed only after the measured producer/model loop.
        current=buffers[(warmup+count-1)%2];old=buffers[1-(warmup+count-1)%2]
        gpu_motion=current.motion.cpu().numpy();gpu_valid=current.validity.cpu().numpy()[:,:,0]
        cpu_motion,cpu_valid=motion_cpu.derive_camera_motion(
            current.depth.cpu().numpy()[:,:,0],current.alpha.cpu().numpy()[:,:,0],current.intrinsic,current.view,
            old.depth.cpu().numpy()[:,:,0],old.alpha.cpu().numpy()[:,:,0],old.intrinsic,old.view)
        cpu_motion*=np.array([width,height],dtype=np.float32)
        mask_mismatch=int(np.count_nonzero((gpu_valid>.5)!=cpu_valid))
        motion_error=float(np.max(np.abs(gpu_motion-cpu_motion)))
        if mask_mismatch or motion_error>1e-4:
            raise AssertionError(f"real GPU camera guide differs: mask={mask_mismatch}, max_pixels={motion_error}")
        q=runtime.query()
        return {"frames":frames,"measured_frames":count,"warmup_frames":warmup,
            "elapsed_seconds":elapsed,"host_fps":count/elapsed,"input_boundary":"real gsplat render+CUDA guides+NR+CPU-readable output",
            "producer_overlap":pipelined,"producer_buffers":2,"producer_stream":producer_stream.cuda_stream,
            "resolution":[width,height],"sessions":1,"max_outstanding_nr":1,
            "motion_valid_fraction_last":float(gpu_valid.mean()),"motion_max_abs_pixels_last":float(np.abs(gpu_motion).max()),
            "runtime_identity":q,"real_camera_guide_cpu_control":{"validity_mismatch_pixels":mask_mismatch,"motion_max_abs_error_pixels":motion_error},
            "timing_scope":"steady throughput after16 warmups; per-frame producer_to_cpu latency includes producer prefetch lead; all output SHA, previous hash overlaps current model",
            "generated_timeline":"static trained scene, synthetic camera translation0.0005 scene units/frame at30fps; captured chronology not inferred"}
    finally:
        if previous_delivery is not None:previous_delivery[0].release()
        try:
            session.close();runtime.close()
        finally:
            producer_stream.synchronize()


def pinned_readback_ab(arrays,resource,anchor,adapter,library,work,context,report):
    # Keep scene, SDK, stream priority, and all model controls fixed.
    os.environ["RW_NR_STREAM_PRIORITY"]="high"
    key="RW_NR_PINNED_READBACK"
    prior=os.environ.get(key)
    expected=None
    labels=("pageable_a","pinned_a","pinned_b","pageable_b")
    try:
        for label in labels:
            pinned=label.startswith("pinned")
            os.environ[key]="1" if pinned else "0"
            stage=continuous(arrays,resource,anchor,adapter,library,work,context,600,pipelined=True)
            report["stages"][label]=stage
            stage["pinned_readback"]=pinned
            observed=[f["output_sha256"] for f in stage["frames"]]
            if expected is None:expected=observed
            if observed!=expected:raise AssertionError(f"{label} changes real sequence outputs")
            slots=stage["runtime_identity"]["slots"]
            calls=sum(s["readback_pinned_jobs"] for s in slots)
            if calls!=(616 if pinned else 0):raise AssertionError(f"{label} exercised {calls} pinned reads")
    finally:
        if prior is None:os.environ.pop(key,None)
        else:os.environ[key]=prior
    stages=report["stages"]
    report["readback_ab"]={
        "order":list(labels),"sampled_frames_per_stage":600,"all_output_hashes_exact":True,
        "pinned_both_runs_at_least_30fps":all(stages[n]["host_fps"]>=30 for n in ("pinned_a","pinned_b")),
    }
    report["selected_continuous_stage"]=min(("pinned_a","pinned_b"),key=lambda n:stages[n]["host_fps"])



def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--library",required=True,type=Path)
    parser.add_argument("--guide-library",required=True,type=Path)
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--frames",type=int,default=64)
    parser.add_argument("--scene",default="room")
    parser.add_argument("--input-manifest",type=Path)
    parser.add_argument("--asset-manifest",type=Path)
    parser.add_argument("--batched-mma-ab",action="store_true")
    parser.add_argument("--public-reference",action="store_true")
    parser.add_argument("--sdr-temporal-head-ab",action="store_true")
    parser.add_argument("--pinned-readback-ab",action="store_true")
    args=parser.parse_args();work=STORAGE/"runs"/args.run_id;work.mkdir(parents=True,exist_ok=False)
    report={"project_id":"renderweave","task_id":"RW-20261117-650","run_id":args.run_id,
            "status":"FAIL","slurm_job_id":os.environ["SLURM_JOB_ID"],"stages":{},"started_utc":datetime.now(timezone.utc).isoformat()}
    try:
        import ctypes as c
        # Pin the project's measured CUDA math dependencies before Torch loads
        # its bundled libraries into this same GPU producer/SDK process.
        sdk_preload=c.CDLL(str(args.library),mode=c.RTLD_GLOBAL)
        math_library=c.CDLL("libcublas.so.13")
        math_library.cublasGetProperty.argtypes=[c.c_int,c.POINTER(c.c_int)]
        math_library.cublasGetProperty.restype=c.c_int
        properties=[]
        for field in range(3):
            value=c.c_int();status=math_library.cublasGetProperty(field,c.byref(value))
            if status:raise RuntimeError("query preloaded CUDA math version: "+str(status))
            properties.append(value.value)
        report["preloaded_cublas_version"]=properties
        import numpy as np
        import torch
        from renderweave.h200_profile import configure_h200_fast
        from renderweave.gsplat_cuda import CudaGuideAdapter
        import renderweave.native_nr as nr
        torch.set_grad_enabled(False);torch.set_num_threads(4);torch.cuda.set_device(0)
        if torch.cuda.device_count()!=1:raise RuntimeError("one physical CUDA device required")
        configure_h200_fast();os.environ.update(RW_NR_EXPERIMENTAL_FEATURE_MODE="noise",RW_NR_EXPERIMENTAL_INTENSITY="1")
        if args.sdr_temporal_head_ab or args.pinned_readback_ab:os.environ["RW_NR_SDR_TEMPORAL_HEAD"]="0"
        if args.batched_mma_ab or args.pinned_readback_ab:os.environ["RW_NR_HOPPER_BATCHED_MMA"]="0"
        report["effective_rw_nr_env"]={k:v for k,v in os.environ.items() if k.startswith("RW_NR_")}
        report["environment"]={"torch":torch.__version__,"cuda":torch.version.cuda,"device":torch.cuda.get_device_name(0),
            "uuid":str(torch.cuda.get_device_properties(0).uuid),"visible_device_count":torch.cuda.device_count()}
        report["mapped_cuda_math_libraries"]=sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "libcublas" in line})
        context=io.execution_context(work/"inputs.json")[1]
        report["execution_context"]=context
        if args.public_reference:
            report["stages"]["public_reference"]=public_reference(args.library,work,context,args.batched_mma_ab)
        frames,asset,identity=registered_inputs(args.scene,4,args.input_manifest,args.asset_manifest)
        report.update(source_commit=context["source_commit"],inputs=identity,asset=asset)
        arrays,resource=renderer.load_graphdeco_ply(Path(asset["path"]),torch.device("cuda:0"),10000000)
        report["gaussian_resources"]=resource
        adapter=CudaGuideAdapter(args.guide_library)
        report["stages"]["motion_math"]=motion_control(adapter,"cuda:0")
        with torch.inference_mode():
            records=photo_quality(arrays,resource,frames,adapter,args.library,work,context)
            report["stages"]["photo_transport"]=[json.loads((work/f"quality-{i}.json").read_text()) for i in range(len(records))]
            if args.pinned_readback_ab:
                pinned_readback_ab(arrays,resource,frames[0],adapter,args.library,work,context,report)
            else:
                report["stages"]["continuous_serial"]=continuous(arrays,resource,frames[0],adapter,args.library,work,context,args.frames)
                report["stages"]["continuous"]=continuous(arrays,resource,frames[0],adapter,args.library,work,context,args.frames,pipelined=True)
                serial_sha=[f["output_sha256"] for f in report["stages"]["continuous_serial"]["frames"]]
                pipeline_sha=[f["output_sha256"] for f in report["stages"]["continuous"]["frames"]]
                report["producer_overlap_output_exact"]=serial_sha==pipeline_sha
                if serial_sha!=pipeline_sha:raise AssertionError("producer overlap changes real sequence outputs")
                os.environ["RW_NR_STREAM_PRIORITY"]="high"
                report["stages"]["continuous_high_priority"]=continuous(arrays,resource,frames[0],adapter,args.library,work,context,args.frames,pipelined=True)
                high_sha=[f["output_sha256"] for f in report["stages"]["continuous_high_priority"]["frames"]]
                report["priority_output_exact"]=high_sha==serial_sha
                if high_sha!=serial_sha:raise AssertionError("stream priority changes real sequence outputs")
                selected="high" if report["stages"]["continuous_high_priority"]["host_fps"]>=report["stages"]["continuous"]["host_fps"] else "normal"
                report["selected_nr_stream_priority"]=selected
                os.environ["RW_NR_STREAM_PRIORITY"]=selected
                selected_result=report["stages"]["continuous_high_priority" if selected=="high" else "continuous"]
                report["selected_continuous_stage"]="continuous_high_priority" if selected=="high" else "continuous"
                if args.batched_mma_ab:
                    os.environ["RW_NR_HOPPER_BATCHED_MMA"]="1"
                    candidate=continuous(arrays,resource,frames[0],adapter,args.library,work,context,args.frames,pipelined=True)
                    report["stages"]["continuous_batched_mma"]=candidate
                    candidate_sha=[f["output_sha256"] for f in candidate["frames"]]
                    report["batched_mma_output_exact"]=candidate_sha==serial_sha
                    if candidate_sha!=serial_sha:raise AssertionError("batched WGMMA changes real sequence outputs")
                    use_candidate=candidate["host_fps"]>selected_result["host_fps"]
                    report["selected_batched_mma"]=use_candidate
                    os.environ["RW_NR_HOPPER_BATCHED_MMA"]="1" if use_candidate else "0"
                    if use_candidate:
                        selected_result=candidate
                        report["selected_continuous_stage"]="continuous_batched_mma"
                if selected_result["host_fps"]>=30 and args.frames<600:
                    report["stages"]["continuous_600"]=continuous(arrays,resource,frames[0],adapter,args.library,work,context,600,pipelined=True)
                    prefix=[f["output_sha256"] for f in report["stages"]["continuous_600"]["frames"][:args.frames]]
                    report["continuous_600_prefix_exact"]=prefix==serial_sha
                    if prefix!=serial_sha:raise AssertionError("600-frame real pipeline changes sequence prefix")
                if args.sdr_temporal_head_ab:
                    os.environ["RW_NR_SDR_TEMPORAL_HEAD"]="1"
                    try:
                        if args.public_reference:
                            report["stages"]["public_reference_sdr_temporal_reset"]=public_reference(
                                args.library,work,context,False,sdr_temporal_head=True)
                        candidate=continuous(arrays,resource,frames[0],adapter,args.library,
                                             work,context,args.frames,pipelined=True)
                        report["stages"]["continuous_sdr_temporal_head"]=candidate
                        report["temporal_head_different_frames"]=sum(
                            f["output_sha256"]!=control["output_sha256"]
                            for f,control in zip(candidate["frames"],selected_result["frames"],strict=True))
                        if candidate["host_fps"]>=30 and args.frames<600:
                            sustained=continuous(arrays,resource,frames[0],adapter,args.library,work,context,600,pipelined=True)
                            report["stages"]["continuous_sdr_temporal_head_600"]=sustained
                            expected=[f["output_sha256"] for f in candidate["frames"]]
                            actual=[f["output_sha256"] for f in sustained["frames"][:args.frames]]
                            if actual!=expected:raise AssertionError("SDR temporal head restart changes sequence prefix")
                    finally:
                        os.environ["RW_NR_SDR_TEMPORAL_HEAD"]="0"
        report["cuda_input_live_leases_after_close"]=len(nr._cuda_owners)
        if nr._cuda_owners:raise RuntimeError("borrowed inputs not released after closed runtimes")
        # Metrics run on CPU after the timed GPU loop; original public weights.
        metrics=quality_module.QualityMetrics(PUBLIC/"weights")
        report["metric_weights"]=metrics.identities
        report["metric_contract"]={"domain":"native captured SDR sRGB RGB code values [0,1], no resizing",
            "psnr":"MAX=1,all RGB samples","ssim":"Gaussian11,sigma1.5,population covariance,C1=.01^2,C2=.03^2,valid5px border",
            "lpips":"AlexNet v0.1,original backbone/calibration,CPU functional evaluation,normalize=True",
            "native_vendor_reference":False,"comparison":"same-camera rendered input and NR output vs independent photo; not an input→input score"}
        report["quality"]=[]
        for index,record in enumerate(records):
            baseline=metrics.pair(record["render"],record["reference"])
            enhanced=metrics.pair(record["enhanced"],record["reference"])
            report["quality"].append({"frame":record["frame"],"baseline":baseline,"nr":enhanced,
                "delta_psnr_db":enhanced["psnr_db"]-baseline["psnr_db"],
                "delta_ssim":enhanced["ssim"]-baseline["ssim"],
                "delta_lpips":enhanced["lpips_alex_v01"]-baseline["lpips_alex_v01"],
                "cpu_cuda_output_exact":record["cpu_cuda_exact"],"job_info":record["job_info"],
                "output":str(work/f"quality-{index}.nr.f32.npy")})
        zero=metrics.pair(records[0]["reference"],records[0]["reference"])
        assert zero["lpips_alex_v01"]==0 and zero["numerically_identical"] and abs(zero["ssim"]-1)<1e-10
        report["metric_identity_control"]=zero
        # A derived comparison figure only; metrics above use native pixels.
        from PIL import Image,ImageDraw
        import base64
        thumb_width=640;row_height=round(thumb_width*records[0]["reference"].shape[0]/records[0]["reference"].shape[1])+30
        board=Image.new("RGB",(3*thumb_width,row_height*len(records)),"white")
        draw=ImageDraw.Draw(board)
        for row,(record,scores) in enumerate(zip(records,report["quality"])):
            for col,(key,label) in enumerate([("reference","Reference photo"),("render","GSplat"),("enhanced","NR")]):
                image=Image.fromarray(np.rint(np.clip(record[key],0,1)*255).astype(np.uint8))
                image.thumbnail((thumb_width,row_height-30))
                board.paste(image,(col*thumb_width,row*row_height+30))
                if col:label+=f" PSNR={scores['baseline' if col==1 else 'nr']['psnr_db']:.3f} dB"
                draw.text((col*thumb_width+8,row*row_height+8),label,fill="black")
        preview=work/"quality_comparison.jpg";board.save(preview,quality=88)
        report["quality_preview"]={"path":str(preview),"sha256":io.sha256(preview),"format":"jpeg",
            "scope":"derived display comparison; original public photos remain in dataset; metrics use original sizes",
            "base64":base64.b64encode(preview.read_bytes()).decode()}
        report["status"]="PASS"
    except BaseException as error:
        import traceback
        report["error"]={"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()}
    report["finished_utc"]=datetime.now(timezone.utc).isoformat();save(work/"report.json",report)
    print(json.dumps({"status":report["status"],"report":str(work/"report.json"),"error":report.get("error")},ensure_ascii=False))
    return 0 if report["status"]=="PASS" else 1


if __name__=="__main__":
    raise SystemExit(main())
