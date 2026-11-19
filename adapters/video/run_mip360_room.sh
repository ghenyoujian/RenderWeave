#!/bin/bash
# One real779x519 MIP360 gsplat view -> lossless video -> native SDK -> lossless video.
set -euo pipefail
usage() { echo 'Usage: run_mip360_room.sh --output-dir ABS_PROJECT_RUN_SUBDIR --run-id RW_ID [--contracts] [--sdk-library ABS_PROJECT_LIBRARY]' >&2; exit 64; }
output_dir='' run_id='' contracts=0 sdk_library=''
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) [[ $# -ge 2 ]] || usage; output_dir=$2; shift 2 ;;
        --run-id) [[ $# -ge 2 ]] || usage; run_id=$2; shift 2 ;;
        --contracts) contracts=1; shift ;;
        --sdk-library) [[ $# -ge 2 ]] || usage; sdk_library=$2; shift 2 ;;
        *) usage ;;
    esac
done
[[ -n "$output_dir" && -n "$run_id" && "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || usage
source_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
source "$source_root/tools/dev/env.sh"
[[ "$PROJECT_EXECUTION_ROLE" == h200_release && "${source_root##*/}" == "$(git -C "$source_root" rev-parse HEAD)" ]] || exit 65
[[ -z "$(git -C "$source_root" status --porcelain)" ]] || { echo 'Published release is dirty' >&2; exit 65; }
media_python="$PROJECT_STORAGE_ROOT/envs/renderweave-media/bin/python"
[[ -x "$media_python" ]] || { echo 'Install project-private media environment first' >&2; exit 69; }
[[ -n "$sdk_library" ]] || sdk_library="$PROJECT_STORAGE_ROOT/envs/renderweave/lib/librw_nr_sdk.so"
render_manifest="$PROJECT_STORAGE_ROOT/runs/RW-20261101-012-room-render-r01/gsplat-room/inputs.json"
model_manifest="$PROJECT_STORAGE_ROOT/runs/RW-20261101-011-parameters-r02/native_nr_parameters_ht.json"
weights="$PROJECT_DOWNLOADED_WEIGHTS_ROOT/dlss5/310.9.16.0/ht/weights.safetensors"
reference_source="$PROJECT_STORAGE_ROOT/runtime/reference-src/3beef7935a8ef976b47ab78b906e75600c52d2de"
"$media_python" - "$source_root" "$output_dir" "$render_manifest" "$model_manifest" "$sdk_library" <<'PY'
from pathlib import Path
import hashlib,json,os,sys
source,output,render,model,library=map(Path,sys.argv[1:]); storage=Path(os.environ['PROJECT_STORAGE_ROOT'])
if not output.is_absolute() or not output.resolve().is_relative_to(storage/'runs'): raise SystemExit('Output escapes project runs')
for path in (render,model,library):
 if not path.resolve().is_relative_to(storage): raise SystemExit('Private input/library escapes project')
 if not path.is_file(): raise SystemExit('Required input/library absent: '+str(path))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
if sha(render)!='24ecdf01011f9e995d7633edb68a977a9d8c824910f484e2acd8a227e389084b': raise SystemExit('Registered gsplat render manifest changed')
if sha(model)!='9a40bd6b3e8967137ce74e4119eb066ea80be5bca34b314730083722d03979ee': raise SystemExit('Original R02 parameter registration changed')
r=json.loads(render.read_text());m=json.loads(model.read_text())
if r.get('status')!='COMPLETED' or r.get('renderer')!='gsplat' or len(r['frames'])!=1: raise SystemExit('Expected completed single-view gsplat capture')
frame=r['frames'][0]
if [frame['width'],frame['height']]!=[779,519] or frame['sha256']!='02cc22b0fc46b7eb1ddfb18eb6c0d1ab32bc76a0bef3b6dcbe6ef11f3b9295c7': raise SystemExit('Real frame geometry/hash differs')
if m['variant']!='ht' or m['parameters_sha256']!='4f236d8a58a48653cde064d4a526d194cd1d9f97c65b26bba3e2e49e598ce353': raise SystemExit('Original HT parameter identity differs')
output.mkdir(parents=True,exist_ok=False)
with (output/'room_run_contract.json').open('x') as stream:
 json.dump({'project_id':'renderweave','task_id':'RW-20261101-021','source_commit':source.name,
 'slurm_job_id':os.environ['SLURM_JOB_ID'],'real_input_size':[779,519],'resize_performed':False,
 'render_manifest_sha256':sha(render),'model_manifest_sha256':sha(model),'SDK_library_sha256':sha(library),
 'model':'original_HT','mode':'stateless_reset_each_frame','generated_timeline':'explicit15Hz_not_camera_capture_PTS',
 'device_budget_bytes':17179869184,'SDK_host_budget_bytes':2147483648,'slots':1,'sessions':1,
 'workspace_allowance_bytes':1073741824,'reserve_allowance_bytes':2147483648,
 'requests_are_not_measured_peaks':True,'qualification_passed':False},stream,indent=2);stream.write('\n')
PY
"$media_python" "$source_root/tools/graph/nr_semantic_plan.py" \
    --source-root "$reference_source" --network ht --width 779 --height 519 --weights "$weights" \
    --slots 1 --sessions 1 --workspace-bytes 1073741824 --reserve-bytes 2147483648 \
    --device-budget-bytes 17179869184 --output "$output_dir/semantic-plan-779x519.json" \
    >"$output_dir/planner.log" 2>&1
"$media_python" - "$output_dir/semantic-plan-779x519.json" <<'PY'
from pathlib import Path
import json,sys
p=json.loads(Path(sys.argv[1]).read_text())
assert p['input']['width']==779 and p['input']['height']==519
assert p['geometry']['width']==832 and p['geometry']['height']==576, 'Pinned source geometry differs; do not bucket or resize'
assert p['resources']['execution_slots']==1 and p['resources']['resident_sessions']==1
assert p['resources']['fits_modelled_budget'] is True, 'Resource derivation does not fit preregistered16GiB; stop before GPU'
assert p['network']=='ht' and p['layer_count']==152 and p['unknown_layer_kind_count']==0
print(json.dumps({'geometry':p['geometry'],'resources':p['resources'],'lower_bounds':p['lower_bounds']}))
PY
"$media_python" "$source_root/adapters/video/run.py" encode \
    --sequence-manifest "$render_manifest" --generated-fps 15 --color-profile srgb-bt709-full-rgb8 \
    --max-frames 1 --output-dir "$output_dir/source-video" --run-id "${run_id}:source"
"$media_python" "$source_root/adapters/video/run.py" enhance \
    --input-video "$output_dir/source-video/input.nut" --input-provenance "$output_dir/source-video/adapter.json" \
    --model "$model_manifest" --plan "$output_dir/semantic-plan-779x519.json" --sdk-library "$sdk_library" \
    --device-budget-mib 16384 --host-budget-mib 2048 --color-profile srgb-bt709-full-rgb8 \
    --max-frames 1 --save-raw --output-dir "$output_dir/native-video" --run-id "${run_id}:native"
if [[ "$contracts" == 1 ]]; then
    "$media_python" "$source_root/adapters/video/codec_contracts.py" \
        --sequence-manifest "$render_manifest" --model "$model_manifest" \
        --plan "$output_dir/semantic-plan-779x519.json" --sdk-library "$sdk_library" \
        --frames 8 --output-dir "$output_dir/codec-contracts" --run-id "${run_id}:contracts"
fi
printf 'Real779x519 gsplat→video→nativeSDK→video evidence: %s\n' "$output_dir"
