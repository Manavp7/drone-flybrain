"""Download a specific official YOLOX ONNX artifact and record local provenance.

A computed SHA256 locks downloaded bytes; it is not a publisher-signed checksum.
This script never loads or executes the model, and never replaces existing files.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=['s','m','l','tiny','nano'],default='s')
    parser.add_argument('--output-dir',type=Path,default=Path('models'))
    args=parser.parse_args()
    name=f'yolox_{args.variant}.onnx'
    url='https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/'+name
    args.output_dir.mkdir(parents=True,exist_ok=True)
    destination=args.output_dir/name
    manifest=destination.with_suffix('.manifest.json')
    if destination.exists() or manifest.exists():
        raise FileExistsError('Model or manifest already exists; choose a new output directory')
    digest=hashlib.sha256(); count=0
    with tempfile.TemporaryDirectory(prefix='download-',dir=args.output_dir) as temporary:
        staged=Path(temporary)/name
        with urllib.request.urlopen(url,timeout=30) as response,staged.open('xb') as stream:
            if not response.geturl().startswith('https://'):
                raise ValueError('Model download must remain HTTPS')
            while block:=response.read(1024*1024):
                count+=len(block)
                if count>512*1024*1024:
                    raise ValueError('Artifact exceeds 512 MiB limit')
                digest.update(block); stream.write(block)
            stream.flush(); os.fsync(stream.fileno())
        if count<100_000:
            raise ValueError('Artifact too small for expected YOLOX model')
        data={'artifact':name,'source_url':url,'sha256':digest.hexdigest(),'bytes':count,
              'input_size':416 if args.variant in ('tiny','nano') else 640,
              'output_format':'raw_yolox','preprocessing':'yolox_bgr_114',
              'downloaded_utc':datetime.now(timezone.utc).isoformat(),
              'provenance':'local_digest_of_https_download_not_publisher_signature',
              'inference_executed':False,
              'upstream_repository':'https://github.com/Megvii-BaseDetection/YOLOX',
              'upstream_code_license':'Apache-2.0'}
        # Exclusive hard-link creation prevents overwriting an existing model.
        os.link(staged,destination)
        with manifest.open('x') as stream:
            json.dump(data,stream,indent=2); stream.write('\n')
    print(json.dumps(data,indent=2))


if __name__=='__main__': main()
