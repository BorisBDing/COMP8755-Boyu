import os
import hashlib
import shutil
import urllib

_TRANSFORMER_MODELS ={
    'clip-vit-base-patch32':("pretrained/clip-vit-base-patch32"),
    
}


_OPENCLIP_S3_BUCKET = 'https://clip-as-service.s3.us-east-2.amazonaws.com/models/torch'
_OPENCLIP_MODELS = {
    'ViT-B-32::openai': ('ViT-B-32.pt', '3ba34e387b24dfe590eeb1ae6a8a122b'),
    'roberta-ViT-B-32::laion2b-s12b-b32k': (
        'roberta-ViT-B-32-laion2b-s12b-b32k.bin',
        '76d4c9d13774cc15fa0e2b1b94a8402c',
    ),
}


_VISUAL_MODEL_IMAGE_SIZE = {
    'RN50': 224,
    'RN101': 224,
    'RN50x4': 288,
    'RN50x16': 384,
    'RN50x64': 448,
    'ViT-B-32': 224,
    'roberta-ViT-B-32': 224,
    'xlm-roberta-base-ViT-B-32': 224,
    'ViT-B-16': 224,
    'Vit-B-16Plus': 240,
    'ViT-B-16-plus-240': 240,
    'ViT-L-14': 224,
    'ViT-L-14-336': 336,
    'ViT-H-14': 224,
    'xlm-roberta-large-ViT-H-14': 224,
    'ViT-g-14': 224,
}


def md5file(filename: str):
    hash_md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def get_model_url_md5(name: str):
    model_pretrained = _OPENCLIP_MODELS[name]
    if len(model_pretrained) == 0:  # not on s3
        return None, None
    else:
        return (_OPENCLIP_S3_BUCKET + '/' + model_pretrained[0], model_pretrained[1])


def download_model(
    url: str,
    target_folder: str = os.path.expanduser("pretrained"),
    md5sum: str = None,
    with_resume: bool = True,
    max_attempts: int = 3,
) -> str:
    os.makedirs(target_folder, exist_ok=True)
    filename = os.path.basename(url)

    download_target = os.path.join(target_folder, filename)

    if os.path.exists(download_target):
        if not os.path.isfile(download_target):
            raise FileExistsError(f'{download_target} exists and is not a regular file')

        actual_md5sum = md5file(download_target)
        if (not md5sum) or actual_md5sum == md5sum:
            return download_target

    from rich.progress import (
        DownloadColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    progress = Progress(
        " \n",  # divide this bar from Flow's bar
        TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
    )

    with progress:
        task = progress.add_task('download', filename=filename, start=False)

        for _ in range(max_attempts):
            tmp_file_path = download_target + '.part'
            resume_byte_pos = (
                os.path.getsize(tmp_file_path) if os.path.exists(tmp_file_path) else 0
            )

            try:
                # resolve the 403 error by passing a valid user-agent
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                total_bytes = int(
                    urllib.request.urlopen(req).info().get('Content-Length', -1)
                )
                mode = 'ab' if (with_resume and resume_byte_pos) else 'wb'

                with open(tmp_file_path, mode) as output:
                    progress.update(task, total=total_bytes)
                    progress.start_task(task)

                    if resume_byte_pos and with_resume:
                        progress.update(task, advance=resume_byte_pos)
                        req.headers['Range'] = f'bytes={resume_byte_pos}-'

                    with urllib.request.urlopen(req) as source:
                        while True:
                            buffer = source.read(8192)
                            if not buffer:
                                break

                            output.write(buffer)
                            progress.update(task, advance=len(buffer))

                actual_md5 = md5file(tmp_file_path)
                if (md5sum and actual_md5 == md5sum) or (not md5sum):
                    shutil.move(tmp_file_path, download_target)
                    return download_target
                else:
                    os.remove(tmp_file_path)
                    raise RuntimeError(
                        f'MD5 mismatch: expected {md5sum}, got {actual_md5}'
                    )

            except Exception as ex:
                progress.console.print(
                    f'Failed to download {url} with {ex!r} at the {_}th attempt'
                )
                progress.reset(task)

        raise RuntimeError(
            f'Failed to download {url} within retry limit {max_attempts}'
        )
