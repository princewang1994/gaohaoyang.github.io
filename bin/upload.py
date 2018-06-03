#!/usr/local/bin/python3
import re
import os
import time
import argparse
import random
import copy
import subprocess
import configparser
from PIL import Image

# valid image postfix
types = ['png', 'jpg', 'jpeg', 'gif']

# image postifx regex
img_postfix_re = '\.' + '|'.join(types)

# markdown image regex
md_img_re = r'\!\[.*\]\((.*?)\)'

# global variables
ak, sk, url, bucket, args = None, None, None, None, None

def make_filename(type):
    prefix = 'debug' if debug else 'blog'
    rnd = random.randint(0, 999)
    return prefix + '_' + time.strftime('%Y%m%d%H%M%S') + '%04d' % rnd + '.' + type

def make_blank_config(config_file):
    """Generate blank config
    Parameters:
        ak: app key
        sk: secret key
        url: prefix
        bucket: bucket in QiNiu
    Example:
        [qiniu]
        ak=F9-lVHzO3KWKAAbJA_FYTL1l8WEx2fJuNubSJXRv
        sk=wEg_xzmOCaYqzZQEEE87sfkJ2-wfYCYfgKy7Bi7Y
        url=http://oodd7tmt5.bkt.clouddn.com/
        bucket=mypicbed
    """

    config = configparser.RawConfigParser(allow_no_value=True)
    config.add_section('qiniu')

    config.set('qiniu', 'ak', value='your QiNiu AppKey')
    config.set('qiniu', 'sk', value='your QiNiu Secret Key')
    config.set('qiniu', 'url', value='your http prefix')
    config.set('qiniu', 'bucket', value='your Bucket')
    config.write(open(config_file, 'w'))

    return config

def register(config):
    """
    Params:
        config: qiniu config
    """

    global ak, sk, url, bucket

    ak = config.get('qiniu', 'ak')
    sk = config.get('qiniu', 'sk')
    url = config.get('qiniu', 'url')
    bucket = config.get('qiniu', 'bucket')

    # authenticate
    ret = os.system('./bin/qshell account {} {}'.format(ak, sk))

    if ret == 0:
        print('Register Success!')
    else:
        raise Exception('Register Error')

def upload_to_qiniu(file_path, config):
    """
    Params:
        file_path: target file path
        config: qiniu config
    Return:
        Uploaded url on QiNiu
    """

    global url, debug
    # make random filename
    file_type = os.path.basename(file_path).split('.')[-1]
    filename = make_filename(file_type)

    # upload
    if debug:
        ret = subprocess.call(['./bin/qshell', 'fput', bucket, filename, file_path])
    else:
        dev_null = open('/dev/null', "w")
        ret = subprocess.call(['./bin/qshell', 'fput', bucket, filename, file_path], stdout=dev_null, stderr=dev_null)
        dev_null.close()

    if ret == 0:
        _url = os.path.join(url, filename)
        print('{} -> {} Uploaded.'.format(file_path, _url))
        return _url
    else:
        raise Exception('Error')

def get_par_dir(path):

    basename = os.path.basename(path)
    path = path[:-len(basename)]

    if path == '':
        return '.'
    else:
        return path

def compress_image(path, thresh=0.2):
    """Compress Image
    """
    
    file_stat = os.stat(path)
    basename = os.path.basename(path)
    file_type = basename.split('.')[-1]

    # dont compress gif file
    if file_type == 'gif':
        return path

    file_size = file_stat.st_size / 1024.**2
    # if file is larger than 200k
    if file_size > 0.2:
        print('Image `{}` is larger than 200k({:.2}m), need to compress.'.format(path, file_size))
        import tempfile
        temp_dir = tempfile.gettempdir()

        img = Image.open(path)
        H, W = img.size
        if H > W:
            w = int(thresh * 2048 * 2)
            h = int(H * w / W)
        else:
            h = int(thresh * 2048 * 2)
            w = int(W * h / H)

        new_path = os.path.join(temp_dir, 'compress_' + basename)
        img.resize((h, w)).save(new_path)

        return new_path
    else:
        return path


def main(args):

    global md_img
    
    config_dir = os.path.join(os.environ['HOME'], '.qiniu')
    config_path = os.path.join(config_dir, 'qiniu_config.ini')

    # if config file not exist, make blank one
    if not os.path.exists(config_path):
        print('Config file not found, making blank config file to `{}`'.format(config_path))
        if not os.path.exists(config_dir):
            os.mkdir(config_dir)
        make_blank_config(config_path)
    
    config = configparser.RawConfigParser(allow_no_value=True)
    config.read(config_path)

    # get parent dir
    parent_path = get_par_dir(args.md_path)

    with open(args.md_path, 'r') as f:
        text = f.read()
        ori_text = copy.copy(text)

    # find markdown image in text
    md_imgs = re.findall(md_img_re, text)

    try:
        if len(md_imgs):
            print('{} images found:'.format(len(md_imgs)))
            for img in md_imgs:
                print('-> {}'.format(img))
            # register qiniu
            register(config)
        else:
            print('No images to deploy')
            exit(0)

        uploaded_img = 0

        for md_img in md_imgs:
            ori_img = md_img
            md_img = md_img.replace('%20', ' ')        
            path = os.path.join(parent_path, md_img)

            if os.path.exists(path) and re.findall(img_postfix_re, md_img): # exist in local file system and is image file format
                path = compress_image(path)
                url = upload_to_qiniu(path, config)
                text = text.replace(ori_img, url)
                uploaded_img += 1 
            else:
                print('{} -> no need to upload'.format(md_img))
        
        # if there are images uploaded successfully
        if uploaded_img > 0:
            print('Upload complete, writing back to {} ...'.format(args.md_path))
            # write back
            with open(args.md_path, 'w') as f:
                f.write(text)
            print('Complete')
        else:
            print('There is no image to upload, finish.')

    # if there is any exception, write back original text
    except Exception: 
        print('Unknown error')
        with open(args.md_path, 'w') as f:
            f.write(ori_text)

if __name__ == '__main__':
    global debug 
    parser = argparse.ArgumentParser()
    parser.add_argument('md_path', type=str, help='Markdown file path')
    parser.add_argument('-d', '--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    debug = args.debug
    main(args)
