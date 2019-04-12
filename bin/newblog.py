#!/usr/local/bin/python3
import argparse
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('title', nargs='+', type=str, help='Title of new blog')
parser.add_argument('--category', '-c', default='DeepLearning', type=str, help='Blog category')
parser.add_argument('--tags', '-t', nargs='+', type=str, default=['None'], help='Tags')
parser.add_argument('--layout', '-l', default='post', type=str, choices=['post', 'page', 'draft'], help='Layout')
parser.add_argument('--author', default='Prince', type=str, help='Layout')

layout_map = {
    'post': '_posts',
    'page': 'page',
    'draft': '_drafts'
}

def write_md(md, md_lines):
    with open(md, 'w') as f:
        for line in md_lines:
            if len(line) > 0:
                f.write(line)
            f.write('\n')

def toc_template():
    return [
        '* content',
        '{:toc}',
        '',
        'abstract，TOC和abstract空一行，abstract和标题1之前空4行，abstract中可以加入图片，会显示在首页'
    ]

def content_template():
    return [
        '## 标题1',
        '',
        '### 标题1.1',
        '',
        'content',
        '',
        '## 标题2',
        '',
        '### 标题2.1',
        '',
        'content'
    ]

def format_header(header):
    lines = ['---']
    for key in ['layout', 'title', 'date', 'categories', 'tags', 'mathjax', 'author']:
        if not isinstance(header[key], list):
            header[key] = [header[key]]
        lines.append('{}: {}'.format(key, ' '.join(header[key])))
    lines.append('---')
    return lines

def format_title(title):
    if not isinstance(title, list):
        title = [title]

    date = time.strftime("%Y-%m-%d", time.gmtime())
    title = '-'.join(title)
    return '{}-{}.md'.format(date, title)

def new_blog(args):

    md_lines = []

    header = {
        'layout': args.layout,
        'title': '"{}"'.format(' '.join(args.title)),
        'date':  time.strftime("%Y-%m-%d", time.gmtime()),
        'categories': args.category,
        'tags': ' '.join(args.tags),
        'mathjax': 'true',
        'author': args.author
    }

    md_lines += format_header(header)
    md_lines += ['']
    md_lines += toc_template()
    md_lines += [''] * 4
    md_lines += content_template()

    filename = format_title(args.title)
    file_path = os.path.join(layout_map[args.layout], args.category, filename)
    print('Writing to {}'.format(filename))
    if not os.path.exists(file_path):
        write_md(file_path, md_lines)
    else:
        print('Blog `{}` exist, skip'.format(file_path))
    print('Done.')

if __name__ == '__main__':
    dirs = os.listdir('.')
    if 'page' in dirs and '_posts' in dirs and '_drafts' in dirs:
        args = parser.parse_args()
        new_blog(args)
    else:
        print('Please run this in $BLOG_ROOT')