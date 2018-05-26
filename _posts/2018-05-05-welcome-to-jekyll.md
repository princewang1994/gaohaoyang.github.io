---
layout: post
title:  "hello jekyll!"
date:   2018-05-05
categories: jekyll
tags: jekyll markdown
excerpt: Jekyll的markdown用法说明，方便以后查阅
mathjax: true
---

* content
{:toc}

##  TOC部分


jekyll似乎不支持`[TOC]`，很遗憾，希望更新的时候能加上。这里使用了[原repo](https://github.com/Gaohaoyang/gaohaoyang.github.io)中的写法，只要在正文(不包含markdown的title部分)开始的时候加上以下markdown代码即可生成侧边的TOC：

```markdown
* content
{:toc}
```

##  Markdown title部分说明

```
---
layout: post
title:  "hello jekyll!"
date:   2018-05-05
categories: jekyll
tags: jekyll markdown
excerpt: Jekyll的markdown用法说明，方便以后查阅
mathjax: true
---
```



## Markdown基本语法

### 多级标题
```
# 一级标题

## 二级标题

### 三级标题
```
一级标题的字体太大，不太美观，这里推荐从二级标题开始(markdown的二级标题会自动在标题下面加一个下划线)



### bullet使用方法

```markdown
- bullet_name
```

- `bullet1`: 在bullet的冒号前面可以使用标签
- `bullet2`: 在bullet的冒号前面可以使用标签

### 划线的使用

```markdown
- ~~被划掉的文字~~
- ~~被划掉的文字~~
```

- ~~被划掉的文字~~
- ~~被划掉的文字~~

### 序号使用方法

```markdown
1. number1
2. number2
```

1. number1
2. number2

### 复选框使用方法

```markdown
- [x] 选中了的框
- [ ] 未选中的框
```

- [ ] checkbox1
- [x] checkbox2
- [ ] checkbox3

### 阴影样式使用方法

```markdown
> 文字
> 文字
```

> 我是一个标签
>
> 我是标签的第二行



## 链接使用

```markdown
[文字内容](链接)
```

[文字内容](链接)



## 代码块的使用

```markdown
​```python
	code
​```
```

Jekyll also offers powerful support for code snippets:

```python
#!/usr/bin/python
def print_hi(name):
	return 'Hello {}'.format(name)
print_hi('World!') # say hello!
```

## 公式的使用

```markdown
单行公式
$$
	mathjax
$$
行内公式$mathjax$
```

单行公式


$$
f(x) = ax + b
$$

$$
a^2 + b^2 = c^2
$$

行内公式 $a \neq b$



## 图片插入

将图片放在`$BLOG_ROOT/images/`中，并使用`![image_name](url)`插入图片

![](http://oodo7tmt3.bkt.clouddn.com/blog_2018052617561527328619.png)



关于图片import的问题想了一下主要有几种解决方法：

1. 第三方图床（如七牛，新浪等）然后直接使用url引入，这种方法缺点是需要预先手动上传（也会有API接口，可以自己写一个小工具）。
2. 直接放在username.github.io这个repo里面，但是需要这里import的时候使用`/images/v2-14562dcdfdbc5ca3c505d9277d77fe8e_r.jpg`这样的url，这种url形式在使用本地编辑器（如typora）的时候无法显示，似乎不够优雅。
3. 在本地的时候使用本地路径作为写markdown的图片url，然后自己写一个脚本统一修改所有的images的路径到2.的格式。
4. 在Mac上使用[iPic](https://github.com/toolinbox/iPic)等快捷工具可以使用，不过iPic的免费版只能使用新浪微博的公共图床

本博客最后采用的是第一种方法，不过没有用现有的轮子（试了几个好像不行），**重新实现了一个alfred的workflow**，[这里是项目主页](https://github.com/princewang1994/markdown-img-upload)，欢迎使用！



## 其他


更多jekyll的使用方法请查看 [Jekyll docs][jekyll] . 想了解Jekyll的特性和请查看 [Jekyll’s GitHub repo][jekyll-gh]. 

[jekyll]:      http://jekyllrb.com
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-help]: https://github.com/jekyll/jekyll-help


如果新的常使用的格式接下来也会慢慢添加到这个模板中😆