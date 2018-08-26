---
layout: post
title:  "Jekyll Markdown模板"
date:   2018-05-05
categories: jekyll
tags: jekyll markdown
mathjax: true
---

* content
{:toc}

![jekyll](http://oodo7tmt3.bkt.clouddn.com/blog_201807222053570342.png)
本篇模板是Jekyll的markdown用法说明，针对Jekyll框架可以使用的Markdown语法进行总结，除少数语法（在本文中为额外说明）之外，其他的语法均可以在各种常见Markdown编辑器中使用（本人墙裂推荐[Typora](https://typora.io)一款极简的MarkDown编辑器），因此也有参考价值，Markdown在技术博客写作中运用十分广泛（另一个比较著名的是RST，但用户远不如Markdown）



##  TOC部分（仅Jekyll）


jekyll似乎不支持`[TOC]`，很遗憾，希望更新的时候能加上。这里使用了[原repo](https://github.com/Gaohaoyang/gaohaoyang.github.io)中的写法，只要在正文(不包含markdown的title部分)开始的时候加上以下markdown代码即可生成侧边的滑动TOC：

```markdown
* content
{:toc}
```

> 注意，TOC底下是abstract部分，abstract与TOC之间空一行，abstract和第一个标题之间空四行，这样，abstract会显示在首页，而从第一个标题开始不会显示在首页上。相比于在标题中加入excerpt关键字，这种方法可以再摘要中加入图片和其他的语法。

另外大部分的markdown语法支持TOC，只需要在文章开始的地方敲以下的代码即可：

```markdown
[TOC]
```

##  Markdown header部分说明（仅jekyll）

这部分主要是为了给博客分类，

```markdown
---
layout: post
title:  "hello jekyll!"
date:   2018-05-05
categories: jekyll
tags: jekyll markdown
excerpt: Jekyll的markdown用法说明，方便以后查阅(与TOC一小节中的方法二选一，推荐上面一个)
mathjax: true
---
```

Markdown头部字段说明：

- `layout`: 文章归于哪个layout中，一般是`post`
- `title`：文章标题
- `date`: 文章日期（用于博客排序）
- `categories`：文章分类，便于检索
- `tags`：标签，一篇文章可以有很多歌标签
- `excerpt`: 摘要，和TOC一章的方法可以二选一，但是TOC中可以加入图片连接，因此建议使用第一种方法
- `mathjax`: 文章中是否启用latex公式语法，建议打开 

## Markdown基本语法

### 多级标题
```
# 一级标题

## 二级标题

### 三级标题
```
一级标题的字体太大，不太美观，推荐从二级标题开始(markdown的二级标题在Github的样式下会自动在标题下面加一个下划线)

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

![](http://oodo7tmt3.bkt.clouddn.com/blog_20180526220800.jpg)



关于图片import的问题想了一下主要有几种解决方法：

1. 第三方图床（如七牛，新浪等）然后直接使用url引入，这种方法缺点是需要预先手动上传（也会有API接口，可以自己写一个小工具。
2. 直接放在username.github.io这个repo里面，但是需要这里markdown里面的时候使用`/images/v2-14562dcdfdbc5ca3c505d9277d77fe8e_r.jpg`这样的url，这种url形式在使用本地编辑器（如typora）的时候无法显示。
3. 在本地的时候使用本地路径作为写markdown的图片url，然后自己写一个脚本统一修改所有的images的路径到2.的格式，**在本博客的[Github Repo](https://github.com/princewang1994/princewang1994.github.io)里面我已经写了一个将md中的所有图片是上传至七牛云的[小工具](https://github.com/princewang1994/princewang1994.github.io/blob/master/bin/upload.py)，如果需要可以使用**）。
4. 在Mac上使用[iPic](https://github.com/toolinbox/iPic)等快捷工具可以使用，不过iPic的免费版只能使用新浪微博的公共图床

本博客最后采用的是第一种方法，不过没有用现有的轮子（试了几个好像不行），**重新实现了一个alfred的[workflow](https://github.com/princewang1994/markdown-img-upload)**，欢迎使用！



## 其他


更多jekyll的使用方法请查看 [Jekyll docs][jekyll] . 想了解Jekyll的特性和请查看 [Jekyll’s GitHub repo][jekyll-gh]. 

[jekyll]:      http://jekyllrb.com
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-help]: https://github.com/jekyll/jekyll-help


如果新的常使用的格式接下来也会慢慢添加到这个模板中😆