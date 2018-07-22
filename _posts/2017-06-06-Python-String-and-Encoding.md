---
layout: post
title:  "深入理解Python2中的字符串和编码"
date:   2018-05-05
categories: Python
tags: Python 计算机基础
mathjax: true
---

* content
{:toc}

python2对于中文字符的支持非常不友好，其原因可能是历史遗留问题，Python2的开发年份较早，因此默认采用ASCII编码。这给日后Python在除英语外的语言中的运用挖了一个非常大的坑。典型的场景是中文爬虫，或是中文自然语言处理上，因此理解`字符串`，`编码`，`解码`等概念以及思想非常重要，当然这种思想不光适用于python，也给其他语言及应用一些启示。




## 什么是字符串

众所周知，字符串就是字符序列。由于计算机只认识二进制，所以字符串中的每个字符都需要使用编码来表示。**字符的编码其实就是把字符转换为字节存储的过程**，**解码就是从字节数据映射为字符数据的过程**，换句话说，解码相当于“查字典”。

## ASCII编码及扩展ASCII编码

最早的时候，只有英文和一些简单的拉丁符号，字符集很小，编码很简单，于是ASCII码就诞生了，这种只7位二进制的编码非常易于存储，**计算机只能存储字节（8位）为单位的数据，因此存储的时候只需要在最高位加上一个0表示这是ASCII码，余下的7位就是这个符号的编码**。缺点是容量非常小，只有128个，对于英文这种语言绰绰有余。

但随着计算机被推广至全世界以及互联网的出现，越来越多的语言开始被纳入字符集，一些国家开始使用**扩展ASCII码**即使用该字节的最高位是0，表示使用扩展ASCII码，其他的继续编码，这种方法能够使ASCII的容量翻倍，变成256个字符的编码。

## Unicode编码

当然这对于汉语这样的语言来说是远远不够的，汉字常用字就有7000+，于是一种统一编码——Unicode就诞生了，这种编码对世界上所有的符号进行编码，并且兼容ASCII码，对于ASCII码存在的字符编码都在128以下，更神奇的是，随着Unicode的发展，不光包括了颜文字，现在还包含了emoji表情。

然而Unicode并没有流行起来，因为**Unicode只规定了字符的编码，并没有规定字符的存储方式**，Unicode中有一些字符的编码较大，直接转换为二进制需要用3或4个字节，如果所有的字符，包括ASCII码中的那些，都是用这么长的定长编码的话，那么文件将变为原来的三到四倍，这无疑是无法接受的。这也是Unicode无法流行的原因。

## UTF-8

于是许多Unicode的实现被提出来，其中最著名的当属UTF-8编码，目前被运用最广泛的Unicode实现，记住这里强调的关系是**UTF-8、UTF-16都是Unicode的一种实现**，UTF8能够流行是因为它是一种变长编码，对于编码较小的字符，存储起来也更小，编码较大的字符，存储起来较大。具体来说如下：

- 对于单字节字符编码0开头，余下7位是其ASCII编码
- 对于n（n>1）字节字符编码，前n位是1然后跟一个0表示前缀，后n-1个字节，每个字节以10开头,其余的位置全部为Unicode编码，不足的位置以0补全。如下表所示。

| Unicode符号范围 | UTF-8编码方式 |
| --------------- | ------------- |
| 十六进制        | 二进制        |
|0000 0000-0000 007F | 0xxxxxxx |
|0000 0080-0000 07FF | 110xxxxx 10xxxxxx |
|0000 0800-0000 FFFF | 1110xxxx 10xxxxxx 10xxxxxx |
|0001 0000-0010 FFFF | 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx |

这种编码存储很灵活，所以很快流行开来，是目前最流行的编码之一。

## GB2312和GBK编码

为了解决中文编码问题，中国自行定义了国家标准，GB2312编码就是中国国标编码，其编码方式是：对于所有字符均采用双字节编码，而低字节是ASCII编码。但gb2312还是不够的，于是再继续拓展了GB2312编码，只要是编码大于127的全都变成了中文，这就是中文的GBK编码，当然后续还有GB18030之类的拓展编码，GB系列的主要特点是：一个汉字使用两个字节编码，连数字，标点符号也增加了双字节编码的格式，也就是我们常说的“全角”，所以“逗号”会有两种，一种是单字节的，一种是双字节的。

## Python中的字符串

![](http://oodo7tmt3.bkt.clouddn.com/blog_20180526214318.jpg)


前面说了，Python的字符串由于历史遗留问题，不得不采用ASCII作为str的默认编码，为了兼容unicode的编码，python中多了一个叫做unicode的类，具体就是在引号前面加上一个u（图来源于网络，侵删）

```python
>>> s = u'你好'
>>> s
u'\u4f60\u597d'
```

这样在Python中，s其实是使用unicode编码的方式来表示，“你好”其实是\u4f60\u597d也就是unicode的编码，不涉及存储，至于内部是如何存储的，暂时还不太清楚。

这样显示起来都没问题，但是遇到写入文件的时候就会出问题，因为前面说了，unicode无法直接存储，需要转换为可以存储的格式——ascii或者是utf-8，gbk都是可以直接存储的，这就涉及到编码问题，**所谓编码，就是把unicode转换为可存储的编码问题**需要写入或者传输的时候，必须进行encode，如上面的“你好”转换为utf8编码：

```python
>>> s.encode('utf-8')
'\xe4\xbd\xa0\xe5\xa5\xbd'

>>> s.encode('gbk')
'\xc4\xe3\xba\xc3'

>>> s.encode('ascii')

UnicodeEncodeError                        
Traceback (most recent call last)
<ipython-input-5-443ef5516bed> in <module>()
----> 1 s.encode('ascii')

UnicodeEncodeError: 'ascii' codec can't encode characters in position 0-1: ordinal not in range(128)
```

这里我们遇到了第一个常见的错误**UnicodeEncodeError**这个错误发生在把unicode编码为str的时候，unicode含有目标编码中没有的字符，同理，当我们使用str函数强制转换时，将隐式调用ascii编码，也可能发生这样的错误：

```python
>>> str(s) # 相当于 s.encode('ascii')

UnicodeEncodeError Traceback (most recent call last)
<ipython-input-6-d22ffcdd2ee9> in <module>()
----> 1 str(s)

UnicodeEncodeError: 'ascii' codec can't encode characters in position 0-1: ordinal not in range(128)
```

我们从网络上下载，爬取，或者从数据库中读取的数据必须是二进制数据，也就是说，**不可能是unicode**，只可能是ascii，utf-8，gbk这样的可存储编码，如下代码我们从文件中读取“你好啊我的朋友”编辑的时候，文件是使用utf-8存储的。那么我们读取出来就是这样的。

```python
>>> f = open('aFile')

>>> ss = f.read()
'\xe4\xbd\xa0\xe5\xa5\xbd\xe5\x95\x8a\xe6\x88\x91\xe7\x9a\x84\xe6\x9c\x8b\xe5\x8f\x8b\n'

>>> print ss
你好啊我的朋友
```

这个时候需要把它变成其他的编码的话可以使用decode方法把它变成unicode，然后encode为其他编码。

```python
>>> s_u = ss.decode('utf8')

>>> s_u
 u'\u4f60\u597d\u554a\u6211\u7684\u670b\u53cb\n'
 
>>> s_gbk = ss.decode('gbk')

UnicodeDecodeError 
Traceback (most recent call last)
<ipython-input-25-99953b745005> in <module>()
----> 1 s_u = ss.decode('gbk')

UnicodeDecodeError: 'gbk' codec can't decode bytes in position 20-21: illegal multibyte sequence
```

这里出现了第二个常见错误**UnicodeDecodeError**，这个错误发生在把str decode成为unicode的时候，str的格式不符合所使用的编码格式，也就是无法解码，需要用其他的格式解码，如上面的代码，ss其实是utf8的编码，但是强行用gbk来解码，就会出错。

第三种常见错误出现在py脚本文件中，如果在脚本中输入以下代码：

```python
#!/usr/bin/python

print u'你好'

```

执行脚本`python a.py`的时候，就会出错:

```
File "a.py", line 3
SyntaxError: Non-ASCII character '\xe4' in file a.py on line 3, but no encoding declared; see http://python.org/dev/peps/pep-0263/ for details
```

原因是python默认脚本的编码是ascii，如果使用ascii编码之外的字符出现在脚本中，那么在编译的时候就会出错。于是需要更改脚本的编码，通过以下注释可更改脚本编码：

```python
#!/usr/bin/python
#coding:utf-8

print u'你好'
```

```python
#!/usr/bin/python
# -*- coding: utf-8 -*-

print u'你好'
```

```python
#!/usr/bin/python
# encoding=utf8

print u'你好'
```

这几种方法都是可以的。

## 后记

不过说到底，python2也是一种面临淘汰的语言，接下来有时间也会讲一下Python3中的字符串与Python2中的区别

## 本文参考

- [阮一峰博客](http://www.ruanyifeng.com/blog/2007/10/ascii_unicode_and_utf-8.html)
- [Python编码为什么这么蛋疼](https://foofish.net/why-python-encoding-is-tricky.html)








