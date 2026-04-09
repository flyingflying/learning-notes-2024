
from lxml import etree 

HTML_CONTENT = """\
<div id="main">
    hello, world
    <a href="https://www.baidu.com">搜索链接</a>
    你好世界
</div>\
"""

root_node = etree.fromstring(HTML_CONTENT)
child_node = root_node.getchildren()[0]

print(repr(root_node.text))    # '\n    hello, world\n    '
print(repr(child_node.text))   # '搜索链接'
print(repr(child_node.tail))   # '\n    你好世界\n'

print(root_node.attrib)        # {'id': 'main'}
print(child_node.attrib)       # {'href': 'https://www.baidu.com'}
