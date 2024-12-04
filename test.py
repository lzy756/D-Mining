import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体路径（请根据实际情况替换字体路径）
font_path = 'C:/Windows/Fonts/msyh.ttc'  # 微软雅黑字体路径（Windows系统为例）
prop = font_manager.FontProperties(fname=font_path)

# 示例：生成中文词云图
plt.figure(figsize=(10, 6))
# plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("中文词云示例", fontproperties=prop)  # 使用指定字体
plt.show()
