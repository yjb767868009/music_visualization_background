# 2019中间件课程大作业后台服务代码
### 音乐分析和风格识别

#### 环境安装

```bash
pip install -r requirements.txt
```

#### 配置
In `server.py` 配置IP和端口
```python
    app.run('0.0.0.0', port=8081)
```

#### 返回消息格式
>'message',返回结果成功与否
>
>'beat_times',关键点的列表
>
>'db_list',关键点的频谱,每个关键点40个值
>
>'style',音乐的风格