# MERT-v1-330M 情感分析 Demo

这个 demo 使用本地下载的 `m-a-p/MERT-v1-330M` 读取 FLAC 音乐，并输出情感分析 JSON。

注意：`MERT-v1-330M` 是自监督音乐表征模型，不是开箱即用的情感分类模型。脚本支持两种模式：

- `classifier`：加载你微调好的情感分类头，输出真正的 MERT 下游分类结果。
- `heuristic`/`auto`：没有分类头时，仍会加载本地 MERT 提取特征，然后用启发式音频情感代理跑通 demo。这个模式适合演示流程，不适合作为可靠情感判断。

## 安装依赖

```powershell
pip install -r requirements.txt
```

## 运行

```powershell
python .\mert_emotion_demo.py `
  --model-dir D:\models\MERT-v1-330M `
  --audio D:\music\demo.flac
```

保存结果：

```powershell
python .\mert_emotion_demo.py `
  --model-dir D:\models\MERT-v1-330M `
  --audio D:\music\demo.flac `
  --output-json .\emotion_result.json
```

只分析前 30 秒：

```powershell
python .\mert_emotion_demo.py `
  --model-dir D:\models\MERT-v1-330M `
  --audio D:\music\demo.flac `
  --max-seconds 30
```

使用微调后的情感分类头：

```powershell
python .\mert_emotion_demo.py `
  --model-dir D:\models\MERT-v1-330M `
  --audio D:\music\demo.flac `
  --mode classifier `
  --head-checkpoint .\emotion_head.pt
```

`emotion_head.pt` 可以是以下格式之一：

```python
{
    "labels": ["calm", "happy", "sad", "angry"],
    "state_dict": {
        "classifier.weight": ...,
        "classifier.bias": ...,
    },
}
```

也可以直接保存 `EmotionHead.state_dict()`。

## 输出示例

```json
{
  "audio": "D:\\music\\demo.flac",
  "method": "heuristic_audio_affect_proxy",
  "duration_seconds": 30.0,
  "mert": {
    "chunks": 6,
    "embedding_dim": 1024,
    "layer": "mean"
  },
  "top_emotions": [
    {
      "label": "energetic",
      "score": 0.34
    }
  ]
}
```

