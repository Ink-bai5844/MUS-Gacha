# QCloud Song Store 爬虫说明

`qcloud_song_store.py` 用本地编译的 `QCloudMusicApi` 动态库调用网易云接口，批量保存歌曲信息到 SQLite，并为每首歌导出 JSON 快照。

它适合做这些事：

- 按歌曲 ID 抓取歌曲详情、歌词、播放链接、评论、相似歌曲、专辑等信息。
- 按歌单 ID 抓取歌单内歌曲。
- 用 `cookie.txt` 登录后，按自己的歌单名抓取，例如 `Ink_bai喜欢的音乐`。
- 在本机 DNS 解析网易云域名失败时，用 `--resolve-host` 走本地解析覆盖代理。

## 环境准备

需要先安装 Qt，并编译 `QCloudMusicApi` 动态库。

在 Windows + Qt MinGW 环境下可以使用：

```powershell
git -C .\QCloudMusicApi submodule update --init --recursive

$env:Path='D:\Qt\Tools\Ninja;D:\Qt\Tools\mingw1310_64\bin;D:\Qt\6.11.0\mingw_64\bin;D:\Qt\Tools\CMake_64\bin;' + $env:Path

& 'D:\Qt\Tools\CMake_64\bin\cmake.exe' `
  -S QCloudMusicApi `
  -B QCloudMusicApi\build `
  -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_PREFIX_PATH=D:\Qt\6.11.0\mingw_64 `
  -DCMAKE_C_COMPILER=D:\Qt\Tools\mingw1310_64\bin\gcc.exe `
  -DCMAKE_CXX_COMPILER=D:\Qt\Tools\mingw1310_64\bin\g++.exe `
  -DQCLOUDMUSICAPI_BUILD_TEST=OFF `
  -DQCLOUDMUSICAPI_BUILD_SHARED=ON

& 'D:\Qt\Tools\CMake_64\bin\cmake.exe' --build QCloudMusicApi\build --config Release -j 4
```

编译成功后动态库通常在：

```text
QCloudMusicApi\build\QCloudMusicApi\QCloudMusicApi.dll
```

运行脚本前建议把 Qt/MinGW 和动态库目录加入当前 PowerShell 的 `PATH`：

```powershell
$env:Path='D:\Qt\Tools\mingw1310_64\bin;D:\Qt\6.11.0\mingw_64\bin;D:\Code\Python\MUS-Gacha\QCloudMusicApi\build\QCloudMusicApi;' + $env:Path
```

## Cookie

爬自己的歌单或登录态接口时，需要在项目根目录创建 `cookie.txt`，内容是一整行网易云网页 cookie。

注意：

- 不要提交或分享 `cookie.txt`。
- cookie 为空时，`--my-playlist-name` 无法使用。
- 公开歌曲、公开歌单的部分信息不一定需要 cookie，但喜欢歌单通常需要。

## DNS 解析覆盖

如果脚本返回类似：

```text
Host interface.music.163.com not found
Host music.163.com not found
```

可以加：

```powershell
--resolve-host interface.music.163.com=117.135.207.67 `
--resolve-host music.163.com=112.29.230.13
```

脚本会启动一个本地 HTTP CONNECT 代理，只覆盖这两个域名的解析，不修改系统 hosts。

## 快速测试

按歌曲 ID 抓一首：

```powershell
python .\qcloud_song_store.py 2058263032 `
  --library .\QCloudMusicApi\build\QCloudMusicApi\QCloudMusicApi.dll `
  --endpoints check_music,lyric_new,song_url_v1 `
  --levels standard `
  --resolve-host interface.music.163.com=117.135.207.67 `
  --resolve-host music.163.com=112.29.230.13 `
  --db data\source\qcloud_songs_test.sqlite3 `
  --json-dir data\source\qcloud_song_json_test
```

只测试自己的喜欢歌单前三首：

```powershell
python .\qcloud_song_store.py `
  --my-playlist-name "Ink_bai喜欢的音乐" `
  --cookie-file .\cookie.txt `
  --library .\QCloudMusicApi\build\QCloudMusicApi\QCloudMusicApi.dll `
  --endpoints check_music `
  --max-songs 3 `
  --resolve-host interface.music.163.com=117.135.207.67 `
  --resolve-host music.163.com=112.29.230.13 `
  --db data\source\ink_bai_liked_test.sqlite3 `
  --json-dir data\source\ink_bai_liked_test_json
```

## 爬自己的喜欢歌单

全量爬 `Ink_bai喜欢的音乐`：

```powershell
python .\qcloud_song_store.py `
  --my-playlist-name "Ink_bai喜欢的音乐" `
  --cookie-file .\cookie.txt `
  --library .\QCloudMusicApi\build\QCloudMusicApi\QCloudMusicApi.dll `
  --resolve-host interface.music.163.com=117.135.207.67 `
  --resolve-host music.163.com=112.29.230.13 `
  --workers 4 `
  --db data\source\ink_bai_liked.sqlite3 `
  --json-dir data\source\ink_bai_liked_json
```

建议第一次先加 `--max-songs 20`，确认输出正常后再全量跑。

## 按歌单 ID 爬取

如果已经知道歌单 ID：

```powershell
python .\qcloud_song_store.py `
  --playlist-id 2925790341 `
  --cookie-file .\cookie.txt `
  --library .\QCloudMusicApi\build\QCloudMusicApi\QCloudMusicApi.dll `
  --max-songs 50 `
  --resolve-host interface.music.163.com=117.135.207.67 `
  --resolve-host music.163.com=112.29.230.13 `
  --db data\source\playlist_2925790341.sqlite3 `
  --json-dir data\source\playlist_2925790341_json
```

## 按用户歌单名爬取

如果知道某个用户的 uid 和公开歌单名：

```powershell
python .\qcloud_song_store.py `
  --user-playlist-uid 123456789 `
  --playlist-name "某个公开歌单名" `
  --library .\QCloudMusicApi\build\QCloudMusicApi\QCloudMusicApi.dll `
  --max-songs 50 `
  --resolve-host interface.music.163.com=117.135.207.67 `
  --resolve-host music.163.com=112.29.230.13 `
  --db data\source\user_playlist.sqlite3 `
  --json-dir data\source\user_playlist_json
```

## 可抓取接口

`--endpoints default` 默认抓：

```text
song_detail
check_music
lyric_new
lyric
song_music_detail
song_wiki_summary
song_dynamic_cover
song_chorus
comment_music
simi_song
simi_playlist
song_url_v1
album
album_detail_dynamic
```

`song_detail` 总会被抓取，因为它用于建立歌曲基础信息。

抓更多接口：

```powershell
--endpoints all
```

只抓指定接口：

```powershell
--endpoints check_music,lyric_new,song_url_v1
```

播放链接质量等级由 `--levels` 控制：

```powershell
--levels standard,exhigh,lossless,hires
```

## 输出结构

SQLite 默认路径：

```text
data/source/qcloud_songs.sqlite3
```

主要表：

- `songs`：歌曲基础信息，包括歌名、专辑、时长、歌手、原始详情 JSON。
- `api_results`：每个接口的原始返回 JSON、参数、采集时间、错误信息。

JSON 快照默认目录：

```text
data/source/qcloud_song_json
```

每首歌一个文件：

```text
data/source/qcloud_song_json/2058263032.json
```

## 常见问题

### 找不到 QCloudMusicApi.dll

确认已经编译，并传入：

```powershell
--library .\QCloudMusicApi\build\QCloudMusicApi\QCloudMusicApi.dll
```

如果提示依赖 DLL 找不到，把 Qt/MinGW 目录加入当前 `PATH`。

### 找不到 `Ink_bai喜欢的音乐`

先确认：

- `cookie.txt` 不是空文件。
- cookie 对应的账号就是 `Ink_bai`。
- 歌单名完全一致。
- `--playlist-limit` 足够大，默认是 1000。

### 返回 `Host ... not found`

加上：

```powershell
--resolve-host interface.music.163.com=117.135.207.67 `
--resolve-host music.163.com=112.29.230.13
```

### 想减少请求量

使用：

```powershell
--endpoints check_music
--max-songs 20
```

### 想慢一点爬

使用：

```powershell
--sleep 0.5
```

默认每个接口调用之间睡眠 `0.15` 秒。

### 想并行爬取

使用：

```powershell
--workers 4
```

`--workers` 按歌曲并行，默认是 `1`。建议先从 `2` 或 `4` 开始，配合精简接口测试：

```powershell
--endpoints check_music,lyric_new,song_url_v1 --levels standard --workers 4
```

为了避免网易云限流、Qt 网络层异常和本地代理压力，脚本会把并发上限限制为 `16`。

如果遇到网络错误、接口限流或 Qt 运行时异常，降回：

```powershell
--workers 1
```
