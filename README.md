# 🐑 羊量每日选股

> A股多因子选股 + 集合竞价验证，浏览器一键操作，手机也能用

---

## 🚀 快速部署（3 步）

### 1. 安装 Python

下载安装 **Python 3.12**：https://www.python.org/downloads/

⚠️ 安装时勾选 **"Add Python to PATH"**

### 2. 下载代码

```bash
git clone https://github.com/rsjhcy/yangquant.git
cd yangquant
pip install -r requirements.txt
```

如果 GitHub 打不开，直接下载 ZIP：  
https://github.com/rsjhcy/yangquant/archive/refs/heads/master.zip

### 3. 启动

```bash
python web_app.py
```

浏览器打开 **http://localhost:5888**

---

## 📱 手机访问

电脑和手机连同一个 WiFi 时，手机浏览器访问：`http://你的电脑IP:5888`

需要外网访问？注册一个 ngrok 账号，然后：

```bash
ngrok http 5888
```

会生成一个公网网址，手机用 4G 也能访问。

---

## 📊 功能说明

| 功能 | 说明 |
|------|------|
| **收盘筛选** | 收盘后（15:30）点，自动扫描全A股主板，多因子打分选出 Top6 |
| **竞价分析** | 次日盘前（9:20）点，用集合竞价数据验证昨日候选 |
| **邮件推送** | 勾选📧后自动发送推荐到邮箱 |

### 选股策略

- **平衡型**：重风险控制（35%）+ 趋势稳定，适合稳健持仓
- **激进型**：重动量（45%）+ 量价活跃，适合短期追涨
- 两种策略不会重复推荐，共 6 只不同股票

### 样本量选择

- 200/500/1200 只（系统采样）
- 🔥 全A股（~3000只，约 2 分钟）

---

## 🖥 系统要求

- Windows 10/11
- Python 3.10+
- 4GB+ 内存
- 能访问东方财富/腾讯行情 API（国内网络即可）

---

## 📂 文件结构

```
yangquant/
├── web_app.py          ← 主程序（你只需要这个）
├── requirements.txt    ← Python依赖
├── data/               ← 筛选结果缓存
├── quant/              ← 核心模块
│   ├── notify/         ← 邮件发送
│   └── screener/       ← 打分模型+卖出建议
└── render.yaml         ← Render云部署配置
```
