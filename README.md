# 🐑 羊量量化平台

> 不用写代码，也能做量化。用命令行就够了。

---

## 🤔 这是什么？

一个帮你**自动分析 A 股**的工具。你可以用它：

1. 📥 **自动下载**股票历史数据到本地
2. 📈 **跑回测** — 用历史数据验证你的交易想法能不能赚钱
3. 🔄 **每天自动更新**数据，为实盘做准备

**不需要会编程**，跟着下面的步骤敲命令就行。

---

## 🛠 第一步：安装 Python

你的电脑需要一个叫 "Python" 的软件才能运行这个工具。

### 安装方法

1. 打开浏览器，访问 **https://www.python.org/downloads/**
2. 点击黄色大按钮 **Download Python 3.12.x**
3. 下载完成后双击安装
4. ⚠️ **重要**: 安装界面底部勾选 **"Add Python to PATH"**，然后点 Install
5. 装完后关掉安装窗口

### 验证安装

按 `Win + R`，输入 `cmd`，回车。在弹出的黑窗口里输入：

```
python --version
```

如果显示 `Python 3.12.x` 就说明装好了。

---

## 📦 第二步：安装这个工具

在黑窗口中依次输入：

```bash
# 1. 进入项目文件夹
cd "d:\Claude code\quant-platform"

# 2. 安装需要的软件包（第一次需要几分钟）
python -m pip install -r requirements.txt
```

看到 `Successfully installed ...` 就完成了。

---

## 🚀 第三步：下载数据，跑起来！

### 3.1 设置你要关注的股票

```bash
python cli.py data watch --set 000001,600519,000858,601318,000333,600036
```

这行命令告诉工具："我关注这6只股票"。

> **股票代码说明**: `000001` = 平安银行, `600519` = 贵州茅台, `000858` = 五粮液, `601318` = 中国平安, `000333` = 美的集团, `600036` = 招商银行

### 3.2 下载历史数据

```bash
python cli.py data download --symbols 000001,600519,000858,601318,000333,600036 --start 2023-01-01
```

第一次下载会慢一些（大概 30 秒到 1 分钟），因为要从网上下载一年多的数据。数据会存在 `data/daily/` 文件夹里。

### 3.3 跑一个回测，看看效果

```bash
python cli.py backtest run --strategy alpha_momentum --symbols 000001,600519,000858,601318,000333,600036 --start 2023-01-01 --capital 1000000
```

你会看到类似这样的输出：

```
BACKTEST RESULTS
================================
  总收益率:      +15.74%
  年化收益:      +15.74%
  夏普比率:      1.10
  最大回撤:      -12.79%
  交易笔数:      5
```

**解读**：
- **总收益率** = 赚了多少钱。+15.74% 意思是 100 万变成 115.7 万
- **夏普比率** = 大于 1 说明这个策略靠谱，大于 2 就是优秀
- **最大回撤** = 中途最多亏过多少。12.79% 意思是 100 万最惨的时候只剩 87 万
- **交易笔数** = 这一年买卖了多少次

---

## 🔄 第四步：每天更新数据（为实盘做准备）

量化交易需要**持续不断的数据**。设置好之后，工具会每天自动帮你下载最新行情。

### 方法一：每天手动更新

每天收盘后（下午 3 点半之后）跑一次：

```bash
python cli.py data update
```

它只会下载本地没有的新数据，不会重复下载。

### 方法二：自动定时更新

让电脑每天自动跑一次更新。在 Windows 上：

1. 按 `Win + R`，输入 `taskschd.msc`，回车（打开任务计划程序）
2. 右侧点 **"创建基本任务"**
3. 名称填：`量化数据更新`
4. 触发器选 **"每天"**，时间设为 **16:00**
5. 操作选 **"启动程序"**
6. 程序和参数填：

```
程序: C:\Users\你的用户名\AppData\Local\Programs\Python\Python312\python.exe
参数: cli.py data update
起始于: d:\Claude code\quant-platform
```

7. 点完成

现在每天下午 4 点，电脑会自动拉取最新数据。

### 查看数据覆盖情况

```bash
python cli.py data update
```

会显示每只股票的最新数据日期，一眼看出哪些需要更新。

---

## 📖 常用命令速查

| 我想做什么 | 命令 |
|-----------|------|
| 设置关注的股票 | `python cli.py data watch --set 000001,600519` |
| 添加股票 | `python cli.py data watch --add 000858` |
| 查看关注列表 | `python cli.py data watch` |
| 下载历史数据 | `python cli.py data download --symbols 000001,600519 --start 2023-01-01` |
| 增量更新数据 | `python cli.py data update` |
| 跑双均线回测 | `python cli.py backtest run --strategy ma_cross --symbols 000001,600519 --start 2023-01-01` |
| 跑动量轮动回测 | `python cli.py backtest run --strategy alpha_momentum --symbols ... --start 2023-01-01` |
| 跑均值回归回测 | `python cli.py backtest run --strategy mean_reversion --symbols ... --start 2023-01-01` |
| 打开可视化面板 | `python cli.py dashboard` |

---

## 🎯 三种内置策略怎么选

| 策略 | 适合行情 | 原理 |
|------|---------|------|
| **alpha_momentum** | 大部分行情 | 选近期涨得好的股票，定期换仓 |
| **ma_cross** | 趋势明显的行情 | 短期均线上穿长期均线时买入 |
| **mean_reversion** | 震荡行情 | 跌多了买，涨回正常就卖 |

> 新手建议先用 **alpha_momentum**，表现最稳定。

---

## 🖥 打开可视化面板

```bash
python cli.py dashboard
```

浏览器会自动打开 `http://localhost:8501`，你可以在网页上：
- 浏览股票列表
- 下载数据
- 调整参数跑回测
- 看净值曲线和回撤图

---

## ❓ 常见问题

**Q: 下载数据时报错 "Connection aborted"？**

> 这是数据源暂时限流了。等 1-2 分钟再试就好。工具已内置了自动重试和备用数据源。

**Q: 回测结果为 0 交易？**

> 说明在那个时间段里策略没找到合适的买卖点。换个时间段（比如 --start 2020-01-01）或者换只股票试试。

**Q: 怎么添加更多股票？**

> ```bash
> python cli.py data watch --add 000002,600900,600887
> python cli.py data update
> ```

**Q: 数据占多少空间？**

> 一只股票一年大约 500KB。关注 20 只股票，3 年数据大约 30MB。完全可以接受。

**Q: 能不能连券商自动交易？**

> 目前支持模拟交易。实盘自动交易需要接入券商 QMT/XTQuant，代码接口已预留。

---

## 📂 文件说明

```
quant-platform/
├── cli.py              ← 命令行入口 (你主要用的)
├── config.yaml         ← 配置文件 (可改初始资金、佣金等)
├── data/               ← 下载的数据存这里
├── logs/               ← 日志文件
├── dashboard/          ← 可视化面板
└── quant/              ← 核心代码 (不用管)
```

---

**🐑 从小白到量化能手，其实没那么难。**
