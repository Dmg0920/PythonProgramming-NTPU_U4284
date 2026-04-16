# Python 課程筆記與作業

以 Jupyter Notebook 為主的 Python 課程學習紀錄，涵蓋基礎語法、流程控制、函式與類別，以及 NumPy、pandas 等常用套件介紹。

## 專案結構

```text
.
├── Sec 1 - Basic.ipynb                    # 基礎語法
├── Sec 2 - Flow Control.ipynb             # 流程控制
├── Sec 3 - Function, Class.ipynb          # 函式與類別
├── Sec 3.1 - case test.py                 # match-case 範例
├── Sec 4.1 - Package Intro - numpy.ipynb  # NumPy 套件介紹
├── Sec 4.2 - Package Intro - pandas.ipynb # pandas 套件介紹
├── homework/
│   ├── HW1.ipynb
│   ├── HW2.ipynb
│   └── HW3.py
├── pyproject.toml          # uv 專案設定
├── requirements.txt        # pip 相容依賴清單
└── pyrightconfig.json      # 型別檢查設定
```

## 環境需求

- Python 3.12+
- macOS / Linux / Windows

## 快速開始

### 使用 uv（推薦）

```bash
uv sync
```

### 使用 pip + venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 開啟 Notebook

```bash
jupyter notebook
```

在瀏覽器中開啟對應的 `.ipynb` 即可閱讀或執行。

## 執行作業

```bash
python homework/HW3.py
```

## 主要套件

| 套件 | 用途 |
|------|------|
| numpy | 數值計算 |
| pandas | 資料處理與分析 |
| matplotlib | 資料視覺化 |
| seaborn | 統計繪圖 |
| scipy | 科學計算 |
| openpyxl | Excel 讀寫 |
| schemdraw | 電路圖繪製 |
| pydantic | 資料驗證 |
