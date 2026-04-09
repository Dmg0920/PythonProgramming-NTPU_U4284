# Python 課程筆記與作業（Jupyter Notebooks）

這個資料夾主要放 Python 課程筆記（`.ipynb`）與作業檔案（`homework/`），並用 `requirements.txt` 管理需要的套件。

## 專案內容

- **課程筆記**：`Sec 1 - Basic.ipynb`、`Sec 2 - Flow Control.ipynb`、`Sec 3 - Function, Class.ipynb`、`Sec 4.1 - Package Intro - numpy.ipynb`
- **作業**：`homework/`（包含 `HW1.ipynb`、`HW2.ipynb`、`HW3.py` 與輸出檔）
- **相依套件**：`requirements.txt`
- **型別檢查設定**：`pyrightconfig.json`

## 需求

- Python 3（建議 3.10+）
- macOS / Linux / Windows 皆可

## 安裝與環境建立（推薦使用 venv）

在此資料夾根目錄執行：

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> 若你已經有 `venv/`，只需要執行 `source venv/bin/activate` 後再 `pip install -r requirements.txt` 更新套件即可。

## 開啟 Jupyter Notebook

如果你的環境還沒裝 Jupyter：

```bash
pip install jupyter
```

啟動：

```bash
jupyter notebook
```

然後在瀏覽器開啟你要的 `.ipynb`（例如 `Sec 1 - Basic.ipynb`）。

## 執行作業程式（以 HW3 為例）

```bash
python homework/HW3.py
```

## 專案結構（概要）

```text
.
├── requirements.txt
├── pyrightconfig.json
├── Sec 1 - Basic.ipynb
├── Sec 2 - Flow Control.ipynb
├── Sec 3 - Function, Class.ipynb
├── Sec 4.1 - Package Intro - numpy.ipynb
├── homework/
│   ├── HW1.ipynb
│   ├── HW2.ipynb
│   └── HW3.py
└── venv/              # 虛擬環境（不建議提交到 git）
```

## 常見問題

### 1) `ModuleNotFoundError` / 找不到套件

- 確認你已啟用虛擬環境（看到命令列前面有 `(venv)`）
- 重新安裝相依：

```bash
pip install -r requirements.txt
```

### 2) Notebook 的 Kernel 不在 venv

- 先安裝 ipykernel：

```bash
pip install ipykernel
python -m ipykernel install --user --name venv --display-name "Python (venv)"
```

然後在 Notebook 內切換 Kernel 到 **Python (venv)**。

## 版本控制建議

此資料夾已包含 `.gitignore`，其中會忽略常見的：

- `venv/`、`.venv/`
- `.ipynb_checkpoints/`
- `__pycache__/`
- `.DS_Store`

如需避免提交 notebook output，可在 `.gitignore` 解除 `# *.ipynb` 的註解（會改成忽略所有 `.ipynb`）。

