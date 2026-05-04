# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案定位

以 Jupyter Notebook 為主的 Python 課程學習紀錄。根目錄的 `Sec N - ....ipynb` 是課程筆記，`homework/` 存放作業（`.ipynb` 與 `.py` 雙版本），`MidExam.ipynb` 為期中考。

## 環境設定

```bash
# 建立虛擬環境並安裝依賴
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 或使用 uv（推薦）
uv sync
```

## 常用指令

```bash
# 啟動 Notebook
jupyter notebook

# 執行 Python 腳本作業
python homework/HW3.py
python "Sec 3.1 - case test.py"

# 型別檢查
pyright
```

## 重要慣例

**Notebook 編輯：**
- 修改 notebook 後務必從頭到尾重新執行一遍（Restart & Run All）確認無隱藏狀態錯誤
- 保留原有章節結構（`Sec X - ...` 的教學脈絡），不要隨意重新命名
- 不要把大量輸出或圖表寫入版本控制，除非使用者明確要求

**依賴管理：**
- 新增套件必須同步更新 `requirements.txt`
- 不要提交 `.venv/`、`__pycache__/`、`.ipynb_checkpoints/`

**檔案命名：**
- 課程筆記：`Sec N - Topic.ipynb`
- 作業：`HWN.ipynb` / `HWN.py`

## 套件概覽

| 套件 | 用途 |
|------|------|
| numpy | 數值計算、陣列操作 |
| pandas | 資料處理與分析 |
| matplotlib / seaborn | 資料視覺化 |
| scipy / scikit-learn | 科學計算與機器學習 |
| openpyxl | Excel 讀寫 |
