# spec.md — 通用 Notebook 增補與註解規格

## Goal

把既有 Jupyter notebook 原地整理成「考前可快速查閱、看 code 就能理解資料流」的版本。

適用範圍：

- Python 基礎語法 notebook
- Flow control / simulation / numerical method notebook
- Function / class / OOP notebook
- pandas / numpy / visualization 等套件 notebook
- homework 或 lecture notebook 的教學增補

核心要求：

- 不改變既有程式邏輯。
- 不改變既有變數名稱、函式簽章、class 名稱或 cell 順序，除非任務明確要求。
- 不修改既有 cell 的 `outputs` 欄位。
- 註解必須幫讀者理解「這行 code 做了什麼、資料如何變化、為什麼需要這樣寫」。
- 禁止只把語法名稱翻成中文的空話註解。

---

## 工作流程

### 1. Notebook 盤點

開始前先讀取 notebook 結構，列出：

- cell 總數
- 每個 markdown/code cell 的 index
- 每個 section 的主題
- 哪些 code cell 已有註解、哪些註解是低品質模板
- 哪些 cell 依賴外部檔案、網路、圖片、特殊 kernel 或互動功能

輸出心中 mapping 後再動手，不要憑印象插 cell。

### 2. Section Mapping

依 notebook 內容建立 section，不要硬套固定模板。

建議格式：

| Section | Cell range | Topic | Action |
|---------|------------|-------|--------|
| A | 0-3 | 基礎變數與型別 | inline comments |
| B | 4-8 | 迴圈與條件 | pre Markdown + inline comments |
| C | 9-12 | 函式與回傳值 | inline comments + extended variant |

Action 可包含：

- `pre Markdown`：在某段主題前補概念整理。
- `inline comments`：改寫或新增 code 註解。
- `extended variant`：在段落後新增可獨立執行的延伸例子。
- `cleanup only`：只移除或改寫低品質註解。

### 3. 編輯順序

1. 先移除低品質註解。
2. 再補真正有用的 inline comments。
3. 再補必要的 Markdown 說明。
4. 最後才加入 extended variant。
5. 完成後 Restart & Run All 或用 `nbconvert --execute` 驗證。

---

## 註解品質標準

### 必須寫出的內容

好的 inline comment 至少符合以下一種：

- 說明這行 code 對資料做了什麼轉換。
- 說明這行結果會被下一行如何使用。
- 說明容易誤解的 index、slice、scope、mutation、copy、type conversion。
- 說明某個參數造成的行為差異。
- 說明錯誤處理、邊界條件或保護條件的目的。
- 說明輸出為什麼會是現在看到的結果。

### 禁止的註解

不要寫這種註解：

```python
# `=` assignment：把右側結果綁定到變數供後續重用。
x = 10

# `class`：宣告類別，封裝資料與方法。
class Dog:
    pass

# `return`：回傳函式結果並結束函式。
return value

# `display(...)`：在 notebook 同時展示多個輸出做對照。
display(x)

# `print(...)`：把結果輸出到標準輸出，方便檢查目前值。
print(x)
```

原因：這些只是在翻譯語法，沒有幫讀者理解該行 code 的具體作用。

### 合格註解範例

```python
a = 100
print(a - 4)  # 只計算 100 - 4，沒有改到 a
a = a - 4     # 把扣掉 4 的結果重新存回 a
a += 3        # 等價於 a = a + 3，直接更新目前的 a
```

```python
dL = [3, 1, 3, 4, 5, 6, 7, 1, 3]
display(dL[0:2],   # 取 index 0,1；右端 2 不包含
        dL[::3],   # 從頭開始每隔 3 個取一次
        dL[::-3],  # 從尾端反向每隔 3 個取一次
        dL[::-1])  # 完整反轉 list
```

```python
class ExamD(ExamB, ExamC):
    pass

display([cls.__name__ for cls in ExamD.__mro__],  # MRO 決定 method lookup 的搜尋順序
        ExamD().who())                            # 先找到 ExamB.who()，所以回傳 "B"
```

```python
exam_gen = (n * n for n in range(3))
display(list(exam_gen), list(exam_gen))  # generator 只能消耗一次，第二次會是空 list
```

```python
report_str = "The winners are: User9, UserN, User8"
display(re.findall(r'User\d', report_str),  # `\d` 只吃數字，所以抓到 User9、User8
        re.findall(r'User\D', report_str),  # `\D` 只吃非數字，所以抓到 UserN
        re.findall(r'User\w', report_str))  # `\w` 吃英數底線，所以三個都會抓到
```

```python
thm['MVT'] = 'Moving average theorem'  # 指定既有 key 會覆寫原本的 value
display(thm.get('BIP'),                # key 不存在時回傳 None
        thm.get('BIP', 'invalid key')) # 第二個參數是找不到 key 時的預設值
```

---

## Markdown 說明格標準

Markdown cell 用來整理一段主題，不要把每行 code 都搬進 Markdown 重講。

應包含：

- 這段主題的核心觀念。
- 常見語法形狀。
- 容易出錯的 gotcha。
- 何時使用這個寫法。

避免：

- 大段冗長文章。
- 和 code comment 重複。
- 只列 API 名稱，沒有說明用途。
- emoji、裝飾性前綴、`WHY:`、`ADD:`、`EXAM:` 等標籤。

範例：

```markdown
**`loc` vs `iloc`**

- `loc` 用 index label 選資料，slice 會包含右端點。
- `iloc` 用整數位置選資料，slice 不包含右端點。

Gotcha：如果 Series 的 index 本身是整數，`s[3]` 容易被誤會成位置 3；
考試或教學中應明確寫 `s.loc[3]` 或 `s.iloc[3]`。
```

---

## Extended Variant 標準

Extended Variant 是段落後新增的 code cell，用來展示常見變形或考題變形。

要求：

- 必須可在空白 kernel 中獨立執行，除非該段明確依賴前文資料。
- 若依賴外部檔案、網路或本機資料，必須用註解或條件保護，避免 Restart & Run All 中斷。
- 每個 extended cell 要有明確目的，不要只是多塞 API。
- import 寫在該 cell 內，避免依賴前面隱性狀態。
- 註解仍然要說明資料流與行為，不要變成 API 翻譯。

範例：

```python
# 延伸：用 `None` 避免 mutable default argument 共用同一個 list
def append_score(score: int, scores: list[int] | None = None) -> list[int]:
    if scores is None:
        scores = []       # 每次沒傳 scores 時都建立新的 list
    scores.append(score)  # 只改這次呼叫使用的 list
    return scores

print(append_score(90))
print(append_score(80))
```

---

## 語言與風格

- Markdown 說明使用繁體中文。
- Code identifier、API、syntax name 保留英文，例如 `split()`, `dict`, `yield`, `nonlocal`, `groupby()`。
- Code comment 使用中文敘述，但要明確提到相關 code 名稱。
- 不使用 emoji。
- 不使用裝飾性標籤，如 `WHY:`, `ADD:`, `EXAM:`。
- 不重複解釋同一個語法；同一 cell 內講一次即可。
- 句子短，重點放在這行 code 的效果。

---

## Editing Rules

### 保留內容

- 不刪原本教學內容，除非只是移除低品質註解。
- 不改既有輸出。
- 不改題目、圖片、公式、原本 markdown 的核心內容。
- 不任意 rename notebook。
- 不新增 machine-specific absolute path。

### 可以修改

- 可以改寫既有註解。
- 可以刪除空話註解。
- 可以在 code cell 內加入少量有用註解。
- 可以插入 Markdown cell。
- 可以插入 extended variant code cell。
- 可以修正明顯 typo，但不得改變程式行為。

---

## Edge Cases Checklist

處理 notebook 時要特別檢查：

| 類型 | 風險 | 註解應說明 |
|------|------|------------|
| mutation | `list.sort()`, `append()`, `dict.update()` 會原地改物件 | 哪個物件被改、回傳值是否可用 |
| copy | shallow copy / deep copy 行為不同 | 哪些層級共用同一個物件 |
| scope | local / global / nonlocal 易混淆 | 變數實際綁在哪個 scope |
| OOP | inheritance / override / MRO | method lookup 找到哪個版本 |
| generator | generator 只能消耗一次 | 第一次與第二次 `list(gen)` 的差異 |
| regex | pattern 不直觀 | 實際抓到哪些 token |
| slicing | 右端點是否包含、step 方向 | 實際取哪些 index |
| pandas index | label vs position | `loc` / `iloc` 的差異 |
| missing values | `None`, `np.nan`, `pd.NA` 行為不同 | 運算或 dtype 如何受影響 |
| file I/O | 本機檔案可能不存在 | 用條件或註解避免整本中斷 |
| random simulation | 結果不固定 | seed、樣本數、估計量意義 |
| numerical methods | 可能不收斂 | 停止條件與錯誤處理 |

---

## Verification

完成後必須驗證：

1. Notebook 可 Restart & Run All，零例外。
2. 若可用 CLI，執行：

```bash
jupyter nbconvert --to notebook --execute "<NOTEBOOK>.ipynb" --ExecutePreprocessor.kernel_name=<KERNEL>
```

3. 若原 notebook 使用 `python3` kernel 但本機沒有，先查：

```bash
jupyter kernelspec list
```

再選可用的 course kernel，例如 `python-programming-venv`。

4. 刪除驗證產生的 `*.executed.ipynb` 臨時檔。
5. 確認只留下目標 notebook 與必要文件變更。

---

## Final Review Checklist

交付前逐項檢查：

- 沒有 `assignment：把右側結果綁定` 這類模板註解。
- 沒有 `宣告函式`, `呼叫函式`, `display 輸出` 這類空話註解。
- 每個新增註解都能回答「這行 code 對資料或流程造成什麼影響」。
- 同一 cell 沒有重複解釋同一語法。
- Markdown 沒有 emoji 或裝飾性 prefix。
- Extended Variant 可獨立執行，或已清楚標明依賴。
- 驗證命令已跑過，或清楚記錄無法驗證的原因。
