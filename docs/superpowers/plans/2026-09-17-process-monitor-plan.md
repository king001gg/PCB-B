# 喷砂工艺参数监测模块 — 实现计划

- **日期**：2026-09-17
- **对应设计**：`docs/superpowers/specs/2026-09-17-process-monitor-design.md`
- **项目**：`E:\PCB-B版本` PCB 阻焊前喷砂质量在线检测系统（PySide6 + OpenCV + scikit-image）

---

## 前置说明

设计文档中的三条硬规则在实现时以注释形式标在代码中，便于后人追溯：

- **R1** 量化 8 级 / 距离 1 / 4 角度取平均
- **R2** 只喂原始灰度，不经过 `Preprocessor`
- **R3** 离线与在线统一降采样至长边 640

计划共 9 个任务，按依赖顺序排列。任务 1-4 可独立完成并测试，不触碰现有功能；任务 5 起开始接入主链路。

---

## 任务 1 — `core/process_monitor.py`：8 级 GLCM 提取器

**文件**：新建 `core/process_monitor.py`

### 实现要点

```python
@dataclass
class ProcessGLCMFeatures:
    contrast: float          # 对比度
    correlation: float       # 相关性
    energy: float            # 能量值 = sqrt(ASM)
    dissimilarity: float     # 差异性
    homogeneity: float       # 同质性
    asm: float               # ASM 值 = sum(p^2)
```

`ProcessGLCMExtractor.__init__(levels=8, distance=1, angles=(0,45,90,135), resize_long_side=640)`

`compute(img)` 流程：

1. **R2** — 若 `img.ndim == 3` 则 `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`；已是单通道则直接用。**不得**调用 `Preprocessor`
2. **R3** — 长边超过 `resize_long_side` 时按比例缩放（`cv2.INTER_AREA`），否则原样
3. **R1** — `np.floor(gray * (levels / 256.0)).astype(np.uint8)` 量化到 8 级
4. `graycomatrix(..., distances=[distance], angles=[弧度], levels=levels, symmetric=True, normed=True)`
5. 逐角度用 `graycoprops` 取 6 个特征，**再对 4 个角度取平均**

### 关键实现约束

**必须用 `graycoprops` 而非手写公式。** 继电器原实现用的就是 `graycoprops`，其中 `'energy'` 返回的是 `sqrt(ASM)`。手写公式极易把 `energy` 写成 `sum(p**2)`（即 ASM），导致与继电器数值不可比、阈值失效。这是本模块最易出错的一点。

**skimage 缺失时直接抛错，不做 numpy 回退。** `requirements.txt:18` 已将 `scikit-image>=0.22.0` 列为必需依赖，且 `core/texture.py` 的 `LBPExtractor.multi_radius_histogram()` 本就无条件调用 skimage。再加一条手写 GLCM 路径只会带来与主路径数值漂移的风险。在 `__init__` 中检查 `HAS_SKIMAGE`，缺失则抛 `RuntimeError("工艺监测模块依赖 scikit-image，请先安装")`。

**`energy` 与 `asm` 必须是两个独立字段。** PCB 现有 `core/texture.py:53` 的 `GLCMFeatures.energy` 字段名不副实，实际存的是 `sum(glcm**2)`。本模块不复用该命名。

### 验证

```bash
python -c "
import numpy as np
from core.process_monitor import ProcessGLCMExtractor
e = ProcessGLCMExtractor()
flat = np.full((200,200), 128, np.uint8)
print('常数图:', e.compute(flat))
ck = np.indices((200,200)).sum(0) % 2 * 255
print('棋盘图:', e.compute(ck.astype(np.uint8)))
"
```

预期：常数图的 `contrast ≈ 0`；棋盘图的 `contrast` 显著更高。

---

## 任务 2 — `core/process_monitor.py`：工艺判定规则

**文件**：同 `core/process_monitor.py`

```python
@dataclass
class ProcessVerdict:
    level: str        # "正常" | "偏小" | "偏大" | "异常"
    suggestion: str
    alarm: bool       # 是否触发 500ms 闪烁
    color: str        # 十六进制色值
```

`ProcessMonitor.__init__(config)` 读取 `config["inspection"]["process_monitor"]`，该段整体缺失时回退内置默认值（保证旧配置文件可加载）。

`evaluate(features) -> ProcessVerdict` 分支：

| 条件 | level | 颜色 | alarm | 建议模板 |
|---|---|---|---|---|
| `c < contrast_low_alarm` | 异常 | 红 `#e74c3c` | `True` | `advice.alarm` |
| `c < contrast_low_warn` | 偏小 | 橙 `#f39c12` | `False` | `advice.low` |
| `c > contrast_high` | 偏大 | 红 `#e74c3c` | `False` | `advice.high` |
| 其余 | 正常 | 绿 `#2ecc71` | `False` | `advice.normal` |

### 两处修正（对应设计 §4）

1. **`c < contrast_low_alarm` 分支的文本** —— 继电器原实现此处写 `"正常"` 却同时点红灯，自相矛盾，改为「异常」
2. **建议速度钳制** —— `speed = speed_base - contrast`，输出前 `speed = max(0.1, min(speed, speed_base))`，避免出现零或负速度

建议文本走 `advice` 配置模板，用 `str.format(speed=...)` 渲染。`advice.normal` 不含 `{speed}` 占位符，渲染时需容错——先判断模板中是否含 `{speed` 再决定传参，或统一用 `format_map` 配合 `defaultdict`。**推荐**：统一构造 `{"speed": speed}` 后调用 `.format(**kw)`，`normal` 模板忽略多余关键字参数即可（`str.format` 允许多余关键字）。

### 验证

边界值 0.29 / 0.31 / 0.49 / 0.51 / 1.99 / 2.01 逐一断言 `level` 与 `alarm`。

---

## 任务 3 — `config/default.yaml`：新增配置段

**文件**：`config/default.yaml`，在 `inspection:` 下新增（建议置于 `texture:` 段之后、`defects:` 段之前）：

```yaml
  # --- 工艺参数监测（继电器口径 GLCM，8 级量化） ---
  process_monitor:
    enabled: true
    glcm:
      levels: 8                  # 须为 8，阈值 0.3/0.5/2.0 依赖此口径
      distance: 1
      angles: [0, 45, 90, 135]
    resize_long_side: 640        # 统一分辨率口径（R3）
    thresholds:
      contrast_low_alarm: 0.3
      contrast_low_warn: 0.5
      contrast_high: 2.0
      speed_base: 3.0
    advice:
      alarm: "提高速度到 {speed:.2f}"
      low: "提高速度到 {speed:.2f}"
      high: "降低速度到 {speed:.2f}"
      normal: "纹理清晰，检测环境理想"
```

`utils/validators.py` 的 `validate_config` 只检查必需键存在、不拒绝多余键，**无需改动**。

---

## 任务 4 — `tests/test_process_monitor.py`

**文件**：新建 `tests/test_process_monitor.py`

四组测试：

1. **提取器正确性** — 常数图 `contrast ≈ 0`；棋盘图 `contrast` 显著更高；6 个字段均为有限值
2. **口径一致性（R3 回归测试）** — 构造长边 >640 的图 `A`，预缩放至长边 640 得 `B`，分别 `compute()`，断言 `contrast` 相对偏差在容差内。这条同时验证「提取器内部降采样」与「调用方预降采样」等价
3. **判定边界** — 0.29/0.31/0.49/0.51/1.99/2.01；断言 `< 0.3` 分支 `alarm=True` 且 `level == "异常"`；断言大 contrast 下建议速度非负
4. **配置回退** — 传入不含 `process_monitor` 段的 config，应正常构造并使用默认值

**验证**：`pytest tests/test_process_monitor.py -v`

---

## 任务 5 — `core/pipeline.py`：接入流水线

**文件**：`core/pipeline.py`

1. `InspectionResult` 新增字段：`process_features: Optional[ProcessGLCMFeatures] = None`
2. `InspectionPipeline.__init__` 中实例化 `self.process_monitor = ProcessMonitor(config)`
3. `run()` 中新增阶段（建议放在 Phase 2 纹理分析之后，Phase 3 之前）：

```python
# --- Phase 2.5: 工艺参数监测 ---
if self.process_monitor.enabled:
    raw_gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = self.process_monitor.extractor.compute(image)
    result.process_features = features
```

注意：**必须从 `image` 取灰度，不能用 `result.gray`**。`result.gray` 是 `Preprocessor` 的输出（含 Retinex + CLAHE），违反 R2。

### 顺带修复：`core/pipeline.py` 缺失 `import cv2`（既有 bug）

`core/pipeline.py` 第 165 行：

```python
color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) if hasattr(cv2, 'cvtColor') else gray
```

模块顶部（第 9-18 行）**没有 `import cv2`**。因此一旦执行到这一行，`hasattr(cv2, ...)` 会因 `cv2` 未定义而抛 `NameError`——`hasattr` 只吞 `AttributeError`，吞不掉 `NameError`，那个看似防御性的 `hasattr` 实际不起作用。

当前未暴露的原因是该分支只在输入为单通道图像时进入，而 CLI（`main.py:112` 转 RGB）和 GUI（`_set_image()` 转 RGB）都传三通道。但 `config/default.yaml:128` 的相机像素格式是 `Mono8`，一旦走 `process_acquisition()` 接入真实工业相机，传入的就是二维灰度图，此分支必然被触发。

本任务新增的工艺特征计算同样需要 `cv2` 且同样要处理灰度输入，因此**必须一并修复**：在 `core/pipeline.py` 顶部补 `import cv2`。

**验证**：`python main.py --cli <样本图>` 正常完成；再用灰度图直接调用 `pipeline.run(gray_2d)` 确认不再抛 `NameError`。

---

## 任务 6 — `ui/workers.py`：离线路径计算

**文件**：`ui/workers.py`

在 `DetectionWorker.run()` 中，第 (5) 步缺陷检测之前插入工艺特征计算，结果填入 `InspectionResult(process_features=...)`。

放在 worker 线程内计算，避免阻塞 GUI（8 级 GLCM 约 5 ms，但整图未降采样时会更久）。

**验证**：GUI 中加载图像并检测，确认 `result.process_features` 非 None。

---

## 任务 7 — `ui/process_panel.py`：右侧常驻区块

**文件**：新建 `ui/process_panel.py`

```python
class ProcessPanel(QWidget):
    def __init__(self, parent=None): ...
    def update_features(self, f: ProcessGLCMFeatures) -> None: ...
    def update_verdict(self, v: ProcessVerdict) -> None: ...
    def reset(self) -> None: ...
```

### 控件构成

- 标题「实时分析数据」
- 6 个只读 `QLineEdit`：对比度 / 相关性 / 能量值 / 差异性 / 同质性 / ASM值
- 工艺判定输入框（只读）
- 报警状态指示灯 `QLabel`（20×20 圆点）+ `QTimer` 驱动 500 ms 闪烁
- 智能维护建议输入框（只读）

### 样式

沿用 PCB 现有浅色主题，**不使用**继电器系统的深色主题。仅报警指示灯使用任务 2 定义的红/橙/绿三色。

实际生效的全局样式表在 `main.py:50-85`（应用级 `app.setStyleSheet`）。注意 `ui/main_window.py:639-649` 有一份内容相同的副本，位于 `if __name__ == "__main__":` 块内——只有直接运行 `ui/main_window.py` 时才生效，正常从 `main.py` 启动时不会走到。两份重复属既有问题，本次不动，但改样式时需知道以 `main.py` 那份为准。

闪烁实现参考继电器 `_toggle_alarm_flash()`，但需修正其一个缺陷：原实现在闪烁时**内联设置 `setStyleSheet`**，会覆盖全局样式表且无法恢复。本次改为通过 `setProperty("alarm", True/False)` + `style().unpolish/polish()` 切换，或直接对指示灯 `setStyleSheet` 但只覆盖背景色一项。

`reset()` 应停止闪烁定时器并将所有字段复位为占位文本，避免残留上一次的报警状态。

---

## 任务 8 — `ui/main_window.py`：接入主窗口

**文件**：`ui/main_window.py`

### 8.1 插入面板

在 `_init_ui()` 中，`right_layout.addWidget(right_tabs)` **之前**插入 `self.process_panel = ProcessPanel()`。这样面板位于 `QTabWidget` 上方，切换标签页时报警灯始终可见（决策 D5）。

### 8.2 离线路径刷新

在 `_on_detection_finished()` 中：

```python
if result.process_features is not None:
    self.process_panel.update_features(result.process_features)
    self.process_panel.update_verdict(
        self.process_monitor.evaluate(result.process_features)
    )
```

`self.process_monitor` 在 `_init_core_modules()` 中实例化。

### 8.3 在线路径

`_live_grab()`（`ui/main_window.py:580`）中，在 `self.image_viewer.set_image(frame_rgb)` 之后追加：

```python
feats = self.process_monitor.extractor.compute(frame)   # frame 为 BGR
self.process_panel.update_features(feats)
self.process_panel.update_verdict(self.process_monitor.evaluate(feats))
```

注意 `frame` 是 BGR，`compute()` 内部会自行转灰度（R2）并降采样（R3），**不要**传 `frame_rgb`。

### 8.4 两条路径互斥（设计 §5 的落地机制）

**在线模式下禁用「开始检测」按钮。** 现状 `toggle_live_mode()` 不禁用 `detect_btn`，用户可同时触发两条路径争抢面板写入。

- `toggle_live_mode()` 进入实时预览时：`self.detect_btn.setEnabled(False)`；退出时恢复为 `self.current_raw_image is not None`
- `load_image()` / `_set_image()` 中若处于 `self._live_mode`，同样保持禁用

退出实时预览时**不清空**面板，保留最后一批读数，避免数值跳空。

### 8.5 加载新图像时复位

在 `_set_image()` 中已有的 `self.result_text.clear()` 附近，追加 `self.process_panel.reset()`。

---

## 任务 9 — `main.py`：CLI 输出（可选）

**文件**：`main.py`

在 `run_cli()` 的报告输出段落后，追加工艺判定打印：

```python
if report_result.process_features:
    verdict = process_monitor.evaluate(report_result.process_features)
    print(f"\n工艺判定: {verdict.level} — {verdict.suggestion}")
```

需要将 `InspectionPipeline` 的产出接入 `run_cli()`，或在 CLI 中直接实例化 `ProcessMonitor`。此项非必需，可延后。

---

## 完成标准

全部任务完成后应满足：

1. `pytest tests/` 全部通过（含既有测试，确认无回归）
2. GUI 中加载样本图 → 点击检测 → 右侧顶部面板填出 6 个 GLCM 值与工艺判定
3. 切换到「热力图」「统计」标签页时，报警指示灯仍可见
4. 开启实时预览后，「开始检测」按钮变灰；预览期间面板数值随帧刷新
5. 触发 `contrast < 0.3` 时红灯 500 ms 闪烁，文本显示「异常」而非「正常」
6. 加载旧版不含 `process_monitor` 段的配置文件，系统正常启动并使用默认值

---

## 风险与注意事项

| 风险 | 说明 | 应对 |
|---|---|---|
| `graycoprops` 的 `energy` 语义 | 返回 `sqrt(ASM)` 而非 ASM，手写公式极易搞反 | 任务 1 强制使用 `graycoprops`，并在测试中断言 `energy ≈ sqrt(asm)` |
| 误用预处理后的灰度 | `result.gray` 含 Retinex + CLAHE，会让特征漂移 | 任务 5 明确从 `image` 取灰度 |
| `pipeline.py` 缺 `import cv2` | 既有 latent bug，单通道输入时抛 `NameError`，`hasattr` 守护无效 | 任务 5 一并补上 import；接入 Mono8 相机前必须修复 |
| 在线路径性能 | 未降采样的大图 GLCM 会拖慢帧率 | R3 在提取器内部兜底降采样，调用方无需关心 |
| 报警灯样式被覆盖 | 继电器原实现内联 `setStyleSheet` 会破坏全局样式 | 任务 7 限定只覆盖背景色 |
| 阈值可移植性 | 0.3/0.5/2.0 源自继电器样品，PCB 样品上未必合理 | 已列入设计 §7 配置化，可标定后调整而无需改代码 |
