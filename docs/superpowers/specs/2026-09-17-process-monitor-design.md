# 喷砂工艺参数监测模块 — 设计文档

- **日期**：2026-09-17
- **状态**：已评审通过，待实现
- **目标**：将电磁继电器动触簧缺陷识别系统的「实时分析数据」面板及其 GLCM 算法移植到 PCB 阻焊前喷砂质量在线检测系统

---

## 1. 背景与动机

继电器系统（`（精简版）电磁继电器动触簧表面缺陷识别系统3.py`）界面上有一块「实时分析数据」区域，包含 6 个 GLCM 纹理特征值、综合判定、报警状态指示灯和智能维护建议。

PCB 喷砂项目其实**已经计算了这 6 个 GLCM 特征**：

- `core/texture.py:50` 的 `GLCMFeatures` 数据类已含 `contrast` / `energy` / `entropy` / `homogeneity` / `correlation` / `dissimilarity`
- `core/pipeline.py:143` 将其存入 `InspectionResult.texture_features`
- `ui/workers.py:71` 也原样回传

但 `ui/main_window.py` 从未将其显示出来。因此本次工作的主体是**把已有数据接出来，并新增一路独立口径的 GLCM 用于工艺判定**。

---

## 2. 已确认的设计决策

| # | 决策点 | 结论 |
|---|---|---|
| D1 | GLCM 数值口径 | **两套并存**。PCB 现有 256 级 GLCM 继续用于缺陷检测与分类器，不动；新增一路 8 级「继电器口径」GLCM 专供工艺面板 |
| D2 | 实时程度 | **两条路径都要**。离线检测完成后刷新；在线摄像头模式下逐帧刷新 |
| D3 | 与现有判定的关系 | **分工**。`quality.ok_ng` 仍是唯一主判定，驱动 PLC 与现有评分 UI；继电器那套降级为「工艺参数监测」，走独立指示灯，不参与 OK/NG |
| D4 | 窗口布局 | **保持现状**：左图像 + 右面板 |
| D5 | 面板位置 | 右侧 `QTabWidget` **上方**的常驻区块，切换标签页时报警灯仍可见 |

### D1 的why：为什么不能直接沿用现有 GLCM

继电器系统的报警阈值（`contrast < 0.3` / `< 0.5` / `> 2.0`）是在 **8 级灰度量化**下标定的。量化步长为 32 个灰度级，相邻像素常相差 1 级，`(i-j)² = 1`；而 256 级量化下相邻像素常相差约 10 个灰度级，`(i-j)² ≈ 100`。因此 PCB 现有 GLCM 的 `contrast` 会大 1~2 个数量级，直接套用继电器阈值将导致**永远报警或永远不报警**。

结论：新开一路 8 级口径的提取器，保证数值与继电器系统可比、阈值可直接沿用。

---

## 3. 数值口径硬规则

这三条是本模块的正确性基础，任何一条违反都会使阈值失效。

### R1 — 量化与邻域参数固定

灰度量化 **8 级**，距离 **1**，角度 **[0, 45, 90, 135]** 四点，对四个角度**取平均**输出单组 6 特征。

### R2 — 只喂原始灰度，不喂预处理后的图

`core/preprocessing.py` 的 `Preprocessor` 会执行多尺度 Retinex 与 CLAHE（见 `config/default.yaml:28-37`），显著改变灰度分布，进而使 GLCM 特征漂移。

继电器系统是直接在原始 `cv_image` 上计算的。本模块照此执行，输入固定为 `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` 的结果，**不经过 Preprocessor**。

### R3 — 离线与在线统一降采样

在线路径逐帧处理，离线路径处理整图。若两者分辨率不同，同一块板在两条路径下会得出不同的特征值，共用一套阈值将自相矛盾。

因此两条路径均先将图像长边缩放到 `resize_long_side`（默认 640）后，再进入 GLCM 计算。

---

## 4. 工艺判定规则

以下规则移植自继电器系统 `update_analysis_results()`，标注处为本次建议的修正。

| contrast 区间 | 显示文本 | 维护建议 | 颜色 | 闪烁 |
|---|---|---|---|---|
| `< 0.3` | 异常（见修正 1） | 提高速度到 `speed_base - contrast` | 红 | 500ms |
| `[0.3, 0.5)` | 偏小 | 提高速度到 `speed_base - contrast` | 橙 | 无 |
| `[0.5, 2.0]` | 正常 | 纹理清晰，检测环境理想 | 绿 | 无 |
| `> 2.0` | 偏大 | 降低速度到 `speed_base - contrast` | 红 | 无 |

`speed_base` 默认 `3.0`。

### 对原始实现的两处修正

**修正 1 — `< 0.3` 分支的文本矛盾。** 原实现该分支写入 `"正常: {contrast}"`，却同时启动红灯闪烁，自相矛盾。判定为原代码笔误，改显示「异常」。

**修正 2 — 建议速度值需钳制下界。** 原公式 `speed_base - contrast` 在 contrast ≥ speed_base 时产出零或负速度，无物理意义。输出前钳制到 `[0.1, speed_base]`。

### 关于工艺名词

继电器系统调节的对象是喷砂枪移动速度。PCB 喷砂线上实际操作量可能是移动速度、喷砂压力或磨料粒度，尚未确认。

为免此处成为阻塞点，**建议文案模板全部走配置**（见 §7 `advice` 段）。变更措辞只需改 YAML，无需改动代码。

---

## 5. 数据流

### 离线路径

```
加载图像 → [开始检测] → DetectionWorker（QThread 后台）
    ├─ 现有流水线：256 级 GLCM + LBP + Gabor + 缺陷检测 + 质量评分
    │     → 质量评分卡 / 缺陷列表 / 热力图 / 统计
    └─ ProcessMonitor：8 级 GLCM（约 5 ms）
          → 工艺面板
```

特征计算放在 worker 线程内，避免阻塞 GUI。

### 在线路径

```
摄像头帧 → 100 ms 定时器 → 降采样至长边 640 → 8 级 GLCM
    → 工艺面板
```

在线路径**不执行** Gabor 滤波器组、SVM 分类与缺陷检测。原因是 `GaborFilterBank` 需对 72 个卷积核逐一执行 `scipy.signal.convolve2d`，单帧耗时达秒级，无法满足实时性。

### 两路互不干扰的约束

在线路径只更新工艺面板，**不触碰** `ok_ng`、不生成缺陷标注、不写入统计面板。工艺报警灯与质量评分 UI 在视觉上分区呈现，不产生矛盾信号。

---

## 6. 模块设计

### 6.1 `core/process_monitor.py`（新增）

纯算法模块，无 Qt 依赖，可独立测试。

```python
@dataclass
class ProcessGLCMFeatures:
    """继电器口径的 6 个 GLCM 特征。"""
    contrast: float          # 对比度
    correlation: float       # 相关性
    energy: float            # 能量值 = sqrt(ASM)
    dissimilarity: float     # 差异性
    homogeneity: float       # 同质性
    asm: float               # ASM 值 = sum(p^2)


@dataclass
class ProcessVerdict:
    """工艺判定结果。"""
    level: str               # "正常" | "偏小" | "偏大" | "异常"
    suggestion: str          # 维护建议文本
    alarm: bool              # 是否触发闪烁报警
    color: str               # 显示色（十六进制）


class ProcessGLCMExtractor:
    def __init__(self, levels=8, distance=1, angles=(0,45,90,135),
                 resize_long_side=640): ...
    def compute(self, bgr_or_gray: np.ndarray) -> ProcessGLCMFeatures: ...


class ProcessMonitor:
    def __init__(self, config: dict): ...
        # 读取 config["inspection"]["process_monitor"]，
        # 该段缺失时回退到内置默认值
    def evaluate(self, features: ProcessGLCMFeatures) -> ProcessVerdict: ...
```

**关于 `energy` 与 `asm` 的区分。** 继电器面板同时显示「能量值」与「ASM值」两个数，二者为平方关系（`energy = sqrt(ASM)`）。PCB 现有 `GLCMFeatures.energy` 字段名虽为 energy，实际存储的却是 `sum(glcm**2)`，即 ASM。

本模块**不复用** PCB 的字段命名，按继电器口径输出两个独立字段，避免语义混淆。

### 6.2 `ui/process_panel.py`（新增）

```python
class ProcessPanel(QWidget):
    """「实时分析数据」常驻区块。"""
    def __init__(self, parent=None): ...
    def update_features(self, f: ProcessGLCMFeatures) -> None: ...
    def update_verdict(self, v: ProcessVerdict) -> None: ...
    def reset(self) -> None: ...
```

控件构成：6 个只读 `QLineEdit`（对比度 / 相关性 / 能量值 / 差异性 / 同质性 / ASM值）、工艺判定输入框、报警状态指示灯 `QLabel`、智能维护建议输入框。

样式沿用 PCB 现有浅色主题（不使用继电器系统的深色主题），仅报警指示灯使用红 / 橙 / 绿三色。闪烁由内部 `QTimer` 驱动，周期 500 ms。

### 6.3 改动清单

| 文件 | 改动内容 |
|---|---|
| `core/pipeline.py` | `InspectionResult` 新增 `process_features: Optional[ProcessGLCMFeatures] = None` 字段；`run()` 中新增工艺特征计算阶段 |
| `ui/main_window.py` | 右侧 `QTabWidget` 上方插入 `ProcessPanel`；在线模式挂载 100 ms 轻量帧定时器；`_on_detection_finished` 中刷新面板 |
| `ui/workers.py` | `DetectionWorker.run()` 中增加一次 `ProcessMonitor` 调用，结果填入 `InspectionResult` |
| `config/default.yaml` | 新增 `inspection.process_monitor` 配置段 |
| `main.py` | CLI 模式输出工艺判定（可选） |
| `utils/validators.py` | 无需改动。现有 `validate_config` 仅检查必需键存在，不拒绝多余键 |

---

## 7. 配置

新增于 `config/default.yaml` 的 `inspection` 段下：

```yaml
  # --- 工艺参数监测（继电器口径 GLCM） ---
  process_monitor:
    enabled: true
    glcm:
      levels: 8                  # 灰度量化级数（须为 8，阈值依赖此口径）
      distance: 1                # 像素距离
      angles: [0, 45, 90, 135]   # 角度（度），输出取其平均
    resize_long_side: 640        # 统一分辨率口径（R3）
    thresholds:
      contrast_low_alarm: 0.3    # 低于此值触发闪烁报警
      contrast_low_warn: 0.5     # 低于此值判「偏小」
      contrast_high: 2.0         # 高于此值判「偏大」
      speed_base: 3.0            # 建议速度公式基数
    advice:
      alarm: "提高速度到 {speed:.2f}"
      low: "提高速度到 {speed:.2f}"
      high: "降低速度到 {speed:.2f}"
      normal: "纹理清晰，检测环境理想"
```

配置缺失时使用内置默认值，保证旧配置文件仍可加载。

---

## 8. 测试

`tests/test_process_monitor.py`：

1. **提取器正确性**
   - 常数灰度图 → `contrast ≈ 0`，`energy` 接近其上界
   - 高频棋盘图 → `contrast` 显著升高
   - 断言输出 6 个字段均为有限值

2. **口径一致性（验证 R3）**
   - 构造一张长边显著大于 640 的图像 `A`；将 `A` 预先缩放至长边 640 得到 `B`
   - 分别对 `A`、`B` 调用 `compute()`，断言两次结果的 `contrast` 相对偏差在容差内
   - 这验证了「提取器内部会自行降采样」与「调用方预先降采样」两种情形结果一致，
     即离线与在线路径共用阈值不会互相矛盾

3. **判定规则边界**
   - 逐一覆盖 `0.29 / 0.31 / 0.49 / 0.51 / 1.99 / 2.01`
   - 断言 `< 0.3` 分支 `alarm=True` 且文本为「异常」（修正 1）
   - 断言大 contrast 下建议速度被钳制为非负（修正 2）

4. **配置回退**
   - 传入不含 `process_monitor` 段的配置，应正常构造并使用默认值

---

## 9. 明确不做的事

- 不改动 PCB 现有 256 级 GLCM 及其下游的缺陷检测、SVM 分类器
- 不改动 `ok_ng` 判定逻辑与 PLC 输出语义
- 不将在线路径接入完整检测流水线
- 不重构 `ui/workers.py` 与 `core/pipeline.py` 之间已有的流程重复（属既有问题，与本次目标无关）

---

## 10. 实施记录（2026-09-17，实施后回填）

实施过程中有三处与本文档前述内容不符，以本节为准。

### 10.1 阈值**不能**沿用继电器原值（推翻第 1 节结论）

第 1 节结论曾写「保证数值与继电器系统可比、阈值可直接沿用」。实测推翻了这一条：

把 `data/samples` 下 6 张样本按 8 级口径跑一遍，`contrast` 为 **0.084 ~ 0.182**（中位 0.141），
**没有一张能到 0.3**。若沿用 `0.3 / 0.5 / 2.0`，6 张全部落入报警区间，报警灯恒亮 —— 等于没有报警。

原因：口径正确只保证了两套数值**可比**，不保证**同一区间**。继电器触簧与 PCB 喷砂面的
纹理强度本就相差一个量级，阈值必须在 PCB 自己的数据上重新标定。

另外，继电器那段 GLCM 代码本身很可能是坏的：它把二维切片传给 `graycoprops`，
在 scikit-image 0.26 上抛 `ValueError`，被宽 `except Exception` 吞掉后回退到硬编码预设值
（`contrast=1.3785...`）——该值恰好落在「正常」区间。也就是说 `0.3 / 0.5 / 2.0` 这组阈值
**很可能从未在真实数据上验证过**（参考：继电器面板实测始终为 ≈0.20）。

当前处置：`config/default.yaml` 里放了一组**临时值**（`0.08 / 0.11 / 0.22`，`speed_base` 由
`3.0` 缩放到 `0.5`），并在配置注释中标明「临时值，待现场标定数据替换」。
`core/process_monitor.py` 的 `DEFAULT_THRESHOLDS` 仍保留继电器原值，仅作为配置缺失时的
兜底，不再代表推荐取值。**上线前必须用真实良品/不良品图像重新标定。**

### 10.2 新增硬规则 R4：三通道输入的颜色顺序

`_to_gray` 最初固定使用 `cv2.COLOR_BGR2GRAY`，但本项目内部约定是 **RGB**
（`main.py` 与 `MainWindow._set_image()` 都先做 `BGR2RGB`，`DefectDetector` 也按 RGB 处理）。
后果是离线路径把 RGB 数据当 BGR 解释，R/B 权重对调：

| 样本 | 旧 离线 | 旧 在线 | 相对误差 |
|---|---|---|---|
| board_01_simple | 0.1230 | 0.1678 | 27% |
| board_06_dense | 0.1320 | 0.1816 | 27% |
| board_02_grid | 0.1042 | 0.1112 | 6% |

同一块板在两条路径下算出不同数值，直接违反 R3。故新增：

    R4  三通道输入默认按 RGB 解释；离线与在线必须传入相同的 color_order

`compute()` 增加 `color_order` 参数（默认 `"rgb"`），`_live_grab` 传 `frame_rgb`。
修复后两条路径逐位一致（已用假相机喂真帧做同图交叉验证）。

注意：用灰度图的三通道副本（`COLOR_GRAY2BGR`）**测不出**这个问题，那里 R/B 本就相同。

### 10.3 「开始检测」按钮的门控收拢到 _start_live / _stop_live

最初只在 `toggle_live_mode()` 里禁用/恢复 `detect_btn`，但菜单的「离线模式」走的是
`_set_mode()` → `_stop_live()`，不经过 `toggle_live_mode`。若在线模式下从菜单切回离线，
`detect_btn` 会永久卡在禁用状态，用户既不能预览也不能检测。

改为由 `_start_live()` / `_stop_live()` 这两个真正的状态迁移点负责门控，
并让 `_stop_live()` 一并复位 `_live_mode` 与预览按钮的选中态和文案。
相机打开失败时按 `current_raw_image` 决定按钮状态，避免「预览失败且无法检测」。
