# Python 版本封装与调用示例

这个仓库新增了 `pybind11` 封装模块：`shape_based_matching_py`，可直接在 Python 中调用 C++ 的模板训练与匹配流程。

## 1. 构建

在仓库根目录执行：

```bash
cmake -S . -B build -DBUILD_PYTHON=ON
cmake --build build -j
```

编译成功后会在 `build/` 目录下生成 Python 扩展模块（如 `shape_based_matching_py*.so`）。

## 2. Python 中导入

方式 A：在仓库根目录执行脚本（推荐，示例脚本已按此假设）。

方式 B：手动把 `build/` 加到 `PYTHONPATH`：

```bash
export PYTHONPATH="$PWD/build:$PYTHONPATH"
python3 python_example.py
```

## 3. 核心 API

```python
import shape_based_matching_py as sbm

detector = sbm.Detector(
    num_features=128,
    T=[4, 8],
    weak_thresh=30.0,
    strong_thresh=60.0,
)
```

- `add_template(image_path, class_id, mask_path="", num_features=0) -> int`
  - 从图像添加模板，返回 `template_id`，失败时为 `-1`。
- `add_rotated_template(class_id, zero_id, theta, center_x, center_y) -> int`
  - 复用零角模板进行旋转模板扩增。
- `match(image_path, threshold, class_ids=[], mask_path="") -> list[dict]`
  - 返回匹配结果，元素包含：`x/y/similarity/class_id/template_id`。
- `get_template_info(class_id, template_id, pyramid_level=0) -> dict`
  - 返回模板宽高和偏移信息：`width/height/tl_x/tl_y/pyramid_level`。
- `write_classes(format="templates_%s.yml.gz")`
  - 保存模板。
- `read_classes(class_ids, format="templates_%s.yml.gz")`
  - 读取模板。

## 4. 最小调用示例

见仓库根目录：`python_example.py`。

运行：

```bash
python3 python_example.py
```

