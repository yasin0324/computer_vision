# 评估模块使用指南

本模块提供了完整的模型评估功能，包括基础模型评估、注意力模型评估、性能分析、模型比较和报告生成。

## 模块结构

```
src/evaluation/
├── __init__.py          # 模块初始化文件
├── evaluator.py         # 模型评估器（基础和注意力模型）
├── metrics.py           # 指标计算和性能分析
├── comparator.py        # 模型比较器
├── reporter.py          # 评估报告生成器
└── README.md           # 使用说明（本文件）
```

## 主要功能

### 1. 模型评估器 (evaluator.py)

#### ModelEvaluator - 基础模型评估器

```python
from src.evaluation import ModelEvaluator

# 创建评估器
evaluator = ModelEvaluator(model_path="path/to/model.pth")

# 运行评估
results = evaluator.evaluate()

# 打印结果
evaluator.print_results(results)
```

#### AttentionModelEvaluator - 注意力模型评估器

```python
from src.evaluation import AttentionModelEvaluator

# 创建注意力模型评估器
evaluator = AttentionModelEvaluator(
    model_path="path/to/senet_model.pth",
    model_type="senet"  # 或 "cbam"
)

# 运行评估（包含注意力分析）
results = evaluator.evaluate_with_attention(save_attention=True)

# 打印结果
evaluator.print_results(results)
```

### 2. 指标计算器 (metrics.py)

#### MetricsCalculator - 指标计算

```python
from src.evaluation import MetricsCalculator
import numpy as np

# 创建指标计算器
calculator = MetricsCalculator(class_names=['class1', 'class2', 'class3'])

# 计算所有指标
metrics = calculator.calculate_all_metrics(y_true, y_pred, y_prob)

# 访问不同类型的指标
basic_metrics = metrics['basic_metrics']  # 基础指标
per_class_metrics = metrics['per_class_metrics']  # 每个类别的指标
roc_metrics = metrics['roc_metrics']  # ROC相关指标
pr_metrics = metrics['pr_metrics']  # Precision-Recall指标
```

#### PerformanceAnalyzer - 性能分析

```python
from src.evaluation import PerformanceAnalyzer

# 创建性能分析器
analyzer = PerformanceAnalyzer(class_names=['class1', 'class2', 'class3'])

# 生成完整的性能报告
report = analyzer.generate_performance_report(y_true, y_pred, y_prob, image_paths)

# 生成性能分析图表
analyzer.plot_performance_charts(report, output_dir="path/to/output")
```

### 3. 模型比较器 (comparator.py)

```python
from src.evaluation import ModelComparator

# 创建比较器
comparator = ModelComparator(class_names=['class1', 'class2', 'class3'])

# 添加模型结果
comparator.add_model_from_dict(baseline_results, "baseline")
comparator.add_model_from_dict(senet_results, "senet")
comparator.add_model_from_dict(cbam_results, "cbam")

# 打印比较摘要
comparator.print_comparison_summary()

# 生成比较图表
comparator.plot_comparison_charts("path/to/output")

# 保存比较报告
comparator.save_comparison_report("comparison_report.json")
```

### 4. 报告生成器 (reporter.py)

```python
from src.evaluation import EvaluationReporter

# 创建报告生成器
reporter = EvaluationReporter(class_names=['class1', 'class2', 'class3'])

# 生成HTML报告
reporter.generate_html_report(
    evaluation_results,
    output_file="report.html",
    include_charts=True
)

# 生成Markdown报告
reporter.generate_markdown_report(
    evaluation_results,
    output_file="report.md"
)

# 生成模型比较报告
reporter.generate_comparison_report(
    comparison_results,
    output_file="comparison.html",
    format_type="html"
)
```

## 使用示例

### 快速开始

1. **评估单个模型**

```python
from src.evaluation import ModelEvaluator

# 评估基线模型
evaluator = ModelEvaluator("outputs/models/baseline/best_checkpoint.pth")
results = evaluator.evaluate()
evaluator.print_results(results)
```

2. **比较多个模型**

```python
from src.evaluation import ModelComparator

# 创建比较器
comparator = ModelComparator(['bacterial_spot', 'septoria_leaf_spot', 'target_spot', 'healthy'])

# 添加模型结果（从评估结果中获取）
comparator.add_model_from_dict(baseline_results)
comparator.add_model_from_dict(senet_results)

# 生成比较报告
comparator.print_comparison_summary()
comparator.plot_comparison_charts("outputs/comparison")
```

3. **生成详细报告**

```python
from src.evaluation import EvaluationReporter

reporter = EvaluationReporter(['bacterial_spot', 'septoria_leaf_spot', 'target_spot', 'healthy'])

# 生成HTML报告
reporter.generate_html_report(results, "detailed_report.html")
```

### 综合评估脚本

使用提供的综合评估脚本：

```bash
# 评估所有可用模型
python scripts/comprehensive_evaluation.py --models all --compare --generate_reports

# 评估特定模型
python scripts/comprehensive_evaluation.py --models baseline senet --compare

# 包含注意力分析
python scripts/comprehensive_evaluation.py --models senet cbam --save_attention
```

### 示例脚本

运行示例脚本来了解各个模块的功能：

```bash
python scripts/evaluation_example.py
```

## 输出结果

### 评估结果目录结构

```
outputs/evaluation/
├── baseline_evaluation_20231201_120000/
│   ├── confusion_matrix.png
│   ├── confusion_matrix.json
│   ├── detailed_predictions.csv
│   ├── error_analysis.csv
│   ├── evaluation_summary.json
│   ├── baseline_report.html
│   └── baseline_report.md
├── senet_evaluation_20231201_120500/
│   ├── confusion_matrix.png
│   ├── attention_analysis.json
│   ├── detailed_predictions.csv
│   └── ...
└── model_comparison/
    ├── comparison_report.json
    ├── comparison_report.html
    ├── overall_performance_comparison.png
    ├── per_class_f1_comparison.png
    ├── performance_radar.png
    └── performance_distribution.png
```

### 主要输出文件说明

1. **evaluation_summary.json** - 评估结果摘要
2. **detailed_predictions.csv** - 详细的预测结果
3. **confusion_matrix.png** - 混淆矩阵可视化
4. **error_analysis.csv** - 错误样本分析
5. **attention_analysis.json** - 注意力分析（仅注意力模型）
6. **model_report.html/md** - 详细的评估报告
7. **comparison_report.json** - 模型比较报告

## 配置选项

### 评估器配置

-   `device`: 计算设备 (cuda/cpu)
-   `batch_size`: 批处理大小
-   `num_workers`: 数据加载器工作进程数

### 报告生成配置

-   `include_charts`: 是否在 HTML 报告中包含图表
-   `format_type`: 报告格式 ('html' 或 'markdown')

### 比较器配置

-   `class_names`: 类别名称列表
-   `output_dir`: 输出目录

## 注意事项

1. **模型路径**: 确保模型文件存在且路径正确
2. **数据路径**: 确保测试数据文件存在 (`data/processed/test_split.csv`)
3. **内存使用**: 大型模型评估可能需要较多内存
4. **GPU 支持**: 注意力分析功能建议使用 GPU 加速
5. **依赖项**: 确保安装了所有必要的依赖包

## 扩展功能

### 自定义指标

可以通过继承 `MetricsCalculator` 类来添加自定义指标：

```python
class CustomMetricsCalculator(MetricsCalculator):
    def calculate_custom_metric(self, y_true, y_pred):
        # 实现自定义指标计算
        pass
```

### 自定义报告格式

可以通过继承 `EvaluationReporter` 类来支持新的报告格式：

```python
class CustomReporter(EvaluationReporter):
    def generate_pdf_report(self, results, output_file):
        # 实现PDF报告生成
        pass
```

## 故障排除

### 常见问题

1. **模型加载失败**

    - 检查模型文件路径
    - 确认模型架构匹配
    - 检查设备兼容性

2. **数据加载错误**

    - 确认测试数据文件存在
    - 检查数据格式
    - 验证类别映射

3. **内存不足**

    - 减少批处理大小
    - 使用 CPU 而非 GPU
    - 分批处理大型数据集

4. **图表生成失败**
    - 检查 matplotlib 后端设置
    - 确认输出目录权限
    - 验证数据格式

### 调试建议

1. 使用示例脚本测试功能
2. 检查日志输出中的错误信息
3. 验证输入数据格式
4. 确认所有依赖项已正确安装

## 更新日志

-   v1.0.0: 初始版本，包含基础评估功能
-   v1.1.0: 添加注意力模型评估支持
-   v1.2.0: 增加模型比较和报告生成功能
-   v1.3.0: 完善性能分析和可视化功能
