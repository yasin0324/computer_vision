# 植物叶片病害识别系统 - Web 应用

基于深度学习的植物叶片病害智能识别 Web 平台，提供用户友好的界面进行病害识别、模型训练、评估和比较。

## 功能特性

### 🔍 病害识别

-   支持拖拽上传图像文件
-   实时预测结果展示
-   多模型选择（ResNet50、SE-Net、CBAM）
-   置信度分析和概率分布
-   病害类型详细说明

### 🛠️ 模型训练

-   可视化训练参数配置
-   实时训练进度监控
-   支持多种模型架构
-   训练状态实时更新

### 📊 模型评估

-   全面的性能指标计算
-   详细的评估报告生成
-   注意力权重分析
-   可视化图表展示

### ⚖️ 模型比较

-   多模型性能对比
-   交互式比较表格
-   统计分析结果
-   排名和推荐

### 📈 系统仪表板

-   系统运行状态监控
-   数据集统计信息
-   模型使用情况
-   实时性能图表

## 技术架构

### 后端技术

-   **Flask**: Web 框架
-   **PyTorch**: 深度学习框架
-   **PIL/Pillow**: 图像处理
-   **NumPy/Pandas**: 数据处理
-   **Matplotlib/Seaborn**: 可视化

### 前端技术

-   **Bootstrap 5**: UI 框架
-   **jQuery**: JavaScript 库
-   **Chart.js**: 图表库
-   **Font Awesome**: 图标库

### 支持的模型

-   **ResNet50**: 基线模型
-   **SE-Net**: 通道注意力机制
-   **CBAM**: 双重注意力机制

### 支持的病害类型

-   细菌性斑点病 (Bacterial Spot)
-   褐斑病 (Septoria Leaf Spot)
-   靶斑病 (Target Spot)
-   健康叶片 (Healthy)

## 安装和运行

### 1. 安装依赖

```bash
# 安装Web应用依赖
pip install -r webapp/requirements.txt

# 或使用项目根目录的requirements.txt
pip install -r requirements.txt
```

### 2. 启动应用

```bash
# 方法1: 使用启动脚本
python run_webapp.py

# 方法2: 直接运行Flask应用
python webapp/app.py

# 方法3: 使用Flask命令
export FLASK_APP=webapp.app
flask run --host=0.0.0.0 --port=5000
```

### 3. 访问应用

打开浏览器访问: http://localhost:5000

## 目录结构

```
webapp/
├── __init__.py              # 模块初始化
├── app.py                   # Flask主应用
├── utils.py                 # 工具类
├── requirements.txt         # 依赖文件
├── README.md               # 说明文档
├── templates/              # HTML模板
│   ├── base.html           # 基础模板
│   ├── index.html          # 主页
│   ├── predict.html        # 预测页面
│   ├── train.html          # 训练页面
│   ├── evaluate.html       # 评估页面
│   ├── compare.html        # 比较页面
│   ├── dashboard.html      # 仪表板
│   └── error.html          # 错误页面
└── uploads/                # 上传文件目录
```

## API 接口

### 预测接口

-   **POST** `/api/predict`
    -   上传图像进行病害识别
    -   参数: `file` (图像文件), `model_type` (模型类型)
    -   返回: 预测结果和置信度

### 训练接口

-   **POST** `/api/train`

    -   启动模型训练
    -   参数: 训练配置参数
    -   返回: 训练 ID

-   **GET** `/api/training_status/<training_id>`
    -   获取训练状态
    -   返回: 训练进度和指标

### 评估接口

-   **POST** `/api/evaluate`
    -   评估模型性能
    -   参数: 模型路径和配置
    -   返回: 评估结果

### 比较接口

-   **POST** `/api/compare`
    -   比较多个模型
    -   参数: 模型列表
    -   返回: 比较报告

### 数据接口

-   **GET** `/api/models` - 获取可用模型列表
-   **GET** `/api/datasets` - 获取数据集信息
-   **GET** `/api/dashboard_data` - 获取仪表板数据

## 使用说明

### 病害识别

1. 访问"病害识别"页面
2. 上传植物叶片图像（支持拖拽）
3. 选择识别模型
4. 点击"开始识别"
5. 查看预测结果和置信度

### 模型训练

1. 访问"模型训练"页面
2. 配置训练参数（模型类型、轮数、学习率等）
3. 点击"开始训练"
4. 实时监控训练进度

### 模型评估

1. 访问"模型评估"页面
2. 选择要评估的模型
3. 配置评估选项
4. 查看详细评估报告

### 模型比较

1. 访问"模型比较"页面
2. 添加要比较的模型
3. 配置模型信息
4. 查看比较结果表格

## 配置说明

### 文件上传限制

-   支持格式: JPG, PNG, GIF, BMP
-   最大文件大小: 16MB
-   自动图像预处理和缩放

### 模型配置

-   自动检测可用模型文件
-   支持模型热加载
-   GPU/CPU 自适应

### 安全设置

-   文件类型验证
-   路径安全检查
-   错误处理和日志记录

## 故障排除

### 常见问题

1. **模型文件未找到**

    - 确保模型文件在 `models/` 目录下
    - 检查文件权限和路径

2. **GPU 不可用**

    - 检查 CUDA 安装
    - 系统会自动降级到 CPU 模式

3. **端口被占用**

    - 修改 `app.run()` 中的端口号
    - 或终止占用端口的进程

4. **依赖包缺失**
    - 重新安装 requirements.txt 中的依赖
    - 检查 Python 版本兼容性

### 日志查看

-   应用日志: 控制台输出
-   训练日志: `logs/` 目录
-   错误日志: Flask 自动记录

## 开发说明

### 扩展功能

-   添加新的模型类型
-   扩展支持的病害类别
-   增加新的评估指标
-   自定义报告模板

### 性能优化

-   模型缓存机制
-   异步任务处理
-   数据库集成
-   负载均衡

### 部署建议

-   使用 Gunicorn 作为 WSGI 服务器
-   配置 Nginx 反向代理
-   设置 SSL 证书
-   容器化部署

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进项目。

## 联系方式

如有问题或建议，请通过以下方式联系：

-   项目仓库: [GitHub 链接]
-   邮箱: [联系邮箱]
