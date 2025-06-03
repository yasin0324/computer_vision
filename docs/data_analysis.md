# PlantVillage数据集分析报告

**生成时间**: 2025-06-03 14:51:32

---

## 1. 数据集结构分析
**总类别数: 15**
**总图像数: 41276**
**平均每类图像数: 2751.7**
## 2. 按作物分类统计
### Tomato
- **总图像数: 32022**
- **病害类别数: 20**
- **- _Tomato_YellowLeaf__Curl_Virus: 3208 张**
- **- _Tomato_YellowLeaf__Curl_Virus: 3208 张**
- **- Bacterial_spot: 2127 张**
- **- Bacterial_spot: 2127 张**
- **- Late_blight: 1909 张**
- **- Late_blight: 1909 张**
- **- Septoria_leaf_spot: 1771 张**
- **- Septoria_leaf_spot: 1771 张**
- **- Spider_mites_Two_spotted_spider_mite: 1676 张**
- **- Spider_mites_Two_spotted_spider_mite: 1676 张**
- **- healthy: 1591 张**
- **- healthy: 1591 张**
- **- _Target_Spot: 1404 张**
- **- _Target_Spot: 1404 张**
- **- Early_blight: 1000 张**
- **- Early_blight: 1000 张**
- **- Leaf_Mold: 952 张**
- **- Leaf_Mold: 952 张**
- **- _Tomato_mosaic_virus: 373 张**
- **- _Tomato_mosaic_virus: 373 张**
### Potato
- **总图像数: 4304**
- **病害类别数: 6**
- **- Early blight: 1000 张**
- **- Early blight: 1000 张**
- **- Late blight: 1000 张**
- **- Late blight: 1000 张**
- **- healthy: 152 张**
- **- healthy: 152 张**
### Pepper  Bell
- **总图像数: 4950**
- **病害类别数: 4**
- **- healthy: 1478 张**
- **- healthy: 1478 张**
- **- Bacterial spot: 997 张**
- **- Bacterial spot: 997 张**
## 3. 细粒度识别候选子集
### Tomato - 细粒度识别候选
  Blight 相关病害:
- **- Early_blight: 1000 张**
- **- Late_blight: 1909 张**
- **- Early_blight: 1000 张**
- **- Late_blight: 1909 张**
  Spot 相关病害:
- **- _Target_Spot: 1404 张**
- **- Spider_mites_Two_spotted_spider_mite: 1676 张**
- **- Septoria_leaf_spot: 1771 张**
- **- Bacterial_spot: 2127 张**
- **- _Target_Spot: 1404 张**
- **- Spider_mites_Two_spotted_spider_mite: 1676 张**
- **- Septoria_leaf_spot: 1771 张**
- **- Bacterial_spot: 2127 张**
  Leaf 相关病害:
- **- _Tomato_YellowLeaf__Curl_Virus: 3208 张**
- **- Leaf_Mold: 952 张**
- **- Septoria_leaf_spot: 1771 张**
- **- _Tomato_YellowLeaf__Curl_Virus: 3208 张**
- **- Leaf_Mold: 952 张**
- **- Septoria_leaf_spot: 1771 张**
  Bacterial 相关病害:
- **- Bacterial_spot: 2127 张**
- **- Bacterial_spot: 2127 张**
  Virus 相关病害:
- **- _Tomato_YellowLeaf__Curl_Virus: 3208 张**
- **- _Tomato_mosaic_virus: 373 张**
- **- _Tomato_YellowLeaf__Curl_Virus: 3208 张**
- **- _Tomato_mosaic_virus: 373 张**
  Mold 相关病害:
- **- Leaf_Mold: 952 张**
- **- Leaf_Mold: 952 张**
### Potato - 细粒度识别候选
  Blight 相关病害:
- **- Early blight: 1000 张**
- **- Early blight: 1000 张**
- **- Late blight: 1000 张**
- **- Late blight: 1000 张**
### Pepper  Bell - 细粒度识别候选
  Spot 相关病害:
- **- Bacterial spot: 997 张**
- **- Bacterial spot: 997 张**
  Bacterial 相关病害:
- **- Bacterial spot: 997 张**
- **- Bacterial spot: 997 张**
## 4. 图像质量分析
**分析样本数: 50**
**损坏图像数: 0**
### 图像尺寸统计
- **宽度: 256 - 256 (平均: 256.0)**
- **高度: 256 - 256 (平均: 256.0)**
### 图像格式分布
- **JPEG: 50 张 (100.0%)**
### 图像质量统计
- **清晰度 (Laplacian方差): 62.5 - 19478.1**
- **平均清晰度: 5012.7**
- **文件大小: 7.7KB - 25.9KB**
### 可能的低质量图像 (清晰度 < 378.3)
- **8add9891-86bd-4b82-a37c-be8691b0002e___UF.GRC_YLCV_Lab 02230.JPG: 清晰度 62.5**
- **7c036479-9d89-40bc-8107-182c35aa5309___UF.GRC_YLCV_Lab 01600.JPG: 清晰度 148.2**
- **812b614b-bad8-4d10-9191-9a614bbdaef3___UF.GRC_YLCV_Lab 01418.JPG: 清晰度 157.6**
- **76b25fb0-e2fa-47ef-9b70-19efa3d65a9a___GHLB Leaf 1.4 Day 12.JPG: 清晰度 141.4**
- **d6e6897a-5083-4914-9903-804c5684a956___GHLB2 Leaf 102.JPG: 清晰度 167.5**
## 5. 数据集使用建议
基于分析结果，以下是针对细粒度识别研究的建议：
### 推荐的细粒度识别子集
## 1. Tomato 细粒度识别子集:
- **- Blight 相关病害组 (5818 张图像):**
- *** Early_blight: 1000 张**
- *** Late_blight: 1909 张**
- *** Early_blight: 1000 张**
- *** Late_blight: 1909 张**
- **✅ 数据相对平衡 (比例: 0.52)**
- **- Spot 相关病害组 (13956 张图像):**
- *** _Target_Spot: 1404 张**
- *** Spider_mites_Two_spotted_spider_mite: 1676 张**
- *** Septoria_leaf_spot: 1771 张**
- *** Bacterial_spot: 2127 张**
- *** _Target_Spot: 1404 张**
- *** Spider_mites_Two_spotted_spider_mite: 1676 张**
- *** Septoria_leaf_spot: 1771 张**
- *** Bacterial_spot: 2127 张**
- **✅ 数据相对平衡 (比例: 0.66)**
- **- Leaf 相关病害组 (11862 张图像):**
- *** _Tomato_YellowLeaf__Curl_Virus: 3208 张**
- *** Leaf_Mold: 952 张**
- *** Septoria_leaf_spot: 1771 张**
- *** _Tomato_YellowLeaf__Curl_Virus: 3208 张**
- *** Leaf_Mold: 952 张**
- *** Septoria_leaf_spot: 1771 张**
- **⚠️  数据不平衡 (比例: 0.30)**
- **- Bacterial 相关病害组 (4254 张图像):**
- *** Bacterial_spot: 2127 张**
- *** Bacterial_spot: 2127 张**
- **✅ 数据相对平衡 (比例: 1.00)**
- **- Virus 相关病害组 (7162 张图像):**
- *** _Tomato_YellowLeaf__Curl_Virus: 3208 张**
- *** _Tomato_mosaic_virus: 373 张**
- *** _Tomato_YellowLeaf__Curl_Virus: 3208 张**
- *** _Tomato_mosaic_virus: 373 张**
- **⚠️  数据不平衡 (比例: 0.12)**
- **- Mold 相关病害组 (1904 张图像):**
- *** Leaf_Mold: 952 张**
- *** Leaf_Mold: 952 张**
- **✅ 数据相对平衡 (比例: 1.00)**
## 2. Potato 细粒度识别子集:
- **- Blight 相关病害组 (4000 张图像):**
- *** Early blight: 1000 张**
- *** Early blight: 1000 张**
- *** Late blight: 1000 张**
- *** Late blight: 1000 张**
- **✅ 数据相对平衡 (比例: 1.00)**
## 3. Pepper  Bell 细粒度识别子集:
- **- Spot 相关病害组 (1994 张图像):**
- *** Bacterial spot: 997 张**
- *** Bacterial spot: 997 张**
- **✅ 数据相对平衡 (比例: 1.00)**
- **- Bacterial 相关病害组 (1994 张图像):**
- *** Bacterial spot: 997 张**
- *** Bacterial spot: 997 张**
- **✅ 数据相对平衡 (比例: 1.00)**
### 数据预处理建议
## 1. 图像尺寸标准化: 建议统一调整为 224x224 或 256x256
## 2. 数据增强: 使用旋转、翻转、亮度调整等技术增加数据多样性
## 3. 数据平衡: 对样本数量较少的类别进行过采样或数据增强
## 4. 质量检查: 移除或修复损坏的图像文件
### 模型训练建议
## 1. 数据划分: 建议使用 70% 训练、15% 验证、15% 测试的比例
## 2. 交叉验证: 使用分层抽样确保各类别在训练/验证/测试集中的比例一致
## 3. 注意力机制: 重点关注病斑区域，可考虑使用空间注意力和通道注意力
## 4. 损失函数: 对于不平衡数据，考虑使用加权交叉熵或Focal Loss
