import os
import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from datetime import datetime
import sys

class ReportWriter:
    """报告输出类，支持控制台和文件输出"""
    
    def __init__(self, output_file=None, output_format='txt'):
        self.output_file = output_file
        self.output_format = output_format.lower()
        self.content = []
        
        if self.output_file:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(self.output_file) if os.path.dirname(self.output_file) else '.', exist_ok=True)
    
    def write(self, text="", end="\n"):
        """写入文本到控制台和缓存"""
        print(text, end=end)
        self.content.append(text + end)
    
    def save_to_file(self):
        """保存报告到文件"""
        if not self.output_file:
            return
            
        try:
            if self.output_format == 'html':
                self._save_as_html()
            elif self.output_format == 'md' or self.output_format == 'markdown':
                self._save_as_markdown()
            else:
                self._save_as_txt()
            print(f"\n✅ 报告已保存到: {self.output_file}")
        except Exception as e:
            print(f"❌ 保存报告失败: {str(e)}")
    
    def _save_as_txt(self):
        """保存为文本格式"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(f"PlantVillage数据集分析报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            f.writelines(self.content)
    
    def _save_as_html(self):
        """保存为HTML格式"""
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PlantVillage数据集分析报告</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .section {{ margin-bottom: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 8px; }}
        .stats {{ background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .warning {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 4px solid #ffc107; }}
        .success {{ background-color: #d4edda; padding: 10px; border-radius: 5px; border-left: 4px solid #28a745; }}
        pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        .timestamp {{ color: #6c757d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>PlantVillage数据集分析报告</h1>
    <p class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <div class="section">
        <pre>{''.join(self.content)}</pre>
    </div>
</body>
</html>
"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _save_as_markdown(self):
        """保存为Markdown格式"""
        # 处理内容，转换为Markdown格式
        markdown_content = self._convert_to_markdown()
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(f"# PlantVillage数据集分析报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            f.write(markdown_content)
    
    def _convert_to_markdown(self):
        """将内容转换为Markdown格式"""
        markdown_lines = []
        content_text = ''.join(self.content)
        lines = content_text.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            
            # 跳过空行和标题行
            if not line or line.startswith('==='):
                i += 1
                continue
            
            # 处理主要章节标题 (数字开头)
            if line and line[0].isdigit() and '. ' in line:
                markdown_lines.append(f"## {line}\n")
            
            # 处理分隔线
            elif line.startswith('-' * 10):
                markdown_lines.append("")  # 用空行替代分隔线
            
            # 处理作物标题 (以大写字母开头，以冒号结尾)
            elif line.endswith(':') and not line.startswith(' '):
                markdown_lines.append(f"### {line[:-1]}\n")
            
            # 处理细粒度识别候选标题
            elif '细粒度识别候选:' in line:
                markdown_lines.append(f"### {line}\n")
            
            # 处理统计信息 (包含数字和冒号)
            elif ':' in line and any(char.isdigit() for char in line):
                if line.startswith('  '):
                    # 二级统计信息
                    markdown_lines.append(f"- **{line.strip()}**\n")
                else:
                    # 一级统计信息
                    markdown_lines.append(f"**{line.strip()}**\n")
            
            # 处理列表项
            elif line.startswith('    - ') or line.startswith('     * '):
                # 病害列表项
                item = line.strip()[2:]  # 移除 "- " 或 "* "
                markdown_lines.append(f"  - {item}\n")
            
            elif line.startswith('   - '):
                # 病害组标题
                item = line.strip()[2:]
                markdown_lines.append(f"- **{item}**\n")
            
            # 处理建议列表
            elif line.strip() and line[0].isdigit() and '. ' in line and ('建议' in line or '划分' in line or '验证' in line or '机制' in line or '函数' in line):
                markdown_lines.append(f"- {line.strip()}\n")
            
            # 处理警告和成功标记
            elif '⚠️' in line or '✅' in line:
                if '⚠️' in line:
                    markdown_lines.append(f"> ⚠️ **警告**: {line.strip()}\n")
                else:
                    markdown_lines.append(f"> ✅ **成功**: {line.strip()}\n")
            
            # 处理错误信息
            elif '❌' in line:
                markdown_lines.append(f"> ❌ **错误**: {line.strip()}\n")
            
            # 处理代码块或特殊格式
            elif line.startswith('损坏的图像:'):
                markdown_lines.append(f"```\n{line}\n```\n")
            
            # 处理普通文本
            else:
                if line.strip():
                    markdown_lines.append(f"{line}\n")
                else:
                    markdown_lines.append("\n")
            
            i += 1
        
        return ''.join(markdown_lines)

def analyze_plantvillage_dataset(data_path, writer):
    """
    全面分析PlantVillage数据集的结构和特征
    """
    writer.write("=== PlantVillage数据集分析报告 ===\n")
    
    # 1. 统计类别和图像数量
    writer.write("1. 数据集结构分析")
    writer.write("-" * 50)
    
    dataset_stats = {}
    crop_disease_mapping = defaultdict(list)
    total_images = 0
    
    # 遍历数据集目录
    for root, dirs, files in os.walk(data_path):
        if files and any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
            class_name = os.path.basename(root)
            if class_name == 'PlantVillage':
                continue
                
            # 统计图像数量
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            num_images = len(image_files)
            
            if num_images > 0:
                dataset_stats[class_name] = num_images
                total_images += num_images
                
                # 解析作物和病害信息
                if '___' in class_name:
                    parts = class_name.split('___')
                    crop = parts[0].replace('_', ' ')
                    disease = parts[1].replace('_', ' ') if len(parts) > 1 else 'healthy'
                else:
                    # 处理其他命名格式
                    if 'healthy' in class_name.lower():
                        crop = class_name.replace('_healthy', '').replace('_', ' ')
                        disease = 'healthy'
                    else:
                        # 尝试从类名中提取作物和病害
                        parts = class_name.split('_')
                        crop = parts[0]
                        disease = '_'.join(parts[1:]) if len(parts) > 1 else 'unknown'
                
                crop_disease_mapping[crop].append((disease, num_images))
    
    # 显示统计结果
    writer.write(f"总类别数: {len(dataset_stats)}")
    writer.write(f"总图像数: {total_images}")
    writer.write(f"平均每类图像数: {total_images / len(dataset_stats):.1f}")
    writer.write()
    
    # 按作物分组显示
    writer.write("2. 按作物分类统计")
    writer.write("-" * 50)
    for crop, diseases in crop_disease_mapping.items():
        writer.write(f"\n{crop.title()}:")
        crop_total = sum(count for _, count in diseases)
        writer.write(f"  总图像数: {crop_total}")
        writer.write(f"  病害类别数: {len(diseases)}")
        for disease, count in sorted(diseases, key=lambda x: x[1], reverse=True):
            writer.write(f"    - {disease}: {count} 张")
    
    # 3. 识别细粒度识别候选子集
    writer.write("\n3. 细粒度识别候选子集")
    writer.write("-" * 50)
    
    fine_grained_candidates = []
    
    for crop, diseases in crop_disease_mapping.items():
        if len(diseases) >= 3:  # 至少有3种不同状态（包括健康）
            disease_names = [d[0] for d in diseases]
            
            # 查找相似病害（包含相似关键词）
            similar_groups = []
            keywords = ['blight', 'spot', 'leaf', 'bacterial', 'virus', 'mold']
            
            for keyword in keywords:
                similar_diseases = [(name, count) for name, count in diseases 
                                if keyword.lower() in name.lower()]
                if len(similar_diseases) >= 2:
                    similar_groups.append((keyword, similar_diseases))
            
            if similar_groups:
                fine_grained_candidates.append((crop, similar_groups))
                writer.write(f"\n{crop.title()} - 细粒度识别候选:")
                for keyword, group in similar_groups:
                    writer.write(f"  {keyword.title()} 相关病害:")
                    for disease, count in group:
                        writer.write(f"    - {disease}: {count} 张")
    
    return dataset_stats, crop_disease_mapping, fine_grained_candidates

def check_image_quality(data_path, writer, sample_size=50):
    """
    检查图像质量和格式
    """
    writer.write("\n4. 图像质量分析")
    writer.write("-" * 50)
    
    image_info = []
    corrupted_images = []
    
    # 收集样本图像
    sample_images = []
    for root, dirs, files in os.walk(data_path):
        if files:
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                class_name = os.path.basename(root)
                if class_name != 'PlantVillage':
                    # 从每个类别中随机选择几张图像
                    import random
                    selected = random.sample(image_files, min(3, len(image_files)))
                    for img_file in selected:
                        sample_images.append(os.path.join(root, img_file))
                        if len(sample_images) >= sample_size:
                            break
        if len(sample_images) >= sample_size:
            break
    
    # 分析样本图像
    for img_path in sample_images:
        try:
            # 使用PIL检查图像
            with Image.open(img_path) as img:
                width, height = img.size
                format_type = img.format
                mode = img.mode
                
                # 使用OpenCV检查图像质量
                cv_img = cv2.imread(img_path)
                if cv_img is not None:
                    # 计算图像质量指标
                    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()  # 清晰度
                    
                    image_info.append({
                        'path': img_path,
                        'width': width,
                        'height': height,
                        'format': format_type,
                        'mode': mode,
                        'sharpness': laplacian_var,
                        'file_size': os.path.getsize(img_path)
                    })
                else:
                    corrupted_images.append(img_path)
                    
        except Exception as e:
            corrupted_images.append(img_path)
            writer.write(f"损坏的图像: {img_path} - {str(e)}")
    
    # 统计分析
    if image_info:
        df = pd.DataFrame(image_info)
        
        writer.write(f"分析样本数: {len(image_info)}")
        writer.write(f"损坏图像数: {len(corrupted_images)}")
        writer.write()
        
        writer.write("图像尺寸统计:")
        writer.write(f"  宽度: {df['width'].min()} - {df['width'].max()} (平均: {df['width'].mean():.1f})")
        writer.write(f"  高度: {df['height'].min()} - {df['height'].max()} (平均: {df['height'].mean():.1f})")
        writer.write()
        
        writer.write("图像格式分布:")
        format_counts = df['format'].value_counts()
        for fmt, count in format_counts.items():
            writer.write(f"  {fmt}: {count} 张 ({count/len(df)*100:.1f}%)")
        writer.write()
        
        writer.write("图像质量统计:")
        writer.write(f"  清晰度 (Laplacian方差): {df['sharpness'].min():.1f} - {df['sharpness'].max():.1f}")
        writer.write(f"  平均清晰度: {df['sharpness'].mean():.1f}")
        writer.write(f"  文件大小: {df['file_size'].min()/1024:.1f}KB - {df['file_size'].max()/1024:.1f}KB")
        
        # 识别可能有问题的图像
        low_quality_threshold = df['sharpness'].quantile(0.1)
        low_quality_images = df[df['sharpness'] < low_quality_threshold]
        if len(low_quality_images) > 0:
            writer.write(f"\n可能的低质量图像 (清晰度 < {low_quality_threshold:.1f}):")
            for _, img in low_quality_images.iterrows():
                writer.write(f"  {os.path.basename(img['path'])}: 清晰度 {img['sharpness']:.1f}")
    
    return image_info, corrupted_images

def generate_recommendations(crop_disease_mapping, fine_grained_candidates, writer):
    """
    生成数据集使用建议
    """
    writer.write("\n5. 数据集使用建议")
    writer.write("-" * 50)
    
    writer.write("基于分析结果，以下是针对细粒度识别研究的建议：")
    writer.write()
    
    # 推荐的细粒度识别子集
    writer.write("推荐的细粒度识别子集:")
    for i, (crop, similar_groups) in enumerate(fine_grained_candidates, 1):
        writer.write(f"\n{i}. {crop.title()} 细粒度识别子集:")
        
        for keyword, group in similar_groups:
            if len(group) >= 2:
                total_images = sum(count for _, count in group)
                writer.write(f"   - {keyword.title()} 相关病害组 ({total_images} 张图像):")
                for disease, count in group:
                    writer.write(f"     * {disease}: {count} 张")
                
                # 评估数据平衡性
                counts = [count for _, count in group]
                balance_ratio = min(counts) / max(counts)
                if balance_ratio < 0.3:
                    writer.write(f"     ⚠️  数据不平衡 (比例: {balance_ratio:.2f})")
                else:
                    writer.write(f"     ✅ 数据相对平衡 (比例: {balance_ratio:.2f})")
    
    writer.write("\n数据预处理建议:")
    writer.write("1. 图像尺寸标准化: 建议统一调整为 224x224 或 256x256")
    writer.write("2. 数据增强: 使用旋转、翻转、亮度调整等技术增加数据多样性")
    writer.write("3. 数据平衡: 对样本数量较少的类别进行过采样或数据增强")
    writer.write("4. 质量检查: 移除或修复损坏的图像文件")
    
    writer.write("\n模型训练建议:")
    writer.write("1. 数据划分: 建议使用 70% 训练、15% 验证、15% 测试的比例")
    writer.write("2. 交叉验证: 使用分层抽样确保各类别在训练/验证/测试集中的比例一致")
    writer.write("3. 注意力机制: 重点关注病斑区域，可考虑使用空间注意力和通道注意力")
    writer.write("4. 损失函数: 对于不平衡数据，考虑使用加权交叉熵或Focal Loss")

if __name__ == "__main__":
    # 设置数据路径
    data_path = "data/raw/PlantVillage"
    
    # 配置输出选项
    output_file = None
    output_format = 'txt'
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print("使用方法:")
            print("  python dataset_analysis.py                      # 仅控制台输出")
            print("  python dataset_analysis.py report.txt           # 输出到文本文件")
            print("  python dataset_analysis.py report.html          # 输出到HTML文件")
            print("  python dataset_analysis.py report.md            # 输出到Markdown文件")
            print("  python dataset_analysis.py report.txt txt       # 指定格式为txt")
            print("  python dataset_analysis.py report.html html     # 指定格式为html")
            print("  python dataset_analysis.py report.md markdown   # 指定格式为markdown")
            sys.exit(0)
        
        output_file = sys.argv[1]
        
        # 从文件扩展名推断格式
        if output_file.lower().endswith('.html'):
            output_format = 'html'
        elif output_file.lower().endswith('.txt'):
            output_format = 'txt'
        elif output_file.lower().endswith('.md'):
            output_format = 'md'
        
        # 如果提供了第二个参数，使用它作为格式
        if len(sys.argv) > 2:
            output_format = sys.argv[2].lower()
    
    # 创建报告写入器
    writer = ReportWriter(output_file, output_format)
    
    try:
        # 执行分析
        dataset_stats, crop_disease_mapping, fine_grained_candidates = analyze_plantvillage_dataset(data_path, writer)
        
        # 检查图像质量
        image_info, corrupted_images = check_image_quality(data_path, writer)
        
        # 生成建议
        generate_recommendations(crop_disease_mapping, fine_grained_candidates, writer)
        
        writer.write("\n=== 分析完成 ===")
        
        # 保存到文件
        if output_file:
            writer.save_to_file()
            
    except Exception as e:
        writer.write(f"\n❌ 分析过程中出现错误: {str(e)}")
        if output_file:
            writer.save_to_file() 