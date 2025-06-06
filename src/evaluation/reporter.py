#!/usr/bin/env python3
"""
评估报告生成器模块

提供详细的评估报告生成功能，包括HTML、PDF和Markdown格式
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import base64
from io import BytesIO


class EvaluationReporter:
    """评估报告生成器"""
    
    def __init__(self, class_names: List[str]):
        """
        初始化报告生成器
        
        Args:
            class_names: 类别名称列表
        """
        self.class_names = class_names
    
    def generate_html_report(self, evaluation_results: Dict[str, Any], 
                           output_file: str, include_charts: bool = True) -> str:
        """生成HTML格式的评估报告"""
        
        # 生成HTML内容
        html_content = self._create_html_template()
        
        # 填充报告内容
        html_content = html_content.replace('{{TITLE}}', f"模型评估报告 - {evaluation_results.get('model_name', 'Unknown')}")
        html_content = html_content.replace('{{GENERATION_TIME}}', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # 添加摘要信息
        summary_html = self._generate_summary_html(evaluation_results)
        html_content = html_content.replace('{{SUMMARY}}', summary_html)
        
        # 添加详细指标
        metrics_html = self._generate_metrics_html(evaluation_results)
        html_content = html_content.replace('{{METRICS}}', metrics_html)
        
        # 添加每个类别的详细信息
        per_class_html = self._generate_per_class_html(evaluation_results)
        html_content = html_content.replace('{{PER_CLASS}}', per_class_html)
        
        # 添加错误分析
        if 'error_analysis' in evaluation_results:
            error_html = self._generate_error_analysis_html(evaluation_results['error_analysis'])
            html_content = html_content.replace('{{ERROR_ANALYSIS}}', error_html)
        else:
            html_content = html_content.replace('{{ERROR_ANALYSIS}}', '<p>错误分析数据不可用</p>')
        
        # 添加图表
        if include_charts:
            charts_html = self._generate_charts_html(evaluation_results)
            html_content = html_content.replace('{{CHARTS}}', charts_html)
        else:
            html_content = html_content.replace('{{CHARTS}}', '')
        
        # 保存HTML文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_file
    
    def generate_markdown_report(self, evaluation_results: Dict[str, Any], 
                               output_file: str) -> str:
        """生成Markdown格式的评估报告"""
        
        md_content = []
        
        # 标题
        model_name = evaluation_results.get('model_name', 'Unknown')
        md_content.append(f"# 模型评估报告 - {model_name}")
        md_content.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_content.append("\n---\n")
        
        # 摘要
        md_content.append("## 评估摘要")
        md_content.append(f"- **模型路径**: {evaluation_results.get('model_path', 'N/A')}")
        md_content.append(f"- **测试准确率**: {evaluation_results.get('test_accuracy', 0):.4f} ({evaluation_results.get('test_accuracy', 0)*100:.2f}%)")
        md_content.append(f"- **总样本数**: {evaluation_results.get('total_samples', 0)}")
        md_content.append(f"- **正确预测**: {evaluation_results.get('correct_predictions', 0)}")
        md_content.append(f"- **错误预测**: {evaluation_results.get('incorrect_predictions', 0)}")
        md_content.append("")
        
        # 整体指标
        md_content.append("## 整体性能指标")
        md_content.append("| 指标 | 数值 |")
        md_content.append("|------|------|")
        md_content.append(f"| 准确率 | {evaluation_results.get('test_accuracy', 0):.4f} |")
        md_content.append(f"| 宏平均精确率 | {evaluation_results.get('macro_precision', 0):.4f} |")
        md_content.append(f"| 宏平均召回率 | {evaluation_results.get('macro_recall', 0):.4f} |")
        md_content.append(f"| 宏平均F1分数 | {evaluation_results.get('macro_f1', 0):.4f} |")
        md_content.append(f"| 微平均精确率 | {evaluation_results.get('micro_precision', 0):.4f} |")
        md_content.append(f"| 微平均召回率 | {evaluation_results.get('micro_recall', 0):.4f} |")
        md_content.append(f"| 微平均F1分数 | {evaluation_results.get('micro_f1', 0):.4f} |")
        md_content.append("")
        
        # 每个类别的详细指标
        md_content.append("## 各类别详细指标")
        if 'per_class_metrics' in evaluation_results:
            md_content.append("| 类别 | 精确率 | 召回率 | F1分数 | 支持样本数 |")
            md_content.append("|------|--------|--------|--------|-----------|")
            
            for class_name, metrics in evaluation_results['per_class_metrics'].items():
                md_content.append(f"| {class_name} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f} | {metrics['support']} |")
        md_content.append("")
        
        # 错误分析
        if 'error_analysis' in evaluation_results:
            error_analysis = evaluation_results['error_analysis']
            md_content.append("## 错误分析")
            md_content.append(f"- **总错误数**: {error_analysis.get('total_errors', 0)}")
            md_content.append(f"- **错误率**: {error_analysis.get('error_rate', 0):.4f}")
            
            # 各类别错误率
            if 'error_by_class' in error_analysis:
                md_content.append("\n### 各类别错误率")
                md_content.append("| 类别 | 错误数 | 总数 | 错误率 |")
                md_content.append("|------|--------|------|--------|")
                
                for class_name, error_info in error_analysis['error_by_class'].items():
                    md_content.append(f"| {class_name} | {error_info['errors']} | {error_info['total']} | {error_info['error_rate']:.4f} |")
            
            # 最容易混淆的类别对
            if 'most_confused_pairs' in error_analysis and error_analysis['most_confused_pairs']:
                md_content.append("\n### 最容易混淆的类别对")
                md_content.append("| 真实类别 | 预测类别 | 混淆次数 |")
                md_content.append("|----------|----------|----------|")
                
                for (true_class, pred_class), count in error_analysis['most_confused_pairs'][:5]:
                    md_content.append(f"| {true_class} | {pred_class} | {count} |")
            
            md_content.append("")
        
        # 置信度分析
        if 'confidence_analysis' in evaluation_results:
            conf_analysis = evaluation_results['confidence_analysis']
            md_content.append("## 置信度分析")
            
            if 'overall_stats' in conf_analysis:
                stats = conf_analysis['overall_stats']
                md_content.append("### 整体置信度统计")
                md_content.append("| 统计量 | 数值 |")
                md_content.append("|--------|------|")
                md_content.append(f"| 平均值 | {stats['mean']:.4f} |")
                md_content.append(f"| 标准差 | {stats['std']:.4f} |")
                md_content.append(f"| 最小值 | {stats['min']:.4f} |")
                md_content.append(f"| 最大值 | {stats['max']:.4f} |")
                md_content.append(f"| 中位数 | {stats['median']:.4f} |")
            
            md_content.append("")
        
        # 保存Markdown文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_content))
        
        return output_file
    
    def generate_comparison_report(self, comparison_results: Dict[str, Any], 
                                 output_file: str, format_type: str = 'html') -> str:
        """生成模型比较报告"""
        
        if format_type.lower() == 'html':
            return self._generate_comparison_html_report(comparison_results, output_file)
        elif format_type.lower() == 'markdown':
            return self._generate_comparison_markdown_report(comparison_results, output_file)
        else:
            raise ValueError("format_type must be 'html' or 'markdown'")
    
    def _generate_comparison_html_report(self, comparison_results: Dict[str, Any], 
                                       output_file: str) -> str:
        """生成HTML格式的比较报告"""
        
        html_content = self._create_comparison_html_template()
        
        # 填充基本信息
        html_content = html_content.replace('{{TITLE}}', "模型性能比较报告")
        html_content = html_content.replace('{{GENERATION_TIME}}', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # 添加摘要
        summary = comparison_results.get('summary', {})
        summary_html = f"""
        <div class="summary-card">
            <h3>比较摘要</h3>
            <p><strong>比较模型数量:</strong> {summary.get('total_models', 0)}</p>
            <p><strong>参与比较的模型:</strong> {', '.join(summary.get('model_names', []))}</p>
            <p><strong>比较时间:</strong> {summary.get('comparison_date', 'N/A')}</p>
        </div>
        """
        html_content = html_content.replace('{{SUMMARY}}', summary_html)
        
        # 添加整体性能比较表格
        overall_perf = comparison_results.get('overall_performance', [])
        if overall_perf:
            table_html = self._create_performance_table(overall_perf)
            html_content = html_content.replace('{{PERFORMANCE_TABLE}}', table_html)
        else:
            html_content = html_content.replace('{{PERFORMANCE_TABLE}}', '<p>性能数据不可用</p>')
        
        # 添加排名信息
        rankings = comparison_results.get('rankings', {})
        if rankings:
            ranking_html = self._create_ranking_html(rankings)
            html_content = html_content.replace('{{RANKINGS}}', ranking_html)
        else:
            html_content = html_content.replace('{{RANKINGS}}', '<p>排名数据不可用</p>')
        
        # 保存HTML文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_file
    
    def _generate_comparison_markdown_report(self, comparison_results: Dict[str, Any], 
                                           output_file: str) -> str:
        """生成Markdown格式的比较报告"""
        
        md_content = []
        
        # 标题
        md_content.append("# 模型性能比较报告")
        md_content.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_content.append("\n---\n")
        
        # 摘要
        summary = comparison_results.get('summary', {})
        md_content.append("## 比较摘要")
        md_content.append(f"- **比较模型数量**: {summary.get('total_models', 0)}")
        md_content.append(f"- **参与比较的模型**: {', '.join(summary.get('model_names', []))}")
        md_content.append(f"- **比较时间**: {summary.get('comparison_date', 'N/A')}")
        md_content.append("")
        
        # 整体性能比较
        overall_perf = comparison_results.get('overall_performance', [])
        if overall_perf:
            md_content.append("## 整体性能比较")
            
            # 创建表格头
            headers = list(overall_perf[0].keys())
            md_content.append("| " + " | ".join(headers) + " |")
            md_content.append("| " + " | ".join(["---"] * len(headers)) + " |")
            
            # 添加数据行
            for row in overall_perf:
                values = []
                for header in headers:
                    value = row[header]
                    if isinstance(value, float):
                        values.append(f"{value:.4f}")
                    else:
                        values.append(str(value))
                md_content.append("| " + " | ".join(values) + " |")
            
            md_content.append("")
        
        # 排名信息
        rankings = comparison_results.get('rankings', {})
        if rankings:
            md_content.append("## 性能排名")
            for metric, ranking in rankings.items():
                md_content.append(f"### {metric.replace('_', ' ').title()}")
                for i, model_name in enumerate(ranking, 1):
                    md_content.append(f"{i}. {model_name}")
                md_content.append("")
        
        # 性能分析
        analysis = comparison_results.get('performance_analysis', {})
        if analysis:
            md_content.append("## 性能分析")
            
            if 'best_model' in analysis:
                best = analysis['best_model']
                md_content.append(f"- **最佳模型**: {best['name']} (准确率: {best['accuracy']:.4f})")
            
            if 'worst_model' in analysis:
                worst = analysis['worst_model']
                md_content.append(f"- **最差模型**: {worst['name']} (准确率: {worst['accuracy']:.4f})")
            
            if 'performance_improvement' in analysis:
                improvement = analysis['performance_improvement']
                md_content.append(f"- **准确率提升**: {improvement['accuracy']:.4f}")
                md_content.append(f"- **宏F1提升**: {improvement['macro_f1']:.4f}")
            
            md_content.append("")
        
        # 保存Markdown文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_content))
        
        return output_file
    
    def _create_html_template(self) -> str:
        """创建HTML报告模板"""
        return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }
        .summary-card {
            background-color: #e8f5e8;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 5px solid #4CAF50;
        }
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .metrics-table th, .metrics-table td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        .metrics-table th {
            background-color: #4CAF50;
            color: white;
        }
        .metrics-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .chart-container {
            margin: 30px 0;
            text-align: center;
        }
        .error-analysis {
            background-color: #fff3cd;
            padding: 20px;
            border-radius: 8px;
            border-left: 5px solid #ffc107;
            margin: 20px 0;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{TITLE}}</h1>
        <p><strong>生成时间:</strong> {{GENERATION_TIME}}</p>
        
        <h2>比较摘要</h2>
        {{SUMMARY}}
        
        <h2>整体性能比较</h2>
        {{PERFORMANCE_TABLE}}
        
        <h2>性能排名</h2>
        {{RANKINGS}}
        
        <div class="footer">
            <p>此报告由植物叶片病害识别系统自动生成</p>
        </div>
    </div>
</body>
</html>
        """
    
    def _create_comparison_html_template(self) -> str:
        """创建比较报告HTML模板"""
        return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #333;
            border-bottom: 2px solid #2196F3;
            padding-bottom: 10px;
        }
        .summary-card {
            background-color: #e3f2fd;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 5px solid #2196F3;
        }
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .comparison-table th, .comparison-table td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: center;
        }
        .comparison-table th {
            background-color: #2196F3;
            color: white;
        }
        .comparison-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .ranking-section {
            background-color: #fff8e1;
            padding: 20px;
            border-radius: 8px;
            border-left: 5px solid #ff9800;
            margin: 20px 0;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{TITLE}}</h1>
        <p><strong>生成时间:</strong> {{GENERATION_TIME}}</p>
        
        <h2>比较摘要</h2>
        {{SUMMARY}}
        
        <h2>整体性能比较</h2>
        {{PERFORMANCE_TABLE}}
        
        <h2>性能排名</h2>
        {{RANKINGS}}
        
        <div class="footer">
            <p>此报告由植物叶片病害识别系统自动生成</p>
        </div>
    </div>
</body>
</html>
        """
    
    def _generate_summary_html(self, results: Dict[str, Any]) -> str:
        """生成摘要HTML"""
        return f"""
        <div class="summary-card">
            <h3>评估摘要</h3>
            <p><strong>模型路径:</strong> {results.get('model_path', 'N/A')}</p>
            <p><strong>测试准确率:</strong> {results.get('test_accuracy', 0):.4f} ({results.get('test_accuracy', 0)*100:.2f}%)</p>
            <p><strong>总样本数:</strong> {results.get('total_samples', 0)}</p>
            <p><strong>正确预测:</strong> {results.get('correct_predictions', 0)}</p>
            <p><strong>错误预测:</strong> {results.get('incorrect_predictions', 0)}</p>
        </div>
        """
    
    def _generate_metrics_html(self, results: Dict[str, Any]) -> str:
        """生成指标HTML表格"""
        metrics_data = [
            ('准确率', results.get('test_accuracy', 0)),
            ('宏平均精确率', results.get('macro_precision', 0)),
            ('宏平均召回率', results.get('macro_recall', 0)),
            ('宏平均F1分数', results.get('macro_f1', 0)),
            ('微平均精确率', results.get('micro_precision', 0)),
            ('微平均召回率', results.get('micro_recall', 0)),
            ('微平均F1分数', results.get('micro_f1', 0))
        ]
        
        table_html = '<table class="metrics-table"><thead><tr><th>指标</th><th>数值</th></tr></thead><tbody>'
        
        for metric_name, value in metrics_data:
            table_html += f'<tr><td>{metric_name}</td><td>{value:.4f}</td></tr>'
        
        table_html += '</tbody></table>'
        return table_html
    
    def _generate_per_class_html(self, results: Dict[str, Any]) -> str:
        """生成每个类别的详细指标HTML"""
        if 'per_class_metrics' not in results:
            return '<p>每个类别的详细指标不可用</p>'
        
        table_html = '''
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>类别</th>
                    <th>精确率</th>
                    <th>召回率</th>
                    <th>F1分数</th>
                    <th>支持样本数</th>
                </tr>
            </thead>
            <tbody>
        '''
        
        for class_name, metrics in results['per_class_metrics'].items():
            table_html += f'''
                <tr>
                    <td>{class_name}</td>
                    <td>{metrics['precision']:.4f}</td>
                    <td>{metrics['recall']:.4f}</td>
                    <td>{metrics['f1_score']:.4f}</td>
                    <td>{metrics['support']}</td>
                </tr>
            '''
        
        table_html += '</tbody></table>'
        return table_html
    
    def _generate_error_analysis_html(self, error_analysis: Dict[str, Any]) -> str:
        """生成错误分析HTML"""
        html = f'''
        <div class="error-analysis">
            <h3>错误统计</h3>
            <p><strong>总错误数:</strong> {error_analysis.get('total_errors', 0)}</p>
            <p><strong>错误率:</strong> {error_analysis.get('error_rate', 0):.4f}</p>
        '''
        
        # 各类别错误率
        if 'error_by_class' in error_analysis:
            html += '<h4>各类别错误率</h4>'
            html += '<table class="metrics-table"><thead><tr><th>类别</th><th>错误数</th><th>总数</th><th>错误率</th></tr></thead><tbody>'
            
            for class_name, error_info in error_analysis['error_by_class'].items():
                html += f'''
                    <tr>
                        <td>{class_name}</td>
                        <td>{error_info['errors']}</td>
                        <td>{error_info['total']}</td>
                        <td>{error_info['error_rate']:.4f}</td>
                    </tr>
                '''
            
            html += '</tbody></table>'
        
        # 最容易混淆的类别对
        if 'most_confused_pairs' in error_analysis and error_analysis['most_confused_pairs']:
            html += '<h4>最容易混淆的类别对</h4>'
            html += '<table class="metrics-table"><thead><tr><th>真实类别</th><th>预测类别</th><th>混淆次数</th></tr></thead><tbody>'
            
            for (true_class, pred_class), count in error_analysis['most_confused_pairs'][:5]:
                html += f'''
                    <tr>
                        <td>{true_class}</td>
                        <td>{pred_class}</td>
                        <td>{count}</td>
                    </tr>
                '''
            
            html += '</tbody></table>'
        
        html += '</div>'
        return html
    
    def _generate_charts_html(self, results: Dict[str, Any]) -> str:
        """生成图表HTML（占位符）"""
        return '''
        <div class="chart-container">
            <p>图表将在此处显示（需要实际的图表文件）</p>
            <p>可以包含混淆矩阵、ROC曲线、PR曲线等可视化内容</p>
        </div>
        '''
    
    def _create_performance_table(self, performance_data: List[Dict]) -> str:
        """创建性能比较表格HTML"""
        if not performance_data:
            return '<p>性能数据不可用</p>'
        
        headers = list(performance_data[0].keys())
        
        table_html = '<table class="comparison-table"><thead><tr>'
        for header in headers:
            table_html += f'<th>{header}</th>'
        table_html += '</tr></thead><tbody>'
        
        for row in performance_data:
            table_html += '<tr>'
            for header in headers:
                value = row[header]
                if isinstance(value, float):
                    table_html += f'<td>{value:.4f}</td>'
                else:
                    table_html += f'<td>{value}</td>'
            table_html += '</tr>'
        
        table_html += '</tbody></table>'
        return table_html
    
    def _create_ranking_html(self, rankings: Dict[str, List[str]]) -> str:
        """创建排名HTML"""
        html = '<div class="ranking-section">'
        
        for metric, ranking in rankings.items():
            html += f'<h4>{metric.replace("_", " ").title()}</h4>'
            html += '<ol>'
            for model_name in ranking:
                html += f'<li>{model_name}</li>'
            html += '</ol>'
        
        html += '</div>'
        return html 