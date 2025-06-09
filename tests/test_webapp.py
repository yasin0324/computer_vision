#!/usr/bin/env python3
"""
植物叶片病害识别系统Web应用测试脚本
"""

import os
import sys
import time
import requests
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_webapp_basic():
    """测试Web应用基本功能"""
    
    print("🧪 开始测试Web应用...")
    
    # 测试主页
    try:
        response = requests.get('http://localhost:5000', timeout=5)
        if response.status_code == 200:
            print("✅ 主页访问正常")
        else:
            print(f"❌ 主页访问失败: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ 无法连接到Web应用: {e}")
        print("💡 请确保Web应用正在运行 (python run_webapp.py)")
        return False
    
    # 测试API端点
    api_endpoints = [
        '/api/models',
        '/api/datasets', 
        '/api/dashboard_data'
    ]
    
    for endpoint in api_endpoints:
        try:
            response = requests.get(f'http://localhost:5000{endpoint}', timeout=5)
            if response.status_code == 200:
                print(f"✅ API {endpoint} 正常")
            else:
                print(f"⚠️ API {endpoint} 返回状态码: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ API {endpoint} 请求失败: {e}")
    
    # 测试页面
    pages = [
        '/predict',
        '/train', 
        '/evaluate',
        '/compare',
        '/dashboard'
    ]
    
    for page in pages:
        try:
            response = requests.get(f'http://localhost:5000{page}', timeout=5)
            if response.status_code == 200:
                print(f"✅ 页面 {page} 正常")
            else:
                print(f"⚠️ 页面 {page} 返回状态码: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ 页面 {page} 请求失败: {e}")
    
    print("🎉 Web应用测试完成!")
    return True

def check_dependencies():
    """检查依赖包"""
    
    print("📦 检查依赖包...")
    
    required_packages = [
        'flask',
        'torch',
        'torchvision', 
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'sklearn',
        'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 缺少依赖包: {', '.join(missing_packages)}")
        print("💡 请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def check_project_structure():
    """检查项目结构"""
    
    print("📁 检查项目结构...")
    
    required_dirs = [
        'src',
        'webapp',
        'webapp/templates',
        'data',
        'models',
        'outputs',
        'logs'
    ]
    
    required_files = [
        'webapp/app.py',
        'webapp/utils.py',
        'webapp/templates/base.html',
        'webapp/templates/index.html',
        'run_webapp.py'
    ]
    
    missing_items = []
    
    # 检查目录
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"❌ 目录缺失: {dir_path}")
            missing_items.append(dir_path)
        else:
            print(f"✅ {dir_path}")
    
    # 检查文件
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ 文件缺失: {file_path}")
            missing_items.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_items:
        print(f"\n⚠️ 缺少项目文件/目录: {len(missing_items)} 个")
        return False
    
    print("✅ 项目结构完整")
    return True

def main():
    """主函数"""
    
    print("=" * 60)
    print("🌱 植物叶片病害识别系统 - Web应用测试")
    print("=" * 60)
    
    # 检查项目结构
    if not check_project_structure():
        print("\n❌ 项目结构检查失败")
        return
    
    # 检查依赖包
    if not check_dependencies():
        print("\n❌ 依赖包检查失败")
        return
    
    # 测试Web应用
    print("\n" + "=" * 40)
    test_webapp_basic()
    
    print("\n" + "=" * 60)
    print("📋 测试总结:")
    print("1. 如果所有测试都通过，Web应用运行正常")
    print("2. 如果有测试失败，请检查相应的配置")
    print("3. 确保已启动Web应用: python run_webapp.py")
    print("4. 访问地址: http://localhost:5000")
    print("=" * 60)

if __name__ == '__main__':
    main() 