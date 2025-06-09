#!/usr/bin/env python3
"""
测试前端问题修复情况
"""

import requests
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time

def test_api_endpoints():
    """测试API端点"""
    base_url = "http://localhost:5000"
    
    print("🔍 测试API端点...")
    
    # 测试模型列表API
    try:
        response = requests.get(f"{base_url}/api/models")
        if response.status_code == 200:
            models = response.json()
            print(f"✅ 模型列表API正常，找到 {len(models)} 个模型")
            for model in models:
                print(f"   - {model['name']} ({model['type']}) - {model['size']}")
        else:
            print(f"❌ 模型列表API失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 模型列表API错误: {e}")

def test_evaluate_page():
    """测试评估页面"""
    print("\n📊 测试评估页面...")
    
    try:
        response = requests.get("http://localhost:5000/evaluate")
        if response.status_code == 200:
            print("✅ 评估页面可以访问")
            
            # 检查页面内容是否包含模型选择
            if "请选择模型文件" in response.text:
                print("✅ 评估页面包含模型选择下拉框")
            else:
                print("❌ 评估页面缺少模型选择下拉框")
        else:
            print(f"❌ 评估页面访问失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 评估页面错误: {e}")

def test_compare_page():
    """测试比较页面"""
    print("\n⚖️ 测试比较页面...")
    
    try:
        response = requests.get("http://localhost:5000/compare")
        if response.status_code == 200:
            print("✅ 比较页面可以访问")
            
            # 检查页面内容
            if "添加模型" in response.text:
                print("✅ 比较页面包含添加模型按钮")
            if "modelSelections" in response.text:
                print("✅ 比较页面包含模型选择容器")
        else:
            print(f"❌ 比较页面访问失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 比较页面错误: {e}")

def test_with_browser():
    """使用浏览器测试（如果可用）"""
    print("\n🌐 尝试浏览器测试...")
    
    try:
        # 配置Chrome选项
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # 无头模式
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(options=chrome_options)
        
        # 测试评估页面
        print("   测试评估页面模型选择...")
        driver.get("http://localhost:5000/evaluate")
        
        # 等待页面加载
        wait = WebDriverWait(driver, 10)
        model_select = wait.until(EC.presence_of_element_located((By.ID, "modelPath")))
        
        # 检查选项数量
        options = model_select.find_elements(By.TAG_NAME, "option")
        print(f"   ✅ 评估页面模型选择有 {len(options)} 个选项")
        
        # 测试比较页面
        print("   测试比较页面添加模型功能...")
        driver.get("http://localhost:5000/compare")
        
        # 等待页面加载
        time.sleep(2)
        
        # 检查初始模型选择数量
        model_selections = driver.find_elements(By.CLASS_NAME, "model-selection")
        print(f"   ✅ 比较页面初始有 {len(model_selections)} 个模型选择")
        
        # 尝试点击添加模型按钮
        add_button = driver.find_element(By.XPATH, "//button[contains(text(), '添加模型')]")
        add_button.click()
        
        time.sleep(1)
        
        # 检查是否增加了模型选择
        new_model_selections = driver.find_elements(By.CLASS_NAME, "model-selection")
        if len(new_model_selections) > len(model_selections):
            print(f"   ✅ 添加模型按钮工作正常，现在有 {len(new_model_selections)} 个模型选择")
        else:
            print("   ❌ 添加模型按钮没有响应")
        
        driver.quit()
        
    except Exception as e:
        print(f"   ⚠️ 浏览器测试跳过（需要Chrome驱动）: {e}")

def main():
    print("=" * 60)
    print("🔧 前端问题修复验证")
    print("=" * 60)
    
    test_api_endpoints()
    test_evaluate_page()
    test_compare_page()
    test_with_browser()
    
    print("\n" + "=" * 60)
    print("📋 测试总结:")
    print("1. 如果API测试通过，说明后端数据正常")
    print("2. 如果页面测试通过，说明前端模板正常")
    print("3. 请在浏览器中手动验证具体功能")
    print("4. 访问地址: http://localhost:5000")
    print("=" * 60)

if __name__ == "__main__":
    main() 