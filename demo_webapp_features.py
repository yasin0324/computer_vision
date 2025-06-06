#!/usr/bin/env python3
"""
植物叶片病害识别系统 - Web应用功能演示
"""

import requests
import json
import time

def test_web_app():
    """测试Web应用的主要功能"""
    base_url = "http://localhost:5000"
    
    print("=" * 60)
    print("🌱 植物叶片病害识别系统 - 功能演示")
    print("=" * 60)
    
    # 1. 测试主页
    print("\n1. 📄 测试主页访问...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("   ✅ 主页访问成功")
        else:
            print(f"   ❌ 主页访问失败: {response.status_code}")
    except Exception as e:
        print(f"   ❌ 主页访问错误: {e}")
    
    # 2. 测试模型列表API
    print("\n2. 📋 测试模型列表API...")
    try:
        response = requests.get(f"{base_url}/api/models")
        if response.status_code == 200:
            models = response.json()
            print(f"   ✅ 找到 {len(models)} 个模型")
            for model in models[:3]:  # 显示前3个
                print(f"      - {model['name']} ({model['type']}) - {model['size']}")
        else:
            print(f"   ❌ 模型列表获取失败: {response.status_code}")
    except Exception as e:
        print(f"   ❌ 模型列表API错误: {e}")
    
    # 3. 测试数据集信息API
    print("\n3. 📊 测试数据集信息API...")
    try:
        response = requests.get(f"{base_url}/api/datasets")
        if response.status_code == 200:
            dataset_info = response.json()
            print(f"   ✅ 数据集状态: {dataset_info.get('status', '未知')}")
            print(f"      - 总样本数: {dataset_info.get('total_samples', 0)}")
            print(f"      - 训练样本: {dataset_info.get('train_samples', 0)}")
            print(f"      - 验证样本: {dataset_info.get('val_samples', 0)}")
            print(f"      - 测试样本: {dataset_info.get('test_samples', 0)}")
        else:
            print(f"   ❌ 数据集信息获取失败: {response.status_code}")
    except Exception as e:
        print(f"   ❌ 数据集信息API错误: {e}")
    
    # 4. 测试训练功能
    print("\n4. 🚀 测试模型训练功能...")
    try:
        training_data = {
            "model_type": "baseline",
            "epochs": 5,  # 短时间演示
            "learning_rate": 0.001,
            "batch_size": 32
        }
        
        response = requests.post(
            f"{base_url}/api/train",
            json=training_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            training_id = result['training_id']
            print(f"   ✅ 训练已启动，ID: {training_id[:8]}...")
            
            # 监控训练状态
            print("   📈 监控训练进度...")
            for i in range(10):  # 监控10次
                time.sleep(2)
                status_response = requests.get(f"{base_url}/api/training_status/{training_id}")
                if status_response.status_code == 200:
                    status = status_response.json()
                    progress = status.get('progress', 0)
                    current_epoch = status.get('current_epoch', 0)
                    total_epochs = status.get('total_epochs', 0)
                    train_acc = status.get('train_acc', 0)
                    val_acc = status.get('val_acc', 0)
                    
                    print(f"      Epoch {current_epoch}/{total_epochs} - "
                          f"进度: {progress:.1f}% - "
                          f"训练准确率: {train_acc:.3f} - "
                          f"验证准确率: {val_acc:.3f}")
                    
                    if status.get('status') == 'completed':
                        print("   🎉 训练完成!")
                        break
                else:
                    print(f"      ❌ 状态查询失败: {status_response.status_code}")
                    break
        else:
            print(f"   ❌ 训练启动失败: {response.status_code}")
    except Exception as e:
        print(f"   ❌ 训练功能错误: {e}")
    
    # 5. 测试仪表板数据
    print("\n5. 📊 测试仪表板数据...")
    try:
        response = requests.get(f"{base_url}/api/dashboard_data")
        if response.status_code == 200:
            dashboard_data = response.json()
            print("   ✅ 仪表板数据获取成功")
            print(f"      - 可用模型数: {dashboard_data.get('models_count', 0)}")
            print(f"      - 数据样本数: {dashboard_data.get('dataset_samples', 0)}")
            print(f"      - GPU可用: {'是' if dashboard_data.get('gpu_available') else '否'}")
        else:
            print(f"   ❌ 仪表板数据获取失败: {response.status_code}")
    except Exception as e:
        print(f"   ❌ 仪表板数据错误: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 功能演示完成!")
    print("💡 您可以在浏览器中访问 http://localhost:5000 体验完整功能")
    print("=" * 60)

if __name__ == "__main__":
    test_web_app() 