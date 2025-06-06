#!/usr/bin/env python3
"""
植物叶片病害识别系统Web应用启动脚本
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 设置环境变量
os.environ['FLASK_APP'] = 'webapp.app'
os.environ['FLASK_ENV'] = 'development'

if __name__ == '__main__':
    from webapp.app import app
    
    print("=" * 60)
    print("🌱 植物叶片病害识别系统")
    print("=" * 60)
    print("🚀 启动Web应用...")
    print("📍 访问地址: http://localhost:5000")
    print("📍 仪表板: http://localhost:5000/dashboard")
    print("📍 病害识别: http://localhost:5000/predict")
    print("=" * 60)
    
    # 启动应用
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=True
    ) 