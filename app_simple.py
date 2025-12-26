"""
IP102昆虫识别系统 - 修复版Web应用
运行: python app_final.py
"""
# ==================== 1. 修复中文字体问题 ====================
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import os

def setup_chinese_font():
    """设置中文字体，确保图表能正常显示中文"""
    try:
        # Windows系统常见中文字体路径
        font_candidates = [
            'C:/Windows/Fonts/msyh.ttc',      # 微软雅黑
            'C:/Windows/Fonts/simhei.ttf',    # 黑体
            'C:/Windows/Fonts/simsun.ttc',    # 宋体
            'C:/Windows/Fonts/simkai.ttf',    # 楷体
            'C:/Windows/Fonts/msjh.ttc',      # 微軟正黑體
        ]
        
        # 查找并添加第一个可用的中文字体
        added_font = None
        for font_path in font_candidates:
            if os.path.exists(font_path):
                try:
                    # 直接添加到字体管理器
                    fm.fontManager.addfont(font_path)
                    font_prop = fm.FontProperties(fname=font_path)
                    added_font = font_prop.get_name()
                    print(f"✅ 成功添加字体: {added_font} ({font_path})")
                    break
                except Exception as e:
                    print(f"⚠️  添加字体失败 {font_path}: {e}")
        
        if added_font:
            # 设置Matplotlib使用这个字体
            matplotlib.rcParams['font.sans-serif'] = [added_font, 'DejaVu Sans', 'Arial']
            matplotlib.rcParams['axes.unicode_minus'] = False
            print(f"📝 当前字体设置: {matplotlib.rcParams['font.sans-serif'][0]}")
            return True
        else:
            print("⚠️  未找到系统中文字体，使用默认英文字体")
            matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            return False
            
    except Exception as e:
        print(f"❌ 字体设置出错: {e}")
        # 确保至少使用默认字体
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        return False

# 立即执行字体设置
print("🔄 正在配置中文字体...")
font_setup_success = setup_chinese_font()
print(f"字体配置状态: {'✅ 成功' if font_setup_success else '⚠️ 使用英文后备方案'}")
print("-" * 50)

# ==================== 2. 导入其他库 ====================
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# ==================== 3. 配置参数和加载模型 ====================
MODEL_PATH = "training_simple_20251226_164303/models/best_model.h5"
IMG_SIZE = (224, 224)
NUM_CLASSES = 102

# 加载模型
print("🔄 正在加载训练好的模型...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ 成功加载模型: {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ 模型加载失败: {e}")
    print("⚠️ 将使用演示模式（随机结果）")
    model = None

# ==================== 4. 核心功能函数 ====================
def predict_insect(image):
    """使用训练好的模型进行预测"""
    if image is None:
        return {}
    
    # 1. 预处理图片（与训练时完全一致）
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # 添加批次维度
    
    # 2. 使用模型预测
    if model is not None:
        try:
            predictions = model.predict(img_array, verbose=0)[0]
            top_indices = np.argsort(predictions)[-5:][::-1]  # 获取Top-5
            
            # 创建类别名称（真实项目中应有具体昆虫名称）
            class_names = [f"昆虫类别_{i:03d}" for i in range(NUM_CLASSES)]
            
            results = {}
            for idx in top_indices:
                confidence = float(predictions[idx])
                class_name = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
                results[class_name] = confidence
            
            return results
        except Exception as e:
            print(f"预测出错: {e}")
            return {"预测错误": 1.0}
    
    # 3. 如果模型不可用，返回演示结果
    return {
        "蚂蚁 (示例)": 0.35, 
        "蜜蜂 (示例)": 0.25, 
        "蝴蝶 (示例)": 0.20, 
        "甲虫 (示例)": 0.15, 
        "蜻蜓 (示例)": 0.05
    }

def create_confidence_plot(predictions):
    """创建置信度分布图"""
    if not predictions:
        return None
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(9, 5))
    
    classes = list(predictions.keys())
    confidences = list(predictions.values())
    
    # 使用漂亮的渐变色
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(classes)))
    
    bars = ax.barh(classes, confidences, color=colors, height=0.6)
    ax.set_xlabel('置信度', fontsize=11, fontweight='bold')
    ax.set_title('Top-5 预测结果 - 置信度分布', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(0, 1.05)  # 留出一点空间显示标签
    
    # 添加网格
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # 在条形上添加数值
    for bar, conf in zip(bars, confidences):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
               f'{conf:.1%}', 
               ha='left', va='center', 
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"))
    
    # 添加背景色
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    # 调整布局
    plt.tight_layout()
    
    return fig

def analyze_training_log():
    """分析训练日志文件"""
    log_file = "training_simple_20251226_164303/training_log.csv"
    if os.path.exists(log_file):
        try:
            df = pd.read_csv(log_file)
            info = f"""
**训练分析报告**
- 训练周期: {len(df)} 个
- 最佳验证准确率: {df['val_accuracy'].max():.2%}
- 最终验证准确率: {df['val_accuracy'].iloc[-1]:.2%}
- 最佳模型周期: {df['val_accuracy'].idxmax() + 1}
"""
            return info
        except Exception as e:
            print(f"分析训练日志时出错: {e}")
            return "训练日志可用，但分析时遇到问题。"
    return "未找到训练日志文件。"

# ==================== 5. 创建Gradio界面 ====================
def create_gradio_interface():
    """创建Gradio界面"""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🐛 IP102昆虫智能识别系统")
        gr.Markdown("基于深度学习的昆虫分类模型，使用ResNet50V2架构在IP102数据集上训练。")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 图片上传区域
                image_input = gr.Image(
                    type="pil",
                    label="📤 上传昆虫图片",
                    sources=["upload", "clipboard"],
                    height=300
                )
                
                # 控制按钮
                with gr.Row():
                    predict_btn = gr.Button("🔍 识别昆虫", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ 清空", variant="secondary")
                
                # 训练信息
                with gr.Accordion("📊 训练详情", open=False):
                    training_info = analyze_training_log()
                    gr.Markdown(training_info)
                    gr.Markdown(f"**模型路径:** `{MODEL_PATH}`")
                
                # 使用说明
                with gr.Accordion("📖 使用说明", open=False):
                    gr.Markdown("""
                    1. **上传图片**: 点击上传区域或拖拽昆虫图片
                    2. **开始识别**: 点击"识别昆虫"按钮
                    3. **查看结果**: 右侧将显示Top-5识别结果和置信度
                    
                    **支持的格式**: JPG, PNG, BMP
                    **最佳效果**: 确保昆虫主体清晰、居中
                    """)
            
            with gr.Column(scale=2):
                # 识别结果
                label_output = gr.Label(
                    label="🔍 识别结果 (Top-5)",
                    num_top_classes=5,
                    container=True
                )
                
                # 置信度图表
                plot_output = gr.Plot(label="📈 置信度分布")
        
        # 事件处理
        def process_image(image):
            """处理图片并返回预测结果"""
            predictions = predict_insect(image)
            plot = create_confidence_plot(predictions)
            return predictions, plot
        
        predict_btn.click(
            fn=process_image,
            inputs=image_input,
            outputs=[label_output, plot_output]
        )
        
        clear_btn.click(
            fn=lambda: (None, None, None),
            inputs=[],
            outputs=[image_input, label_output, plot_output]
        )
        
        # 页脚
        gr.Markdown("---")
        gr.Markdown("**技术栈**: TensorFlow, ResNet50V2, Gradio  |  **数据集**: IP102 (102类, 75,222张图片)")
        
        return demo

# ==================== 6. 主程序 ====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 IP102昆虫识别系统启动中...")
    print("="*60)
    
    # 创建示例图片目录（如果不存在）
    example_dir = "examples"
    os.makedirs(example_dir, exist_ok=True)
    
    # 创建一些示例图片
    example_files = []
    for i, name in enumerate(["example1.jpg", "example2.jpg", "example3.jpg"]):
        path = os.path.join(example_dir, name)
        if not os.path.exists(path):
            # 创建简单的示例图片
            img_array = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(img_array).save(path)
        example_files.append([path])
    
    print(f"📁 创建了 {len(example_files)} 个示例图片")
    print(f"🌐 请在浏览器中打开: http://127.0.0.1:7860")
    print("🛑 按 Ctrl+C 停止应用")
    print("="*60)
    
    # 创建并启动应用
    demo = create_gradio_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )