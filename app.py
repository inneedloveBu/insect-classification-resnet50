"""
IP102 Insect Classification - Hugging Face Spaces Version
简单、干净的版本，专为Hugging Face部署设计
"""

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# 配置参数
MODEL_PATH = "models/best_model.h5"  # 确保模型文件在这个路径
IMG_SIZE = (224, 224)
NUM_CLASSES = 102

# 昆虫类别名称（示例，可以根据你的实际类别修改）
INSECT_CLASSES = {
    0: "蚂蚁 Ant",
    1: "蜜蜂 Bee",
    2: "蝴蝶 Butterfly",
    3: "甲虫 Beetle",
    4: "蜻蜓 Dragonfly",
    # ... 其他类别
}

# 加载模型
print("🔄 Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully: {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Model loading failed: {e}")
    model = None

def predict_insect(image):
    """预测昆虫类别"""
    if image is None:
        return {}
    
    try:
        # 预处理图片
        img = image.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        if model is not None:
            # 预测
            predictions = model.predict(img_array, verbose=0)[0]
            top_indices = np.argsort(predictions)[-5:][::-1]  # Top-5
            
            # 构建结果字典
            results = {}
            for idx in top_indices:
                confidence = float(predictions[idx])
                # 获取类别名称
                class_name = INSECT_CLASSES.get(idx, f"Insect Class {idx}")
                results[class_name] = confidence
            
            return results
            
    except Exception as e:
        print(f"Prediction error: {e}")
    
    # 如果失败，返回演示结果
    return {
        "Ant (Example)": 0.35,
        "Bee (Example)": 0.25,
        "Butterfly (Example)": 0.20,
        "Beetle (Example)": 0.15,
        "Dragonfly (Example)": 0.05
    }

def create_confidence_plot(predictions):
    """创建置信度图表（英文版）"""
    if not predictions:
        return None
    
    classes = list(predictions.keys())
    confidences = list(predictions.values())
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(classes)))
    
    bars = ax.barh(classes, confidences, color=colors, height=0.6)
    ax.set_xlabel('Confidence', fontsize=11, fontweight='bold')
    ax.set_title('Top-5 Predictions - Confidence Distribution', 
                fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(0, 1.05)
    
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
               bbox=dict(boxstyle="round,pad=0.3", 
                        facecolor="white", alpha=0.8, edgecolor="gray"))
    
    # 美化
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    
    return fig

# 创建Gradio界面
with gr.Blocks(theme=gr.themes.Soft(), title="IP102 Insect Classifier") as demo:
    gr.Markdown("# 🐛 IP102 Insect Classification System")
    gr.Markdown("A deep learning model for insect classification using ResNet50V2.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # 图片上传
            image_input = gr.Image(
                type="pil",
                label="📤 Upload Insect Image",
                sources=["upload", "clipboard"],
                height=300
            )
            
            # 按钮
            with gr.Row():
                predict_btn = gr.Button("🔍 Identify Insect", variant="primary")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
            
            # 模型信息
            with gr.Accordion("📊 Model Information", open=False):
                gr.Markdown("""
                **Model Architecture**: ResNet50V2  
                **Dataset**: IP102 (102 classes, ~75k images)  
                **Training Samples**: 1,000 images  
                **Purpose**: Demonstrate deep learning for insect classification
                """)
        
        with gr.Column(scale=2):
            # 结果显示
            label_output = gr.Label(
                label="🔍 Identification Results (Top-5)",
                num_top_classes=5,
                container=True
            )
            
            # 图表显示
            plot_output = gr.Plot(label="📈 Confidence Distribution")
    
    # 事件处理
    def process_image(image):
        predictions = predict_insect(image)
        plot = create_confidence_plot(predictions)
        return predictions, plot
    
    predict_btn.click(
        fn=process_image,
        inputs=image_input,
        outputs=[label_output, plot_output]
    )
    
    clear_btn.click(
        fn=lambda: (None, {}, None),
        inputs=[],
        outputs=[image_input, label_output, plot_output]
    )
    
    # 页脚
    gr.Markdown("---")
    gr.Markdown("""
    **Tech Stack**: TensorFlow, ResNet50V2, Gradio  
    **For QMUL Application**: Demonstrates Computer Vision & Machine Learning skills
    """)

# 启动应用
if __name__ == "__main__":
    demo.launch(share=False)