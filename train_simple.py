"""
简化但可靠的IP102训练脚本
运行: python train_simple.py
"""
from PIL import Image
import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

# 设置路径和参数
DATA_PATH = 'ip102_v1.1'
IMAGES_DIR = os.path.join(DATA_PATH, 'images')
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 102
EPOCHS = 5  # 先训练5个周期

def load_annotation_file(filename):
    """加载标注文件"""
    filepath = os.path.join(DATA_PATH, filename)
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                filename = parts[0]
                class_id = int(parts[1])
                filepath_img = os.path.join(IMAGES_DIR, filename)
                if os.path.exists(filepath_img):
                    data.append({'filename': filename, 'class_id': class_id, 'filepath': filepath_img})
    return pd.DataFrame(data)

def create_simple_model():
    """创建简化模型"""
    # 使用预训练的ResNet50V2
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False  # 冻结预训练层
    
    # 构建完整模型
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_and_preprocess_image(filepath, label):
    """加载和预处理单张图片"""
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img, label

def main():
    print("=" * 60)
    print("📊 IP102昆虫分类 - 简化训练脚本")
    print("=" * 60)
    
    # 1. 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"training_simple_{timestamp}"
    model_dir = os.path.join(save_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    print(f"✅ 保存目录: {save_dir}")
    
    # 2. 加载数据
    print("\n[1/4] 加载数据...")
    try:
        train_df = load_annotation_file('train.txt')
        val_df = load_annotation_file('val.txt')
        test_df = load_annotation_file('test.txt')
        
        print(f"✅ 数据加载成功:")
        print(f"   训练集: {len(train_df)} 张图片")
        print(f"   验证集: {len(val_df)} 张图片")
        print(f"   测试集: {len(test_df)} 张图片")
        
        # 取前1000张作为快速训练（完整训练可去掉这个限制）
        if len(train_df) > 1000:
            train_df = train_df.sample(1000, random_state=42)
            print(f"   使用前1000张图片进行快速训练")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 3. 准备数据管道
    print("\n[2/4] 准备数据管道...")
    
    # 转换数据为TensorFlow格式
    train_images = train_df['filepath'].values
    train_labels = tf.keras.utils.to_categorical(train_df['class_id'].values, NUM_CLASSES)
    
    val_images = val_df['filepath'].values[:200]  # 只用200张验证
    val_labels = tf.keras.utils.to_categorical(val_df['class_id'].values[:200], NUM_CLASSES)
    
    # 创建TensorFlow数据集
    def create_dataset(image_paths, labels, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        
        def load_wrapper(filepath, label):
            return load_and_preprocess_image(filepath, label)
        
        dataset = dataset.map(load_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    train_dataset = create_dataset(train_images, train_labels, shuffle=True)
    val_dataset = create_dataset(val_images, val_labels, shuffle=False)
    
    # 4. 构建模型
    print("\n[3/4] 构建模型...")
    try:
        model = create_simple_model()
        model.summary()
        print("✅ 模型构建成功")
    except Exception as e:
        print(f"❌ 模型构建失败: {e}")
        return
    
    # 5. 设置回调函数
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(save_dir, 'training_log.csv')
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # 6. 开始训练
    print("\n[4/4] 开始训练...")
    print(f"   训练周期: {EPOCHS}")
    print(f"   批次大小: {BATCH_SIZE}")
    print(f"   开始时间: {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 60)
    
    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        # 保存最终模型
        final_model_path = os.path.join(model_dir, 'final_model.h5')
        model.save(final_model_path)
        
        print("\n" + "=" * 60)
        print("🎉 训练成功完成!")
        print("=" * 60)
        print(f"   最终验证准确率: {history.history['val_accuracy'][-1]:.4f}")
        print(f"   最佳验证准确率: {max(history.history['val_accuracy']):.4f}")
        print(f"   模型保存位置: {model_dir}/")
        print(f"   1. 最佳模型: {model_dir}/best_model.h5")
        print(f"   2. 最终模型: {final_model_path}")
        print(f"   3. 训练日志: {save_dir}/training_log.csv")
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()    


    # 7. 新增：测试集评估（添加在训练完成后）
    # ##############################################
    print("\n[5/5] 测试集评估...")
    
    try:
        # 准备测试数据（和验证集类似）
        test_images = test_df['filepath'].values[:200]  # 用200张测试
        test_labels = tf.keras.utils.to_categorical(
            test_df['class_id'].values[:200], 
            NUM_CLASSES
        )
        
        # 创建测试数据集
        test_dataset = create_dataset(test_images, test_labels, shuffle=False)
        
        # 加载最佳模型进行评估
        best_model_path = os.path.join(model_dir, 'best_model.h5')
        if os.path.exists(best_model_path):
            best_model = tf.keras.models.load_model(best_model_path)
            
            # 评估模型
            test_loss, test_accuracy = best_model.evaluate(test_dataset, verbose=1)
            
            print(f"\n📊 测试集性能:")
            print(f"   测试损失: {test_loss:.4f}")
            print(f"   测试准确率: {test_accuracy:.4f}")
            
            # 详细评估（可选）
            print("\n📋 详细分析:")
            detailed_evaluation(best_model, test_df, num_samples=50)
            
        else:
            print("⚠️  最佳模型文件未找到")
            
    except Exception as e:
        print(f"⚠️  测试评估失败: {e}")
    






        # 8. 绘制训练曲线
        plot_training_history(history, save_dir)
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()





def plot_training_history(history, save_dir):
    """绘制训练历史"""
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 准确率
        ax1.plot(history.history['accuracy'], 'b-o', label='训练', markersize=4)
        ax1.plot(history.history['val_accuracy'], 'r-s', label='验证', markersize=4)
        ax1.set_title('模型准确率')
        ax1.set_xlabel('训练周期')
        ax1.set_ylabel('准确率')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 损失
        ax2.plot(history.history['loss'], 'b-o', label='训练', markersize=4)
        ax2.plot(history.history['val_loss'], 'r-s', label='验证', markersize=4)
        ax2.set_title('模型损失')
        ax2.set_xlabel('训练周期')
        ax2.set_ylabel('损失')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        plot_path = os.path.join(save_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"   4. 训练曲线: {plot_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"⚠️  绘制图表失败: {e}")

# 测试代码 - 验证数据加载
def test_data_loading():
    """测试数据加载是否正常"""
    print("🧪 测试数据加载...")
    
    # 测试文件是否存在
    if not os.path.exists(DATA_PATH):
        print(f"❌ 数据集路径不存在: {DATA_PATH}")
        return False
    
    if not os.path.exists(IMAGES_DIR):
        print(f"❌ 图片目录不存在: {IMAGES_DIR}")
        return False
    
    # 测试加载一张图片
    try:
        df = load_annotation_file('train.txt')
        if len(df) == 0:
            print("❌ 标注文件为空")
            return False
        
        # 测试第一张图片
        first_img = df.iloc[0]['filepath']
        if os.path.exists(first_img):
            print(f"✅ 数据加载测试通过!")
            print(f"   找到 {len(df)} 张训练图片")
            print(f"   示例图片: {first_img}")
            return True
        else:
            print(f"❌ 图片文件不存在: {first_img}")
            return False
            
    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        return False

# a) 增加测试集评估
def evaluate_model(model, test_df):
    """评估模型在测试集上的性能"""
    test_images = test_df['filepath'].values[:200]  # 取200张测试
    test_labels = tf.keras.utils.to_categorical(
        test_df['class_id'].values[:200], 
        NUM_CLASSES
    )
    
    test_dataset = create_dataset(test_images, test_labels, shuffle=False)
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    
    print(f"\n📊 测试集性能:")
    print(f"   测试损失: {test_loss:.4f}")
    print(f"   测试准确率: {test_accuracy:.4f}")
    
    return test_accuracy

# b) 增加模型性能分析
from sklearn.metrics import classification_report, confusion_matrix

def detailed_evaluation(model, test_df, num_samples=50):
    """详细评估模型"""
    from sklearn.metrics import classification_report
    
    # 随机选择一些测试样本
    sample_df = test_df.sample(min(num_samples, len(test_df)), random_state=42)
    
    predictions = []
    true_labels = []
    
    print(f"   正在分析 {len(sample_df)} 个样本...")
    
    for _, row in sample_df.iterrows():
        try:
            # 加载和预处理图片
            img = tf.io.read_file(row['filepath'])
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, IMG_SIZE)
            img = img / 255.0
            img = tf.expand_dims(img, axis=0)
            
            # 预测
            pred = model.predict(img, verbose=0)[0]
            pred_class = np.argmax(pred)
            
            predictions.append(pred_class)
            true_labels.append(row['class_id'])
            
        except Exception as e:
            continue
    
    if predictions:
        # 计算准确率
        correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
        accuracy = correct / len(predictions)
        
        print(f"   Top-1准确率: {accuracy:.2%}")
        print(f"   正确/总数: {correct}/{len(predictions)}")
        
        # 打印分类报告（简单版）
        unique_classes = set(true_labels)
        if len(unique_classes) <= 10:  # 只显示少量类别的报告
            print("\n   分类报告:")
            print(classification_report(true_labels, predictions, digits=3))



if __name__ == "__main__":
    # 先测试数据加载
    if test_data_loading():
        # 数据加载正常，开始训练
        main()
    else:
        print("\n⚠️  数据加载失败，请检查:")
        print(f"   1. 确保 '{DATA_PATH}' 文件夹存在")
        print(f"   2. 确保 '{IMAGES_DIR}' 中有图片文件")
        print(f"   3. 确保 '{DATA_PATH}' 中有 train.txt, val.txt, test.txt 文件")