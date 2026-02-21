"""
IP102昆虫分类 - 最终优化训练脚本
运行: python train_advanced3.py
"""

import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
# 推荐：直接从 tensorflow.keras.applications 导入
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# ==================== 配置参数 ====================
DATA_PATH = 'ip102_v1.1'                # 数据集根目录
IMAGES_DIR = os.path.join(DATA_PATH, 'images')
IMG_SIZE = (224, 224)                   # 输入图像尺寸
BATCH_SIZE = 32                          # 批次大小（根据GPU显存调整）
NUM_CLASSES = 102                        # 昆虫类别数
USE_FULL_VAL = True                       # 是否使用完整验证集（True 建议）
EPOCHS_PHASE1 = 20                        # 第一阶段训练轮数（冻结基础层）

LR_PHASE1 = 5e-5  #1e-4                          # 第一阶段学习率
USE_LABEL_SMOOTHING = True                 # 是否使用标签平滑
LABEL_SMOOTHING = 0.1                      # 标签平滑因子
SAMPLE_SIZE = None  # 使用全部数据
EPOCHS_PHASE2 = 30  # 可适当增加

LR_PHASE2 = 2e-5    # 提高一点   # 第二阶段学习率

# ==================== 数据加载 ====================

def check_annotation_file(filename):
    filepath = os.path.join(DATA_PATH, filename)
    with open(filepath, 'r') as f:
        for i, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) != 2:
                print(f"行 {i}: 格式错误 -> {line.strip()}")
            else:
                try:
                    label = int(parts[1])
                    if label < 1 or label > 102:
                        print(f"行 {i}: 标签超出范围 {label} -> {line.strip()}")
                except:
                    print(f"行 {i}: 标签非数字 -> {line.strip()}")

def load_annotation_file(filename):
    """加载标注文件，返回包含 'filename', 'class_id', 'filepath' 的 DataFrame"""
    filepath = os.path.join(DATA_PATH, filename)
    data = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            parts = line.split()
            if len(parts) >= 2:
                fname = parts[0]
                try:
                    class_id = int(parts[1]) - 1
                except:
                    print(f"警告: 标签解析失败: {line}")
                    continue
                # 检查标签范围
                if class_id < 0 or class_id >= NUM_CLASSES:
                    print(f"警告: 标签 {class_id+1} 超出范围，跳过该行: {line}")
                    continue
                full_path = os.path.join(IMAGES_DIR, fname)
                if os.path.exists(full_path):
                    data.append({
                        'filename': fname,
                        'class_id': class_id,
                        'filepath': full_path
                    })
    return pd.DataFrame(data)

# ==================== 数据预处理与增强 ====================
def load_and_preprocess_image(filepath, label):
    """读取、解码、调整大小，并使用预训练模型要求的预处理"""
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)               # 关键：使用 ResNet50V2 的预处理
    return img, label

def create_data_augmentation():
    """训练数据增强策略"""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.1),
    ])

def create_dataset(image_paths, labels, shuffle=True, augment=False):
    """创建 tf.data.Dataset，支持数据增强"""
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(image_paths), 5000))
    if augment:
        aug = create_data_augmentation()
        ds = ds.map(lambda x, y: (aug(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# ==================== 模型构建 ====================
# def create_advanced_model():
#     """构建改进的模型（ResNet50V2 + 自定义分类头）"""
#     base_model = ResNet50V2(
#         include_top=False,
#         weights='imagenet',
#         input_shape=(224, 224, 3),
#         pooling='avg'                     # 全局平均池化
#     )
#     base_model.trainable = False           # 第一阶段冻结

#     # 在 create_advanced_model() 中，加载 base_model 后
#     print("Base model weights mean:", np.mean(base_model.get_weights()[0]))

#     inputs = tf.keras.Input(shape=(224, 224, 3))
#     x = base_model(inputs, training=False)
#     x = tf.keras.layers.Dropout(0.5)(x)
#     x = tf.keras.layers.Dense(1024, activation='relu')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Dropout(0.5)(x)
#     x = tf.keras.layers.Dense(512, activation='relu')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Dropout(0.3)(x)
#     outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

#     model = tf.keras.Model(inputs, outputs)
#     return model, base_model




def create_advanced_model():
    """构建改进的模型（简化分类头）"""
    base_model = ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)  # 只保留一层全连接
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model




# ==================== 训练曲线绘制 ====================
def plot_training_history(history, save_dir):
    """绘制并保存训练曲线"""
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        # 准确率
        ax1.plot(history['accuracy'], 'b-o', label='Training', markersize=4)
        ax1.plot(history['val_accuracy'], 'r-s', label='Validation', markersize=4)
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # 损失
        ax2.plot(history['loss'], 'b-o', label='Training', markersize=4)
        ax2.plot(history['val_loss'], 'r-s', label='Validation', markersize=4)
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"   训练曲线已保存至: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"   绘制曲线失败: {e}")

# ==================== 主训练流程 ====================
def main():

    print("检查训练集...")
    check_annotation_file('train.txt')
    print("检查验证集...")
    check_annotation_file('val.txt')
    print("=" * 70)
    print("🏋️  IP102 昆虫分类 - 终极优化训练")
    print(f"   训练集采样: {SAMPLE_SIZE if SAMPLE_SIZE else '全部'} 张")
    print(f"   验证集: {'完整' if USE_FULL_VAL else '采样'}")
    print(f"   第一阶段轮数: {EPOCHS_PHASE1}, 学习率: {LR_PHASE1}")
    print(f"   第二阶段轮数: {EPOCHS_PHASE2}, 学习率: {LR_PHASE2}")
    print("=" * 70)

    # 1. 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"training_advanced_{timestamp}"
    model_dir = os.path.join(save_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    print(f"✅ 保存目录: {save_dir}")

    # 2. 加载数据
    print("\n[1/5] 加载数据...")
    try:
        train_df = load_annotation_file('train.txt')
        val_df = load_annotation_file('val.txt')
        print(f"   完整训练集: {len(train_df):,} 张")
        print(f"   完整验证集: {len(val_df):,} 张")

        # 训练集采样（可选）
        # if SAMPLE_SIZE and len(train_df) > SAMPLE_SIZE:
        #     # 尝试保持类别平衡的采样
        #     samples_per_class = SAMPLE_SIZE // NUM_CLASSES
        #     sampled = []
        #     for cls in range(NUM_CLASSES):
        #         cls_df = train_df[train_df['class_id'] == cls]

        #         print("Class ID range:", train_df['class_id'].min(), "-", train_df['class_id'].max())


        #         if len(cls_df) > 0:
        #             n = min(len(cls_df), max(1, samples_per_class))
        #             sampled.append(cls_df.sample(n, random_state=42))
        #     train_df = pd.concat(sampled, ignore_index=True)
        #     print(f"   采样后训练集: {len(train_df):,} 张 (平衡采样)")

        # 验证集处理
        if not USE_FULL_VAL and len(val_df) > 3000:
            val_df = val_df.sample(3000, random_state=42)
        print(f"   最终验证集: {len(val_df):,} 张")

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 3. 计算类别权重（处理不平衡）
    print("\n[2/5] 计算类别权重...")
    classes = train_df['class_id'].values
    class_weights = compute_class_weight('balanced',
                                         classes=np.unique(classes),
                                         y=classes)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"   类别权重范围: [{min(class_weights):.3f}, {max(class_weights):.3f}]")

    # 4. 准备数据管道
    print("\n[3/5] 准备数据管道...")
    train_images = train_df['filepath'].values
    train_labels = tf.keras.utils.to_categorical(train_df['class_id'].values, NUM_CLASSES)
    val_images = val_df['filepath'].values
    val_labels = tf.keras.utils.to_categorical(val_df['class_id'].values, NUM_CLASSES)

    train_dataset = create_dataset(train_images, train_labels,
                                   shuffle=True, augment=False)# ##################################################
    val_dataset = create_dataset(val_images, val_labels,
                                 shuffle=False, augment=False)
    

    # 在创建数据集后，立即检查一个 batch
    for images, labels in train_dataset.take(1):
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        print("Labels sample:", labels[0].numpy())  # 应该是 one-hot 向量
        # 可以显示一张图（可选）
        import matplotlib.pyplot as plt
        plt.imshow(images[0].numpy() * 0.5 + 0.5)  # 因为 preprocess_input 将像素范围变为 [-1,1] 左右，需要反标准化才能正确显示
        plt.title("Sample image")
        plt.show()










    # 5. 构建模型
    print("\n[4/5] 构建模型...")
    model, base_model = create_advanced_model()
    model.summary()

    # 6. 设置回调
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(save_dir, 'training_log.csv')
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # 7. 第一阶段训练（冻结基础层）
    print("\n[5/5] 第一阶段训练 (冻结基础层)...")
    loss_fn = (tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
               if USE_LABEL_SMOOTHING else 'categorical_crossentropy')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_PHASE1),
        loss=loss_fn,
        metrics=['accuracy']
    )

    history1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        # epochs=EPOCHS_PHASE1,
        epochs=10,
        # class_weight=class_weight_dict,###################################################
        callbacks=callbacks,
        verbose=1
    )

    # 8. 第二阶段微调
    print("\n🔧 第二阶段：微调部分层")
    base_model.trainable = True
    # 只微调最后30层
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_PHASE2),
        loss=loss_fn,
        metrics=['accuracy']
    )

    history2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS_PHASE2,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # 9. 保存最终模型
    final_model_path = os.path.join(model_dir, 'final_model.h5')
    model.save(final_model_path)
    print(f"✅ 最终模型已保存: {final_model_path}")

    # 10. 输出结果与绘图
    print("\n" + "=" * 70)
    print("🎉 训练完成！")
    print("=" * 70)

    # 合并训练历史
    history = {
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
        'loss': history1.history['loss'] + history2.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss']
    }
    final_val_acc = history['val_accuracy'][-1]
    best_val_acc = max(history['val_accuracy'])

    print(f"   最终验证准确率: {final_val_acc:.2%}")
    print(f"   最佳验证准确率: {best_val_acc:.2%}")
    print(f"   模型保存位置: {model_dir}/")

    plot_training_history(history, save_dir)
    print(f"✅ 所有输出保存在: {save_dir}")

if __name__ == "__main__":
    # 设置 TensorFlow 日志级别（可选）
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()