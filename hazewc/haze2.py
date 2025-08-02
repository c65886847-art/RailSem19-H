import cv2
import numpy as np
import os
from skimage.exposure import adjust_gamma
from itertools import product
import traceback

# 固定参数组合
# 城市雾参数范围（适度浓度+灰白基调）
BETA_LIST = [0.6, 0.6, 0.7, 1]  # 消光系数（浓度）
A_LIST = [0.92, 0.95, 0.98]      # 散射反照率（灰度）

# 按真实物理特性排序：浓度优先，反照率负向加权
all_combinations = list(product(BETA_LIST, A_LIST))
sorted_combinations = sorted(
    all_combinations,
    key=lambda x: (x[0] * 4 - x[1]),  # 提升浓度权重，降低高反照率收益
    reverse=True
)

# 优化后的暗调雾效参数（按视觉权重排序）
# 更合理的参数组合（物理真实 + 视觉多样性）
PARAM_COMBINATIONS = [ # 已将 'd' 重命名为 PARAM_COMBINATIONS 以提高可读性
    (0.6, 0.5),     # 薄暮雾（低浓度 + 高反照率，冷色调）
    (0.7, 0.5),     # 薄雾（低浓度 + 高反照率）
    (0.6, 0.52),    # 城市雾（中等浓度）
    (0.8, 0.90),    # 浓雾（中等浓度 + 略低反照率）
    (0.9, 0.85),    # 雾层过渡（中等偏高浓度 + 较低反照率）
    (0.9, 0.80),    # 强浓雾（高浓度 + 较低反照率）
    (1.0, 0.75),    # 极端浓雾（非常高浓度 + 低反照率）
    (1.0, 0.70),    # 沙尘暴（极高浓度 + 极低反照率）
    (0.8, 0.55),    # 雾中光束（中等浓度 + 丁达尔效应）
    (0.9, 0.83)     # 晨雾（低反照率，轻微散射）
]
# 其他参数范围
OTHER_PARAMS = {
    'gamma': (0.7, 1.3),
    'noise': (0.02, 0.05)
}

def generate_fog_effect(rgb_path, depth_path, output_path, beta, A):
    """生成单张雾效图像"""
    try:
        # 验证输入文件
        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"RGB图像不存在: {rgb_path}")
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"深度图不存在: {depth_path}")

        # 读取图像
        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

        if rgb is None:
            raise ValueError(f"无法读取RGB图像: {rgb_path}")
        if depth is None:
            raise ValueError(f"无法读取深度图: {depth_path}")

        # 转换到[0,1]范围
        rgb = rgb.astype(np.float32) / 255.0
        depth = depth.astype(np.float32) / 255.0

        # 深度图处理
        depth = cv2.GaussianBlur(depth, (7,7), 2.0)
        depth = 1.0 - np.power(depth, 1.5)

        # 生成随机参数（固定种子）
        # 使用文件名哈希值作为 gamma/noise 的种子，以确保对于给定文件的一致性
        np.random.seed(abs(hash(os.path.basename(rgb_path))) % (2**32))
        gamma = np.random.uniform(*OTHER_PARAMS['gamma'])
        noise_level = np.random.uniform(*OTHER_PARAMS['noise'])

        # 物理雾效模型
        transmission = np.exp(-beta * depth)
        atmosphere = A * np.array([0.89, 0.93, 1.0])  # 冷色调

        # 合成雾效
        foggy = rgb * transmission[..., np.newaxis] + atmosphere * (1 - transmission[..., np.newaxis])
        foggy = np.clip(foggy, 0, 1)

        # 后处理
        foggy = adjust_gamma(foggy, gamma)
        foggy += np.random.normal(scale=noise_level, size=foggy.shape)
        foggy = np.clip(foggy, 0, 1)

        # 保存结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not cv2.imwrite(output_path, (foggy * 255).astype(np.uint8)):
            raise IOError(f"无法保存图像到: {output_path}")

        return True

    except Exception as e:
        print(f"处理失败: {str(e)}")
        traceback.print_exc()
        return False

def batch_process():
    # 路径配置（根据实际结构调整）
    root = "/media/scdx/E2/wc/hazewc"
    input_dir = os.path.join(root, "test/gt")       # 原始图像目录
    depth_dir = os.path.join(root, "depth")         # 深度图目录
    output_dir = os.path.join(root, "fog_2")        # 雾效输出目录

    # 验证目录
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    if not os.path.exists(depth_dir):
        raise FileNotFoundError(f"深度图目录不存在: {depth_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # 处理统计
    success_count = 0
    fail_count = 0

    # 处理每张图片
    for img_file in sorted(os.listdir(input_dir)):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # 提取基础文件名（如rs00001）
        base_name = os.path.splitext(img_file)[0]

        # 构建深度图路径（depth_rs00001.png）
        depth_file = f"depth_{base_name}.png"
        rgb_path = os.path.join(input_dir, img_file)
        depth_path = os.path.join(depth_dir, depth_file)

        # 检查深度图是否存在
        if not os.path.exists(depth_path):
            print(f"[警告] 缺失深度图: {depth_path}")
            fail_count += 1
            continue

        # 为每张图片随机选择一个雾效参数组合
        # 为了确保相同图片每次运行都能得到相同的随机结果（可复现性），
        # 我们使用图片文件名作为随机种子。如果您希望每次运行都生成完全不同的雾图，
        # 可以移除下方设置种子的行。
        np.random.seed(abs(hash(base_name)) % (2**32 - 1)) # 使用图片名作为种子，确保每张图片的参数选择是固定的随机值

        # 从 PARAM_COMBINATIONS 中随机选择一个参数组合的索引
        selected_combination_index = np.random.choice(len(PARAM_COMBINATIONS))

        # 获取选定的 beta 和 A 值
        beta, A = PARAM_COMBINATIONS[selected_combination_index]

        # 构建输出文件名，包含选定的参数组合索引，方便识别
        output_file = f"{base_name}_fog_{selected_combination_index}.png"
        output_path = os.path.join(output_dir, output_file)

        if generate_fog_effect(rgb_path, depth_path, output_path, beta, A):
            print(f"[成功] {img_file} -> {output_file} (β={beta:.2f}, A={A:.2f})")
            success_count += 1
        else:
            print(f"[失败] {img_file} (参数组合 {selected_combination_index})")
            fail_count += 1

    # 打印最终报告
    print("\n" + "="*50)
    print(f"处理完成！成功: {success_count} 张，失败: {fail_count} 张")
    if fail_count > 0:
        print("\n失败可能原因：")
        print("1. 深度图命名不符合'depth_rsXXXXX.png'格式")
        print("2. 图像文件损坏或权限不足")
        print("3. 内存不足（尝试减小PARAM_COMBINATIONS数量）")

if __name__ == "__main__":
    batch_process()