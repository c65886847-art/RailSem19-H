"""Compute depth map for abstract images with size correction"""
import os
import torch
import utils
import numpy as np
import cv2
from midas.model_loader import load_model

# 固定配置
MODEL_WEIGHTS = "/media/scdx/E2/wc/hazewc/weights/dpt_hybrid_384.pt"
MODEL_TYPE = "dpt_hybrid_384"

def process_abstract_image(input_path, output_path):
    """处理抽象风格图像"""
    # 验证输入文件
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    if not input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise ValueError("仅支持PNG/JPG/JPEG格式")

    # 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model, transform, _, _ = load_model(
        device,
        MODEL_WEIGHTS,
        MODEL_TYPE,
        optimize=False,
        height=None,
        square=False
    )

    # 读取图像并验证尺寸
    orig_image = utils.read_image(input_path)  # [0,1]范围
    if orig_image.shape != (1024, 1536, 3):
        raise ValueError(f"输入尺寸应为1024x1536x3，但得到的是{orig_image.shape}")

    # 生成深度图
    with torch.no_grad():
        input_tensor = transform({"image": orig_image})["image"]
        sample = torch.from_numpy(input_tensor).to(device).unsqueeze(0)
        prediction = model.forward(sample)

        # 尺寸修正处理
        depth_map = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(1024, 1536),
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    # 后处理
    depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
    depth_map = np.power(depth_map, 0.5)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存结果
    cv2.imwrite(output_path, (depth_map * 65535).astype(np.uint16), 
               [cv2.IMWRITE_PNG_COMPRESSION, 0])

if __name__ == "__main__":
    # 路径配置
    input_dir = "/media/scdx/E2/wc/hazewc/test/gt"
    output_dir = "/media/scdx/E2/wc/hazewc/depth"
    
    # 统计处理进度
    total_files = sum(1 for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    processed = 0
    
    # 处理目录中的所有图像
    for img_name in sorted(os.listdir(input_dir)):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, img_name)
            output_path = os.path.join(output_dir, f"depth_{img_name}")
            
            # 跳过已处理的文件
            if os.path.exists(output_path):
                print(f"[{processed}/{total_files}] 跳过已处理文件: {img_name}")
                processed += 1
                continue
                
            try:
                print(f"[{processed}/{total_files}] 正在处理: {img_name}")
                process_abstract_image(input_path, output_path)
                processed += 1
            except Exception as e:
                print(f"处理 {img_name} 失败: {str(e)}")
                # 失败时删除可能生成的不完整文件
                if os.path.exists(output_path):
                    os.remove(output_path)
    
    print(f"处理完成！共处理 {processed}/{total_files} 个文件")