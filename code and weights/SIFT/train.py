#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from batch2 import EnhancedTransformReuseStitcher

def main():
    """主函数演示顺序拼接功能"""
    
    # 1. 创建拼接器实例
    print("=== EnhancedTransformReuseStitcher 顺序拼接示例 ===\n")
    
    stitcher = EnhancedTransformReuseStitcher(feature_detector='SIFT')
    print("已创建拼接器，使用 SIFT 特征检测器\n")
    
    # 2. 设置路径
    image_folder = "F:/dect/tezhengdian/image"
    result_folder = "F:/dect/tezhengdian/result"
    os.makedirs(result_folder, exist_ok=True)
    
    output_path = os.path.join(result_folder, "sequential_panorama.jpg")
    intermediate_folder = os.path.join(result_folder, "intermediate_steps")
    transforms_save_path = os.path.join(result_folder, "sequential_transforms.json")
    
    print(f"输入文件夹: {image_folder}")
    print(f"输出文件: {output_path}")
    print(f"变换参数文件: {transforms_save_path}\n")
    
    # 3. 检查输入文件夹是否存在
    if not os.path.exists(image_folder):
        print(f"错误: 输入文件夹 '{image_folder}' 不存在")
        return
    
    # 4. 分析输入图像
    print("--- 分析输入图像 ---")
    try:
        analysis = stitcher.analyze_folder_images(image_folder)
        print(f"共找到 {analysis['total_images']} 张图像")
        
        if analysis['total_images'] < 2:
            print("错误: 需要至少2张图像才能进行拼接")
            return
    except Exception as e:
        print(f"分析图像时出错: {e}")
        return
    
    # 5. 执行顺序拼接（包含后处理）
    print("\n--- 执行顺序拼接并保存变换参数 ---")
    try:
        result_with_transforms = stitcher.stitch_multiple_images_sequential(
            image_folder=image_folder,
            output_path=output_path,
            save_intermediate=True,
            intermediate_folder=intermediate_folder,
            save_transforms=True,
            transforms_save_path=transforms_save_path,
            # 新增后处理参数
        
        )
        
        if result_with_transforms is not None:
            print(f"✓ 拼接完成，结果尺寸: {result_with_transforms.shape}")
            print(f"✓ 变换参数已保存到: {transforms_save_path}")
        else:
            print("✗ 拼接失败")
            return
    except Exception as e:
        print(f"拼接时出错: {e}")
        return
    
    # 6. 演示不同的后处理方法
    print("\n--- 演示不同的后处理方法 ---")
    if result_with_transforms is not None:
        demonstrate_post_processing_methods(stitcher, result_with_transforms, result_folder)
    
    # 7. 使用保存的变换参数处理另一个文件夹的图像
    print("\n--- 使用变换参数处理新文件夹图像 ---")
    new_image_folder = "F:/dect/tezhengdian/keshi"
    applied_transforms_folder = os.path.join(result_folder, "applied_transforms")
    
    if os.path.exists(new_image_folder):
        try:
            new_analysis = stitcher.analyze_folder_images(new_image_folder)
            print(f"新文件夹中找到 {new_analysis['total_images']} 张图像")
            
            if new_analysis['total_images'] >= analysis['total_images']:
                successful_files = stitcher.apply_sequential_transforms_to_folder(
                    source_folder=new_image_folder,
                    transforms_file=transforms_save_path,
                    output_folder=applied_transforms_folder
                )
                
                print(f"\n✓ 变换参数应用完成!")
                print(f"✓ 成功处理了 {len(successful_files)} 个文件")
                
            else:
                print(f"警告: 新文件夹中的图像数量不足")
        except Exception as e:
            print(f"应用变换参数时出错: {e}")
    else:
        print(f"新图像文件夹 '{new_image_folder}' 不存在")
    
    print(f"\n=== 全部完成 ===")

def demonstrate_post_processing_methods(stitcher, original_image, result_folder):
    """演示不同的后处理方法"""
    methods_folder = os.path.join(result_folder, "post_processing_methods")
    os.makedirs(methods_folder, exist_ok=True)
    
    # 定义不同的处理方法和参数
    processing_configs = [
        {
            'name': '16:9_居中裁剪',
            'target_aspect_ratio': 16/9,
            'method': 'crop_center',
            'filename': 'panorama_16_9_crop.jpg'
        },
        {
            'name': '4:3_居中裁剪', 
            'target_aspect_ratio': 4/3,
            'method': 'crop_center',
            'filename': 'panorama_4_3_crop.jpg'
        },
        {
            'name': '16:9_等比缩放',
            'target_aspect_ratio': 16/9,
            'method': 'resize_fit',
            'max_width': 1920,
            'max_height': 1080,
            'filename': 'panorama_16_9_fit.jpg'
        },
        {
            'name': '16:9_拉伸填充',
            'target_aspect_ratio': 16/9,
            'method': 'resize_fill',
            'max_width': 1920,
            'max_height': 1080,
            'filename': 'panorama_16_9_fill.jpg'
        },
        {
            'name': '16:9_填充',
            'target_aspect_ratio': 16/9,
            'method': 'pad',
            'filename': 'panorama_16_9_pad.jpg'
        }
    ]
    
    print("正在生成不同后处理方法的示例...")
    
    for config in processing_configs:
        print(f"\n处理方法: {config['name']}")
        save_path = os.path.join(methods_folder, config['filename'])
        
        try:
            processed = stitcher.normalize_panorama_aspect_ratio(
                original_image,
                target_aspect_ratio=config['target_aspect_ratio'],
                method=config['method'],
                max_width=config.get('max_width'),
                max_height=config.get('max_height')
            )
            
            # 先自动裁剪黑边
            processed = stitcher.auto_crop_black_borders(processed)
            
            # 保存结果
            cv2.imwrite(save_path, processed)
            print(f"✓ 已保存: {config['filename']}")
            
        except Exception as e:
            print(f"✗ 处理失败: {e}")
    
    print(f"\n所有后处理示例已保存到: {methods_folder}")

def manual_post_process_existing_image():
    """手动对已存在的全景图进行后处理"""
    print("\n" + "="*50)
    print("=== 手动后处理现有全景图 ===")
    print("="*50)
    
    stitcher = EnhancedTransformReuseStitcher()
    
    # 输入和输出路径
    input_image_path = "F:/dect/tezhengdian/result/sequential_panorama.jpg"
    output_folder = "F:/dect/tezhengdian/result/manual_processed"
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"输入图像: {input_image_path}")
    print(f"输出文件夹: {output_folder}")
    
    if not os.path.exists(input_image_path):
        print("输入图像不存在，请先运行主函数生成全景图")
        return
    
    # 读取图像
    image = cv2.imread(input_image_path)
    if image is None:
        print("无法读取输入图像")
        return
    
    print(f"原始图像尺寸: {image.shape[1]}x{image.shape[0]}")
    
    # 不同的后处理配置
    configs = [
        ("16_9_HD", 16/9, 'crop_center', 1920, 1080),
        ("4_3_standard", 4/3, 'crop_center', 1600, 1200),
        ("21_9_ultrawide", 21/9, 'crop_center', 2560, 1080),
        ("square", 1/1, 'crop_center', 1080, 1080),
        ("original_cropped", None, 'crop_center', None, None)  # 只裁剪黑边
    ]
    
    for name, aspect_ratio, method, max_w, max_h in configs:
        print(f"\n处理配置: {name}")
        output_path = os.path.join(output_folder, f"panorama_{name}.jpg")
        
        if aspect_ratio is None:
            # 只裁剪黑边
            processed = stitcher.auto_crop_black_borders(image)
        else:
            # 完整后处理
            processed = stitcher.process_panorama_complete(
                image,
                auto_crop_borders=True,
                normalize_aspect=True,
                target_aspect_ratio=aspect_ratio,
                method=method,
                max_width=max_w,
                max_height=max_h,
                save_path=output_path
            )
    
    print(f"\n手动后处理完成，结果保存在: {output_folder}")

if __name__ == "__main__":   
    print("\n" + "="*50 + "\n")
    
    # 运行主示例
    main()
    
    # 可选：手动后处理现有图像
    # manual_post_process_existing_image()